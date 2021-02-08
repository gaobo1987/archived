import os
import torch
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from config_qa import ConfigQA


class BertQA:
    def __int__(self, config_path):
        self.config = ConfigQA(config_path)
        self.logger = self.config.logger
        self.writer = SummaryWriter()
        # Load pretrained model and tokenizer
        self.model_config = BertConfig.from_pretrained(self.config.model_path)
        self.model_tokenizer = BertTokenizer.from_pretrained(self.config.model_path,
                                                             do_lower_case=self.config.do_lower_case)
        self.model = BertForQuestionAnswering.from_pretrained(self.config.model_path,
                                                              config=self.model_config)
        self.model.to(self.config.device)

    def train(self, train_dataset, model, tokenizer):
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.config.train_batch_size)
        t_total = len(train_dataloader) // self.config.gradient_accumulation_steps * self.config.num_train_epochs
        # Prepare optimizer and schedule (linear warm-up and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=t_total
        )
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.config.model_path, "optimizer.pt")) and \
                os.path.isfile(os.path.join(self.config.model_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.config.model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.config.model_path, "scheduler.pt")))

        # multi-gpu training (should be after apex fp16 initialization)
        if self.config.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataset))
        self.logger.info("  Num Epochs = %d", self.config.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.config.per_gpu_train_batch_size)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.config.train_batch_size
            * self.config.gradient_accumulation_steps,
        )
        self.logger.info("  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.config.model_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = self.config.model_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // self.config.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // self.config.gradient_accumulation_steps)

                self.logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                self.logger.info("  Continuing training from epoch %d", epochs_trained)
                self.logger.info("  Continuing training from global step %d", global_step)
                self.logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(epochs_trained, int(self.config.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(self.config.device) for t in batch)
                inputs = {
                    # input token indices corresponding to positions in vocab
                    "input_ids": batch[0],
                    # optional, used when batching input sequences, binary numbers showing if a token is real or padding
                    "attention_mask": batch[1],
                    # segmentation ids: separation of two sequences: question, context
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }
                outputs = model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]
                if self.config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if self.config.evaluate_during_training:
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                self.writer.add_scalar("eval_{}".format(key), value, global_step)
                        self.writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        self.writer.add_scalar("loss", (tr_loss - logging_loss) / self.config.logging_steps, global_step)
                        logging_loss = tr_loss

                    # Save model checkpoint
                    if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                        output_dir = os.path.join(self.config.output_dir, "checkpoint-{}".format(global_step))
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(self.config.to_dict(), os.path.join(output_dir, "training_args.bin"))
                        self.logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        self.logger.info("Saving optimizer and scheduler states to %s", output_dir)

        self.writer.close()
        # Save the trained model and the tokenizer
        self.save()

        return global_step, tr_loss / global_step

    def save(self):
        self.logger.info("Saving model checkpoint to %s", self.config.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model.save_pretrained(self.config.output_dir)
        self.model_tokenizer.save_pretrained(self.config.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.config.to_dict(), os.path.join(self.config.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        # model = BertForQuestionAnswering.from_pretrained(self.config.output_dir)  # , force_download=True)
        # tokenizer = BertTokenizer.from_pretrained(self.config.output_dir, do_lower_case=self.config.do_lower_case)
        # model.to(self.config.device)

    def evaluate(args, model, tokenizer, prefix=""):
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                    del inputs["token_type_ids"]

                feature_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                        )
                outputs = model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        # XLNet and XLM use a more complex post-processing procedure
        if args.model_type in ["xlnet", "xlm"]:
            start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
            end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

            predictions = compute_predictions_log_probs(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                start_n_top,
                end_n_top,
                args.version_2_with_negative,
                tokenizer,
                args.verbose_logging,
            )
        else:
            predictions = compute_predictions_logits(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                args.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                args.verbose_logging,
                args.version_2_with_negative,
                args.null_score_diff_threshold,
                tokenizer,
            )

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        return results
