import os
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset, RandomSampler
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from nlp.bert.utils import InputExample, InputFeatures, \
    read_ner_examples_from_file, strip_accents, strip_accents_and_lower_case


class BertModel:
    BERT_TYPE = 'bert'
    TOKENIZER_ARGS = {
        'do_lower_case': False,
        'strip_accents': True,
        'keep_accents': False,
        'use_fast': False
    }
    PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 1
    ADAM_EPSILON = 1e-8
    LEARNING_RATE = 5e-5
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, model_path,
                 bert_type=BERT_TYPE,
                 tokenizer_args=None,
                 pad_token_label_id=PAD_TOKEN_LABEL_ID,
                 max_seq_length=MAX_SEQ_LENGTH,
                 batch_size=BATCH_SIZE,
                 num_epochs=NUM_EPOCHS,
                 adam_epsilon=ADAM_EPSILON,
                 learning_rate=LEARNING_RATE,
                 gradient_accumulation_step=GRADIENT_ACCUMULATION_STEPS,
                 max_grad_norm=MAX_GRAD_NORM,
                 device=DEVICE):
        self.model_path = model_path
        self._set_entity_labels()
        self.device = device
        self.bert_type = bert_type
        self.tokenizer_args = BertModel.TOKENIZER_ARGS.copy()
        if tokenizer_args is not None:
            self.tokenizer_args.update(tokenizer_args)
        self.pad_token_label_id = pad_token_label_id
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.gradient_accumulation_step = gradient_accumulation_step
        self.max_grad_norm = max_grad_norm
        self.config = AutoConfig.from_pretrained(
            model_path,
            num_labels=len(self.entity_labels),
            id2label={str(i): label for i, label in enumerate(self.entity_labels)},
            label2id={label: i for i, label in enumerate(self.entity_labels)},
            cache_dir=None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=None,
            **self.tokenizer_args)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            config=self.config,
            cache_dir=None,
        )
        self.model.to(device)
        print('========== BERT model ==========')
        print('model_path: ', self.model_path)
        print('tokenizer_args: ', self.tokenizer_args)
        print('entity_labels: ', self.entity_labels)
        print('bert_type: ', self.bert_type)
        print('num_epochs: ', self.num_epochs)
        print('batch_size: ', self.batch_size)
        print('max_seq_length: ', self.max_seq_length)
        print('learning_rate: ', self.learning_rate)
        print('adam_epsilon: ', self.adam_epsilon)
        print('device: ', self.device)

    # Note that it is important that the bert model folder contains the labels.txt,
    # based on which this model was trained/fine-tuned.
    # The sequence of the labels in this file matters
    def _set_entity_labels(self):
        label_path = os.path.join(self.model_path, 'labels.txt')
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                self.entity_labels = f.read().splitlines()
        else:
            print('Could not locate the labels of the model.')
            raise

    def _convert_examples_to_features(self,
                                      examples,
                                      cls_token_at_end=False,
                                      cls_token="[CLS]",
                                      cls_token_segment_id=1,
                                      sep_token="[SEP]",
                                      sep_token_extra=False,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      sequence_a_segment_id=0,
                                      mask_padding_with_zero=True,
                                      ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(self.entity_labels)}

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens = []
            label_ids = []
            for word, label in zip(example.words, example.labels):
                word_tokens = self.tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend([label_map[label]] + [self.pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            # special_tokens_count = tokenizer.num_special_tokens_to_add() # new function name
            # special_tokens_count = tokenizer.num_added_tokens() # old function name
            special_tokens_count = self.tokenizer.num_added_tokens() \
                if hasattr(self.tokenizer, 'num_added_tokens') else self.tokenizer.num_special_tokens_to_add()
            if len(tokens) > self.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [self.pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [self.pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [self.pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [self.pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([self.pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [self.pad_token_label_id] * padding_length

            # if len(input_ids) > max_seq_length:
            #     input_ids = input_ids[:max_seq_length]
            # if len(input_mask) > max_seq_length:
            #     input_mask = input_mask[:max_seq_length]
            # if len(segment_ids) > max_seq_length:
            #     segment_ids = segment_ids[:max_seq_length]
            # if len(label_ids) > max_seq_length:
            #     label_ids = label_ids[:max_seq_length]

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
            )
        return features

    def _create_input_tensors(self, examples):
        features = self._convert_examples_to_features(
            examples,
            cls_token_at_end=bool(self.bert_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.bert_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=bool(self.bert_type in ["roberta1"]),  # disable robert sep_token???
            # roberta uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(self.bert_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id
        )
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset

    def is_lower_case(self):
        return 'do_lower_case' in self.tokenizer_args and self.tokenizer_args['do_lower_case']

    def is_strip_accents(self):
        return 'strip_accents' in self.tokenizer_args and self.tokenizer_args['strip_accents']

    def _create_data_loader(self, data_path):
        examples = read_ner_examples_from_file(data_path,
                                               do_lower_case=self.is_lower_case(),
                                               do_strip_accents=self.is_strip_accents())
        dataset = self._create_input_tensors(examples)
        data_sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=self.batch_size)
        print(f'{len(examples)} examples loaded')
        return data_loader

    def train(self, data_path, output_path):
        summary_writer = SummaryWriter()
        train_dataloader = self._create_data_loader(data_path)
        dataloader_size = len(train_dataloader)
        t_total = dataloader_size // self.num_epochs if dataloader_size > 2000 else dataloader_size
        print('Dataloder size:', len(train_dataloader))
        print('num training steps: ', t_total)
        save_steps = len(train_dataloader)

        # Prepare optimizer and schedule (linear warmup and decay)
        print('...prepare optimizer and scheduler')
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.model_path, "optimizer.pt")) and \
                os.path.isfile(os.path.join(self.model_path, "scheduler.pt")):
            print('...use saved optimizer and scheduler')
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.model_path, "scheduler.pt")))

        # Train!
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        total_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(epochs_trained, int(self.num_epochs), desc="Epoch", disable=False)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.bert_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.bert_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids

                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                loss.backward()

                total_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    summary_writer.add_scalar('Train/loss', loss.item(), global_step)
                    summary_writer.add_scalar('Train/avg_loss', total_loss/global_step, global_step)
                    if save_steps > 0 and global_step % save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(output_path, f'{self.bert_type}-checkpoint-{global_step}')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        self.model.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)
                        print("Saving model checkpoint to %s", output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        print("Saving optimizer and scheduler states to %s", output_dir)
                        w = open(os.path.join(output_dir, 'labels.txt'), 'w')
                        for e in self.entity_labels:
                            w.write(e+'\n')
                        w.close()
                        print(f'Saving entity labels to {output_dir}.')

        return global_step, total_loss / global_step

    def ner(self, words):
        if self.is_lower_case() and self.is_strip_accents():
            words = [strip_accents_and_lower_case(w) for w in words]
        elif self.is_lower_case():
            words = [w.lower() for w in words]
        elif self.is_strip_accents():
            words = [strip_accents(w) for w in words]

        placeholder_labels = ['O' for _ in words]
        input_examples = [InputExample(guid='0', words=words, labels=placeholder_labels)]
        input_tensors = self._create_input_tensors(input_examples)
        sampler = SequentialSampler(input_tensors)
        dataloader = DataLoader(input_tensors, sampler=sampler, batch_size=self.batch_size)
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.bert_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.bert_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = self.model(**inputs)
                _, logits = outputs[:2]

            # shape: 32 (batch_size) x 128 (max_seq_len) x 9 (#labels)
            logits_np = logits.detach().cpu().numpy()
            if preds is None:
                preds = logits_np
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits_np, axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=2)
        label_map = {i: label for i, label in enumerate(self.entity_labels)}
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list[0]
