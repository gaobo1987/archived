import sys; sys.path.insert(0, '..')
import os
import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from ner.bert.utils import convert_examples_to_features, read_examples_from_file, InputExample
from ner.bert.params import LABELS, TOKENIZER_ARGS, NUM_EPOCHS, \
    DATA_DIR, MAX_SEQ_LENGTH, MAX_GRAD_NORM, OUTPUT_DIR, MODEL_TYPE, \
    BATCH_SIZE, LEARNING_RATE, ADAM_EPSILON, MODEL_NAME_OR_PATH, \
    GRADIENT_ACCUMULATION_STEPS, MODEL_NAME, DATA_NAME, \
    DEVICE, PAD_TOKEN_LABEL_ID


def init_model(model_path):
    print('init config...')
    num_labels = len(LABELS)
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
        cache_dir=None,
    )
    print('init tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        **TOKENIZER_ARGS)
    print('init model...')
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config,
        cache_dir=None,
    )
    model.to(DEVICE)
    return config, tokenizer, model


def create_input_tensors(examples, tokenizer):
    print("Creating input tensors... ")
    features = convert_examples_to_features(
        examples,
        LABELS,
        MAX_SEQ_LENGTH,
        tokenizer,
        cls_token_at_end=bool(MODEL_TYPE in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if MODEL_TYPE in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(MODEL_TYPE in ["roberta1"]),  # disable robert sep_token???
        # roberta uses an extra separator b/w pairs of sentences,
        # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(MODEL_TYPE in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_type_id,
        pad_token_label_id=PAD_TOKEN_LABEL_ID,
    )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def train(model_path):
    _config, tokenizer, model = init_model(model_path)
    examples = read_examples_from_file(DATA_DIR, 'train')
    train_dataset = create_input_tensors(examples, tokenizer)
    print('...start training')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
    dataloader_size = len(train_dataloader)
    t_total = dataloader_size // NUM_EPOCHS if dataloader_size > 2000 else dataloader_size
    print('#examples: ', len(examples))
    print('Dataloder size:', len(train_dataloader))
    print('num training steps: ', t_total)
    SAVE_STEPS = len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    print('...prepare optimizer and scheduler')
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPSILON)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(MODEL_NAME_OR_PATH, "optimizer.pt")) and \
            os.path.isfile(os.path.join(MODEL_NAME_OR_PATH, "scheduler.pt")):
        print('...use saved optimizer and scheduler')
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(MODEL_NAME_OR_PATH, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(MODEL_NAME_OR_PATH, "scheduler.pt")))

    # Train!
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(NUM_EPOCHS), desc="Epoch", disable=False)
    count = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if MODEL_TYPE in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if SAVE_STEPS > 0 and global_step % SAVE_STEPS == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}-{DATA_NAME}-checkpoint-{global_step}')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    print("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    print("Saving optimizer and scheduler states to %s", output_dir)

    return global_step, tr_loss / global_step


def evaluate(model_path, mode='test'):
    # mode = 'dev' or 'test'
    _, tokenizer, model = init_model(model_path)
    examples = read_examples_from_file(DATA_DIR, mode)
    eval_dataset = create_input_tensors(examples, tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=BATCH_SIZE)

    print("***** Running evaluation *****")
    print("  Num examples = ", len(eval_dataset))
    print("  Batch size = ", BATCH_SIZE)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if MODEL_TYPE in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        # shape: 32 (batch_size) x 128 (max_seq_len) x 9 (#labels)
        logits_np = logits.detach().cpu().numpy()
        if preds is None:
            preds = logits_np
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits_np, axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)
    label_map = {i: label for i, label in enumerate(LABELS)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != PAD_TOKEN_LABEL_ID:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    print("***** Eval results *****")
    for key in sorted(results.keys()):
        print(key, str(results[key]))

    return results, preds_list


def do_training():
    print('========================= Training =======================================')
    _global_step, _tr_loss = train(MODEL_NAME_OR_PATH)
    print(f'training finished, global_step={_global_step}, tr_loss={_tr_loss}')


def do_evaluation():
    print('========================= Evaluating =======================================')
    _checkpoints = sorted(os.listdir(OUTPUT_DIR))
    for c in _checkpoints:
        if DATA_NAME in c:
            print("Evaluate checkpoint: ", c)
            c_dir = os.path.join(OUTPUT_DIR, c)
            evaluate(c_dir, 'test')


def predict(input_examples, model_path):
    print('#examples:', len(input_examples))
    _, tokenizer, model = init_model(model_path)
    input_tensors = create_input_tensors(input_examples, tokenizer)
    sampler = SequentialSampler(input_tensors)
    dataloader = DataLoader(input_tensors, sampler=sampler, batch_size=BATCH_SIZE)
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if MODEL_TYPE in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
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
    label_map = {i: label for i, label in enumerate(LABELS)}
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != PAD_TOKEN_LABEL_ID:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list


def do_prediction():
    print('========================= Predicting =======================================')
    checkpoints = sorted(os.listdir(OUTPUT_DIR))
    for c in checkpoints:
        print("use checkpoint: ", c)
        c_dir = os.path.join(OUTPUT_DIR, c)
        sentence = 'Een bomaanslag op een trein in de noordoostelijke Indiase deelstaat Assam heeft aan minstens twaalf mensen het leven gekost.'
        sentence = 'Allereerst dank ik de aanwezigen voor hun inbreng en hun komst naar de Tweede Kamer.'
        words = sentence.split(' ')
        labels = ['O' for _ in words]
        input_examples = []
        input_examples.append(InputExample(guid='foobar', words=words, labels=labels))
        predictions = predict(input_examples, c_dir)
        print(words)
        print(predictions)


if __name__ == "__main__":
    do_training()
    do_evaluation()
    # do_prediction()