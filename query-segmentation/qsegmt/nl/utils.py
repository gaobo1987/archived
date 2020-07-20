import os
import torch
import numpy as np
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data import TensorDataset
from qsegmt import logger
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer
)
from qsegmt.nl.params import BERT_LABELS, BERT_TOKENIZER_ARGS, BATCH_SIZE, \
    DEVICE, BERT_MODEL_TYPE, MAX_SEQ_LENGTH, PAD_TOKEN_LABEL_ID


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        # special_tokens_count = tokenizer.num_special_tokens_to_add() # new function name
        # special_tokens_count = tokenizer.num_added_tokens() # old function name
        special_tokens_count = tokenizer.num_added_tokens() \
            if hasattr(tokenizer, 'num_added_tokens') else tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

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
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length


        # if len(input_ids) > max_seq_length:
        #     input_ids = input_ids[:max_seq_length]
        # if len(input_mask) > max_seq_length:
        #     input_mask = input_mask[:max_seq_length]
        # if len(segment_ids) > max_seq_length:
        #     segment_ids = segment_ids[:max_seq_length]
        # if len(label_ids) > max_seq_length:
        #     label_ids = label_ids[:max_seq_length]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )
    return features


def init_model(model_path):
    logger.info('init config...')
    num_labels = len(BERT_LABELS)
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(BERT_LABELS)},
        label2id={label: i for i, label in enumerate(BERT_LABELS)},
        cache_dir=None,
    )
    logger.info('init tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        **BERT_TOKENIZER_ARGS)
    logger.info('init model...')
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config,
        cache_dir=None,
    )
    model.to(DEVICE)
    return model, tokenizer, config


def create_input_tensors(examples, tokenizer):
    logger.info("Creating input tensors... ")
    features = convert_examples_to_features(
        examples,
        BERT_LABELS,
        MAX_SEQ_LENGTH,
        tokenizer,
        cls_token_at_end=bool(BERT_MODEL_TYPE in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if BERT_MODEL_TYPE in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(BERT_MODEL_TYPE in ["roberta1"]),  # disable robert sep_token???
        # roberta uses an extra separator b/w pairs of sentences,
        # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(BERT_MODEL_TYPE in ["xlnet"]),
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


def bert_ner(bert_model, bert_tokenizer, words, placeholder_labels):
    input_examples = [InputExample(guid='0', words=words, labels=placeholder_labels)]
    input_tensors = create_input_tensors(input_examples, bert_tokenizer)
    sampler = SequentialSampler(input_tensors)
    dataloader = DataLoader(input_tensors, sampler=sampler, batch_size=BATCH_SIZE)
    preds = None
    out_label_ids = None
    bert_model.eval()
    for batch in dataloader:
        batch = tuple(t.to(DEVICE) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if BERT_MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if BERT_MODEL_TYPE in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = bert_model(**inputs)
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
    label_map = {i: label for i, label in enumerate(BERT_LABELS)}
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != PAD_TOKEN_LABEL_ID:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list[0]


# combine predictions, e.g.
# input: ['O', 'O', 'B-LOC', 'O', 'B-PER', 'B-PER', 'I-PER', 'O', 'O', 'O', 'I-ORG', 'I-PER', 'I-PER', 'B-PER', 'O']
# output:
# [[(2, 'B-LOC')],
#  [(4, 'B-PER')],
#  [(5, 'B-PER'), (6, 'I-PER')],
#  [(10, 'I-ORG')],
#  [(11, 'I-PER'), (12, 'I-PER')],
#  [(13, 'B-PER')]]
def merge_bert_ner_predictions(preds_list):
    neg_label = 'O'
    pos_labels = BERT_LABELS.copy()
    pos_labels.remove(neg_label)
    whole_words = []
    word = []
    prev_label = preds_list[0] if len(preds_list[0]) < 3 else preds_list[0][2:]
    for i, label in enumerate(preds_list):
        if label in pos_labels:
            time_to_wrap_up = False
            if label[0] == 'B' and len(word) > 0:
                time_to_wrap_up = True
            elif label[2:] != prev_label and len(word) > 0:
                time_to_wrap_up = True

            if time_to_wrap_up:
                whole_words.append(word.copy())
                word = [(i, label)]
            else:
                word.append((i, label))
        elif len(word) > 0:
            whole_words.append(word.copy())
            word = []

        prev_label = label if len(label) < 3 else label[2:]

    return whole_words


def exist_in_output(item_text, output):
    match = None
    for item in output:
        if item['item'] == item_text:
            match = item
            break

    return match
