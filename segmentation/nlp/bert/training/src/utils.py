from __future__ import absolute_import, division, print_function

import os
import logging
import torch
from transformers import BertForTokenClassification
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import moxing.pytorch as mox
mox.file.shift('os', 'mox')

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
from segmt_eval import evaluate_ner

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForNer(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def readfile(filename):
    """
    read BIO file
    """
    data = []
    sentence = []
    label = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if (not line) or (line.startswith('-DOCSTART')):
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1])

        if len(sentence) > 0:
            data.append((sentence, label))
    return data


class NerProcessor:
    """Processor for the CoNLL-2003 data set."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self, train_file_name):
        return self._create_examples(train_file_name, 'train')

    def get_dev_examples(self, dev_file_name):
        return self._create_examples(dev_file_name, 'dev')

    def get_test_examples(self, test_file_name):
        return self._create_examples(test_file_name, 'test')

    def get_labels(self, labels_file_name):
        labels_file = os.path.join(self.data_dir, labels_file_name)
        if os.path.isfile(labels_file):
            labels = ['O']
            with open(labels_file, 'r') as f:
                for line in f:
                    label = line.strip()
                    if label not in labels:
                        labels.append(label)
            extra_labels = ['[CLS]', '[SEP]']
            labels += extra_labels
        else:
            labels = ['O', 'B-MISC', 'I-MISC',  'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', '[CLS]', '[SEP]']
        return labels

    def _create_examples(self, file_name, set_type):
        file_path = os.path.join(self.data_dir, file_name)
        data = readfile(file_path)
        examples = []
        for i, (sentence, label) in enumerate(data):
            guid = '%s-%s' % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append('[CLS]')
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map['[CLS]'])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append('[SEP]')
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map['[SEP]'])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('tokens: %s' % ' '.join(
                    [str(x) for x in tokens]))
            logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
            logger.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
            logger.info(
                    'segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
            # logger.info('label: %s (id = %d)' % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


def get_data_loader(features, batch_size, randomize=True, local_rank=-1):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_valid_ids, all_lmask_ids)
    if randomize:
        if local_rank == -1:
            sampler = RandomSampler(tensor_data)
        else:
            sampler = DistributedSampler(tensor_data)
    else:
        sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(tensor_data, sampler=sampler, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, label_list, device, get_ner_metrics=False):
    model.eval()
    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(dataloader,
                                                                                 desc='Evaluating'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            if get_ner_metrics:
                logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)
            else:
                return model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)

        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
    types_set = set()
    for label in label_list:
        if label not in ('O', '[SEP]', '[CLS]'):
            _, ent_type = label.split('-')
            types_set.add(ent_type)
    results, results_per_label = evaluate_ner(y_true, y_pred, list(types_set))
    return results, results_per_label


def train(model, dataloader, num_train_epochs, gradient_accumulation_steps, n_gpu,
          fp16, optimizer, scheduler, max_grad_norm, device):
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')
    # summary_writer = SummaryWriter()
    global_step = 0
    model.train()
    for i in trange(int(num_train_epochs), desc='Epoch'):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(dataloader, desc='Iteration')):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                # summary_writer.add_scalar('Train/loss', loss.item(), global_step)
                global_step += 1
        logger.info(f'Epoch {i} finished, total loss: {tr_loss}')
    return model
