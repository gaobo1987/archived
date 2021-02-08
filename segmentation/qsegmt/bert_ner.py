"""BERT NER Inference."""

import json
import os
import torch
import torch.nn.functional as F
from transformers import BertForTokenClassification, BertTokenizer


class BertNerModel(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32,
                                   device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        return logits


class BertNer:

    def __init__(self, model_dir: str):
        model_config_path = os.path.join(model_dir, 'model_config.json')
        with open(model_config_path, 'r') as f:
            self.model_config = json.load(f)
        self.model = BertNerModel.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir,
                                                       do_lower_case=self.model_config['do_lower'])
        self.label_map = self.model_config['label_map']
        self.max_seq_length = self.model_config['max_seq_length']
        self.label_map = {int(k): v for k, v in self.label_map.items()}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def entity_labels(self):
        entity_labels = []
        for entity_label in self.model_config['label_map'].values():
            if entity_label not in ('[CLS]', '[SEP]'):
                entity_labels.append(entity_label)
        return entity_labels

    def subword_tokenize(self, words: list):
        """subword tokenize input"""
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for j in range(len(token)):
                if j == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, words: list):
        """ preprocess """
        tokens, valid_positions = self.subword_tokenize(words)
        # insert '[CLS]'
        tokens.insert(0, '[CLS]')
        valid_positions.insert(0, 1)
        # insert '[SEP]'
        tokens.append('[SEP]')
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids, input_mask, segment_ids, valid_positions

    def ner(self, words: list):
        input_ids, input_mask, segment_ids, valid_ids = self.preprocess(words)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        valid_ids = torch.tensor([valid_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask, valid_ids)
        logits = F.softmax(logits, dim=2)
        logits_label = torch.argmax(logits, dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values, label in zip(logits[0], logits_label)]

        logits = []
        pos = 0
        for index, mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos], logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()
        labels = []
        for label, _ in logits:
            labels.append(self.label_map[label])
        assert len(labels) == len(words)
        return labels


