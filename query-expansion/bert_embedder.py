import math
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import numpy as np
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)


class BertEmbedder:
    """
    This class converts texts to BERT vectors
    """
    def __init__(self, model_path, max_seq_length=128):
        print(f'Init BERT, max_seq_length={max_seq_length}')
        self.max_seq_length = max_seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        # Load pre-trained model (weights)
        # output_hidden_states: Whether the model returns all hidden-states.
        self.model = BertModel.from_pretrained(model_path, output_hidden_states=True)
        device_count = torch.cuda.device_count()
        print(f'Using {device_count} GPUs!')
        if device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model.eval()
        self.model.to(self.device)

    def preprocess(self, sentences):
        # list of token indices per sentence
        input_ids_list = []
        # list of segment mask ids per sentence
        token_type_ids_list = []
        # list of lengths of meaningful tokens per sentence
        meaningful_lengths = []
        for s in sentences:
            result = self.tokenizer.encode_plus(s)
            input_ids = result['input_ids']
            token_type_ids = result['token_type_ids']
            mlen = len(input_ids)
            # cut or pad
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[: self.max_seq_length]
                token_type_ids = token_type_ids[: self.max_seq_length]
                mlen = len(input_ids)
            elif len(input_ids) < self.max_seq_length:
                diff = self.max_seq_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * diff
                token_type_ids += [self.tokenizer.pad_token_type_id] * diff

            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            meaningful_lengths.append(mlen)

        tokens_tensors = torch.tensor(input_ids_list).to(self.device)
        segments_tensors = torch.tensor(token_type_ids_list).to(self.device)
        return tokens_tensors, segments_tensors, meaningful_lengths

    def get_raw_outputs(self, sentences):
        tokens_tensors, segments_tensors, meaningful_lengths = self.preprocess(sentences)
        with torch.no_grad():
            outputs = self.model(tokens_tensors, segments_tensors)
            return outputs, meaningful_lengths

    def embed_v1(self, sentences):
        outputs, _ = self.get_raw_outputs(sentences)
        # hidden states from all layers
        # (#bert layers, #sentences, #max_seq_length, #hidden_units)
        # in the case of bert-base models, this shape would be:
        # (13, #sentences, #max_seq_length, 768)
        hidden_states = outputs[2]
        # we take the last layer's output to compose sentence embeddings
        vecs = []
        for i, sent_vecs in enumerate(hidden_states[-1]):
            # approach 1: average over all tokens per sentence
            vec = torch.mean(sent_vecs, dim=0)
            vecs.append(vec.cpu().numpy())
        return vecs

    def embed_v2(self, sentences):
        outputs, meaningful_lengths = self.get_raw_outputs(sentences)
        hidden_states = outputs[2]
        # we take the last layer's output to compose sentence embeddings
        vecs = []
        for i, sent_vecs in enumerate(hidden_states[-1]):
            mlen = meaningful_lengths[i]
            sent_vecs = sent_vecs.cpu().numpy()
            # approach 2: only average over meaningful tokens per sentence
            sent_vecs_meaningful = [sent_vecs[i] for i in range(mlen)]
            vec = np.average(sent_vecs_meaningful, axis=0)
            vecs.append(vec)
        return vecs

    def embed_v3(self, sentences):
        outputs, _ = self.get_raw_outputs(sentences)
        hidden_states = outputs[2]
        vecs = []
        for i, sent_vecs in enumerate(hidden_states[-1]):
            sent_vecs = sent_vecs.cpu().numpy()
            # approach 3: use only the cls token vector
            print('rows', len(vecs), ', cols: ', len(vecs[0]))
            vec = sent_vecs[0]
            vecs.append(vec)
        return vecs
