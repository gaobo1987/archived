import spacy
from .utils import init_model, create_input_tensors, InputExample, bert_ner, \
    merge_bert_ner_predictions, exist_in_output
from .. import Segmenter
from .params import BATCH_SIZE, DEVICE, BERT_MODEL_TYPE, BERT_LABELS, \
    PAD_TOKEN_LABEL_ID, BERT_MODEL_PATH, SPACY_MODEL_PATH

is_from_user_dict_default = 'false'
# load the spaCy dutch nlp model
nlp = spacy.load(SPACY_MODEL_PATH)
# load the bert dutch ner model
bert_model, bert_tokenizer, _ = init_model(BERT_MODEL_PATH)


# Segmentation for NL
class NLSegmenter(Segmenter):
    def search_segment(self, input_string):
        doc = nlp(input_string)
        output = []
        idx_item_map = {}
        words = []
        placeholder_labels = []
        count = 0
        for token in doc:
            length = token.__len__()
            item = {
                'item': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'startOffset': token.idx,
                'endOffset': token.idx + length,
                'ner': '',
                # 'ent_type': token.ent_type_, # spaCy entity type
                # 'ent_iob': token.ent_iob_, # spaCy entity iob/bio tag
                'isFromUserDict': is_from_user_dict_default
            }
            output.append(item)
            idx_item_map[count] = item
            count += 1
            words.append(token.text)
            placeholder_labels.append('O')

        # add NER entries
        preds_list = bert_ner(bert_model, bert_tokenizer, words, placeholder_labels)
        whole_words = merge_bert_ner_predictions(preds_list)
        for whole_word in whole_words:
            size = len(whole_word)
            start_item = idx_item_map[whole_word[0][0]]
            start_offset = start_item['startOffset']
            end_item = idx_item_map[whole_word[size-1][0]]
            end_offset = end_item['endOffset']
            ner_category = whole_word[size-1][1][2:]
            text = ''
            lemma = ''
            pos = end_item['pos']
            for idx, _ in whole_word:
                item = idx_item_map[idx]
                text += item['item'] + ' '
                lemma += item['lemma'] + ' '
            text = text.rstrip()
            lemma = lemma.rstrip()
            match = exist_in_output(text, output)
            if match is None:
                ner_item = {
                    'item': text,
                    'lemma': lemma,
                    'pos': pos,
                    'startOffset': start_offset,
                    'endOffset': end_offset,
                    'ner': [{
                        'ner': ner_category
                    }],
                    'isFromUserDict': is_from_user_dict_default
                }
                output.append(ner_item)
            else:
                match['ner'] = [{
                    'ner': ner_category
                }]

        return output

    def get_instance(self):
        return NLSegmenter()

    def index_segment(self, input_string):
        pass



