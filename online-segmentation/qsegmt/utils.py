import json
import spacy
from nlp.ner.utils import create_spacy_bert_ner_component


def load_spacy_bert_model(spacy_model_path, bert_model_path, bert_model_type):
    model = spacy.load(spacy_model_path, disable=['ner'])
    ner_component = create_spacy_bert_ner_component(bert_model_path, bert_model_type)
    model.add_pipe(ner_component, name="bert_ner", last=True)
    return model


def load_spacy_model(spacy_model_path):
    return spacy.load(spacy_model_path)


# Remark: the spacy_model is language specific
def do_segmentation(spacy_model, input_string, use_custom_ner=True):
    # default keys and values
    item_text_key = 'item'
    lemma_key = 'lemma'
    pos_key = 'pos'
    start_offset_key = 'startOffSet'
    end_offset_key = 'endOffSet'
    ner_key = 'ner'
    is_minimum_token_key = 'isMinimumToken'
    is_minimum_token_val = True
    is_stop_word_key = 'isStopWord'
    # create doc
    doc = spacy_model(input_string)
    output = []
    count = 0
    for token in doc:
        length = token.__len__()
        item = {
            item_text_key: token.text,
            lemma_key: token.lemma_,
            pos_key: token.pos_,
            start_offset_key: token.idx,
            end_offset_key: token.idx + length,
            ner_key: '',
            is_minimum_token_key: is_minimum_token_val,
            is_stop_word_key: token.is_stop
        }
        output.append(item)
        count += 1

    # add NER entries
    ents = doc._.ents if use_custom_ner else doc.ents
    for ent in ents:
        match = exist_in_output(ent.text, output)
        if match is None:
            output.append({
                item_text_key: ent.text,
                lemma_key: None,
                pos_key: None,
                start_offset_key: ent.start_char,
                end_offset_key: ent.end_char,
                ner_key: [{
                    ner_key: ent.label_
                }],
                is_minimum_token_key: not is_minimum_token_val,
                is_stop_word_key: False
            })
        else:
            match[ner_key] = [{
                ner_key: ent.label_
            }]

    return json.dumps(output)


def exist_in_output(item_text, output):
    match = None
    for item in output:
        if item['item'] == item_text:
            match = item
            break

    return match
