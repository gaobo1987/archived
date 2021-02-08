import json
from qsegmt.utils import strip_accents_and_lower_case, exist_in_output


class Segmenter:
    # online segmentation
    def segment(self, input_string):
        raise NotImplementedError


# Remark: the spacy_model is language specific
def do_segmentation(spacy_model, input_string, use_custom_ner=False, label_map={}):
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
            lemma_key: strip_accents_and_lower_case(token.lemma_),
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
        if label_map:
            ent_label = label_map[ent.label_]
        else:
            ent_label = ent.label_
        match = exist_in_output(ent.text, output)
        if match is None:
            output.append({
                item_text_key: ent.text,
                lemma_key: None,
                pos_key: None,
                start_offset_key: ent.start_char,
                end_offset_key: ent.end_char,
                ner_key: [{
                    ner_key: ent_label
                }],
                is_minimum_token_key: not is_minimum_token_val,
                is_stop_word_key: False
            })
        else:
            match[ner_key] = [{
                ner_key: ent_label
            }]

    return json.dumps(output, ensure_ascii=False)
