from nlp.bert.bert_model import BertModel
from spacy.tokens import Doc, Span, Token


def create_spacy_bert_ner_component(model_path, model_type):
    model = BertModel(model_path, bert_type=model_type)
    if not Token.has_extension('ent_label'):
        Token.set_extension("ent_label", default=True)
    if not Doc.has_extension('ents'):
        Doc.set_extension("ents", default=True)

    def ner_component(doc):
        words = [t.text for t in doc]
        preds_list = model.ner(words)
        for i, t in enumerate(doc):
            if i < len(preds_list):
                t._.ent_label = preds_list[i]
            else:
                t._.ent_label = 'O'
        whole_words = _merge_bert_ner_predictions(preds_list, model.entity_labels)
        ents = []
        for w in whole_words:
            size = len(w)
            start_index = w[0][0]
            end_index = w[size - 1][0] + 1
            ent = Span(doc, start=start_index, end=end_index, label=w[0][1][2:])
            ents.append(ent)

        doc._.ents = ents
        return doc

    return ner_component


# combine predictions, e.g.
# input: ['O', 'O', 'B-LOC', 'O', 'B-PER', 'B-PER', 'I-PER', 'O', 'O', 'O', 'I-ORG', 'I-PER', 'I-PER', 'B-PER', 'O']
# output:
# [[(2, 'B-LOC')],
#  [(4, 'B-PER')],
#  [(5, 'B-PER'), (6, 'I-PER')],
#  [(10, 'I-ORG')],
#  [(11, 'I-PER'), (12, 'I-PER')],
#  [(13, 'B-PER')]]
def _merge_bert_ner_predictions(preds_list, entity_labels):
    neg_label = 'O'
    pos_labels = entity_labels.copy()
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
