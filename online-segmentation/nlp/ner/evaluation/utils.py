import spacy
import stanza
import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
from nlp.bert.bert_model import BertModel
from nlp.ner.evaluation.params import CONFIG


def load_model(model_name, model_path):
    print('...loading model', model_name)
    nlp = None
    ner = None
    if 'spacy_default' == model_name:
        nlp = spacy.load(model_path, disable=['tagger', 'parser', 'textcat'])
        ner = spacy_default_ner
    elif 'spacy_custom_blank' == model_name or 'spacy_custom_default' == model_name:
        nlp = spacy.load(model_path)
        ner = spacy_custom_ner
    elif 'stanza_default' == model_name:
        nlp = stanza.Pipeline(lang=model_path, processors='tokenize,mwt,ner', use_gpu=True)
        ner = stanza_default_ner
    elif 'bert' in model_name and 'uncased' in model_name:
        tokenizer_args = {'do_lower_case': True}
        nlp = BertModel(model_path, bert_type='bert', tokenizer_args=tokenizer_args)
        ner = nlp.ner
    elif 'bert' in model_name and 'cased' in model_name:
        tokenizer_args = {'do_lower_case': False}
        nlp = BertModel(model_path, bert_type='bert', tokenizer_args=tokenizer_args)
        ner = nlp.ner

    return nlp, ner


def load_models(lang):
    if lang in CONFIG:
        config = CONFIG[lang]
        models = config.MODELS
        for m in models:
            nlp, ner = load_model(m['name'], m['path'])
            m['nlp'] = nlp
            m['ner'] = ner
        return models
    else:
        return None


def prepare_data(data_path):
    f = open(data_path, 'r')
    result = []
    words = []
    tags = []
    for l in f:
        if '-DOCSTART' in l:
            pass
        else:
            l = l.replace('\n', '')
            l = re.sub(' +', ' ', l)
            if l == '':
                assert len(words) == len(tags)
                result.append({
                    'sentence': convert_words_to_sentence(words),
                    'words': words.copy(),
                    'tags': tags.copy(),
                    'num_tokens': len(words),
                })
                words = []
                tags = []
            else:
                parts = l.split(' ')
                words.append(parts[0])
                tags.append(parts[1])

    return result


def convert_words_to_sentence(words):
    sent = ''
    for w in words:
        sent += w + ' '

    sent = sent[:-1]
    sent = sent.replace('\n', '')
    sent = sent.replace(' , ', ', ')
    sent = sent.replace(' .', '.')
    sent = sent.replace(' !', '!')
    sent = sent.replace(' ?', '?')
    sent = sent.replace(' : ', ': ')
    sent = sent.replace(' ( ', ' (')
    sent = sent.replace(' ) ', ') ')

    return sent


# for the tags that spacy can detect:
# https://spacy.io/api/annotation#named-entities
def convert_spacy_tags_to_iob(spacy_tags):
    output = []
    miscs = ['NORP', 'FAC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW',
             'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
             'QUANTITY', 'ORDINAL', 'CARDINAL', 'MISC']
    miscs.remove('QUANTITY')
    miscs.remove('ORDINAL')
    miscs.remove('CARDINAL')
    miscs.remove('MONEY')
    miscs.remove('PERCENT')
    miscs.remove('DATE')
    miscs.remove('TIME')
    for iob, etype in spacy_tags:
        if iob == 'O':
            output.append(iob)
        else:
            if etype == 'PERSON' or etype == 'PER':
                output.append(iob + '-PER')
            elif etype == 'ORG':
                output.append(iob + '-ORG')
            elif etype in ['GPE', 'LOC']:
                output.append(iob + '-LOC')
            elif etype in miscs:
                output.append(iob + '-MISC')
            else:
                output.append('O')

    return output


def spacy_default_ner(model, input_string):
    doc = model(input_string)
    tags = [(token.ent_iob_, token.ent_type_) for token in doc]
    return convert_spacy_tags_to_iob(tags)


def spacy_custom_ner(model, input_string):
    doc = model(input_string)
    output = []
    for t in doc:
        if t.ent_iob_ == 'O':
            output.append('O')
        else:
            output.append(f'{t.ent_iob_}-{t.ent_type_}')
    return output


def stanza_default_ner(model, input_string):
    doc = model(input_string)
    output = []
    for sent in doc.sentences:
        for token in sent.tokens:
            ner = token.ner
            if ner == 'O':
                output.append(ner)
            else:
                if ner[0:2] == 'E-':
                    output.append('I-'+ner[2:])
                elif ner[0:2] == 'S-':
                    output.append('B-'+ner[2:])
                else:
                    output.append(ner)

    return output


def summarize_dataset(path):
    f = open(path, 'r')
    sents = []
    words = []
    num_tokens = 0
    for l in f:
        if l[:-1] == '':
            sents.append(words.copy())
            words = []
        else:
            num_tokens += 1
            words.append(l[:-1])
    f.close()
    num_sents = len(sents)
    num_tokens = num_tokens / num_sents
    print('#sents\t#tokens_per_sent')
    print(f'{num_sents}\t{num_tokens}')
