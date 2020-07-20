import os

CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CONFIG_DIR, 'models'))
DATA_DIR = os.path.abspath(os.path.join(CONFIG_DIR, 'data'))
MODELS = [
    {
        'name': 'spacy_custom_blank',
        'path': os.path.join(MODELS_DIR, 'spacy_custom_blank'),
        'description': 'The spacy dutch model re-trained from blank.',
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    {
        'name': 'spacy_custom_default',
        'path': os.path.join(MODELS_DIR, 'spacy_custom_default'),
        'description': 'The spacy dutch model fine-tuned based on its default.',
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    {
        'name': 'spacy_default',
        'path': os.path.join(MODELS_DIR, 'spacy_default'),
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    {
        'name': 'stanza_default',
        'path': 'nl',
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    {
        'name': 'bert_base_dutch_cased_ft',
        'path': os.path.join(MODELS_DIR, 'bert_base_dutch_cased_ft'),
        'description': 'The model fine-tuned on the <all> dataset, with cased input, '
                       'based on the <bert-base-dutch-cased> pretrained model.',
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    {
        'name': 'bert_base_dutch_cased_uncased_ft',
        'path': os.path.join(MODELS_DIR, 'bert_base_dutch_cased_uncased_ft'),
        'description': 'The model fine-tuned on the <all> dataset, with uncased input, '
                       'based on the <bert-base-dutch-cased> pretrained model.',
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    {
        'name': 'bert_base_dutch_cased_ft_uncased_ft',
        'path': os.path.join(MODELS_DIR, 'bert_base_dutch_cased_ft_uncased_ft'),
        'description': 'The model fine-tuned on the <all> dataset, with uncased input, '
                       'based on the <bert_base_dutch_cased_ft> fine-tuned model.',
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    {
        'name': 'bert_base_multilingual_uncased_ft',
        'path': os.path.join(MODELS_DIR, 'bert_base_multilingual_uncased_ft'),
        'description': 'The model fine-tuned on the <all> dataset, with uncased input, '
                       'based on the <bert_base_multilingual_uncased> pretrained model.',
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    }
]
SUBSET = 'test'
DATASETS = [
    {
        'name': 'demorgen2000',
        'path': os.path.join(DATA_DIR, 'demorgen2000', f'{SUBSET}.txt')
    },
    {
        'name': 'wikiner',
        'path': os.path.join(DATA_DIR, 'wikiner', f'{SUBSET}.txt')
    },
    {
        'name': 'europeananews',
        'path': os.path.join(DATA_DIR, 'europeananews', f'{SUBSET}.txt')
    },
    {
        'name': 'meantimenews',
        'path': os.path.join(DATA_DIR, 'meantimenews', f'{SUBSET}.txt')
    },
    {
        'name': 'parliamentary',
        'path': os.path.join(DATA_DIR, 'parliamentary', f'{SUBSET}.txt')
    },

]
ENTITY_TYPES = ['PER', 'LOC', 'ORG', 'MISC']
