import os

CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CONFIG_DIR, 'models'))
DATA_DIR = os.path.abspath(os.path.join(CONFIG_DIR, 'data'))
MODELS = [
    {
        'name': 'spacy_default',
        'path': os.path.join(MODELS_DIR, 'spacy_default'),
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    {
        'name': 'bert_base_greek_uncased_ft',
        'path': os.path.join(MODELS_DIR, 'bert_base_greek_uncased_ft'),
        'nlp': None,  # the language model to be loaded in eval.py
        'ner': None  # the ner function to be assigned in eval.py
    },
    {
        'name': 'bert_base_multilingual_uncased_ft',
        'path': os.path.join(MODELS_DIR, 'bert_base_multilingual_uncased_ft'),
        'nlp': None, # the language model to be loaded in eval.py
        'ner': None # the ner function to be assigned in eval.py
    },
    # {
    #     'name': 'bert_base_multilingual_uncased_ft_51',
    #     'path': os.path.join(MODELS_DIR, 'bert_base_multilingual_uncased_ft_51'),
    #     'nlp': None,  # the language model to be loaded in eval.py
    #     'ner': None  # the ner function to be assigned in eval.py
    # },
    # {
    #     'name': 'bert_base_multilingual_uncased_ft_102',
    #     'path': os.path.join(MODELS_DIR, 'bert_base_multilingual_uncased_ft_102'),
    #     'nlp': None,  # the language model to be loaded in eval.py
    #     'ner': None  # the ner function to be assigned in eval.py
    # },
    # {
    #     'name': 'bert_base_multilingual_uncased_ft_153',
    #     'path': os.path.join(MODELS_DIR, 'bert_base_multilingual_uncased_ft_153'),
    #     'nlp': None,  # the language model to be loaded in eval.py
    #     'ner': None  # the ner function to be assigned in eval.py
    # },
    # {
    #     'name': 'bert_base_multilingual_uncased_ft_510',
    #     'path': os.path.join(MODELS_DIR, 'bert_base_multilingual_uncased_ft_510'),
    #     'nlp': None,  # the language model to be loaded in eval.py
    #     'ner': None  # the ner function to be assigned in eval.py
    # }
]
SUBSET = 'test'
DATASETS = [
    {
        'name': 'spacyner2018',
        'path': os.path.join(DATA_DIR, 'spacyner2018', f'{SUBSET}.txt')
    }
]
ENTITY_TYPES = ['PER', 'LOC', 'ORG', 'MISC']
