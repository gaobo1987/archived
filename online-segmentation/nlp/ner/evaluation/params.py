import torch
from nlp.ner.resources.nl import config as nl_config
from nlp.ner.resources.el import config as el_config

SCORE_MODES = ['strict', 'exact', 'type', 'partial']
SCORE_METRICS = ['f1', 'precision', 'recall']
OUTPUT_PATH = 'evaluation_results.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG = {
    'nl': nl_config,
    'el': el_config
}
