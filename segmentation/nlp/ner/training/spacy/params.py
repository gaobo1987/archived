import os
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data
LANG = 'nl'
DATASET = 'all'
SUBSET = 'train'
DATA_PATH = os.path.abspath(os.path.join('..', '..', 'resources', LANG, 'data', DATASET, f'{SUBSET}.txt'))
OUTPUT_JSON_PATH = os.path.abspath(os.path.join(f'{LANG}-{DATASET}-{SUBSET}.json'))

# model
OUTPUT_MODEL_PATH = os.path.abspath(os.path.join('output'))

# training params
NUM_EPOCHS = 100
