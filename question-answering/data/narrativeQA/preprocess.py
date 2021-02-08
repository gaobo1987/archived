"""
This script parses the given datasets
into a format that we can conveniently use
for our open domain QA experiments.
The output format is in JSON

To parse the original data, you need to download it first (see README)
and put it in the same directory as this script, rename it to e.g. "original"
"""

from tqdm import tqdm
from os.path import join
import json
import sys
sys.path.insert(1, join('..'))
from data_utils import *


def parse(input_filename):
    with open(f'{input_filename}', 'r') as f:
        test_items = []
        train_items = []
        dev_items = []
        for i, l in tqdm(enumerate(f)):
            if i > 0:
                parts = l.split(',')
                question_id = parts[0]
                which_set = parts[1]
                question = parts[2]
                answer1 = parts[3]
                answer2 = parts[4]
                item = DataItemQA(question_id, question)
                item.add_answer(answer1)
                item.add_answer(answer2)
                obj = item.to_dict()
                if which_set == 'test':
                    test_items.append(obj)
                elif which_set == 'train':
                    train_items.append(obj)
                elif which_set == 'valid':
                    dev_items.append(obj)

        print(f'Finished processing {input_filename}')
        with open('narrativeQA-test.json', 'w', encoding='utf8') as out:
            json.dump(test_items, out, ensure_ascii=False)
            print('Test:')
            print(f'Number of QA pairs: {len(test_items)*2}')
            print(f'Number of unique questions: {len(test_items)}')
        with open('narrativeQA-train.json', 'w', encoding='utf8') as out:
            json.dump(train_items, out, ensure_ascii=False)
            print('Train:')
            print(f'Number of QA pairs: {len(train_items) * 2}')
            print(f'Number of unique questions: {len(train_items)}')
        with open('narrativeQA-dev.json', 'w', encoding='utf8') as out:
            json.dump(dev_items, out, ensure_ascii=False)
            print('Dev:')
            print(f'Number of QA pairs: {len(dev_items) * 2}')
            print(f'Number of unique questions: {len(dev_items)}')


if __name__ == '__main__':
    parse(join('original', 'qaps.csv'))

    '''
    Expected output:
    Finished processing original/qaps.csv
    Test:
    Number of QA pairs: 21,114
    Number of unique questions: 10,557
    Train:
    Number of QA pairs: 65,494
    Number of unique questions: 32,747
    Dev:
    Number of QA pairs: 6,922
    Number of unique questions: 3,461

    '''
