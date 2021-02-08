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


def parse(input_filename, output_filename):
    items = []
    with open(f'{input_filename}', 'r') as f:
        for i, l in tqdm(enumerate(f)):
            l = l.replace('\n', '')
            obj = json.loads(l)
            answer = obj['answer']
            question = obj['question']
            question_id = obj['uid']
            item = DataItemQA(question_id, question)
            item.add_answer(answer)
            items.append(item.to_dict())

        with open(output_filename, 'w', encoding='utf8') as out:
            json.dump(items, out, ensure_ascii=False)
        print('---------------------------------------')
        print(f'Finished processing {input_filename}')
        print(f'Number of QA pairs: {len(items)}')
        print(f'Number of unique questions: {len(items)}')
        print('---------------------------------------')


if __name__ == '__main__':
    original_data_dir = join('original')
    inputs = [join(original_data_dir, 'dev_questions.json'),
              join(original_data_dir, 'test_questions.json'),
              join(original_data_dir, 'train_questions.json')]
    outputs = ['quasarT-dev.json', 'quasarT-test.json', 'quasarT-train.json']
    for i, o in zip(inputs, outputs):
        parse(i, o)

    '''
    Expected output:
    ---------------------------------------
    Finished processing original/dev_questions.json
    Number of QA pairs: 3000
    Number of unique questions: 3000
    ---------------------------------------
    Finished processing original/test_questions.json
    Number of QA pairs: 3000
    Number of unique questions: 3000
    ---------------------------------------
    Finished processing original/train_questions.json
    Number of QA pairs: 37012
    Number of unique questions: 37012
    ---------------------------------------
    '''