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
from zipfile import ZipFile
import json
import sys
sys.path.insert(1, join('..'))
from data_utils import *


def parse(input_filename, output_filename):
    with ZipFile(input_filename) as zipf:
        items = []
        files = zipf.namelist()
        for filename in tqdm(files):
            with zipf.open(filename) as f:
                obj = json.loads(f.read())
                question = obj['question']
                question_id = obj['id']
                answer = obj['answer']
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
    inputs = [join(original_data_dir, 'val.zip'),
              join(original_data_dir, 'test.zip'),
              join(original_data_dir, 'train.zip')]
    outputs = ['searchQA-dev.json', 'searchQA-test.json', 'searchQA-train.json']
    for i, o in zip(inputs, outputs):
        parse(i, o)

    '''
    Expected output:
    ---------------------------------------
    Finished processing original/val.zip
    Number of QA pairs: 21,613
    Number of unique questions: 21,613
    ---------------------------------------
    Finished processing original/test.zip
    Number of QA pairs: 43,228
    Number of unique questions: 43,228
    ---------------------------------------
    Finished processing original/train.zip
    Number of QA pairs: 151,295
    Number of unique questions: 151,295
    ---------------------------------------
    '''