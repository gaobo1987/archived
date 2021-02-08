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
    qa_dict = {}
    count = 0
    with open(f'{input_filename}', 'r') as f:
        for i, l in tqdm(enumerate(f)):
            if i > 0:
                parts = l.split('\t')
                label = parts[6]
                if int(label) == 1:
                    count += 1
                    question_id = parts[0]
                    answer_text = parts[5]
                    if question_id in qa_dict:
                        qa_item: DataItemQA = qa_dict[question_id]
                        qa_item.add_answer(answer_text)
                    else:
                        question_text = parts[1]
                        qa_item: DataItemQA = DataItemQA(question_id, question_text)
                        qa_item.add_answer(answer_text)
                        qa_dict[question_id] = qa_item

        qas = []
        for qid in qa_dict:
            qas.append(qa_dict[qid].to_dict())
        with open(output_filename, 'w', encoding='utf8') as out:
            json.dump(qas, out, ensure_ascii=False)
        print('---------------------------------------')
        print(f'Finished processing {input_filename}')
        print(f'Number of QA pairs: {count}')
        print(f'Number of unique questions: {len(qa_dict)}')
        print('---------------------------------------')


if __name__ == '__main__':
    original_data_dir = join('original')
    inputs = [join(original_data_dir, 'WikiQA-dev.tsv'),
              join(original_data_dir, 'WikiQA-train.tsv'),
              join(original_data_dir, 'WikiQA-test.tsv')]
    outputs = ['wikiQA-dev.json', 'wikiQA-train.json', 'wikiQA-test.json']
    for i, o in zip(inputs, outputs):
        parse(i, o)
