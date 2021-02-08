"""
This script parses the given datasets
into a format that we can conveniently use
for our open domain QA experiments.
The output format is in JSON

To parse the original data, you need to download it first (see README)
and put it in the same directory as this script, rename it to e.g. "original"
"""

from os.path import join
import ijson
import sys
sys.path.insert(1, join('..'))
from data_utils import *


def parse(input_filename, output_filename):
    answer_prefixes = [
        'Data.item.Answer.Value',
        'Data.item.Answer.Aliases.item',
        'Data.item.Answer.MatchedWikiEntityName',
        'Data.item.Answer.NormalizedAliases',
        'Data.item.Answer.NormalizedMatchedWikiEntityName',
        'Data.item.Answer.NormalizedValue',
        'Data.item.Answer.Value'
    ]
    question_prefix = 'Data.item.Question'
    question_id_prefix = 'Data.item.QuestionId'

    print('---------------------------------------')
    with open(f'{input_filename}', 'r') as f:
        print(f'loading {input_filename}...')
        fw = open(f'{output_filename}', 'w')
        fw.write('[\n')
        parser = ijson.parse(f)
        current_qa_item = None
        total_count = 0
        qa_count = 0
        result_s = ''
        for prefix, event, value in parser:
            if (prefix, event) == ('Data.item', 'start_map'):
                current_qa_item = DataItemQA()
                qa_count += 1
                if qa_count % 1000 == 0:
                    print(qa_count)
            elif event == 'string':
                if prefix in answer_prefixes:
                    current_qa_item.add_answer(value)
                    total_count += 1
                elif prefix == question_prefix:
                    assert current_qa_item.question is None
                    current_qa_item.set_question(value)
                elif prefix == question_id_prefix:
                    assert current_qa_item.question_id is None
                    current_qa_item.set_question_id(value)
            elif (prefix, event) == ('Data.item', 'end_map'):
                result_s += current_qa_item.to_json()+',\n'

        result_s = result_s[:-2]
        fw.write(result_s)
        fw.write(']\n')
        fw.close()
        print(f'Finished processing {input_filename}')
        print(f'Number of QA pairs: {total_count}')
        print(f'Number of unique questions: {qa_count}')
        print('---------------------------------------')


if __name__ == '__main__':
    original_data_dir = join('original')
    inputs = [join(original_data_dir, 'unfiltered-web-dev.json'),
              join(original_data_dir, 'unfiltered-web-train.json')]
    outputs = ['triviaQA-dev.json', 'triviaQA-train.json']
    for i, o in zip(inputs, outputs):
        parse(i, o)

    # parse(join(original_data_dir, 'unfiltered-web-dev.json'), 'triviaQA-dev.json')
    """
    Expected results:
    Finished processing original/unfiltered-web-dev.json
    Number of QA pairs: 200,843
    Number of unique questions: 11,313
    
    Finished processing original/unfiltered-web-train.json
    Number of QA pairs: 1,524,084
    Number of unique questions: 87,622
    """
