"""
This script parses the given datasets
into a format that we can conveniently use
for our open domain QA experiments.
The output format is in JSON

To parse the original data, you need to download it first (see README)
and put it in the same directory as this script, rename it to e.g. "original"

Note that only the questions with answers are extracted.
"""

from tqdm import tqdm
from os.path import join
import json
import sys
sys.path.insert(1, join('..'))
from data_utils import *


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def extract_answer_positions(context_text, answer_text, start_char_position, is_impossible):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    # Split on whitespace so that different tokens may be attributed to their original position.
    for c in context_text:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    start_position, end_position = None, None
    # Start and end positions only has a value during evaluation.
    if start_char_position is not None and not is_impossible:
        start_position = char_to_word_offset[start_char_position]
        end_position = char_to_word_offset[
            min(start_char_position + len(answer_text) - 1, len(char_to_word_offset) - 1)
        ]
    return start_position, end_position


def parse(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        items = []
        total_count = 0
        obj = json.load(f)
        data = obj['data']
        context_tokens_counts = []
        contexts = {}
        for entry in tqdm(data):
            title = entry['title']
            paragraphs = entry['paragraphs']
            for p in paragraphs:
                context_tokens_counts.append(len(p['context'].split()))
                context_id = str(len(contexts))
                context_text = p['context']
                contexts[context_id] = context_text
                for qa in p['qas']:
                    answers = qa['answers']
                    if len(answers) > 0:
                        is_impossible = False if not "is_impossible" in qa else qa["is_impossible"]
                        question_id = qa['id']
                        question = qa['question']
                        item = DataItemQA(question_id, question, context_id=context_id, title=title)
                        for a in answers:
                            answer = a['text']
                            start_char_position = a['answer_start']
                            start_position, end_position = \
                                extract_answer_positions(context_text, answer, start_char_position, is_impossible)
                            item.add_answer(answer, start_position=start_position, end_position=end_position)
                            total_count += 1
                        if not is_impossible:
                            assert len(item.answers_full) > 0
                        items.append(item.to_dict())

        with open(output_filename, 'w', encoding='utf8') as out:
            result = {
                'contexts': contexts,
                'qas': items
            }
            json.dump(result, out, ensure_ascii=False)
        print('---------------------------------------')
        print(f'Finished processing {input_filename}')
        print(f'Number of QA pairs: {total_count}')
        print(f'Number of unique questions: {len(items)}')
        print(f'Number of contexts: ', len(context_tokens_counts))
        print(f'Number of tokens per context: ')
        print(f'    mean={np.mean(context_tokens_counts)}')
        print(f'    min={np.min(context_tokens_counts)}')
        print(f'    max={np.max(context_tokens_counts)}')
        print(f'    std={np.std(context_tokens_counts)}')
        print(f'    median={np.median(context_tokens_counts)}')
        print('---------------------------------------')


if __name__ == '__main__':
    original_data_dir = join('original')
    inputs = [join(original_data_dir, 'dev-v2.0.json'),
              join(original_data_dir, 'train-v2.0.json')]
    outputs = ['squad2-dev-foobar.json', 'squad2-train-foobar.json']
    for i, o in zip(inputs, outputs):
        parse(i, o)

    '''
    Expected output:
    
   ---------------------------------------
    Number of QA pairs: 20302
    Number of unique questions: 5928
    Number of contexts:  1204
    Number of tokens per context: 
        mean=126.57308970099668
        min=25
        max=629
        std=57.437157397647965
        median=113.0
    ---------------------------------------
    Number of QA pairs: 86821
    Number of unique questions: 86821
    Number of contexts:  19035
    Number of tokens per context: 
        mean=116.58550039401104
        min=20
        max=653
        std=49.668850126326525
        median=107.0
    ---------------------------------------
    '''
