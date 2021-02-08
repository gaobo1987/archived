import os
import json
from typing import List
import numpy as np
from tqdm import tqdm


class DataItemQA:
    """
    This class is mainly used in the data preprocessing scripts,
    for producing preprocessed datasets.
    """

    def __init__(self, question_id=None, question=None, **kwargs):
        self.question_id = question_id
        self.question = question
        self.answers = []
        self.answers_full = []
        for attr in kwargs.keys():
            self.__dict__[attr] = kwargs[attr]

    def set_question_id(self, question_id):
        self.question_id = question_id

    def set_question(self, question):
        self.question = question

    def add_answer(self, answer, start_position=None, end_position=None):
        if answer and answer not in self.answers:
            self.answers.append(answer)
            if start_position is not None and end_position is not None:
                self.answers_full.append({
                    'text': answer,
                    'start_position': start_position,
                    'end_position': end_position
                })

    def to_dict(self):
        result = self.__dict__.copy()
        return result

    def to_json(self):
        return json.dumps(self.to_dict())


def save_json(obj_to_save, output_filename):
    with open(output_filename, 'w', encoding='utf8') as out:
        json.dump(obj_to_save, out, ensure_ascii=False)


def stat_vals(dist: []) -> tuple:
    r_num = 1
    if dist:
        return round(np.mean(dist), r_num), round(np.std(dist), r_num), \
               round(np.min(dist), r_num), round(np.max(dist), r_num), \
               round(np.median(dist), r_num)
    else:
        return 0, 0, 0, 0, 0


def stat_strs(dist: List[str]) -> str:
    tokens = [len(e.split(' ')) for e in dist]
    mean_t, std_t, min_t, max_t, median_t = stat_vals(tokens)
    chars = [len(e) for e in dist]
    mean_c, std_c, min_c, max_c, median_c = stat_vals(chars)
    s = f'##### Tokens / Characters \n'
    s += f'* mean: {mean_t} / {mean_c}\n'
    s += f'* std: {std_t} / {std_c}\n'
    s += f'* min: {min_t} / {min_c}\n'
    s += f'* max: {max_t} / {max_c}\n'
    s += f'* median: {median_t} / {median_c}\n'
    return s


def stat_items(items: List, total_questions=None, total_answers=None) -> str:
    qs = [item['question'] for item in items]
    questions_with_answer = []
    for item in items:
        if item['answers']:
            questions_with_answer.append(item)
    num_qs_w_ans = len(questions_with_answer)
    num_qs_w_ans_stat = 0 if num_qs_w_ans == 0 else num_qs_w_ans / len(items)
    num_qs_w_ans_stat = f'({round(num_qs_w_ans_stat * 100, 1)}%)'
    q_perc = '{:.1%}'.format(len(qs) / total_questions) if isinstance(total_questions, int) else None
    q_stat = f'{len(qs)} ({q_perc})' if q_perc is not None else f'{len(qs)}'
    s = f'##### Questions <{q_stat}>:\n'
    s += f'Num.Questions with Answers: {num_qs_w_ans} {num_qs_w_ans_stat}\n'
    s += stat_strs(qs)
    answers = []
    for item in items:
        for a in item['answers']:
            answers.append(a)
    a_perc = '{:.1%}'.format(len(answers) / total_answers) if isinstance(total_answers, int) else None
    a_stat = f'{len(answers)} ({a_perc})' if a_perc is not None else f'{len(answers)}'
    s += f'##### Answers <{a_stat}>:\n'
    s += stat_strs(answers)
    s += '------\n'
    return s


def write_data_stats(parsed_data_path):
    s = ''
    with open(parsed_data_path, 'r', encoding='utf8') as f:
        obj = json.load(f)
        items = obj if isinstance(obj, List) else obj['qas']
        q_dict = {
            'who': [],
            'what': [],
            'why': [],
            'when': [],
            'where': [],
            'which': [],
            'how': [],
            'others': []
        }
        num_questions = len(items)
        num_answers = 0
        for item in items:
            question = item['question'].lower()
            num_answers += len(item['answers'])
            if question.startswith('who'):
                q_dict['who'].append(item)
            elif question.startswith('what') or \
                    question.startswith('in what') or \
                    question.startswith('on what') or \
                    question.startswith('at what') or \
                    question.startswith('under what') or \
                    question.startswith('about what') or \
                    question.startswith('of what'):
                q_dict['what'].append(item)
            elif question.startswith('why'):
                q_dict['why'].append(item)
            elif question.startswith('when'):
                q_dict['when'].append(item)
            elif question.startswith('where'):
                q_dict['where'].append(item)
            elif question.startswith('how'):
                q_dict['how'].append(item)
            elif question.startswith('which') or \
                    question.startswith('in which') or \
                    question.startswith('on which') or \
                    question.startswith('at which') or \
                    question.startswith('under which') or \
                    question.startswith('about which') or \
                    question.startswith('of which'):
                q_dict['which'].append(item)
            else:
                q_dict['others'].append(item)

    s += f'* Average Number of Answers per Question: {round(num_answers / num_questions, 1)}\n'
    s += stat_items(items)
    # 5Ws + 1H
    s += '### Who\n'
    s += stat_items(q_dict['who'], num_questions, num_answers)
    s += '### What\n'
    s += stat_items(q_dict['what'], num_questions, num_answers)
    s += '### Why\n'
    s += stat_items(q_dict['why'], num_questions, num_answers)
    s += '### When\n'
    s += stat_items(q_dict['when'], num_questions, num_answers)
    s += '### Where\n'
    s += stat_items(q_dict['where'], num_questions, num_answers)
    s += '### Which\n'
    s += stat_items(q_dict['which'], num_questions, num_answers)
    s += '### How\n'
    s += stat_items(q_dict['how'], num_questions, num_answers)
    s += '### Others\n'
    s += stat_items(q_dict['others'], num_questions, num_answers)

    output_file = str(os.path.splitext(os.path.abspath(parsed_data_path))[0])
    output_file += '_stats.md'
    with open(output_file, 'w') as f:
        f.write(s)


if __name__ == '__main__':
    data_paths = [
        # 'narrativeQA/narrativeQA-dev.json',
        # 'naturalQuestions/naturalQuestions-dev.json',
        # 'naturalQuestions/naturalQuestions-dev-clean.json',
        # 'quasarT/quasarT-dev.json',
        # 'searchQA/searchQA-dev.json',
        # 'squad2/squad2-dev.json',
        # 'triviaQA/triviaQA-dev.json',
        # 'wikiQA/wikiQA-dev.json',
        # 'wikiQA/wikiQA-test.json'
        'squad1/squad1-dev.json',
    ]
    for p in tqdm(data_paths):
        write_data_stats(p)
