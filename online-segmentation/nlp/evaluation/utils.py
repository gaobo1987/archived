import json
import spacy
from tqdm import tqdm


class ConfusionMatrix:
    def __init__(self):
        self.cm = {}

    def add_value_to_cell(self, row_label, col_label, val):
        key = (row_label, col_label)
        if key in self.cm:
            self.cm[key] += val
        else:
            self.cm[key] = val

    def get_value_from_cell(self, row_label, col_label):
        key = (row_label, col_label)
        return self.cm[key] if key in self.cm else None

    def row_labels(self):
        return list(set([k[0] for k in self.cm.keys()]))

    def col_labels(self):
        return list(set([k[1] for k in self.cm.keys()]))

    # total count per predicted label
    def row_label_counts(self):
        all_row_labels = self.row_labels()
        all_col_labels = self.col_labels()
        result = {}
        for rlbl in all_row_labels:
            count = 0
            for clbl in all_col_labels:
                val = self.get_value_from_cell(rlbl, clbl)
                if val is not None:
                    count += val
            result[rlbl] = count
        return result

    # total count per gold label
    def col_label_counts(self):
        all_row_labels = self.row_labels()
        all_col_labels = self.col_labels()
        result = {}
        for clbl in all_col_labels:
            count = 0
            for rlbl in all_row_labels:
                val = self.get_value_from_cell(rlbl, clbl)
                if val is not None:
                    count += val
            result[clbl] = count
        return result

    # true positive count per matched label
    def matched_label_counts(self):
        all_row_labels = self.row_labels()
        all_col_labels = self.col_labels()
        result = {}
        for lbl in all_row_labels:
            count = 0
            if lbl in all_col_labels:
                val = self.get_value_from_cell(lbl, lbl)
                if val is not None:
                    count += val

            result[lbl] = count

        for lbl in all_col_labels:
            if lbl not in result:
                result[lbl] = 0

        return result

    def precisions(self):
        matched_counts = self.matched_label_counts()
        row_counts = self.row_label_counts()
        result = {}
        for m in matched_counts.keys():
            TP = matched_counts[m]
            if m in row_counts:
                Total_Predicted = row_counts[m]
                result[m] = TP / Total_Predicted
            else:
                result[m] = 0
        return result

    def recalls(self):
        matched_counts = self.matched_label_counts()
        col_counts = self.col_label_counts()
        result = {}
        for m in matched_counts.keys():
            TP = matched_counts[m]
            if m in col_counts:
                Total_Gold = col_counts[m]
                result[m] = TP / Total_Gold
            else:
                result[m] = 0
        return result

    def f1s(self):
        precisions = self.precisions()
        recalls = self.recalls()
        result = {}
        for k in precisions.keys():
            p = precisions[k]
            r = recalls[k]
            if (p + r) > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0
            result[k] = f1
        return result

    def avg(self, _dict):
        count = len(_dict)
        if count > 0:
            _sum = 0
            for k in _dict.keys():
                _sum += _dict[k]
            return _sum / count
        else:
            return 0

    def wgt(self, _dict):
        count = len(_dict)
        if count > 0:
            gold_label_counts = self.col_label_counts()
            total = 0
            for k in gold_label_counts:
                total += gold_label_counts[k]

            _sum = 0
            for k in _dict.keys():
                gold_count = 0
                if k in gold_label_counts.keys():
                    gold_count = gold_label_counts[k]
                _sum += _dict[k] * gold_count
            return _sum / total
        else:
            return 0

    def avg_precision(self):
        return self.avg(self.precisions())

    def avg_recall(self):
        return self.avg(self.recalls())

    def avg_f1(self):
        return self.avg(self.f1s())

    def wgt_precision(self):
        return self.wgt(self.precisions())

    def wgt_recall(self):
        return self.wgt(self.recalls())

    def wgt_f1(self):
        return self.wgt(self.f1s())

    def show(self):
        all_row_labels = self.row_labels()
        all_col_labels = self.col_labels()
        delim = '\t'
        header = ''
        for c in all_col_labels:
            header += delim + c
        print(header)
        for r in all_row_labels:
            row = r
            for c in all_col_labels:
                val = self.get_value_from_cell(r, c)
                row += delim + str(val)
            print(row)


def load_json(data_path):
    data_file = open(data_path, 'r')
    data_str = data_file.read()
    data_file.close()
    return json.loads(data_str)


def save_json(data_path, data):
    data_file = open(data_path, 'w', encoding='utf-8')
    json.dump(data, data_file, ensure_ascii=False)
    data_file.close()


def load_model(model_path):
    model = None
    if 'spacy' in model_path:
        model = spacy.load(model_path, disable=['ner'])
        print('spacy model loaded with pipe: ', model.pipe_names)
    return model


def load_segmenter(lang):
    if lang == 'nl':
        from qsegmt.nl import NLSegmenter
        return NLSegmenter()
    elif lang == 'el':
        from qsegmt.el import ELSegmenter
        return ELSegmenter()
    else:
        return None


def find_gold_term(query_gold: list, item_text: str, item_start_char_index: int):
    match = {}
    for item in query_gold:
        start_offset = int(item['startOffSet'])
        if item['item'] == item_text and start_offset == item_start_char_index:
            match = item
            break
    return match


def compose_evaluation_data(gold_path, save_path, lang):
    segmenter = load_segmenter(lang)
    if segmenter is not None:
        eval_data = load_json(gold_path)
        for q in tqdm(eval_data):
            output = segmenter.segment(q['query'])
            q['pred'] = json.loads(output)
        save_json(save_path, eval_data)


def convert_items_to_bio(items):
    if items == []:
        return []
    if 'startOffSet' in items[0]:
        start_key = 'startOffSet'
    elif 'startOffset' in items[0]:
        start_key = 'startOffset'
    else:
        raise ValueError('no startOffSet key in items')
    if 'endOffSet' in items[0]:
        end_key = 'endOffSet'
    elif 'endOffset' in items[0]:
        end_key = 'endOffset'
    else:
        raise ValueError('no endOffset key in items')

    items = sorted(items, key=lambda it: (it[start_key], -it[end_key]))
    tags = []
    i = 0
    while i < len(items):
        if items[i]['ner'] == '':
            tags.append('O')
            i += 1
            continue
        ner = items[i]['ner'][0]['ner']
        tags.append(f'B-{ner}')
        start, end = items[i][start_key], items[i][end_key]
        i += 1
        while i < len(items) and items[i][end_key] <= end:
            if items[i][start_key] != start:
                tags.append(f'I-{ner}')
            i += 1
    return tags


def evaluate(data_path, key):
    data = load_json(data_path)
    if key == 'pos' or key == 'lemma':
        cm = ConfusionMatrix()
        for q in tqdm(data):
            for t in q['pred']:
                pred = t[key]
                gold_term = find_gold_term(q['gold'], t['item'], int(t['startOffSet']))
                gold = gold_term.get(key, None)
                cm.add_value_to_cell(pred, gold, 1)

        result = {
            'weighted_f1': cm.wgt_f1(),
            'weighted_precision': cm.wgt_precision(),
            'weighted_recall': cm.wgt_recall(),
            'average_f1': cm.avg_f1(),
            'average_precision': cm.avg_precision(),
            'average_recall': cm.avg_recall()
        }
        if key == 'pos':
            result.update({
                'f1s': cm.f1s(),
                'precisions': cm.precisions(),
                'recalls': cm.recalls()
            })
        return result
    elif key == 'ner':
        from nlp.ner.evaluation.ner_evaluation.ner_eval import Evaluator
        gold_labels = []
        pred_labels = []
        label_set = set()
        for q in data:
            # construct bio labels
            sent_gold_labels = convert_items_to_bio(q['gold'])
            sent_pred_labels = convert_items_to_bio(q['pred'])
            if len(sent_pred_labels) == len(sent_gold_labels):
                pass
            elif len(sent_pred_labels) > len(sent_gold_labels):
                diff = len(sent_pred_labels) - len(sent_gold_labels)
                sent_pred_labels = sent_pred_labels[:-diff]
            elif len(sent_gold_labels) > len(sent_pred_labels):
                diff = len(sent_gold_labels) - len(sent_pred_labels)
                sent_gold_labels = sent_gold_labels[:-diff]
            assert len(sent_pred_labels) == len(sent_gold_labels)

            gold_labels.append(sent_gold_labels)
            pred_labels.append(sent_pred_labels)

            # add to label set
            sent_labels = list(set(sent_gold_labels + sent_pred_labels))
            sent_labels = [lbl.replace('B-', '').replace('I-', '') for lbl in sent_labels]
            label_set.update(sent_labels)

        evaluator = Evaluator(gold_labels, pred_labels, list(label_set))
        results, results_agg = evaluator.evaluate()
        for k in results:
            p = results[k]['precision']
            r = results[k]['recall']
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            results[k]['f1'] = f1

        for label in results_agg:
            for k in results_agg[label]:
                p = results_agg[label][k]['precision']
                r = results_agg[label][k]['recall']
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                results_agg[label][k]['f1'] = f1

        return results, results_agg

    else:
        return None

