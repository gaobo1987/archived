import os
import sys
import time
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', '..')))
sys.path.append(os.path.abspath(os.path.join('..', '..', '..')))
from nlp.ner.evaluation.params import SCORE_METRICS, SCORE_MODES, OUTPUT_PATH, CONFIG
from nlp.ner.evaluation.utils import prepare_data, load_models
from nlp.ner.evaluation.ner_evaluation.ner_eval import Evaluator


class NEREvaluator:
    def __init__(self, lang):
        self.lang = lang
        if lang in CONFIG:
            self.config = CONFIG[lang]
            load_models(lang)
        else:
            print(f'{lang} is not supported yet...')
            raise

    def _ner_eval(self, true, pred):
        labels = self.config.ENTITY_TYPES
        evaluator = Evaluator(true, pred, labels)
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

    def evaluate(self, model, data_path):
        ms_per_sent = []
        num_words_per_sent = []
        data = prepare_data(data_path)
        preds = []
        tags = []
        for d in tqdm(data):
            sent = d['sentence']
            sent_words = d['words']
            sent_tags = d['tags']
            num_words_per_sent.append(d['num_tokens'])

            start_ms = int(round(time.time() * 1000))
            sent_preds = model['ner'](model['nlp'], sent) \
                if 'spacy' in model['name'] or 'stanza' in model['name'] else model['ner'](sent_words)
            end_ms = int(round(time.time() * 1000))
            ms_per_sent.append(end_ms - start_ms)

            if len(sent_preds) == len(sent_tags):
                pass
            elif len(sent_preds) > len(sent_tags):
                diff = len(sent_preds) - len(sent_tags)
                sent_preds = sent_preds[:-diff]
            elif len(sent_tags) > len(sent_preds):
                diff = len(sent_tags) - len(sent_preds)
                sent_tags = sent_tags[:-diff]
            assert len(sent_preds) == len(sent_tags)

            preds.append(sent_preds)
            tags.append(sent_tags)

        results, results_agg = self._ner_eval(tags, preds)
        output = {
            'data': data_path,
            'avg_ms_per_sent': sum(ms_per_sent) / len(ms_per_sent),
            'avg_num_words_per_sent': sum(num_words_per_sent) / len(num_words_per_sent)
        }
        output.update(results)
        output.update(results_agg)
        return output

    def write_results_to_csv(self):
        entity_types = [''] + self.config.ENTITY_TYPES
        output = open(OUTPUT_PATH, 'w')
        header = 'experiment_id, model_name, dataset, ms_p_sent, num_tokens_p_sent, '
        for entity_type in entity_types:
            for score_mode in SCORE_MODES:
                for score_metric in SCORE_METRICS:
                    col = score_metric + '_' + score_mode + '_' + entity_type
                    if entity_type == '':
                        col = col[:-1]
                    header += col + ', '
        header = header[:-2]
        output.write(header + '\n')
        count = 0
        for d in self.config.DATASETS:
            dataset_name = d['name']
            dataset_path = d['path']
            for m in self.config.MODELS:
                if m['nlp'] is not None:
                    model_name = m['name']
                    count += 1
                    result = self.evaluate(m, dataset_path)
                    ms_p_sent = result['avg_ms_per_sent']
                    num_tokens_p_sent = result['avg_num_words_per_sent']
                    row = f'{count}, {model_name}, {dataset_name}, {ms_p_sent}, {num_tokens_p_sent}, '
                    for entity_type in entity_types:
                        for score_mode in SCORE_MODES:
                            for score_metric in SCORE_METRICS:
                                _score_mode = 'ent_type' if score_mode == 'type' else score_mode
                                if entity_type == '':
                                    score = result[_score_mode][score_metric]
                                else:
                                    score = result[entity_type][_score_mode][score_metric]
                                row += f'{score}, '
                    row = row[:-2]
                    output.write(row + '\n')
        output.close()

