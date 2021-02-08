import sys, os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas import DataFrame
from tqdm.notebook import tqdm
from typing import List

sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from evaluation import *


def plot_line(px, df, x, y, title, template='plotly_white', **kwargs):
    fig = px.line(df, x=x, y=y, title=title, template=template, **kwargs)
    fig.show()


class OutputReporter:
    def __init__(self, json_path=None, save_cache=True):
        self.json_path = json_path
        self.cache_path = json_path.replace('.json', '_cache.json') if json_path else None
        self.items = []
        self.save_cache = save_cache
        self.cache = None
        if json_path:
            self.items = load_json(json_path)

    def get_recalls(self, field='bm25_ranks', begin=5, end=5000, step=5):
        results = {
            'k': [],
            'recall': []
        }
        if self.items:
            assert field in self.items[0]
            ranks_list = [item[field] for item in self.items]
            for k in tqdm(range(begin, end, step)):
                r = recall_at_k(ranks_list, k)
                results['k'].append(k)
                results['recall'].append(r)
        return results

    def get_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        results = {
            'k': [],
            'precision': []
        }
        if self.items:
            assert field in self.items[0]
            ranks_list = [item[field] for item in self.items]
            for k in tqdm(range(begin, end, step)):
                p = precision_at_k(ranks_list, k)
                results['k'].append(k)
                results['precision'].append(p)
        return results

    def get_recalls_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        result = {
            'k': [],
            'precision': [],
            'recall': []
        }
        if self.cache is None:
            # check disk first
            if self.cache_path and os.path.isfile(self.cache_path):
                self.cache = load_json(self.cache_path)
                return self.cache
            rs = self.get_recalls(field, begin, end, step)
            ps = self.get_precisions(field, begin, end, step)
            for i, k in enumerate(rs['k']):
                result['k'].append(k)
                r = rs['recall'][i]
                result['recall'].append(r)
                p = ps['precision'][i]
                result['precision'].append(p)
            if self.save_cache:
                print('Saving cache ', self.cache_path)
                save_json(result, self.cache_path)
        else:
            for k in range(begin, end, step):
                i = self.cache['k'].index(k)
                if i > 0:
                    result['k'].append(k)
                    r = self.cache['recall'][i]
                    result['recall'].append(r)
                    p = self.cache['precision'][i]
                    result['precision'].append(p)

        return result

    def plot_recalls(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_recalls(field, begin, end, step)
        df = DataFrame(res)
        plot_line(px, df, x='k', y='recall', title='Recalls')

    def plot_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_precisions(field, begin, end, step)
        df = DataFrame(res)
        plot_line(px, df, x='k', y='precision', title='Precisions')

    def plot_recalls_precisions(self, field='bm25_ranks', begin=5, end=5000, step=5):
        res = self.get_recalls_precisions(field, begin, end, step)
        df = DataFrame(res)
        plot_line(px, df, x='recall', y='precision', title='Precision-Recall')


class OutputReporterAgg:
    def __init__(self, json_paths: List[str] = None):
        self.cache = None
        self.reporters: List[OutputReporter] = []
        for _path in tqdm(json_paths):
            # caching
            reporter = OutputReporter(_path)
            reporter.get_recalls_precisions()
            self.reporters.append(reporter)

    def aggregate(self):
        if self.cache is None:
            res = {
                'index': [],
                'k': [],
                'recall': [],
                'precision': []
            }
            for reporter in self.reporters:
                index = reporter.json_path.split('/')[0]
                output = reporter.get_recalls_precisions()
                for i, k in enumerate(output['k']):
                    res['index'].append(index)
                    res['k'].append(k)
                    res['recall'].append(output['recall'][i])
                    res['precision'].append(output['precision'][i])
            self.cache = res
            return res
        else:
            return self.cache

    def aggregate_adjust(self):
        ref_k = 5000
        k_map = {
            'wikipedia_50': ref_k,
            'wikipedia_100': int(ref_k * (50 / 100)),
            'wikipedia_100_stride_50': int(ref_k * (50 / 100)),
            'wikipedia_150': int(ref_k * (50 / 150)),
            'wikipedia_200': int(ref_k * (50 / 200)),
            'wikipedia_paragraph': int(ref_k * (50 / 61)),
            'wikipedia': int(ref_k * (50 / 372)),
        }
        if self.cache is None:
            res = {
                'index': [],
                'k': [],
                'recall': [],
                'precision': []
            }
            for reporter in self.reporters:
                index = reporter.json_path.split('/')[0]
                max_k = k_map[index]
                output = reporter.get_recalls_precisions(end=max_k)
                cur_max_k = max(output['k'])
                cur_min_k = min(output['k'])
                target_max_k = ref_k
                target_min_k = cur_min_k
                for i, k in enumerate(output['k']):
                    res['index'].append(index)
                    target_k = (target_max_k - target_min_k) * (k - cur_min_k) / (cur_max_k - cur_min_k) + target_min_k
                    res['k'].append(int(target_k))
                    res['recall'].append(output['recall'][i])
                    res['precision'].append(output['precision'][i])
            self.cache = res
            return res
        else:
            return self.cache

    def plot_recalls(self, adjusted=True):
        agg = self.aggregate_adjust() if adjusted else self.aggregate()
        df = DataFrame(agg)
        plot_line(px, df, x='k', y='recall', title='Recalls', color='index')

    def plot_precisions(self, adjusted=True):
        agg = self.aggregate_adjust() if adjusted else self.aggregate()
        df = DataFrame(agg)
        plot_line(px, df, x='k', y='precision', title='Precisions', color='index')

    def plot_recalls_precisions(self, adjusted=True):
        agg = self.aggregate_adjust() if adjusted else self.aggregate()
        df = DataFrame(agg)
        plot_line(px, df, x='recall', y='precision', title='Precision-Recall', color='index')


def merge_outputs(json_paths: List[str], merged_filename):
    items = []
    for _path in json_paths:
        items += load_json(_path)
    if items:
        save_json(items, merged_filename)


def find_filename(directory: str, constraints: List[str]):
    result = []
    if os.path.isdir(directory):
        fs = os.listdir(directory)
        for f in fs:
            match = True
            for c in constraints:
                if c not in f:
                    match = False
                    break
            if match:
                result.append(f)
    return result


def plot_pr(adjusted=True):
    index_types = ['', '_50', '_paragraph', '_100', '_100_stride_50', '_150', '_200']
    json_paths = []
    for t in index_types:
        json_paths.append(f'wikipedia{t}/bm25/qa_BM25_wikipedia{t}_5000_merged.json')
    agg = OutputReporterAgg(json_paths)
    agg.plot_recalls(adjusted=adjusted)
    agg.plot_precisions(adjusted=adjusted)
    agg.plot_recalls_precisions(adjusted=adjusted)


def plot_inter_index_performance(dataset='triviaQA', error_bar=False):
    """
    Given a dataset, show its performance F1, EM over different indexes
    :param dataset:
    :return:
    """
    dataset = dataset + '-dev' if 'natural' not in dataset else dataset + '-dev-clean'
    if 'merged' in dataset:
        dataset = 'merged'
    index_types = ['_50', '_paragraph', '_100', '_100_stride_50', '_150', '_200', '']
    # index_types = ['_50', '_paragraph', '_100', '_100_stride_50', '_150', '_200']
    data = {
        'index': [],
        'f1_mean': [],
        'f1_sd': [],
        'em_mean': [],
        'em_sd': []
    }
    json_paths = []
    for t in index_types:
        directory = os.path.join(f'wikipedia{t}', 'bm25+electra')
        found = find_filename(directory, [dataset, '1000.json'])
        print(found)
        assert len(found) == 1
        json_paths.append(os.path.join(directory, found[0]))
    for j in json_paths:
        index = j.split('/')[0]
        items = load_json(j)
        f1s = [item['f1'] for item in items]
        ems = [item['em'] for item in items]
        data['index'].append(index)
        data['f1_mean'].append(np.mean(f1s))
        data['f1_sd'].append(np.std(f1s))
        data['em_mean'].append(np.mean(ems))
        data['em_sd'].append(np.std(ems))

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    f1_error_y = dict(
        type='data',  # value of error bar given in data coordinates
        array=data['f1_sd'],
        color='lightsteelblue',
        visible=True) if error_bar else None
    em_error_y = dict(
        type='data',  # value of error bar given in data coordinates
        array=data['em_sd'],
        color='lightcoral',
        visible=True) if error_bar else None
    marker = dict(size=10, color="steelblue")
    line = dict(width=2, color='lightcoral')
    fig.add_trace(
        go.Scatter(x=data['index'], y=data['f1_mean'],
                   mode='lines+markers',
                   error_y=f1_error_y,
                   marker=marker,
                   name='F1'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['index'], y=data['em_mean'],
                   mode='lines+markers',
                   error_y=em_error_y,
                   line=line,
                   name='EM'),
        row=1, col=1,
        secondary_y=True
    )

    # Add figure title
    fig.update_layout(
        title_text=f"F1 and EM over indexes (dataset: {dataset})"
    )
    # Set x-axis title
    fig.update_xaxes(title_text="<b>Index</b>")
    # Set y-axes titles, range can be adjusted by e.g. range=[-0.1, 1]
    max_val = max(data['f1_mean'] + data['em_mean']) * 1.1
    min_val = min(data['f1_mean'] + data['em_mean']) * 0.9
    fig.update_yaxes(title_text="<b>F1</b>", secondary_y=False, range=[min_val, max_val])
    fig.update_yaxes(title_text="<b>Exact Match</b>", secondary_y=True, range=[min_val, max_val])

    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='darkgrey', gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='darkgrey', gridwidth=1, gridcolor='lightgray')
    fig.show()


def plot_inter_index_performance_dist(dataset='triviaQA', plot_type='box'):
    """
    Given a dataset, show its performance distributions of F1, EM over different indexes
    :param dataset:
    :param plot_type: 'box' | 'strip' | 'violin'
    :return:
    """
    dataset = dataset + '-dev' if 'natural' not in dataset else dataset + '-dev-clean'
    if 'merged' in dataset:
        dataset = 'merged'
    index_types = ['_50', '_paragraph', '_100', '_100_stride_50', '_150', '_200', '']
    index_types = ['_50', '_paragraph', '_100', '_100_stride_50', '_150', '_200']
    data = {
        'index': [],
        'f1': [],
        'em': []
    }
    json_paths = []
    for t in index_types:
        directory = os.path.join(f'wikipedia{t}', 'bm25+electra')
        found = find_filename(directory, [dataset, '.json'])
        assert len(found) == 1
        json_paths.append(os.path.join(directory, found[0]))
    for j in json_paths:
        index = j.split('/')[0]
        items = load_json(j)
        f1s = [item['f1'] for item in items]
        ems = [item['em'] for item in items]
        for i, _ in enumerate(items):
            data['index'].append(index)
            data['f1'].append(f1s[i])
            data['em'].append(ems[i])

    df = DataFrame(data)
    plot_func = px.box
    if plot_type == 'strip':
        plot_func = px.strip
    elif plot_type == 'violin':
        plot_func = px.violin
    fig_f1 = plot_func(df, x='index', y='f1')
    fig_f1.show()


def plot_intra_index_performance(index='wikipedia_100_stride_50', error_bar=False):
    """
    Given an index, show the performances over different datasets
    :param dataset:
    :return:
    """
    datasets = ['naturalQuestions-dev-clean', 'quasarT-dev',
                'searchQA-dev', 'squad2-dev', 'triviaQA-dev', 'wikiQA-dev']
    directory = os.path.join(index, 'bm25+electra')
    data = {
        'dataset': [],
        'f1_mean': [],
        'f1_sd': [],
        'em_mean': [],
        'em_sd': []
    }
    for d in datasets:
        qa_filenames = find_filename(directory, [d, '.json'])
        assert len(qa_filenames) == 1
        items = load_json(os.path.join(directory, qa_filenames[0]))
        f1s = [item['f1'] for item in items]
        ems = [item['em'] for item in items]
        data['dataset'].append(d.replace('-dev-clean', '').replace('-dev', ''))
        data['f1_mean'].append(np.mean(f1s))
        data['f1_sd'].append(np.std(f1s))
        data['em_mean'].append(np.mean(ems))
        data['em_sd'].append(np.std(ems))

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    f1_error_y = dict(
        type='data',  # value of error bar given in data coordinates
        array=data['f1_sd'],
        color='lightsteelblue',
        visible=True) if error_bar else None
    em_error_y = dict(
        type='data',  # value of error bar given in data coordinates
        array=data['em_sd'],
        color='lightcoral',
        visible=True) if error_bar else None
    marker = dict(size=10, color="steelblue")
    line = dict(width=2, color='lightcoral')
    fig.add_trace(
        go.Scatter(x=data['dataset'], y=data['f1_mean'],
                   mode='lines+markers',
                   error_y=f1_error_y,
                   marker=marker,
                   name='F1'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['dataset'], y=data['em_mean'],
                   mode='lines+markers',
                   error_y=em_error_y,
                   line=line,
                   name='EM'),
        row=1, col=1,
        secondary_y=True
    )
    # Add figure title
    fig.update_layout(
        title_text=f"F1 and EM over datasets (index: {index})"
    )
    # Set x-axis title
    fig.update_xaxes(title_text="<b>Index</b>")
    # Set y-axes titles, range can be adjusted by e.g. range=[-0.1, 1]
    max_val = max(data['f1_mean'] + data['em_mean'])*1.1
    min_val = min(data['f1_mean'] + data['em_mean'])*0.9
    fig.update_yaxes(title_text="<b>F1</b>", secondary_y=False, range=[min_val, max_val])
    fig.update_yaxes(title_text="<b>Exact Match</b>", secondary_y=True, range=[min_val, max_val])

    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='darkgrey', gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='darkgrey', gridwidth=1, gridcolor='lightgray')
    fig.show()


if __name__ == '__main__':
    pass
