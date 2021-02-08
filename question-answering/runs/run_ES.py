import os, sys, time
import numpy as np
from tqdm import tqdm

sys.path.insert(1, os.path.join('..', 'common'))
from item_qa import ItemQA2
from utils import *
from evaluation import *
from params import args, DATASETS

ES_HOST = args.host
ES_PORT = args.port
ES_INDEX_NAME = args.index
USE_GPU = args.use_gpu
RETRIEVER_ES_TOP_K = args.retriever_es_k
SEED = args.seed
SUBSET = args.subset


def get_output_filename(data_path):
    retriever = 'BM25_' + ES_INDEX_NAME
    data_name = os.path.basename(data_path).replace('.json', '')
    output_name = f'qa_{retriever}__{RETRIEVER_ES_TOP_K}__{data_name})'
    if SUBSET is not None:
        output_name += f'_{SUBSET}'
    return output_name + '.json'


def retrieve_and_evaluate(gold_qa_entry, retriever_es):
    gold_answers = gold_qa_entry['answers']
    if len(gold_answers) == 0:
        return None
    try:
        question_id = gold_qa_entry['question_id']
        question = gold_qa_entry['question']
        es_ranks = []
        t = 0
        start_time = time.time()
        docs = retriever_es.retrieve(question, top_k=RETRIEVER_ES_TOP_K)
        if docs:
            t = time.time() - start_time
            es_doc_texts = [d.text for d in docs]
            es_ranks = retrieval_ranks_merge(gold_answers, es_doc_texts)

        item = ItemQA2(question_id, question,
                       bm25_ranks=es_ranks,
                       t=t)
        for g_answer in gold_answers:
            item.add_gold_answer(g_answer)
        return item
    except Exception as e:
        print(e)
        return None


def summarize(output_filename: str):
    with open(output_filename, 'r', encoding='utf8') as f:
        output_items = json.load(f)
        summary_filename = output_filename.replace('.json', '_summary.md')
        pipeline_name = output_filename.replace(".json", "").replace("qa_", "")
        with open(summary_filename, 'w', encoding='utf8') as fw:
            round_num = 3
            fw.write('### Pipeline Parameters:\n')
            fw.write(f'* Name: {pipeline_name}\n')
            fw.write(f'* BM25 top K = {RETRIEVER_ES_TOP_K}\n')
            fw.write(f'* USE_GPU = {USE_GPU}\n')
            fw.write(f'* Number of questions: {len(output_items)}\n')
            fw.write(f'* Seed = {SEED}\n')
            fw.write(f'------\n')

            bm25_ranks_list = [item['bm25_ranks'] for item in output_items]
            fw.write('### BM25 Retrieval recall \n')
            fw.write(f'* BM25 Recall @ 5: {round(recall_at_k(bm25_ranks_list, 5), round_num)}\n')
            fw.write(f'* BM25 Recall @ 10: {round(recall_at_k(bm25_ranks_list, 10), round_num)}\n')
            fw.write(f'* BM25 Recall @ 20: {round(recall_at_k(bm25_ranks_list, 20), round_num)}\n')
            fw.write(f'* BM25 Recall @ 50: {round(recall_at_k(bm25_ranks_list, 50), round_num)}\n')
            fw.write(f'* BM25 Recall @ 100: {round(recall_at_k(bm25_ranks_list, 100), round_num)}\n')
            fw.write(f'* BM25 Recall @ 500: {round(recall_at_k(bm25_ranks_list, 500), round_num)}\n')
            fw.write(f'* BM25 Recall @ 1000: {round(recall_at_k(bm25_ranks_list, 1000), round_num)}\n')
            fw.write(f'* BM25 Recall @ 1500: {round(recall_at_k(bm25_ranks_list, 1500), round_num)}\n')
            fw.write(f'* BM25 Recall @ 2000: {round(recall_at_k(bm25_ranks_list, 2000), round_num)}\n')
            fw.write('### BM25 Retrieval precision \n')
            fw.write(f'* BM25 Precision @ 5: {round(precision_at_k(bm25_ranks_list, 5), round_num)}\n')
            fw.write(f'* BM25 Precision @ 10: {round(precision_at_k(bm25_ranks_list, 10), round_num)}\n')
            fw.write(f'* BM25 Precision @ 20: {round(precision_at_k(bm25_ranks_list, 20), round_num)}\n')
            fw.write(f'* BM25 Precision @ 50: {round(precision_at_k(bm25_ranks_list, 50), round_num)}\n')
            fw.write(f'* BM25 Precision @ 100: {round(precision_at_k(bm25_ranks_list, 100), round_num)}\n')
            fw.write(f'* BM25 Precision @ 500: {round(precision_at_k(bm25_ranks_list, 500), round_num)}\n')
            fw.write(f'* BM25 Precision @ 1000: {round(precision_at_k(bm25_ranks_list, 1000), round_num)}\n')
            fw.write(f'* BM25 Precision @ 1500: {round(precision_at_k(bm25_ranks_list, 1500), round_num)}\n')
            fw.write(f'* BM25 Precision @ 2000: {round(precision_at_k(bm25_ranks_list, 2000), round_num)}\n')


def save_output(output_filename: str, output_items: List[ItemQA2], logger):
    logger.info(f'Saving {len(output_items)} items...')
    items_json = [item.json() for item in output_items]
    save_json(items_json, output_filename)
    logger.info(f'Output is saved to {output_filename}')
    summarize(output_filename)


def main():
    logger = get_logger('run_ES_Reader', 'run_ES_Reader.log')
    logger.info('----------------------------')
    logger.info('Parameters:')
    logger.info(f'RETRIEVER_ES_TOP_K = {RETRIEVER_ES_TOP_K}')
    logger.info(f'USE_GPU = {USE_GPU}')
    logger.info('----------------------------')
    logger.info(f'connecting to ElasticSearch {ES_HOST}:{ES_PORT}@{ES_INDEX_NAME}...')
    document_store = get_elastic_search_document_store(ES_HOST, ES_PORT, ES_INDEX_NAME)
    logger.info('loading ES retriever...')
    retriever_es = get_elastic_search_retriever(document_store)
    for data_path in DATASETS:
        output_filename = get_output_filename(data_path)
        logger.info(f'output filename: {output_filename}')
        output_items = []
        count = 0
        try:
            data = load_qa_data(data_path, seed=SEED, subset=SUBSET)
            logger.info(f'{len(data)} QA entries loaded')
            for qa in tqdm(data):
                if count >= 0:
                    item = retrieve_and_evaluate(qa, retriever_es)
                    if item is not None:
                        output_items.append(item)
                count += 1
            save_output(output_filename, output_items, logger)

        except (Exception, KeyboardInterrupt) as e:
            logger.info(e)
            logger.info(f'An error occurred at Count {count}, saving what we have now...')
            save_output(output_filename, output_items, logger)


if __name__ == '__main__':
    main()
