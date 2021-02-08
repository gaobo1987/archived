import os, sys, time
from tqdm import tqdm

sys.path.insert(1, os.path.join('..', 'common'))
from item_qa import ItemQA2
from utils import *
from evaluation import *
from params import args, READERS, DATASETS

ES_HOST = args.host
ES_PORT = args.port
ES_INDEX_NAME = args.index
USE_GPU = args.use_gpu
RETRIEVER_ES_TOP_K = args.retriever_es_k
READER_TOP_K = args.reader_k
SEED = args.seed
SUBSET = args.subset


def get_output_filename(reader_path, data_path):
    retriever = 'BM25_' + ES_INDEX_NAME
    reader_name = os.path.basename(reader_path)
    data_name = os.path.basename(data_path).replace('.json', '')
    output_name = f'qa_{retriever}__{RETRIEVER_ES_TOP_K}__{reader_name}__{data_name}'
    if SUBSET is not None:
        output_name += f'_{SUBSET}'
    return output_name + '.json'


def predict_and_evaluate(gold_qa_entry, retriever_es, reader):
    gold_answers = gold_qa_entry['answers']
    if len(gold_answers) == 0:
        return None
    question_id = gold_qa_entry['question_id']
    question = gold_qa_entry['question']
    pred_answer = ''
    es_ranks = []
    f1 = 0
    p = 0
    r = 0
    em = 0
    t = 0

    start_time = time.time()
    docs = retriever_es.retrieve(question, top_k=RETRIEVER_ES_TOP_K)
    if docs:
        prediction = reader.predict(question=question, documents=docs, top_k=READER_TOP_K)
        pred_answers = prediction['answers']
        if pred_answers:
            pred_answer = pred_answers[0]['answer']
        t = time.time() - start_time
        # eval
        es_doc_texts = [d.text for d in docs]
        es_ranks = retrieval_ranks_merge(gold_answers, es_doc_texts)
        f1, p, r = reader_f1_max(pred_answer, gold_answers)
        em = reader_match_max(exact_match_score, pred_answer, gold_answers)

    item = ItemQA2(question_id, question,
                   bm25_ranks=es_ranks,
                   f1=f1,
                   p=p,
                   r=r,
                   em=em,
                   t=t)
    item.add_pred_answer(pred_answer)
    for g_answer in gold_answers:
        item.add_gold_answer(g_answer)
    return item


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
            fw.write(f'* Reader top K = {READER_TOP_K}\n')
            fw.write(f'* USE_GPU = {USE_GPU}\n')
            fw.write(f'* Number of questions: {len(output_items)}\n')
            fw.write(f'* Seed = {SEED}\n')
            fw.write(f'------\n')

            fw.write('### BM25 Retrieval recall \n')
            bm25_ranks_list = [item['bm25_ranks'] for item in output_items]
            fw.write(f'* BM25 Recall @ 5: {round(recall_at_k(bm25_ranks_list, 5), round_num)}\n')
            fw.write(f'* BM25 Recall @ 10: {round(recall_at_k(bm25_ranks_list, 10), round_num)}\n')
            fw.write(f'* BM25 Recall @ 20: {round(recall_at_k(bm25_ranks_list, 20), round_num)}\n')
            fw.write(f'* BM25 Recall @ 50: {round(recall_at_k(bm25_ranks_list, 50), round_num)}\n')
            fw.write(f'* BM25 Recall @ 100: {round(recall_at_k(bm25_ranks_list, 100), round_num)}\n')
            fw.write('### BM25 Retrieval precision \n')
            fw.write(f'* BM25 Precision @ 5: {round(precision_at_k(bm25_ranks_list, 5), round_num)}\n')
            fw.write(f'* BM25 Precision @ 10: {round(precision_at_k(bm25_ranks_list, 10), round_num)}\n')
            fw.write(f'* BM25 Precision @ 20: {round(precision_at_k(bm25_ranks_list, 20), round_num)}\n')
            fw.write(f'* BM25 Precision @ 50: {round(precision_at_k(bm25_ranks_list, 50), round_num)}\n')
            fw.write(f'* BM25 Precision @ 100: {round(precision_at_k(bm25_ranks_list, 100), round_num)}\n')

            fw.write('### F1 \n')
            f1s = [item['f1'] for item in output_items]
            fw.write(f'* Mean F1 per q: {round(sum(f1s) / len(f1s), round_num)}\n')
            fw.write(f'* Median F1 per q: {round(np.median(f1s), round_num)}\n')
            fw.write(f'* Max F1 per q: {round(max(f1s), round_num)}\n')
            fw.write(f'* Min F1 per q: {round(min(f1s), round_num)}\n')
            fw.write(f'* Std F1 per q: {round(np.std(f1s), round_num)}\n')
            fw.write('### Precision \n')
            ps = [item['p'] for item in output_items]
            fw.write(f'* Mean precision per q: {round(sum(ps) / len(ps), round_num)}\n')
            fw.write(f'* Median precision per q: {round(np.median(ps), round_num)}\n')
            fw.write(f'* Max precision per q: {round(max(ps), round_num)}\n')
            fw.write(f'* Min precision per q: {round(min(ps), round_num)}\n')
            fw.write(f'* Std precision per q: {round(np.std(ps), round_num)}\n')
            fw.write('### Recall \n')
            rs = [item['r'] for item in output_items]
            fw.write(f'* Mean recall per q: {round(sum(rs) / len(rs), round_num)}\n')
            fw.write(f'* Median recall per q: {round(np.median(rs), round_num)}\n')
            fw.write(f'* Max recall per q: {round(max(rs), round_num)}\n')
            fw.write(f'* Min recall per q: {round(min(rs), round_num)}\n')
            fw.write(f'* Std recall per q: {round(np.std(rs), round_num)}\n')
            fw.write('### Exact Match \n')
            ems = [item['em'] for item in output_items]
            fw.write(f'* Mean EM per q: {round(sum(ems) / len(ems), round_num)}\n')
            fw.write(f'* Median EM per q: {round(np.median(ems), round_num)}\n')
            fw.write(f'* Max EM per q: {round(max(ems), round_num)}\n')
            fw.write(f'* Min EM per q: {round(min(ems), round_num)}\n')
            fw.write(f'* Std EM per q: {round(np.std(ems), round_num)}\n')
            fw.write('### Time(s) \n')
            ts = [item['t'] for item in output_items]
            fw.write(f'* Mean time per q: {round(sum(ts) / len(ts), 2)}s\n')
            fw.write(f'* Max time per q: {round(max(ts), 2)}s\n')
            fw.write(f'* Min time per q: {round(min(ts), 2)}s\n')
            fw.write(f'* Std time per q: {round(np.std(ts), 2)}s\n')


def save_output(output_filename: str, output_items: List[ItemQA2], logger):
    items_json = [item.json() for item in output_items]
    save_json(items_json, output_filename)
    logger.info(f'Output is saved to {output_filename}')
    summarize(output_filename)


def main():
    logger = get_logger('run_ES_Reader', 'run_ES_Reader.log')
    logger.info('----------------------------')
    logger.info('Parameters:')
    logger.info(f'RETRIEVER_ES_TOP_K = {RETRIEVER_ES_TOP_K}')
    logger.info(f'READER_TOP_K = {READER_TOP_K}')
    logger.info(f'USE_GPU = {USE_GPU}')
    logger.info('----------------------------')
    logger.info(f'connecting to ElasticSearch {ES_HOST}:{ES_PORT}@{ES_INDEX_NAME}...')
    document_store = get_elastic_search_document_store(ES_HOST, ES_PORT, ES_INDEX_NAME)
    logger.info('loading ES retriever...')
    retriever_es = get_elastic_search_retriever(document_store)
    for reader_path in READERS:
        logger.info(f'loading reader at {reader_path} ...')
        reader = get_neural_reader(reader_path, use_gpu=USE_GPU)
        for data_path in DATASETS:
            output_filename = get_output_filename(reader_path, data_path)
            logger.info(f'output filename: {output_filename}')
            output_items = []
            count = 0
            try:
                data = load_qa_data(data_path, seed=SEED, subset=SUBSET)
                logger.info(f'{len(data)} QA entries loaded')
                for qa in tqdm(data):
                    if count > 615:
                        item = predict_and_evaluate(qa, retriever_es, reader)
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
