import os, sys, time
from haystack import Finder
from haystack.database.base import Document
from IPython.display import display, Markdown

sys.path.insert(1, os.path.join('..', 'common'))
from utils import *
from evaluation import *

logging.disable(logging.WARNING)

HOST = 'localhost'
PORT = 9200
INDEX_NAME = 'wikipedia_100_stride_50'
READER_DiR = '../models/electra-base-squad2'
DPR_MODEL_PATH = '../models/dpr/multi_hf_bert_base.cp'
USE_GPU = True
GPU_INDEX = 0 if USE_GPU else -1

document_store = ElasticsearchDocumentStore(host=HOST, port=PORT, username="", password="", index=INDEX_NAME)
retriever_es = ElasticsearchRetriever(document_store=document_store)
retriever_dpr = DensePassageRetriever(document_store=document_store,
                                      embedding_model=DPR_MODEL_PATH,
                                      use_gpu=USE_GPU,
                                      batch_size=16,
                                      do_lower_case=True)
reader = TransformersReader(model=READER_DiR, tokenizer=READER_DiR, use_gpu=GPU_INDEX)
faiss_index = get_faiss_ip_index(d=768, use_gpu=USE_GPU)
finder = Finder(reader, retriever_es)


def print_demo_answers(prediction):
    s = ''
    answers = prediction['answers']
    for i, a in enumerate(answers):
        rank = i + 1
        s += f'##### Answer {rank}: {a["answer"]}\n'
        s += f'* Context: \"{a["context"]}\"\n'
        s += f'* Article: {a["meta"]["name"]}\n'
        s += f'* URL: {a["meta"]["url"]}\n\n'
    display(Markdown(s))


def find_doc_by_id(_docs, _id) -> (Document, int):
    for i, d in enumerate(_docs):
        if d.id == _id:
            return d, i
    return None, -1


def answer(question, mode, retriever_es_k=100, retriever_dpr_k=10, reader_k=3):
    if mode == 'ranker+reader':
        prediction = finder.get_answers(question=question,
                                        top_k_retriever=retriever_es_k,
                                        top_k_reader=reader_k)
        print_demo_answers(prediction)

    elif mode == 'ranker+reader+scorer':
        es_docs = retriever_es.retrieve(question, top_k=retriever_es_k)
        prediction = reader.predict(question=question, documents=es_docs, top_k=retriever_es_k)
        answers = prediction['answers']
        if answers:
            mu = 0.5
            reader_scores = []
            ranker_scores = []
            result = []
            for i, pred_answer in enumerate(answers):
                doc_id = pred_answer['document_id']
                es_doc, _ = find_doc_by_id(es_docs, doc_id)
                assert es_doc is not None
                reader_score = float(pred_answer['probability'])
                reader_scores.append(reader_score)
                ranker_score = es_doc.query_score
                ranker_scores.append(ranker_score)
                result.append({
                    'reader_score': reader_score,
                    'ranker_score': ranker_score,
                    'answer': pred_answer['answer'],
                    'context': pred_answer['context'],
                    'meta': pred_answer['meta']
                })

            min_reader_s = min(reader_scores)
            max_reader_s = max(reader_scores)
            min_ranker_s = min(ranker_scores)
            max_ranker_s = max(ranker_scores)
            for item in result:
                reader_s = normalize_min_max(min_reader_s, max_reader_s, item['reader_score'])
                ranker_s = normalize_min_max(min_ranker_s, max_ranker_s, item['ranker_score'])
                item['b'] = (1 - mu) * ranker_s + mu * reader_s

            result = sorted(result, key=lambda x: x['b'], reverse=True)
            new_prediction = {
                'answers': result[:reader_k]
            }
            print_demo_answers(new_prediction)

    elif mode == 'ranker+dpr+reader':
        docs = retriever_es.retrieve(question, top_k=retriever_es_k)
        q_vecs = retriever_dpr.embed_queries([question])
        es_doc_texts = [normalize_text(d.text) for d in docs]
        p_vecs = retriever_dpr.embed_passages(es_doc_texts)
        faiss_index.add(np.array(p_vecs))
        D, I = faiss_index.search(np.array(q_vecs), retriever_es_k)
        dpr_docs = [docs[I[0][i]] for i in range(retriever_dpr_k)]
        prediction = reader.predict(question=question, documents=dpr_docs, top_k=reader_k)
        faiss_index.reset()
        print_demo_answers(prediction)
