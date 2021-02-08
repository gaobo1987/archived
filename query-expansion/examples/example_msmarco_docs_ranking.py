import faiss
import numpy as np
from tqdm import tqdm
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from bert_embedder import BertEmbedder


def init_gpu_faiss(d=768):
    res = faiss.StandardGpuResources()
    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    return gpu_index_flat


def init_cpu_faiss(d=768):
    index = faiss.IndexFlatL2(d)  # build the index
    print('index is trained: ', index.is_trained)
    print('ntotal: ', index.ntotal)
    return index


def add_doc_vecs_to_index(index, doc_vecs_file_path):
    batch_size = 3000
    npy = np.load(doc_vecs_file_path, allow_pickle=True)
    data = npy.item()
    print('number of docs: ', len(data))
    dkeys = []
    dvecs = []
    count = 0
    for k in tqdm(data, total=len(data)):
        dkeys.append(str(k))
        dvecs.append(list(data[k]).copy())
        count += 1
        if count == batch_size:
            count = 0
            index.add(np.array(dvecs, dtype=np.float32))
            dvecs.clear()
            data[k] = None

    if len(dvecs) > 0:
        index.add(np.array(dvecs, dtype=np.float32))
        dvecs.clear()
    return dkeys


def read_query_vecs(query_vecs_file_path):
    npy = np.load(query_vecs_file_path, allow_pickle=True)
    data = npy.item()
    print('number of queries: ', len(data))
    qkeys = []
    qvecs = []
    for k in tqdm(data, total=len(data)):
        qkeys.append(k)
        vec = list(data[k])
        qvecs.append(vec)

    # count = 0
    # for k in tqdm(data, total=len(data)):
        # count += 1
        # if count < 6:
        #     qkeys.append(k)
        #     vec = list(data[k])
        #     qvecs.append(vec)
        # else:
        #     break
    return qkeys, np.array(qvecs, dtype=np.float32)


if __name__ == '__main__':
    MAX_SEQ_LEN = 512
    VERSION = 'v2'
    CUT_NUM = 1000 if MAX_SEQ_LEN == 128 else 10000

    # q = 'hot glowing surfaces of stars emit energy in the form of electromagnetic radiation science & mathematics physics'
    # BERT_MODEL_PATH = '../models/bert/bert-base-uncased'
    # MAX_SEQ_LENGTH = 128
    # bert = BertEmbedder(BERT_MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH)
    # qvecs = bert.embed([q])
    # qvecs = np.array(qvecs, dtype=np.float32)

    print('====================================== Load queries ===============================================')
    query_filename = f'msmarco-queries-vecs-{MAX_SEQ_LEN}-{VERSION}.npy'
    print(query_filename)
    datadir = os.path.join('..', 'data')
    qkeys, qvecs = read_query_vecs(os.path.join(datadir, query_filename))

    print('================================ Load docs into index =============================================')
    doc_keys = []
    # index = init_gpu_faiss(d=768)
    index = init_cpu_faiss(d=768)
    # for k in range(1, 12):
    #     print(f'------- K = {k} -----------')
    #     filename = f'msmarco-docs-1000-sub-{k}-vecs-128.npy'
    #     filepath = os.path.join(datadir, filename)
    #     dkeys = add_doc_vecs_to_index(index, filepath)
    #     doc_keys += dkeys
    #     print('length of doc keys: ', len(doc_keys))

    doc_filename = f'msmarco-docs-{CUT_NUM}-vecs-{MAX_SEQ_LEN}-{VERSION}.npy'
    print(doc_filename)
    filepath = os.path.join(datadir, doc_filename)
    dkeys = add_doc_vecs_to_index(index, filepath)
    doc_keys += dkeys
    print('length of doc keys: ', len(doc_keys))

    print('================================== searching ================================================')
    output_filename = f'run-msmarco-doc-bert-{MAX_SEQ_LEN}-{VERSION}.txt'
    output_path = os.path.join('..', 'runs', output_filename)
    output = open(output_path, 'w')
    D, I = index.search(qvecs, 1000)
    print('writing output...', output_filename)
    for query_index, doc_indices in tqdm(enumerate(I), total=len(I)):
        query_id = qkeys[query_index]
        for i, doc_index in enumerate(doc_indices):
            doc_id = doc_keys[doc_index]
            distance = D[query_index][i]
            rank = i+1
            score = 1/distance if distance > 0 else 1000
            l = f'{query_id} Q0 {doc_id} {rank} {score} Bert-128\n'
            output.write(l)
    output.close()



    # with open('../data/msmarco-docs-1000-sub-1.tsv', 'r') as f:
    #     for l in tqdm(f, total=300000):
    #         docid = l.split('\t')[0]
    #         if docid in ['D1236929', 'D3059524', 'D477742']:
    #             print(l)

