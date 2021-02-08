import os
import pyarrow as pa
from pyarrow.lib import Table
from tqdm import tqdm
import numpy as np


def main():
    import os
    import logging
    from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
    from generator import RagSequenceGenerator, RagTokenGenerator
    logging.getLogger('transformers.models.rag.retrieval_rag').setLevel(logging.INFO)

    model_name = 'rag-token-nq'
    model_path = os.path.abspath(os.path.join(model_name))
    generator_class = RagSequenceGenerator if 'sequence' in model_name else RagTokenGenerator
    model = generator_class.from_pretrained(model_path)
    q1 = "how many countries are in europe" # should give 54 => google says either 44 or 51
    q2 = 'who is the president of USA?'
    q3 = 'how big is the earth'
    q4 = 'who wrote the song yesterday?'
    q5 = 'who holds the record in 100m freestyle'
    questions = [q5, q2]
    print(questions)

    answers = model.predict(questions)
    print(answers)


def read_arrow(filename: str) -> Table:
    mmap = pa.memory_map(filename)
    f = pa.ipc.open_stream(mmap)
    table: Table = f.read_all()
    return table


def test_arrow_read():
    data_path = os.path.abspath(os.path.join('index', 'wiki_dpr-train.arrow'))
    print(data_path)
    table = read_arrow(data_path)
    print('column names:', table.column_names)
    embeddings: pa.lib.chunked_array() = table['embeddings']
    ids = table['id']
    print('embeddings type: ', type(embeddings), type(embeddings.to_numpy()))
    print(type(ids[0]), type(embeddings[0]), type(np.array(embeddings[0].as_py())))
    print()


def write_faiss_hnsw():
    import faiss
    print('initialize faiss index')
    d = 768
    ef_construction = 200
    ef_search = 128
    index = faiss.IndexHNSWFlat(d, 128, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    print('read arrow file')
    index_dir = './index'
    arrow_path = os.path.abspath(os.path.join(index_dir, 'wiki_dpr-train.arrow'))
    table = read_arrow(arrow_path)
    print('...len table: ', len(table))

    batch_size = 10
    print(f'add vectors, batch_size={batch_size}')
    for i in tqdm(range(0, len(table), batch_size)):
        vecs = table[i: i+batch_size]['embeddings']
        vecs = vecs.to_pylist()
        vecs = np.array(vecs, dtype=np.float32)
        index.add(vecs)

    output_name = f'index_hnsw_efsearch{ef_search}_efconstruct{ef_construction}.faiss'
    output_path = os.path.join(index_dir, output_name)
    print('save faiss index: ', output_path)
    faiss.write_index(index, output_path)
    print('finished')


def write_faiss_flat_ip():
    import faiss
    print('initialize faiss index')
    d = 768
    index = faiss.IndexFlatIP(d)

    print('read arrow file')
    index_dir = './index'
    arrow_path = os.path.abspath(os.path.join(index_dir, 'wiki_dpr-train.arrow'))
    table = read_arrow(arrow_path)
    print('...len table: ', len(table))

    batch_size = 10
    print(f'add vectors, batch_size={batch_size}')
    for i in tqdm(range(0, len(table), batch_size)):
        vecs = table[i: i + batch_size]['embeddings']
        vecs = vecs.to_pylist()
        vecs = np.array(vecs, dtype=np.float32)
        index.add(vecs)

    output_name = f'index_ip_flat.faiss'
    output_path = os.path.join(index_dir, output_name)
    print('save faiss index: ', output_path)
    faiss.write_index(index, output_path)
    print('finished')


def test_rag_token():
    from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

    tokenizer = RagTokenizer.from_pretrained("./rag-token-nq")
    retriever = RagRetriever.from_pretrained("./rag-token-nq", index_name="exact", use_dummy_dataset=True)
    model = RagTokenForGeneration.from_pretrained("./rag-token-nq", retriever=retriever)

    input_dict = tokenizer.prepare_seq2seq_batch("who holds the record in 100m freestyle", return_tensors="pt")

    generated = model.generate(input_ids=input_dict["input_ids"])
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])

    # should give michael phelps => sounds reasonable


if __name__ == '__main__':
    print('hello world')
    main()
    # test_arrow_read()
    # write_faiss_flat_ip()
