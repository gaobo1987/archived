import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from bert_embedder import BertEmbedder
from pyserini import collection, index
from tqdm import tqdm
import math
import h5py
from utils import *


def show_doc_num_tokens_histogram():
    import matplotlib.pyplot as plt
    threshold = 50000
    total = 3213835
    c = collection.Collection('TrecCollection', '../collections/msmarco-doc/')
    generator = index.Generator('DefaultLuceneDocumentGenerator')
    num_tokens_per_doc = []
    for (i, fs) in enumerate(c):
        for (j, doc) in tqdm(enumerate(fs), total=threshold):
            if j < threshold:
                parsed = generator.create_document(doc)
                docid = parsed.get('id')
                raw = parsed.get('raw')
                content = parsed.get('contents').replace('\n', ' ').replace('\t', ' ').strip()
                docstr = clean(content)
                parts = docstr.split(' ')
                num = len(parts)
                num_tokens_per_doc.append(num)
            else:
                break

    plt.hist(num_tokens_per_doc, bins=50, range=(0, 5000))
    plt.show()


def extract_msmarco_docs():
    c = collection.Collection('TrecCollection', '../collections/msmarco-doc/')
    generator = index.Generator('DefaultLuceneDocumentGenerator')
    f = open('msmarco-doc-strings.txt', 'w', encoding='utf8')
    f500 = open('msmarco-docs-500.txt', 'w', encoding='utf8')
    f1000 = open('msmarco-docs-1000.txt', 'w', encoding='utf8')
    f5000 = open('msmarco-docs-5000.txt', 'w', encoding='utf8')
    f10000 = open('msmarco-docs-10000.txt', 'w', encoding='utf8')


    # docs = []
    for (i, fs) in enumerate(c):
        for (j, doc) in tqdm(enumerate(fs), total=3213835):
            parsed = generator.create_document(doc)
            docid = parsed.get('id')
            raw = parsed.get('raw')
            content = parsed.get('contents').replace('\n', ' ').replace('\t', ' ').strip()
            docstr = clean(content)
            f.write(docid+'\t'+docstr+'\n')
            f500.write(docid+'\t'+docstr[:500]+'\n')
            f1000.write(docid+'\t'+docstr[:1000]+'\n')
            f5000.write(docid+'\t'+docstr[:5000]+'\n')
            f10000.write(docid+'\t'+docstr[:10000]+'\n')


    f.close()
    f500.close()
    f1000.close()
    f5000.close()
    f10000.close()


# now if you already have extracted msmarco-docs-1000.txt, etc.
# split it into k pieces
def split_msmarco_docs():
    num = 10000
    total = 3213835
    size = 300000
    k = math.ceil(total / size)
    fks = []
    fk = None
    doc_filename = f'../data/msmarco-docs-{num}.tsv'
    f = open(doc_filename, 'r')
    for i, l in tqdm(enumerate(f), total=total):
        if i%size == 0:
            fk = open(f'../data/msmarco-docs-{num}-sub-{int(i/size+1)}.tsv', 'w')
            fks.append(fk)
        fk.write(l)

    for fk_i in fks:
        fk_i.close()
    f.close()


split_msmarco_docs()
