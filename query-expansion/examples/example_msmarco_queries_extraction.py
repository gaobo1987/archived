print('====================== loading queries from topics ==============================')
from pyserini.search import get_topics
from tqdm import tqdm
topic_set = 'msmarco_doc_dev'
topics = get_topics(topic_set)
print(f'{len(topics)} topics total')

for id in tqdm(topics):
    query = topics[id]['title']
    print(id, query)


