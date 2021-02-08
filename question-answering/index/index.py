from os import listdir
from os.path import isfile, join
import json
from tqdm import tqdm
from haystack.database.elasticsearch import ElasticsearchDocumentStore
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(data_dir, bulk_size, paragraph=False):
    document_store = ElasticsearchDocumentStore(host="localhost",
                                                username="",
                                                password="",
                                                index="wikipedia" + ("_paragraph" if paragraph else ""))
    if document_store.client.indices.exists(index=document_store.index):
        logger.info(f'{"wikipedia" + ("_paragraph" if paragraph else "")}')
        logger.warning(f"Index {document_store.index} already exists, deleting the index.")
        document_store.client.indices.delete(index=document_store.index)

    # Get all dirs in wikipedia folder
    only_dirs = [f for f in listdir(data_dir) if not isfile(join(data_dir, f))]

    dicts = []
    counts = dict(documents=0,
                  paragraphs=0)
    progress_bar = tqdm(only_dirs)
    for directory in progress_bar:
        sub_dirs = [f for f in listdir(join(data_dir, directory)) if not isfile(join(data_dir, directory))]
        progress_bar.set_description(f"Processing wikipedia folder {directory}")

        for file in sub_dirs:
            f = open(join(data_dir, directory, file), "r")

            # Each text file contains json structures separated by EOL
            articles = f.read().split("\n")

            for article in articles:
                if len(article) == 0:
                    continue

                # Article in json format
                json_formatted_article = json.loads(article)

                base_document = {"id": json_formatted_article["id"],
                                 "name": json_formatted_article["title"],
                                 "url": json_formatted_article["url"],
                                 }
                counts["documents"] += 1
                if paragraph:
                    """
                    - Paragraphs are separated by two new-line characters.
                    - The first paragraph is always the title --> remove!
                    - Some paragraphs only contain whitespace --> ignore
                    """
                    paragraphs = [p.strip() for pid, p in enumerate(json_formatted_article["text"].split("\n\n"))
                                  if pid > 0 and p.strip()]
                    counts["paragraphs"] += len(paragraphs)
                    for pid, p in enumerate(paragraphs):
                        document = {**base_document,
                                    "paragraph_id": pid,
                                    "text": p}

                        # Add document to bulk
                        dicts.append(document)

                else:
                    # Rename keys
                    document = {**base_document,
                                "text": json_formatted_article["text"]}

                    # Add document to bulk
                    dicts.append(document)

                if len(dicts) >= bulk_size:
                    # Index bulk
                    try:
                        document_store.write_documents(dicts)
                    except:
                        logger.warning("Bulk not indexed")

                    # Empty bulk
                    dicts = []

    # index the last partial batch
    if dicts:
        try:
            document_store.write_documents(dicts)
        except:
            logger.warning("Bulk not indexed")

    logger.info("==" * 100)
    logger.info("Indexing done.")
    logger.info(f"# documents: {counts['documents']}")
    if paragraph and counts['documents']:
        logger.info(f"# paragraphs: {counts['paragraphs']}, "
                    f"{counts['paragraphs'] / counts['documents']:.2f} per document")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory where the wikipedia dump is stored")
    parser.add_argument("-b", "--bulk-size",
                        type=int,
                        help="Bulk size of batches to write to ES, defaults to 5000",
                        default=25000)
    parser.add_argument('-p', '--paragraph',
                        dest='paragraph',
                        action='store_true',
                        help="",
                        default=False)
    args = parser.parse_args()

    main(args.dir, args.bulk_size, args.paragraph)
