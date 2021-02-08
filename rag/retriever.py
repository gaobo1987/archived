import numpy as np
import os
from typing import List, Tuple, Optional, Union, Dict
from transformers.modeling_rag import RagRetriever
from transformers.retrieval_rag import HFIndexBase
from transformers import RagConfig, RagTokenizer
from datasets import Dataset, DatasetBuilder, Features, Split, DownloadConfig, GenerateMode, Version, \
    prepare_module, import_main_class, DatasetInfo, ArrowReader
from time import time
import faiss


class RagDprIndex(HFIndexBase):

    def __init__(self, vector_size: int):
        print('retrieval.py:DprRagIndex.__init__()')
        t = time()

        name = 'wiki_dpr'
        split: Split = Split.TRAIN
        index_dir = '/home/bo/workspace/rag/index'
        # load DatasetInfo
        info = DatasetInfo.from_directory(index_dir)
        # read index from Arrow file
        dataset_kwargs = ArrowReader(index_dir, info).read(
            name=name,
            instructions=split,
            split_infos=info.splits.values(),
        )
        dataset: Dataset = Dataset(**dataset_kwargs)

        # load the faiss index file into the dataset
        index_file = 'index_hnsw_efsearch_128_efconstruct_200.faiss'
        # index_file = 'index_ip_flat.faiss'
        index_file = os.path.join(index_dir, index_file)
        dataset.load_faiss_index("embeddings", index_file)
        # or construct the faiss index file into the dataset if it doesn't exist
        # e.g.
        # d = 768
        # index = faiss.IndexHNSWFlat(d, 128, faiss.METRIC_INNER_PRODUCT)
        # index.hnsw.efConstruction = 200
        # index.hnsw.efSearch = 128
        # dataset.add_faiss_index("embeddings", custom_index=index)

        super().__init__(vector_size, dataset, index_initialized=True)
        print(f'retrieval.py:DprRagIndex.__init__(): passages and index loaded in {time() - t} seconds')

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        raise NotImplementedError


class RagDprRetriever(RagRetriever):

    def __init__(self, model_path: str):
        print('retriever.py:DprRagRetriever.__init__()')
        # kwargs.update({
        #     'index_name': "exact",
        #     'use_dummy_dataset': True
        # })
        config = RagConfig.from_pretrained(model_path)
        rag_tokenizer = RagTokenizer.from_pretrained(model_path, config=config)
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator

        print('config.retrieval_vector_size:', config.retrieval_vector_size)
        print('config.dataset:', config.dataset)
        print('config.dataset_split:', config.dataset_split)
        print('config.index_name:', config.index_name)
        print('config.index_path:', config.index_path)
        print('config.use_dummy_dataset:', config.use_dummy_dataset)
        print()
        index = RagDprIndex(vector_size=config.retrieval_vector_size)

        # transformers 4.2.0.dev0 version:
        # super().__init__(config=config,
        #                  question_encoder_tokenizer=question_encoder_tokenizer,
        #                  generator_tokenizer=generator_tokenizer,
        #                  index=index,
        #                  init_retrieval=False)

        # transformers 3.5.1 version
        RagRetriever._init_retrieval = False
        super().__init__(config=config,
                         question_encoder_tokenizer=question_encoder_tokenizer,
                         generator_tokenizer=generator_tokenizer,
                         index=index)

