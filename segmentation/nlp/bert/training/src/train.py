from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import os
import random
import numpy as np
import torch

import moxing.pytorch as mox
mox.file.shift('os', 'mox')

# transformers installation
transformers_dir = os.path.join(os.getcwd(), 'src', 'transformers_pkg')
os.system(f'pip install {transformers_dir}')

# apex installation
apex_dir = os.path.join(os.getcwd(), 'src', 'apex-master')
if os.path.isdir(apex_dir):
    os.system(f'pip install -v --no-cache-dir {apex_dir}')

from transformers import BertConfig, BertForTokenClassification, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from utils import BertForNer, NerProcessor, convert_examples_to_features, get_data_loader, evaluate, train

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    oneai_config_path = os.path.join(os.getcwd(), 'src', 'oneai_project_config.json')
    with open(oneai_config_path, 'r') as f:
        oneai_config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default=oneai_config['data_dir'],
                        type=str,
                        help='The input data dir. Should contain the .tsv files (or other data files) for the task.')
    parser.add_argument('--bert_model', default=oneai_config['bert_model'], type=str,
                        help='Bert pre-trained model selected in the list: bert-base-uncased, '
                             'bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, '
                             'bert-base-multilingual-cased, bert-base-chinese.')
    parser.add_argument('--task_name',
                        default='ner',
                        type=str,
                        help='The name of the task to train.')
    parser.add_argument('--output_dir',
                        default=oneai_config['output_dir'],
                        type=str,
                        help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--train_file_name',
                        default=oneai_config['train_file_name'],
                        type=str,
                        help='train file name excluding the full path')
    parser.add_argument('--test_files_names',
                        nargs='+',
                        default=oneai_config['test_files_names'],
                        type=str,
                        help='list of test files names excluding the full path')
    parser.add_argument('--dev_file_name',
                        default=oneai_config['dev_file_name'],
                        type=str,
                        help='train file name excluding the full path')
    parser.add_argument('--labels_file_name',
                        default=oneai_config['labels_file_name'],
                        type=str,
                        help='labels file name with one NER label per line excluding the full path.')
    parser.add_argument('--data_url',
                        default='',
                        type=str
                        )
    parser.add_argument('--init_method',
                        default='',
                        type=str
                        )
    parser.add_argument('--train_url',
                        default='',
                        type=str
                        )
    parser.add_argument('--max_seq_length',
                        default=oneai_config['max_seq_length'],
                        type=int,
                        help='The maximum total input sequence length after WordPiece tokenization. \n'
                             'Sequences longer than this will be truncated, and sequences shorter \n'
                             'than this will be padded.')
    parser.add_argument('--do_train',
                        default=oneai_config['do_train'],
                        help='Whether to run training.')
    parser.add_argument('--do_eval',
                        default=oneai_config['do_eval'],
                        help='Whether to run eval or not.')
    parser.add_argument('--do_lower_case',
                        default=oneai_config['do_lower_case'],
                        help='Set this to True if you are using an uncased model.')
    parser.add_argument('--train_batch_size',
                        default=oneai_config['train_batch_size'],
                        type=int,
                        help='Total batch size for training.')
    parser.add_argument('--eval_batch_size',
                        default=oneai_config['eval_batch_size'],
                        type=int,
                        help='Total batch size for eval.')
    parser.add_argument('--learning_rate',
                        default=oneai_config['learning_rate'],
                        type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--num_train_epochs',
                        default=oneai_config['num_train_epochs'],
                        type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--warmup_proportion',
                        default=oneai_config['warmup_proportion'],
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10%% of training.')
    parser.add_argument('--weight_decay', default=oneai_config['weight_decay'], type=float,
                        help='Weight deay if we apply some.')
    parser.add_argument('--adam_epsilon', default=oneai_config['adam_epsilon'], type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=oneai_config['max_grad_norm'], type=float,
                        help='Max gradient norm.')
    parser.add_argument('--no_cuda',
                        default=oneai_config['no_cuda'],
                        help='Whether not to use CUDA when available')
    parser.add_argument('--local_rank',
                        type=int,
                        default=oneai_config['local_rank'],
                        help='local_rank for distributed training on gpus')
    parser.add_argument('--seed',
                        type=int,
                        default=oneai_config['seed'],
                        help='random seed for initialization')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=oneai_config['gradient_accumulation_steps'],
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--fp16',
                        default=oneai_config['fp16'],
                        help='Whether to use 16-bit float precision instead of 32-bit')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help='For fp16: Apex AMP optimization level selected in ["O0", "O1", "O2", and "O3"].'
                             'See details at https://nvidia.github.io/apex/amp.html')
    parser.add_argument('--loss_scale',
                        type=float, default=oneai_config['loss_scale'],
                        help='Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n'
                             '0 (default value): dynamic loss scaling.\n'
                             'Positive power of 2: static loss scaling value.\n')

    args = parser.parse_args()

    processors = {'ner': NerProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info('device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError('At least one of `do_train` or `do_eval` must be True.')

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError('Output directory ({}) already exists and is not empty.'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)

    processor = processors[task_name](args.data_dir)
    label_list = processor.get_labels(args.labels_file_name)

    if args.do_train:

        if args.gradient_accumulation_steps < 1:
            raise ValueError('Invalid gradient_accumulation_steps parameter: {}, should be >= 1'.format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        num_labels = len(label_list) + 1

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        train_examples = processor.get_train_examples(args.train_file_name)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        # Prepare model
        config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
        model = BertForNer.from_pretrained(args.bert_model, from_tf=False, config=config)

        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info('***** Running training *****')
        logger.info('  Num examples = %d', len(train_examples))
        logger.info('  Batch size = %d', args.train_batch_size)
        logger.info('  Num steps = %d', num_train_optimization_steps)
        train_dataloader = get_data_loader(train_features, args.train_batch_size,
                                           randomize=True, local_rank=args.local_rank)
        model = train(model, train_dataloader, args.num_train_epochs, args.gradient_accumulation_steps,
                      n_gpu, args.fp16, optimizer, scheduler, args.max_grad_norm, device)
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        model_config = {'bert_model': args.bert_model,
                        'do_lower': args.do_lower_case,
                        'max_seq_length': args.max_seq_length,
                        'num_labels': len(label_list) + 1,
                        'label_map': label_map}
        with open(os.path.join(args.output_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f)
        # Load a trained model and config that you have fine-tuned
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForNer.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        dev_examples = processor.get_dev_examples(args.dev_file_name)
        test_examples_list = [(test_file, processor.get_test_examples(test_file))
                              for test_file in args.test_files_names]
        eval_examples_list = [(args.dev_file_name, dev_examples)] + test_examples_list
        for eval_set, eval_examples in zip(('dev', 'test'), eval_examples_list):
            eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
            logger.info('***** Running evaluation *****')
            logger.info('  Num examples = %d', len(eval_examples))
            logger.info('  Batch size = %d', args.eval_batch_size)
            eval_dataloader = get_data_loader(eval_features, args.eval_batch_size, randomize=False)
            results, results_per_label = evaluate(model, eval_dataloader, label_list, device, get_ner_metrics=True)
            logger.info(f'####EVAL RESULTS on {eval_set} set####')
            for k, v in results.items():
                logger.info(f'{k} --> {v}')
            logger.info(f'####EVAL LABEL RESULTS on {eval_set} set####')
            for k, v in results_per_label.items():
                logger.info(f'{k} --> {v}')
            eval_file = os.path.join(args.output_dir, f'{eval_set}_results.txt')
            with open(eval_file, 'w') as writer:
                writer.write('####GLOBAL RESULTS####\n')
                for k, v in results.items():
                    writer.write(k + '-->' + str(v) + '\n')
                writer.write('####LABEL RESULTS####\n')
                for k, v in results_per_label.items():
                    writer.write(k + '-->' + str(v) + '\n')


if __name__ == '__main__':
    main()
