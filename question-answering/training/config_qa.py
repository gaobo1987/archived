import json, os, logging, torch, random, numpy


class ConfigQA:
    def __init__(self, config_path):
        self.logger = logging.getLogger(__name__)
        params = ConfigQA.load_json(config_path)
        self.output_dir = params['output_dir']
        self.cache_dir = params['cache_dir']
        self.data_dir = params['data_dir']
        self.train_data_file = params['train_data_file']
        self.train_data_path = os.path.join(self.data_dir, self.train_data_file)
        self.dev_data_file = params['dev_data_file']
        self.dev_data_path = os.path.join(self.data_dir, self.dev_data_file)
        self.test_data_file = params['test_data_file']
        self.test_data_path = os.path.join(self.data_dir, self.test_data_file)
        self.model_path = params['model_path']
        self.model_type = params['model_type']
        self.do_lower_case = params['do_lower_case']
        self.doc_stride = params['doc_stride']
        self.max_seq_length = params['max_seq_length']
        self.max_query_length = params['max_query_length']
        self.evaluate_during_training = params['evaluate_during_training']
        self.per_gpu_train_batch_size = params['per_gpu_train_batch_size']
        self.per_gpu_eval_batch_size = params['per_gpu_eval_batch_size']
        self.learning_rate = params['learning_rate']
        self.gradient_accumulation_steps = params['gradient_accumulation_steps']
        self.weight_decay = params['weight_decay']
        self.adam_epsilon = params['adam_epsilon']
        self.max_grad_norm = params['max_grad_norm']
        self.num_train_epochs = params['num_train_epochs']
        self.warmup_steps = params['warmup_steps']
        self.logging_steps = params['logging_steps']
        self.save_steps = params['save_steps']
        self.seed = params['seed']
        self.n_gpu = 0
        self.device = 'cpu'
        self._perform_checks()
        self.logger.info('Loaded configuration: ', self.to_dict())

    def _perform_checks(self):
        if self.doc_stride >= self.max_seq_length - self.max_query_length:
            self.logger.warning(
                "WARNING - You've set a doc stride which may be superior to the document length in some "
                "examples. This could result in errors when building features from the examples. Please reduce the doc "
                "stride or increase the maximum length to ensure the features are correctly built."
            )
        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            raise ValueError(
                f'Output directory ({self.output_dir}) already exists and is not empty.'
            )
        # Setup CUDA, GPU training
        self.n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.warning(
            "Process rank: device: %s, n_gpu: %s",
            self.device,
            self.n_gpu
        )
        self._set_seed()

    def _set_seed(self):
        random.seed(self.seed)
        numpy.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def to_dict(self):
        return self.__dict__.copy()

    @staticmethod
    def load_json(input_filename):
        with open(input_filename, 'r', encoding='utf8') as f:
            return json.load(f)
