# Training BERT NER Model Using OneAI

To successfully train a BERT model using this code, please follow carefully the following steps

**1- Start oneai project with the following configuration**

- Engine: PyTorch-1.0.0-python3.6
- Allow oneai to create training template
- In the training template that is opened in pycharm, replace the "training" directory with the one here
- Feel free to add your datasets if you want under the dataset directory and upload them to your obs bucket

**2- Get the dependencies by following these steps:**

- Download the transformer package [v2.11.0](https://github.com/huggingface/transformers/tree/v2.11.0) and place it under the "src"
directory and rename it to "transformers_pkg"

- If you want mixed precision training feature (not well tested), download [NVIDIA apex](https://github.com/NVIDIA/apex),
and place it under the "src"
directory and rename it to "apex-master". Tested on commit hash: "43a6f9fe91c242170cbc5c8bf13f466eaccab2e4"

- Download the segmentation evaluation tool wheel (v0.2.0) from [here](https://onebox.huawei.com/p/618d3aa0edea71f4e23970b7078af7ae)
and place it also under the "src" directory 

**3- Open the configuration file (in training/src/oneai_project_config.json) and edit it according to your needs**

Important fields that you **have to** edit

- data_dir: obs dir for your dataset
- bert_model: bert model path in obs
- output_dir: where you want to save the fine-tuned model in obs
- num_train_epochs: number of training epochs
- train_file_name: train file name without a full path. It should be located under the data_dir
- dev_file_name: dev file name without a full path. It should be located under the data_dir
- test_file_name: test file name without a full path. It should be located under the data_dir
- labels_file_name: labels file name (one label per line) without a full path. It should be located under the data_dir
- If you want to run evaluation only, make "do_train: false" and "do_eval: true". The evaluation is done using the model
saved in the "output_dir"

**4- Press Run!**