This repository about our method is based on EasyEdit about LoRA.

### 1. Environment Setup

```shell
cd EasyEdit
conda create -n editpgi python=3.10.3
conda activate editpgi
pip install -r requirements.txt
```



### 2. Benchmark

MQuAKE Benchmark (cite in our paper)



### 3. Knowledge Editing

```shell
cd examples
python run_knowedit_llama2.py     --editing_method=LoRA     --hparams_dir=../hparams/LoRA/llama-7b      --data_dir=./data/data/mquake_easyedit_3k_1_add_alias.json     --datatype='mquake' 

# 1 edited, 100 edited and all edited settings are adjusted through the variable '_sequential_edit' in file 'EasyEdit/examples/run_knowedit_llama2.py' and the variable 'chunk_size' in file 'EasyEdit/easyeditor/editors/editor.py':
```

|                  | 1 edited | 100 edited | all edited    |
| ---------------- | -------- | ---------- | ------------- |
| _sequential_edit | False    | True       | True          |
| chunk_size       | None     | 100        | len(requests) |