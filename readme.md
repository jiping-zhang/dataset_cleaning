# dataset clarify

### clarified datasets

we provide some clarified datasets in the folder "./clarified_datasets"

### environment to run the code

only 4 packages are required to run the code.

numpy

pytorch

transformers

tqdm

You can run the code if you have already installed them

or you can install these packages according to the requirements.txt provided by us

### how to use code

```sh
python ./clearify.py --epoch EPOCH --batch_size BATCH_SIZE --lr LR --weight_decay WEIGHT_DECAY --max_len MAX_LEN [--loss_func LOSS_FUNC] --ths THS --dataset DATASET [--dataset_path DATASET_PATH] [--output_folder_path OUTPUT_FOLDER_PATH] [--temp_folder_path TEMP_FOLDER_PATH] [--n_label N_LABEL] [--seed SEED]
```

ths means threshold , samples whose loss (calculated by the specified loss function) is bigger than the threshold will be removed. The smaller it is ,the more sentences will be removed.

(If you want more detailed information about threshold , please turn to our essay)

#### dataset file

dataset file should be like this:

```
I like this movie very much.	1
I have no idea what this movie is about.	0
It's really a waste of time.	0

```

the sentence and the index of class it belongs to should be split by '\\t'.

#### default values:

parameters with [] have default value.

| parameter name     | default value         |
| ------------------ | --------------------- |
| loss_func          | cross entropy loss    |
| dataset_path       | ./{dataset}/train.txt |
| output_folder_path | ./{dataset}/          |
| temp_folder_path   | {output_folder_path}  |
| n_label            | 2                     |
| seed               | 42                    |

### example

```sh
python ./clearify.py --epoch 4 --batch_size 32 --lr 2e-5 --weight_decay 1e-5 --max_len MAX_LEN 256 --ths 0.2 --dataset imdb --dataset_path ./imdb/train.txt --output_folder_path ./imdb/clarify/ --temp_folder_path ./temp/saved_models/imdb/
```
