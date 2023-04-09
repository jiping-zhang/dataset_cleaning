from typing import *

import os
import argparse
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, BertTokenizer, BertForSequenceClassification

from special_datasets import *
from seeds import set_seed
import train
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--lr", default=0.00002, type=float)
    parser.add_argument("--weight_decay", default=0.00001, type=float)
    parser.add_argument("--max_len", required=True, type=int)
    parser.add_argument("--loss_func", default='CE', type=str)
    parser.add_argument("--clean_method",default='ths',type=str)
    parser.add_argument('--ths', default=0.0, type=float)
    parser.add_argument('--ratio',default=0.0,type=float)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--output_folder_path", default=None, type=str)
    parser.add_argument("--temp_folder_path", default=None, type=str)
    parser.add_argument("--temp_model_path", default=None, type=str)
    parser.add_argument("--n_label", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_split", default=2, type=int)
    return parser.parse_args()


DEFAULT_GPU = "cuda" if torch.cuda.is_available() else "cpu"

BY_THRESHOLD="ths"
BY_RATIO="ratio"

args = get_args()
N_LABEL = args.n_label
SEED = args.seed
set_seed(SEED)
MAX_LENGTH = args.max_len
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
CLEAN_METHOD = args.clean_method
THRESHOLD = args.ths
RATIO = args.ratio
CLEAN_METHOD_REP_LETTER='U'
CLEAN_ARG = 0.0 
if CLEAN_METHOD==BY_THRESHOLD:
    assert THRESHOLD>0.0
    CLEAN_METHOD_REP_LETTER='T'
    CLEAN_ARG = THRESHOLD
if CLEAN_METHOD==BY_RATIO:
    assert 0.0<RATIO<1.0
    CLEAN_METHOD_REP_LETTER='R'
    CLEAN_ARG = RATIO
DATASET_NAME = args.dataset
DATASET_PATH = args.dataset_path
if DATASET_PATH is None: DATASET_PATH = f'./dataset/{DATASET_NAME}/train.txt'
VALIDSET_PATH = DATASET_PATH.replace("train.txt", "dev.txt")
LOSS_FUNC_NAME = args.loss_func
TEMP_FOLDER_PATH = args.temp_folder_path
if TEMP_FOLDER_PATH is None: TEMP_FOLDER_PATH = f'./dataset/{DATASET_NAME}/temp/'
LOSS_FUNC_CLASS = {
    'CE': torch.nn.CrossEntropyLoss,
    'MSE': torch.nn.MSELoss,
}[LOSS_FUNC_NAME]
TEMP_FOLDER_PATH = get_folder_abs_path(TEMP_FOLDER_PATH)
create_folder(TEMP_FOLDER_PATH)
OUTPUT_FOLDER_PATH = args.output_folder_path
if OUTPUT_FOLDER_PATH is None: OUTPUT_FOLDER_PATH = f'./dataset/{DATASET_NAME}/cleaned/'
OUTPUT_FOLDER_PATH = get_folder_abs_path(OUTPUT_FOLDER_PATH)
create_folder(OUTPUT_FOLDER_PATH)
TEMP_MODEL_PATH = args.temp_model_path
if TEMP_MODEL_PATH is None:
    TEMP_MODEL_PATH = "./temp/"
TEMP_MODEL_PATH = get_folder_abs_path(TEMP_MODEL_PATH)
create_folder(TEMP_MODEL_PATH)
N_SPLIT = args.n_split
CONFIG = BertConfig(num_labels=N_LABEL)
MODEL_TYPE = "bert-base-uncased"
CLEANED_DATASET_SAVE_FILE_NAME = f"{DATASET_NAME}_cleaned_cm{CLEAN_METHOD_REP_LETTER}({CLEAN_ARG})_split{N_SPLIT}_e{EPOCH}_b{BATCH_SIZE}_ml{MAX_LENGTH}_lr{LEARNING_RATE}_lf{LOSS_FUNC_NAME}_l2{WEIGHT_DECAY}_s{SEED}.txt"
CLEANED_DATASET_SAVE_FILE_PATH = f"{OUTPUT_FOLDER_PATH}{CLEANED_DATASET_SAVE_FILE_NAME}"

def get_datasets(original_dataset: Dataset, split_n: int, seed: int) -> Tuple[list, list]:
    # returns all the used datasets
    # return_value[0] = [dataset(0),dataset(1),...,dataset(n-1)]
    # return_value[1] = [dataset_neg(0),dataset_neg(1),...,dataset_neg(n-1)]
    # suppose there are <cnt> samples in original dataset
    # sd , shuffled_dataset , is the after-shuffle result of original_dataset by using seed to randomly shuffle it
    # dataset(i) means the i-th part of sd , that is , the i*cnt/split_n to (i+1)*cnt/split_n th samples
    # dataset_neg(i) means the sd - dataset(i) , that is , the dataset of all samples of sd which are not in dataset(i)
    # note : this function doesn't include AI computation or IO , it runs very fast.
    sample_cnt = len(original_dataset)
    indices_order = list(range(sample_cnt))
    shuffle_array(indices_order, seed)
    res0 = list()
    res1 = list()
    for i in range(split_n):
        split_start = (i * sample_cnt) // split_n
        split_end = ((i + 1) * sample_cnt) // split_n
        dataset_i = ReorderedDataset(original_dataset, indices_order[split_start:split_end])
        dataset_neg_i = ReorderedDataset(original_dataset, indices_order[:split_start] + indices_order[split_end:])
        res0.append(dataset_i)
        res1.append(dataset_neg_i)
    return (res0, res1)


def clean(tokenizer, model: nn.Module, dataset: Dataset, loss_func_class, threshold: float, gpu: str = DEFAULT_GPU,
           cpu: str = "cpu") -> Tuple[list,list]:
    reserved = []
    deleted = []
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    loss_func = loss_func_class()
    model.eval()
    model.to(gpu)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader)):
            b_x = batch[0]
            b_y = batch[1].to(gpu)
            input_dict = tokenizer(b_x, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH)
            train.to_device(input_dict, gpu)
            output = model(**input_dict)
            loss = loss_func(output.logits, b_y)
            loss = loss.to(cpu)
            if loss.detach().numpy() < threshold:
                reserved.append((b_x[0], b_y[0]))
            else:
                deleted.append((b_x[0], b_y[0]))
    return reserved, deleted


hyper_params = {
    'loss_func_class': torch.nn.CrossEntropyLoss,
    'optimizer_class': torch.optim.Adam,
    'lr': LEARNING_RATE,
    'epoch': EPOCH,
    'batch_size': BATCH_SIZE,
    'max_len': MAX_LENGTH,
    'weight_decay': WEIGHT_DECAY
}

train_set = TxtDataset(DATASET_PATH)
# for data in DataLoader(train_set,batch_size=2,shuffle=False,drop_last=False):
#     print(data)

pos_sets, neg_sets = get_datasets(train_set, N_SPLIT, SEED)


# _,splited_sets = get_datasets(train_set,N_SPLIT,SEED)
# for splited_set in splited_sets:
#     print("splited set:")
#     loader = DataLoader(splited_set,batch_size=2,drop_last=False)
#     for data in loader:
#         print(data)


tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
reserved_samples = []
for i in range(N_SPLIT):
    print(f"cleaning the subset({i})")
    this_model_save_name = f"{DATASET_NAME}_split{N_SPLIT}_neg({i})_e{EPOCH}_b{BATCH_SIZE}_ml{MAX_LENGTH}_lr{LEARNING_RATE}_lf{LOSS_FUNC_NAME}_l2{WEIGHT_DECAY}_s{SEED}.pth"
    this_model_save_path = f"{TEMP_MODEL_PATH}{this_model_save_name}"
    model = BertForSequenceClassification.from_pretrained(MODEL_TYPE, config=CONFIG)
    train.reload_or_train(this_model_save_path, copy_of_dict(hyper_params), tokenizer=tokenizer, model=model,
                          dataset=neg_sets[i], seed=SEED)
    reserved_this,deleted_this = clean(tokenizer,model,dataset=pos_sets[i],loss_func_class=LOSS_FUNC_CLASS,threshold=THRESHOLD)
    reserved_samples.extend(reserved_this)

reserved_dataset = ListDataset(reserved_samples)
save_dataset(reserved_dataset,CLEANED_DATASET_SAVE_FILE_PATH)