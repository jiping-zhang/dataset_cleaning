from typing import *

import pandas as pd

from utils import *
from special_datasets import *

SEED = 42

def predeal(dataset_name:str,seed:int=SEED):
    all_data_path = f"./dataset/{dataset_name}/all.txt"
    all_data = TxtDataset(all_data_path)
    train_valid_split = len(all_data)//2    # 50% train data
    valid_test_split = (3*len(all_data))//5 # 10% valid data + 40% test data
    data_indices = list(range(len(all_data)))
    shuffle_array(data_indices,seed)
    train_set = ReorderedDataset(all_data,data_indices[:train_valid_split])
    valid_set = ReorderedDataset(all_data,data_indices[train_valid_split:valid_test_split])
    test_set = ReorderedDataset(all_data,data_indices[valid_test_split:])
    save_dataset(train_set,f"./dataset/{dataset_name}/train.txt")
    save_dataset(valid_set,f"./dataset/{dataset_name}/dev.txt")
    save_dataset(test_set,f"./dataset/{dataset_name}/test.txt")
    
def parse_imdb():
    raw_data_path = "./dataset/imdb/IMDB Dataset.csv"
    pd_csv_data=pd.read_csv(raw_data_path,encoding="utf-8")
    all_data = list()
    for i in range(len(pd_csv_data["review"])):
        sentence = pd_csv_data["review"][i]
        sentence = sentence.replace("<br />"," ")
        sentence = sentence.replace("<br/>"," ")
        sentence = sentence.replace("\t"," ")
        label = int(pd_csv_data["sentiment"][i]=="positive")
        all_data.append((sentence,label))
    save_dataset(ListDataset(all_data),"./dataset/imdb/all.txt")
    
# predeal_sst2()
parse_imdb()
predeal("imdb")