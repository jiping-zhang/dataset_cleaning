from typing import *

from utils import *
from special_datasets import *

SEED = 42

def predeal_sst2():
    all_data_path = "./dataset/sst2/all.txt"
    all_data = TxtDataset(all_data_path)
    train_valid_split = len(all_data)//2    # 50% train data
    valid_test_split = (3*len(all_data))//5 # 10% valid data + 40% test data
    data_indices = list(range(len(all_data)))
    shuffle_array(data_indices,SEED)
    train_set = ReorderedDataset(all_data,data_indices[:train_valid_split])
    valid_set = ReorderedDataset(all_data,data_indices[train_valid_split:valid_test_split])
    test_set = ReorderedDataset(all_data,data_indices[valid_test_split:])
    save_dataset(train_set,"./dataset/sst2/train.txt")
    save_dataset(valid_set,"./dataset/sst2/dev.txt")
    save_dataset(test_set,"./dataset/sst2/test.txt")
    
predeal_sst2()