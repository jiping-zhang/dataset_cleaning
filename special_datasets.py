from typing import *
from torch.utils.data import Dataset, DataLoader


class TxtDataset(Dataset):
    def __init__(self, path: str):
        super(TxtDataset, self).__init__()
        with open(path) as fin:
            raw = fin.readlines()
            after_process = list()
            for line in raw:
                if len(line)>0 and line.find("\t")>0:
                    after_process.append(line)
            self.lines = after_process

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        temp = line.split('\t')
        if temp==-1 or temp==len(line)-1:
            print(f"error:{index}-th sample {line} is not in standard format")
        sentence = temp[0]
        label = int(temp[1])
        return sentence, label


class ReorderedDataset(Dataset):
    def __init__(self,original_dataset:Dataset,indices:List[int]) -> None:
        super().__init__()
        self.original_dataset = original_dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index: int):
        return self.original_dataset[self.indices[index]]


class ListDataset(Dataset):
    def __init__(self, datas:List[tuple]):
        super(ListDataset, self).__init__()
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index: int):
        data = self.datas[index]
        return data[0], data[1]


def save_dataset(dataset: Dataset, path: str):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    with open(path,'w') as fout:
        for b_x, b_y in data_loader:
            x = b_x[0]
            y = int(b_y[0])
            fout.write(x+"\t"+str(y)+"\n")

# dataset = TxtDataset('./sst2small/train.txt')
# save_dataset(dataset,'./sst2small/train(2).txt')
