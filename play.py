from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizer
from special_datasets import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from train import to_device
import torch

# with open("./pure_test_set/train.txt","w") as fout:
#     for i in range(10):
#         fout.write(str(i)*3+"\t"+str(i%2)+"\n")


# original_dataset = TxtDataset("./dataset/sst2/train.txt")
# smaller_dataset = ReorderedDataset(original_dataset,list(range(5000)))
# save_dataset(smaller_dataset,"./dataset/sst2_small/train.txt")

dataset = TxtDataset("./dataset/sst2/train.txt")
total_len = len(dataset) 
train_set = ReorderedDataset(dataset,list(range(total_len//2)))
test_set = ReorderedDataset(dataset,list(range(total_len//2,total_len)))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
loader = DataLoader(train_set, batch_size=8)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-6},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-6)

# optimizer1 = torch.optim.Adam(model.classifier.weight,lr=0.01,weight_decay=0.001)
# for step, batch in enumerate(tqdm(loader)):
#     b_y = batch[1].to("cuda")
#     input_dict = tokenizer(batch[0], return_tensors='pt', padding=True, truncation=True, max_length=128)
#     to_device(input_dict, "cuda")
#     loss_func = nn.CrossEntropyLoss()
#     output = model(**input_dict)
#     loss = loss_func(output.logits, b_y)
#     if step % 10 == 0:
#         print(loss)
#     optimizer1.zero_grad()
#     loss.backward()
#     optimizer1.step()

model.to("cuda")
loss_func = nn.CrossEntropyLoss()
model.train()
# for epoch in range(2):
#     for step, batch in enumerate(tqdm(loader)):
#         b_y = batch[1].to("cuda")
#         input_dict = tokenizer(batch[0], return_tensors='pt', padding=True, truncation=True, max_length=64)
#         to_device(input_dict, "cuda")
#         output = model(**input_dict)
#         loss = loss_func(output.logits, b_y)
#         if step%20==0:
#             print(loss)
#         optimizer.ero_grad()
#         loss.backward()
#         optimizer.step()
from train import train_model

train_model(tokenizer,model,train_set,torch.nn.CrossEntropyLoss,torch.optim.AdamW,2e-6,2,8,64,1e-6,42)

torch.save(model.state_dict(),"./play2.pth")
    

def evaluate(tokenizer,model:torch.nn.Module,dataset:Dataset,max_len:int,batch_size:int=32,cpu="cpu",gpu="cuda")->float:
    model.to(gpu)
    with torch.no_grad():
        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=False)
        correct = 0
        all = len(dataset)
        model.eval()
        for batch in tqdm(data_loader):
            b_y = batch[1]
            input_dict = tokenizer(batch[0], return_tensors='pt', padding=True, truncation=True, max_length=max_len)
            to_device(input_dict, gpu)
            output = (model(**input_dict).logits).to(cpu)
            correct+=int(b_y.eq(torch.max(output,1)[1]).sum())
        return float(correct)/all
    
print(evaluate(tokenizer,model,test_set,64,8))