import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import argparse
from transformers import BertTokenizer,BertForSequenceClassification,BertConfig
from special_datasets import TxtDataset

def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)


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

# python ./eval.py --max_len 128 --n_label 4 --testset_path ./agnews/test.txt --model_path /disks/sdb/zjp/temp/saved_models/best_origin_agnews_lfCE_ths0.0_e4_b32_lr2e-05_l21e-06_ml128.pth

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,required=True)
    parser.add_argument("--testset_path",type=str,required=True)
    parser.add_argument("--max_len",type=int,required=True)
    parser.add_argument("--n_label",type=int,default=2)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    model_path = args.model_path
    testset_path = args.testset_path
    max_len = args.max_len
    n_label = args.n_label
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_pretrained("bert-base-uncased",num_labels = n_label)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",config=config)
    model.load_state_dict(torch.load(model_path))
    dataset = TxtDataset(testset_path)
    accuracy = evaluate(tokenizer,model,dataset,max_len)
    print(accuracy)
    