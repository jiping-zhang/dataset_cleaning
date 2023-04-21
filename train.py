from typing import *

import os
from tqdm import tqdm
import time  # todo:remove this

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad
from torch.optim.lr_scheduler import CosineAnnealingLR

from seeds import set_seed
from eval import evaluate
from utils import try_save

DEFAULT_GPU = "cuda" if torch.cuda.is_available() else "cpu"


def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)


def delta_update(adv_learning_rate, delta: torch.Tensor, delta_grad: torch.Tensor) -> torch.Tensor:
    denorm = torch.norm(delta_grad.view(
        delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
    denorm = torch.clamp(denorm, min=1e-8)
    delta = (delta + adv_learning_rate * delta_grad / denorm).detach()
    return delta


def train_batch_freelb(model: nn.Module, input_dict, b_y, loss_func, optimizer, asteps: int, adv_lr: float):
    # embedding_layer = model.get_input_embeddings()
    # input_ids
    # input_embeds
    # token_type_ids
    # attention_mask
    embedding_layer: nn.Module = model.get_input_embeddings()
    input_ids = input_dict['input_ids']
    token_type_ids = input_dict['token_type_ids']
    attention_mask = input_dict['attention_mask']

    input_embeds = embedding_layer(input_ids)
    delta_embeds = torch.zeros_like(input_embeds)

    optimizer.zero_grad()

    for astep in range(asteps):
        delta_embeds.requires_grad_()
        adv_batch_input = input_embeds + delta_embeds
        adv_batch_input_dict = {
            'inputs_embeds': adv_batch_input,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        output_logits = model(**adv_batch_input_dict).logits
        losses = loss_func(output_logits, b_y)
        loss = torch.mean(losses)
        loss_ = loss / asteps
        loss_.backward()

        if astep == asteps - 1:
            break

        delta_grad = delta_embeds.grad.clone().detach()

        delta_embeds = delta_update(adv_lr, delta_embeds, delta_grad)
        input_embeds = embedding_layer(input_ids)

    optimizer.step()


def train_batch(model, input_dict, b_y, loss_func, optimizer, print_loss=False):
    output = model(**input_dict)
    loss = loss_func(output.logits, b_y)
    if print_loss:
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_model(tokenizer, model: torch.nn.Module, dataset: Dataset,
                loss_func_class, optimizer_class, lr: float, epoch: int, batch_size: int, max_len: int,
                weight_decay: float, seed: int = 42,
                cpu: str = 'cpu', gpu: str = 'cuda', tqdm_desc: str = None, best_save_path: str = None,
                validset: Dataset = None) -> str:
    print(
        f"start training , hyper params:\nepoch:{epoch},batch_size:{batch_size},max_len{max_len}\nlr:{lr},l2norm:{weight_decay}\nseed:{seed}")

    if best_save_path is not None:
        if validset is None:
            raise ValueError(
                'You have to provide a valid set if you wan\'t to save the best epoch during training')
        pos = best_save_path.rfind('.')
        test_save_path = best_save_path[:pos] + '.test'
        with open(test_save_path, 'w') as fout:
            fout.write("test")

    set_seed(seed)

    best_acc = 0.0

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.to(gpu)

    loss_func = loss_func_class()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = optimizer_class(
        optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
    if tqdm_desc is None:
        tqdm_desc = str(gpu)
    for _ in tqdm(range(epoch), desc=tqdm_desc):
        model.train()
        for step, batch in enumerate(tqdm(data_loader, desc=tqdm_desc)):
            # if step%10==1:
            #     time.sleep(1)
            b_y = batch[1].to(gpu)
            input_dict = tokenizer(
                batch[0], return_tensors='pt', padding=True, truncation=True, max_length=max_len)
            to_device(input_dict, gpu)
            train_batch(model, input_dict, b_y, loss_func, optimizer)
            #train_batch(model, input_dict, b_y, loss_func,optimizer,step%50==0)
        if best_save_path is not None:
            acc = evaluate(tokenizer=tokenizer, model=model, dataset=validset, max_len=max_len, batch_size=batch_size,
                           gpu="cuda")
            if acc > best_acc:
                best_acc = acc
                model.to(cpu)
                torch.save(model.state_dict(), best_save_path)
                model.to(gpu)
    model.eval()
    model.to(cpu)


def train_model_freelb(tokenizer, model: torch.nn.Module, dataset: Dataset,
                       loss_func_class, optimizer_class, lr: float, epoch: int, batch_size: int, max_len: int, weight_decay: float,
                       asteps: int = 5, adv_lr: float = 1e-5,  seed: int = 42,
                       cpu: str = 'cpu', gpu: str = 'cuda', tqdm_desc: str = None, best_save_path: str = None,
                       validset: Dataset = None) -> str:
    print(
        f"start freelb training , hyper params:\nepoch:{epoch},batch_size:{batch_size},max_len{max_len}\nlr:{lr},l2norm:{weight_decay}\nseed:{seed}")

    if best_save_path is not None:
        if validset is None:
            raise ValueError(
                'You have to provide a valid set if you wan\'t to save the best epoch during training')
        pos = best_save_path.rfind('.')
        test_save_path = best_save_path[:pos] + '.test'
        with open(test_save_path, 'w') as fout:
            fout.write("test")

    set_seed(seed)

    best_acc = 0.0

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.to(gpu)

    loss_func = loss_func_class()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = optimizer_class(
        optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
    # scheduler = CosineAnnealingLR(optimizer,len(dataset)//batch_size*epoch)
    if tqdm_desc is None:
        tqdm_desc = str(gpu)
    for _ in tqdm(range(epoch), desc=tqdm_desc):
        model.train()
        for step, batch in enumerate(tqdm(data_loader, desc=tqdm_desc)):
            b_y = batch[1].to(gpu)
            input_dict = tokenizer(
                batch[0], return_tensors='pt', padding=True, truncation=True, max_length=max_len)
            to_device(input_dict, gpu)
            train_batch_freelb(model, input_dict, b_y, loss_func,
                               optimizer, adv_lr=adv_lr, asteps=asteps)
            # output = model(**input_dict)
            # loss = loss_func(output.logits, b_y)
            # optimizer.zero_grad()
            # loss.backward()
            # # clip_grad.clip_grad_norm_(model.parameters(),1.0)
            # optimizer.step()
        if best_save_path is not None:
            acc = evaluate(tokenizer=tokenizer, model=model, dataset=validset, max_len=max_len, batch_size=batch_size,
                           gpu="cuda")
            if acc > best_acc:
                best_acc = acc
                model.to(cpu)
                torch.save(model.state_dict(), best_save_path)
                model.to(gpu)

    model.eval()
    model.to(cpu)


def reload_or_train(path: str, train_params: dict, tokenizer=None, model=None, dataset=None, seed: int = None,
                    cpu: str = "cpu", gpu: str = DEFAULT_GPU, tqdm_desc: str = None, best_save_path: str = None,
                    validset=None, training_type='base') -> None:
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        assert try_save(path) == True, f"cannot save anything to {path}"
        if 'tokenizer' not in train_params:
            train_params['tokenizer'] = tokenizer
        if 'model' not in train_params:
            train_params['model'] = model
        if 'dataset' not in train_params:
            train_params['dataset'] = dataset
        if 'seed' not in train_params:
            train_params['seed'] = seed
        if 'cpu' not in train_params:
            train_params['cpu'] = cpu
        if 'gpu' not in train_params:
            train_params['gpu'] = gpu
        if tqdm_desc is not None and 'tqdm_desc' not in train_params:
            train_params['tqdm_desc'] = tqdm_desc
        if 'best_save_path' not in train_params:
            train_params['best_save_path'] = best_save_path
        if 'validset' not in train_params:
            train_params['validset'] = validset
        if 'weight_decay' not in train_params:
            train_params['weight_decay'] = 0.0
        train_model_func = {
            'base': train_model,
            'freelb': train_model_freelb,
        }[training_type]
        train_model_func(**train_params)
        model.to(cpu)
        torch.save(model.state_dict(), path)
