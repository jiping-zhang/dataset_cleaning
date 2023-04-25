import torch
from transformers import BertConfig,BertTokenizer,BertForSequenceClassification
import textattack

import special_datasets

MODEL_PARAM_PATH = "./model/B_lr2e-05_l21e-05_e4_b8_ml64_sst2.pth"
N_LABEL = 2
MODEL_TYPE = "bert-base-uncased"
CONFIG = BertConfig.from_pretrained(MODEL_TYPE,num_labels = N_LABEL)
