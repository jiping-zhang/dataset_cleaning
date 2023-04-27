import argparse

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import textattack
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification, MaxNumWordsModified
from textattack.constraints.semantics import WordEmbeddingDistance

from utils import get_filename

def get_dataset(path:str)->textattack.datasets.Dataset:
    data_in_tuple=list()
    with open(path,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in lines:
            if len(line)==0:
                continue
            tab_pos = line.rfind("\t")
            if tab_pos==-1:
                continue
            try:
                sentence = line[:tab_pos]
                label = int(line[tab_pos+1:])
                data_in_tuple.append((sentence,label))
            except Exception as e:
                continue
    return textattack.datasets.Dataset(data_in_tuple)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--n_label", type=int, default=2)
arg_parser.add_argument("--model_path", type=str, required=True)
arg_parser.add_argument("--test_set", type=str, required=True)
args = arg_parser.parse_args()

N_LABEL = args.n_label
MODEL_PATH = args.model_path
TEST_SET_PATH = args.test_set

MODEL_TYPE = "bert-base-uncased"
CONFIG = BertConfig.from_pretrained(MODEL_TYPE, num_labels=N_LABEL)

model = BertForSequenceClassification.from_pretrained(
    MODEL_TYPE, config=CONFIG)
model.load_state_dict(torch.load(MODEL_PATH))
tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)
model_wrapped = textattack.models.wrappers.HuggingFaceModelWrapper(
    model, tokenizer)

goal_func = textattack.goal_functions.UntargetedClassification(model_wrapped)
stopwords = set(
    ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn",
        "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
)
constraints = [
    RepeatModification(),
    StopwordModification(stopwords),
    MaxNumWordsModified(3),
    WordEmbeddingDistance(min_cos_sim=0.8)
]
transformation=textattack.transformations.WordSwapEmbedding(max_candidates=20)
search_method=textattack.search_methods.GreedyWordSwapWIR(wir_method="delete")
attack = textattack.Attack(goal_func,constraints,transformation,search_method)
attack_args=textattack.AttackArgs(
    num_examples=500,
    log_to_csv=f"./attack/{get_filename(MODEL_PATH)}_atk_log.csv",
)
attacker = textattack.Attacker(attack,get_dataset(TEST_SET_PATH),attack_args)
attacker.attack_dataset()