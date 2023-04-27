import argparse

from transformers import BertConfig,BertForSequenceClassification,BertTokenizer
import textattack

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--n_label",type=int,default=2)
arg_parser.add_argument("--model_path",type=str,required=True)
arg_parser.add_argument("--test_set",type=str,required=True)
args = arg_parser.parse_args()

N_LABEL = args.n_label
MODEL_PATH = args.model_path