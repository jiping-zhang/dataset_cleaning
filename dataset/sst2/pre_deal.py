from tqdm import tqdm
from typing import *

fin = open("./dev.tsv")
lines = fin.readlines()
line1 = lines[1]
index1 = line1.rfind("1")
index_dot = line1.rfind(".")
spliter = line1[index_dot+1::index1]
del fin


def parse_standard_file(fin_path,fout_path):
    error_cnt = 0
    with open(fout_path,'w') as fout:
        with open(fin_path) as fin:
            lines:List[str] = fin.readlines()
            for line in lines[1:]:
                index_spliter = line.rfind(spliter)
                if index_spliter==-1:
                    error_cnt+=1
                sentence = line[:index_spliter]
                label = line[index_spliter+1:]
                fout.write(f"{sentence}\t{label}")
    print(f"{error_cnt} sentences failed to be parsed")

fnames = ["dev","test","train"]

for fname in fnames:
    parse_standard_file(f"./{fname}.tsv",f"./{fname}.txt")