from typing import *

import os
import random

def get_folder_abs_path(rel_path: str) -> str:
    ans = os.path.abspath(rel_path)
    ans = ans.replace("\\", "/") + "/"
    return ans


def create_folder(path: str):
    path = os.path.abspath(path)
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def shuffle_array(arr: list, seed: int) -> None:
    random.seed(seed)
    for i in range(len(arr)):
        j = random.randint(0, i)
        temp = arr[j]
        arr[j] = arr[i]
        arr[i] = temp


def try_save(path:str) -> bool:
    try:
        with open(path,"w") as fout:
            fout.write("678")
        with open(path,"r") as fin:
            lines = fin.readlines()
            if lines[0]=="678":
                return True
            return False
        return False
    except OSError as err:
        print(str(err))
        return False


def copy_of_dict(d: dict) -> dict:
    ans = {}
    for key in d:
        ans[key] = d[key]
    return ans


# path = "./"+"123"*40+".txt"
# print(try_save(path))