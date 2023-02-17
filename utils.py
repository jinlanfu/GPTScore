import os
import pickle
import sys

import nltk
from mosestokenizer import *
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import json

nltk.download('stopwords')
detokenizer = MosesDetokenizer('en')

def read_demos(json_path):
    asp_demos = json.load(open(json_path))
    asp_dfs, demos = asp_demos["asp_definition"], asp_demos["demo"]
    return demos, asp_dfs


def lower_check(text):
    # The BAGEL dataset uses X to replace named entities.
    if text.startswith("X ") == False:
        text1 = text[0].lower() + text[1:]
    else:
        text1 = text
    return text1

def add_dot(text):
    if text.strip()[-1] != '.':
        text = text.strip() + ' .'
    new_text = text
    return new_text


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')

def detokenize(text: str):
    words = text.split(" ")
    return detokenizer(words)


# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__
