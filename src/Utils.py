import os
import pandas as pd
from parsivar import Normalizer, Tokenizer, FindStems
import re


def scan_directory(ROOT_DIR: str):
    dataset = pd.DataFrame()
    categories = os.listdir(ROOT_DIR)
    for category in categories:
        category_path = os.path.join(ROOT_DIR, category)
        for text_file in os.listdir(category_path):
            file_path = os.path.join(category_path, text_file)
            text = open(file_path, encoding='utf-8', newline='\n').read()
            new_row = {'text': text, 'category': category}
            dataset = dataset.append(new_row, ignore_index=True)
    return dataset


def preprocessing(data: pd.DataFrame):
    stop_words = open('../raw/stop_words.txt', encoding='utf-8', newline='\n').read().split()
    normalizer = Normalizer()
    tokenizer = Tokenizer()
    stemmer = FindStems()
    corpus = []
    for text in data:
        text = normalizer.normalize(text, new_line_elimination=True)
        text = re.sub('[\u200c]', ' ', text)
        text = re.sub('[0-9]', ' ', text)
        words = tokenizer.tokenize_words(text)
        words = filter(lambda word: word not in stop_words, words)
        words = [stemmer.convert_to_stem(word) for word in words]
        text = ' '.join(words)
        corpus.append(text)
    return corpus
