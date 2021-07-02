# -*- coding: utf-8 -*-
# @File  : preprocess.py
# @Author: LVFANGFANG
# @Date  : 2021/6/30 11:55
# @Desc  :

import json
import logging
import multiprocessing
import os
import re
import string
from typing import List

import jieba
import nltk
from gensim import corpora
from gensim.models import word2vec
from zhon.hanzi import punctuation

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_stopwords(path: str) -> List:
    with open(path, encoding='utf-8') as f:
        stopwords = f.readlines()
    stopwords = [i.strip() for i in stopwords]
    return stopwords


pattern = re.compile('|'.join(re.escape(i) for i in punctuation + string.punctuation))


def remove_non_characters(need_clean: str) -> str:
    need_clean = pattern.sub(' ', need_clean)

    return need_clean


def segment_line(input_files: List, output_dir: str):
    clean = lambda x: jieba.cut(remove_non_characters(x))
    for i, input_file in enumerate(input_files):
        segment_out_name = os.path.join(output_dir, f'segment_{i}.txt')
        with open(input_file, encoding='utf-8') as f1, open(segment_out_name, 'w', encoding='utf-8') as f2:
            for line in f1:
                example = json.loads(line)
                corpus = [clean(example['question'])] + [clean(i) for i in example['answers']] \
                         + [clean(doc['title']) for doc in example['documents']] \
                         + [clean(i) for doc in example['documents'] for i in doc['paragraphs']]
                sentences = list(map(lambda x: ' '.join(i for i in x if i != ' ') + '\n', corpus))
                f2.writelines(sentences)


def word_count(input_file: str, output_file: str):
    # dictionary = corpora.Dictionary(line.lower().split() for line in open(input_file, encoding='utf-8'))
    if os.path.isdir(input_file):
        sentences = word2vec.PathLineSentences(input_file)
    else:
        sentences = word2vec.LineSentence(input_file)
    dictionary = corpora.Dictionary(sentences, prune_at=5000000)
    counter = ((dictionary[i], count) for i, count in dictionary.cfs.items())
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in sorted(counter, key=lambda kv: kv[-1], reverse=True):
            f.write(f'{word} {count}\n')


def get_tokens(sentence: str) -> List:
    remove = str.maketrans('', '', string.punctuation)
    without_punctuation = sentence.lower().translate(remove)
    tokens = nltk.word_tokenize(without_punctuation)
    return tokens


def segment(input_file: str, output_file: str):
    with open(input_file) as f:
        data = json.load(f)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data['data']:
            for paragraph in item['paragraphs']:
                context = ' '.join(get_tokens(paragraph['context']))
                sentences = [context + '\n'] + [' '.join(get_tokens(qa['question'])) + '\n' for qa in paragraph['qas']]
                f.writelines(sentences)


def process_squad():
    logging.info('Start process SQuAD......')
    input_file = 'data/SQuAD/train-v2.0.json'
    output_file = 'data/processed/segment_squad.txt'
    wordcount_file = 'data/processed/word_count_squad.txt'
    segment(input_file, output_file)
    word_count(output_file, wordcount_file)
    logging.info('Done!')


def process_dureader():
    logging.info('Start process dureader......')
    input_files = ['data/dureader/raw/trainset/search.train.json',
                   'data/dureader/raw/trainset/zhidao.train.json']
    output_dir = 'data/processed/segment_dureader'
    wordcount_file = 'data/processed/word_count_dureader.txt'
    word2vec_path = 'models/dureader.bin.gz'

    segment_line(input_files, output_dir)
    word_count(output_dir, wordcount_file)
    logging.info('Start train w2v')
    sentences = word2vec.PathLineSentences(output_dir)
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, sg=1, workers=multiprocessing.cpu_count())
    logging.info(f'save model to {word2vec_path}')
    model.wv.save_word2vec_format(word2vec_path, binary=True)
    logging.info('Done!')


if __name__ == '__main__':
    process_squad()
    process_dureader()
