import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from data_loader import TweetProcessor, load_word_matrix
from utils import set_seed, load_vocab
from model import ACN


class Trainer(object):
    def __init__(self, args, train_dataset, dev_dataset, test_dataset):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = TweetProcessor.get_labels()

        self.pad_token_label_id = args.ignore_index

        self.word_vocab, self.char_vocab, self.word_ids_to_tokens, self.char_ids_to_tokens = load_vocab(args)
        self.pretrained_word_matrix = load_word_matrix(args, self.word_vocab)
        self.model = ACN(args, self.pretrained_word_matrix)

    def train(self):
        pass

    def evaluate(self, mode):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass