import os
import re
import random
import logging
from collections import Counter

import gdown
import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

logger = logging.getLogger(__name__)


def download_vgg_features(args):
    """ Download vgg features"""
    vgg_path = os.path.join(args.data_dir, args.img_feature_file)
    if not os.path.exists(vgg_path):
        logger.info("Downloading vgg img features...")
        gdown.download("https://drive.google.com/uc?id=1q9CRm5gCuU9EVEA6Y4xYp6naskTL0bs4", vgg_path, quiet=False)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def preprocess_word(word):
    """
    - Do lowercase
    - Regular expression (number, url, hashtag, user)
        - https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    :param word: str
    :return: word: str
    """
    number_re = r"[-+]?[.\d]*[\d]+[:,.\d]*"
    url_re = r"https?:\/\/\S+\b|www\.(\w+\.)+\S*"
    hashtag_re = r"#\S+"
    user_re = r"@\w+"

    if re.compile(number_re).match(word):
        word = '<NUMBER>'
    elif re.compile(url_re).match(word):
        word = '<URL>'
    elif re.compile(hashtag_re).match(word):
        word = word[1:]  # only erase `#` at the front
    elif re.compile(user_re).match(word):
        word = word[1:]  # only erase `@` at the front

    word = word.lower()

    return word


def build_vocab(args):
    """
    Build vocab from train, dev and test set
    Write all the tokens in vocab. When loading the vocab, limit the size of vocab at that time
    """
    # Read all the files
    words, chars = [], []

    for filename in [args.train_file, args.dev_file, args.test_file]:
        with open(os.path.join(args.data_dir, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != "":
                    if not line.startswith('IMGID:'):
                        word = line.split('\t')[0]
                        word = preprocess_word(word)
                        words.append(word)
                        for char in word:
                            chars.append(char)

    if not os.path.exists(args.vocab_dir):
        os.mkdir(args.vocab_dir)

    word_vocab, char_vocab = [], []

    word_vocab_path = os.path.join(args.vocab_dir, "word_vocab")
    char_vocab_path = os.path.join(args.vocab_dir, "char_vocab")

    word_counts = Counter(words)
    word_vocab.append("[pad]")
    word_vocab.append("[unk]")
    word_vocab.extend([x[0] for x in word_counts.most_common()])
    logger.info("Total word vocabulary size: {}".format(len(word_vocab)))

    with open(word_vocab_path, 'w', encoding='utf-8') as f:
        for word in word_vocab:
            f.write(word + "\n")

    char_counts = Counter(chars)
    char_vocab.append("[pad]")
    char_vocab.append("[unk]")
    char_vocab.extend([x[0] for x in char_counts.most_common()])
    logger.info("Total char vocabulary size: {}".format(len(char_vocab)))

    with open(char_vocab_path, 'w', encoding='utf-8') as f:
        for char in char_vocab:
            f.write(char + "\n")

    # Set the exact vocab size
    # If the original vocab size is smaller than args.vocab_size, then set args.vocab_size to original one
    with open(word_vocab_path, 'r', encoding='utf-8') as f:
        word_lines = f.readlines()
        args.word_vocab_size = min(len(word_lines), args.word_vocab_size)

    with open(char_vocab_path, 'r', encoding='utf-8') as f:
        char_lines = f.readlines()
        args.char_vocab_size = min(len(char_lines), args.char_vocab_size)

    logger.info("args.word_vocab_size: {}".format(args.word_vocab_size))
    logger.info("args.char_vocab_size: {}".format(args.char_vocab_size))


def load_vocab(args):
    word_vocab_path = os.path.join(args.vocab_dir, "word_vocab")
    char_vocab_path = os.path.join(args.vocab_dir, "char_vocab")

    if not os.path.exists(word_vocab_path):
        logger.warning("Please build word vocab first!")
        return

    if not os.path.exists(char_vocab_path):
        logger.warning("Please build char vocab first!")
        return

    word_vocab = dict()
    char_vocab = dict()
    word_ids_to_tokens = []
    char_ids_to_tokens = []

    # Load word vocab
    with open(word_vocab_path, "r", encoding="utf-8") as f:
        # Set the exact vocab size
        # If the original vocab size is smaller than args.vocab_size, then set args.vocab_size to original one
        word_lines = f.readlines()
        args.word_vocab_size = min(len(word_lines), args.word_vocab_size)

        for idx, line in enumerate(word_lines[:args.word_vocab_size]):
            line = line.strip()
            word_vocab[line] = idx
            word_ids_to_tokens.append(line)

    # Load char vocab
    with open(char_vocab_path, "r", encoding="utf-8") as f:
        char_lines = f.readlines()
        args.char_vocab_size = min(len(char_lines), args.char_vocab_size)
        for idx, line in enumerate(char_lines[:args.char_vocab_size]):
            line = line.strip()
            char_vocab[line] = idx
            char_ids_to_tokens.append(line)

    return word_vocab, char_vocab, word_ids_to_tokens, char_ids_to_tokens


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(labels) == len(preds)
    return f1_pre_rec(labels, preds)


def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def report(labels, preds):
    return classification_report(labels, preds)
