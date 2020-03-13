import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


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
