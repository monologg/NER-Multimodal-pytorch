import os
import random
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def load_vocab(args):
    word_vocab_path = os.path.join(args.vocab_dir, "word_vocab_{}".format(args.word_vocab_size))
    char_vocab_path = os.path.join(args.vocab_dir, "char_vocab_{}".format(args.char_vocab_size))

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
        for idx, line in enumerate(f):
            line = line.strip()
            word_vocab[line] = idx
            word_ids_to_tokens.append(line)

    # Load char vocab
    with open(char_vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
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
