import os
import copy
import json
import logging
from collections import Counter

import numpy as np

import torch
from torch.utils.data import TensorDataset

from utils import load_vocab

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, img_id, words, labels):
        self.guid = guid  # int
        self.img_id = img_id  # int
        self.words = words  # list
        self.labels = labels  # list

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, word_ids, char_ids, img_feature, label_ids):
        self.word_ids = word_ids
        self.char_ids = char_ids
        self.img_feature = img_feature
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TweetProcessor(object):
    """Processor for the Tweet data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def get_labels(cls):
        return ["[PAD]", "[UNK]", "O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-OTHER", "I-OTHER"]

    @classmethod
    def get_label_vocab(cls):
        label_vocab = dict()
        for idx, label in enumerate(cls.get_labels()):
            label_vocab[label] = idx

        return label_vocab

    def load_img_features(self):
        return torch.load(os.path.join(self.args.data_dir, self.args.img_feature_file))

    @classmethod
    def _read_file(cls, input_file):
        """Read tsv file, and return words and label as list"""
        with open(input_file, "r", encoding="utf-8") as f:
            sentences = []

            sentence = [[], []]  # [[words], [tags], img_id]
            for line in f:
                if line.strip() == "":
                    continue

                if line.startswith("IMGID:"):
                    if sentence[0]:
                        # Add img_id at last
                        img_id = int(line.replace("IMGID:", "").strip())
                        sentence.append(img_id)
                        sentences.append(sentence)
                        sentence = [[], []]  # Flush
                else:
                    try:
                        word, tag = line.strip().split("\t")
                        sentence[0].append(word)
                        sentence[1].append(tag)
                    except:
                        logger.info("{} cannot be splitted".format(line))
            return sentences

    def _create_examples(self, sentences, set_type):
        """Creates examples for the training dev, and test sets."""
        words_for_vocab, chars_for_vocab = [], []  # For building vocab when it is train set
        examples = []

        for (i, sentence) in enumerate(sentences):
            words, labels, img_id = sentence[0], sentence[1], sentence[2]
            assert len(words) == len(labels)

            if set_type == "train":  # Only save the vocab when it is train set
                words_for_vocab.extend(words)
                for word in words:
                    for char in word:
                        chars_for_vocab.append(char)

            guid = "%s-%s" % (set_type, i)
            if i % 10000 == 0:
                logger.info(sentence)
            examples.append(InputExample(guid=guid, img_id=img_id, words=words, labels=labels))

        if set_type == 'train':
            build_vocab(self.args, words_for_vocab, chars_for_vocab)

        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


def build_vocab(args, words, chars):
    """Save word_vocab, char_vocab as file"""
    if not os.path.exists(args.vocab_dir):
        os.mkdir(args.vocab_dir)

    word_vocab, char_vocab = [], []

    word_vocab_path = os.path.join(args.vocab_dir, "word_vocab_{}".format(args.word_vocab_size))
    char_vocab_path = os.path.join(args.vocab_dir, "char_vocab_{}".format(args.char_vocab_size))

    if not os.path.exists(word_vocab_path):
        word_counts = Counter(words)
        word_vocab.append("[PAD]")
        word_vocab.append("[UNK]")
        word_vocab.extend([x[0] for x in word_counts.most_common()])
        logger.info("Total word vocabulary size: {}".format(len(word_vocab)))
        word_vocab = word_vocab[:args.word_vocab_size]

        with open(word_vocab_path, 'w', encoding='utf-8') as f:
            for word in word_vocab:
                f.write(word + "\n")

    if not os.path.exists(char_vocab_path):
        char_counts = Counter(chars)
        char_vocab.append("[PAD]")
        char_vocab.append("[UNK]")
        char_vocab.extend([x[0] for x in char_counts.most_common()])
        logger.info("Total char vocabulary size: {}".format(len(char_vocab)))
        char_vocab = char_vocab[:args.char_vocab_size]

        with open(char_vocab_path, 'w', encoding='utf-8') as f:
            for char in char_vocab:
                f.write(char + "\n")


def load_word_matrix(args, word_vocab):
    if not os.path.exists(args.wordvec_dir):
        os.mkdir(args.wordvec_dir)

    compressed_w2v_filename = os.path.join(args.wordvec_dir, "word_vector_{}".format(args.word_vocab_size))

    # If the compressed one already exists, load it!
    if os.path.exists(compressed_w2v_filename) and not args.overwrite_w2v:
        word_matrix = np.zeros((args.word_vocab_size, args.word_emb_dim))
        with open(compressed_w2v_filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.rstrip()
                coefs = np.asarray(line, dtype='float32')
                word_matrix[idx] = coefs
        word_matrix = torch.from_numpy(word_matrix)
        return word_matrix

    # Making new word vector
    embedding_index = dict()
    with open(os.path.join(args.wordvec_dir, args.w2v_file), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    word_matrix = np.zeros((args.word_vocab_size, args.word_emb_dim))
    cnt = 0

    for word, i in word_vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            word_matrix[i] = embedding_vector
        else:
            word_matrix[i] = np.random.uniform(-0.25, 0.25, args.word_emb_dim)
            cnt += 1
    print('{} words not in pretrained matrix'.format(cnt))

    # Save the compressed word vector
    np.savetxt(compressed_w2v_filename, word_matrix, delimiter=" ")
    # Convert numpy to tensor
    word_matrix = torch.from_numpy(word_matrix)
    return word_matrix


def convert_examples_to_features(examples,
                                 img_features,
                                 max_seq_len,
                                 max_word_len,
                                 word_vocab,
                                 char_vocab,
                                 label_vocab,
                                 pad_token="[PAD]",
                                 unk_token="[UNK]"):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 1. Load img feature
        try:
            img_feature = img_features[example.img_id]
        except:
            logger.warning("Cannot load image feature! (IMGID: {})".format(example.img_id))
            continue

        # 2. Convert tokens to idx & Padding
        word_pad_idx, char_pad_idx, label_pad_idx = word_vocab[pad_token], char_vocab[pad_token], label_vocab[pad_token]
        word_unk_idx, char_unk_idx, label_unk_idx = word_vocab[unk_token], char_vocab[unk_token], label_vocab[unk_token]

        word_ids = []
        char_ids = []
        label_ids = []

        for word in example.words:
            word_ids.append(word_vocab.get(word, word_unk_idx))
            ch_in_word = []
            for char in word:
                ch_in_word.append(char_vocab.get(char, char_unk_idx))
            # Padding for char
            char_padding_length = max_word_len - len(ch_in_word)
            ch_in_word = ch_in_word + ([char_pad_idx] * char_padding_length)
            ch_in_word = ch_in_word[:max_word_len]
            char_ids.append(ch_in_word)

        for label in example.labels:
            label_ids.append(label_vocab.get(label, label_unk_idx))

        # Padding for word and label
        word_padding_length = max_seq_len - len(word_ids)
        word_ids = word_ids + ([word_pad_idx] * word_padding_length)
        word_ids = word_ids[:max_seq_len]
        label_ids = label_ids + ([label_pad_idx] * word_padding_length)
        label_ids = label_ids[:max_seq_len]

        # Additional padding for char if word_padding_length > 0
        if word_padding_length > 0:
            for i in range(word_padding_length):
                char_ids.append([char_pad_idx] * max_word_len)

        # 3. Verify
        assert len(word_ids) == max_seq_len, "Error with word_ids length {} vs {}".format(len(word_ids), max_seq_len)
        assert len(char_ids) == max_seq_len, "Error with char_ids length {} vs {}".format(len(char_ids), max_seq_len)
        assert len(label_ids) == max_seq_len, "Error with label_ids length {} vs {}".format(len(label_ids), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("word_ids: %s" % " ".join([str(x) for x in word_ids]))
            logger.info("char_ids[0]: %s" % " ".join([str(x) for x in char_ids[0]]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(word_ids=word_ids,
                          char_ids=char_ids,
                          img_feature=img_feature,
                          label_ids=label_ids
                          ))

    return features


def load_and_cache_examples(args, mode):
    processor = TweetProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}'.format(
            mode,
            args.max_seq_len,
            args.max_word_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        word_vocab, char_vocab, _, _ = load_vocab(args)
        label_vocab = processor.get_label_vocab()

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        features = convert_examples_to_features(examples,
                                                processor.load_img_features(),
                                                args.max_seq_len,
                                                args.max_word_len,
                                                word_vocab,
                                                char_vocab,
                                                label_vocab)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_word_ids = torch.tensor([f.word_ids for f in features], dtype=torch.long)
    all_char_ids = torch.tensor([f.char_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_img_feature = torch.stack([f.img_feature for f in features])

    logger.info("all_word_ids.size(): {}".format(all_word_ids.size()))
    logger.info("all_char_ids.size(): {}".format(all_char_ids.size()))
    logger.info("all_label_ids.size(): {}".format(all_label_ids.size()))
    logger.info("all_img_feature.size(): {}".format(all_img_feature.size()))

    dataset = TensorDataset(all_word_ids, all_char_ids, all_label_ids, all_img_feature)
    return dataset
