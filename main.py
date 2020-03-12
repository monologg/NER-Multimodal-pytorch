import argparse

from trainer import Trainer
from utils import init_logger
from data_loader import load_and_cache_examples


def main(args):
    init_logger()

    train_dataset = load_and_cache_examples(args, mode="train")
    dev_dataset = load_and_cache_examples(args, mode="dev")
    test_dataset = load_and_cache_examples(args, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path for saving model")
    parser.add_argument("--wordvec_dir", default="./wordvec", type=str, help="Path for pretrained word vector")
    parser.add_argument("--vocab_dir", default="./vocab", type=str)

    parser.add_argument("--train_file", default="train", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test", type=str, help="Test file")
    parser.add_argument("--w2v_file", default="glove.twitter.27B.200d.txt", type=str, help="Pretrained word vector file")
    parser.add_argument("--img_feature_file", default="img_vgg_features.pt", type=str, help="Filename for preprocessed image features")

    parser.add_argument("--max_seq_len", default=35, type=int, help="Max sentence length")
    parser.add_argument("--max_word_len", default=30, type=int, help="Max word length")

    parser.add_argument("--overwrite_w2v", action="store_true", help="Overwriting word vector")
    parser.add_argument("--word_vocab_size", default=10000, type=int, help="Maximum size of word vocabulary")
    parser.add_argument("--char_vocab_size", default=1000, type=int, help="Maximum size of character vocabulary")

    parser.add_argument("--word_emb_dim", default=200, type=int, help="Word embedding size")
    parser.add_argument("--char_emb_dim", default=30, type=int, help="Character embedding size")
    parser.add_argument("--final_char_dim", default=50, type=int, help="Dimension of character cnn output")
    parser.add_argument("--hidden_dim", default=200, type=int, help="Dimension of BiLSTM output")

    parser.add_argument("--kernel_lst", default="2,3,4", type=str, help="kernel size for character cnn")
    parser.add_argument("--num_filters", default=32, type=int, help=" Number of filters for character cnn")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=0.19, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--slot_pad_label", default="[PAD]", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")
    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--logging_steps', type=int, default=250, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=250, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()

    # For VGG16 img features (DO NOT change this part)
    args.num_img_region = 49
    args.img_feat_dim = 512

    main(args)
