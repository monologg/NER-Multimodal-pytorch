import torch
import torch.nn as nn
from torchcrf import CRF
from data_loader import TweetProcessor


class CharCNN(nn.Module):
    def __init__(self,
                 max_word_len=30,
                 kernel_lst="2,3,4",
                 num_filters=32,
                 char_vocab_size=1000,
                 char_emb_dim=30,
                 final_char_dim=50):
        super(CharCNN, self).__init__()

        # Initialize character embedding
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        nn.init.uniform_(self.char_emb.weight, -0.25, 0.25)

        kernel_lst = list(map(int, kernel_lst.split(",")))  # "2,3,4" -> [2, 3, 4]

        # Convolution for each kernel
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size // 2),
                nn.Tanh(),  # As the paper mentioned
                nn.MaxPool1d(max_word_len - kernel_size + 1),
                nn.Dropout(0.25)  # As same as the original code implementation
            ) for kernel_size in kernel_lst
        ])

        self.linear = nn.Sequential(
            nn.Linear(num_filters * len(kernel_lst), 100),
            nn.ReLU(),  # As same as the original code implementation
            nn.Dropout(0.25),
            nn.Linear(100, final_char_dim)
        )

    def forward(self, x):
        """
        :param x: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, final_char_dim)
        """
        batch_size = x.size(0)
        max_seq_len = x.size(1)
        max_word_len = x.size(2)

        x = self.char_emb(x)  # (b, s, w, d)
        x = x.view(batch_size * max_seq_len, max_word_len, -1)  # (b*s, w, d)
        x = x.transpose(2, 1)  # (b*s, d, w): Conv1d takes in (batch, dim, seq_len), but raw embedded is (batch, seq_len, dim)

        conv_lst = [conv(x) for conv in self.convs]
        conv_concat = torch.cat(conv_lst, dim=-1)  # (b*s, num_filters, len(kernel_lst))
        conv_concat = conv_concat.view(conv_concat.size(0), -1)  # (b*s, num_filters * len(kernel_lst))

        output = self.linear(conv_concat)  # (b*s, final_char_dim)
        output = output.view(batch_size, max_seq_len, -1)  # (b, s, final_char_dim)
        return output


class BiLSTM(nn.Module):
    def __init__(self, args, pretrained_word_matrix):
        super(BiLSTM, self).__init__()
        self.args = args

        self.char_cnn = CharCNN(max_word_len=args.max_word_len,
                                kernel_lst=args.kernel_lst,
                                num_filters=args.num_filters,
                                char_vocab_size=args.char_vocab_size,
                                char_emb_dim=args.char_emb_dim,
                                final_char_dim=args.final_char_dim)

        if pretrained_word_matrix:
            self.word_emb = nn.Embedding.from_pretrained(pretrained_word_matrix)
        else:
            self.word_emb = nn.Embedding(args.word_vocab_size, args.word_emb_dim, padding_idx=0)
            nn.init.uniform_(self.word_emb.weight, -0.25, 0.25)

        self.bi_lstm = nn.LSTM(input_size=args.word_emb_dim + args.final_char_dim,
                               hidden_size=args.hidden_dim // 2,  # Bidirectional will double the hidden_size
                               bidirectional=True,
                               batch_first=True)

    def forward(self, word_ids, char_ids):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :return: (batch_size, max_seq_len, dim)
        """
        w_emb = self.word_emb(word_ids)
        c_emb = self.char_cnn(char_ids)

        w_c_emb = torch.cat([w_emb, c_emb], dim=-1)

        h0 = torch.randn(2, w_c_emb.size(0), self.args.hidden_dim // 2).to(self.args.device)
        c0 = torch.randn(2, w_c_emb.size(0), self.args.hidden_dim // 2).to(self.args.device)

        lstm_output, _ = self.bi_lstm(w_c_emb, (h0, c0))

        return lstm_output


class CoAttention(nn.Module):
    def __init__(self):
        super(CoAttention, self).__init__()

    def forward(self, lstm_features, img_features):
        """
        :param lstm_features: (batch_size, max_seq_len, hidden_dim*2)
        :param img_features: (batch_size, num_img_region, img_feat_dim)
        :return: (batch_size, max_seq_len, dim)
        """


class ACN(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> CRF
    """

    def __init__(self, args, pretrained_word_matrix=None):
        super(ACN, self).__init__()
        self.lstm = BiLSTM(args, pretrained_word_matrix)
        # Transform each img vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.img_feat_dim, args.hidden_dim),
            nn.Tanh()
        )

        self.crf = CRF(num_tags=len(TweetProcessor.get_labels()), batch_first=True)

    def forward(self, word_ids, char_ids, img_features):
        """
        :param word_ids: (batch_size, max_seq_len)
        :param char_ids: (batch_size, max_seq_len, max_word_len)
        :param img_features: [batch_size, num_img_region, img_feat_dim]
        :return:
        """
        lstm_features = self.lstm(word_ids, char_ids)
        img_features = self.dim_match(img_features)  # [batch_size, 49, hidden_dim]
        assert lstm_features.size(-1) == img_features.size(-1)


if __name__ == '__main__':
    word_ids = torch.ones((16, 35), dtype=torch.long)
    char_ids = torch.ones((16, 35, 30), dtype=torch.long)
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.max_seq_len = 35
    args.device = 'cpu'
    args.max_word_len = 30
    args.kernel_lst = "2,3,4"
    args.num_filters = 32
    args.char_vocab_size = 200
    args.char_emb_dim = 30
    args.final_char_dim = 50
    args.word_vocab_size = 1000
    args.word_emb_dim = 200
    args.hidden_dim = 200

    print(args.max_word_len)

    model = BiLSTM(args, None)
    model.eval()
    with torch.no_grad():
        output = model(word_ids, char_ids)

    print(output.size())
