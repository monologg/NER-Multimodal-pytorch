import torch
import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self,
                 max_word_len=30,
                 kernel_lst=[2, 3, 4],
                 num_filters=32,
                 char_vocab_size=1000,
                 char_dim=30,
                 final_char_dim=50):
        super(CharCNN, self).__init__()

        # Initialize character embedding
        self.char_emb = nn.Embedding(char_vocab_size, char_dim, padding_idx=0)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        # Convolution for each kernel
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(char_dim, num_filters, kernel_size),
                nn.Tanh(),  # As the paper mentioned
                nn.MaxPool1d(max_word_len - kernel_size + 1),
                nn.Dropout(0.25)
            ) for kernel_size in kernel_lst
        ])

        self.linear = nn.Sequential(
            nn.Linear(num_filters * len(kernel_lst), 100),
            nn.ReLU(),
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

        output = self.linear(conv_concat)  # (b*s, 50)
        output = output.view(batch_size, max_seq_len, -1)  # (b, s, 50)
        return output
