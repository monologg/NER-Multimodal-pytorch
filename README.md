# NER-Multimodal-pytorch

(Unofficial) Pytorch Implementation of ["Adaptive Co-attention Network for Named Entity Recognition in Tweets" (AAAI 2018)](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16432)

## Model

<p float="left" align="center">
    <img width="700" src="https://user-images.githubusercontent.com/28896432/76892333-de934800-68cd-11ea-93ff-16cb22a5cc3f.png" />  
</p>

## Dependencies

- python>=3.5
- torch==1.3.1
- torchvision==0.4.2
- pillow==7.0.0
- pytorch-crf==0.7.2
- seqeval==0.0.12
- gdown>=3.10.1

```bash
$ pip3 install -r requirements.txt
```

## Data

|           | Train | Dev   | Test  |
| --------- | ----- | ----- | ----- |
| # of Data | 4,000 | 1,000 | 3,257 |

### 1. Pretrained Word Vectors

- Original code's pretrained word embedding can be downloaded at [here](https://pan.baidu.com/s/1boSlljL#list/path=%2F).
- But it takes quite long time to download, so I take out the word vectors that are only in word vocab.
- **It will be downloaded automatically when you run `main.py`.**

### 2. Extracted VGG Features

- Image features are extracted from **last pooling layer** of `VGG16`.
- If you want to extract the feature by yourself, follow as below.

  1. Clone the repo of [original code](https://github.com/jlfu/NERmultimodal).
  2. Copy `data/ner_img` from original code to this repo.
  3. Run as below. (`img_vgg_features.pt` will be saved in `data` dir)

  ```bash
  $ python3 save_vgg_feature.py
  ```

- **Extracted features will be downloaded automatically when you run `main.py`.**

## Detail

- There are some differences between the `paper` and the `original code`, **so I tried to follow the paper's equations as possible.**
- Build the vocab with `train`, `dev`, and `test` dataset. (same as the original code)
  - Making the vocab only with train dataset decreases performance a lot. (about 5%)
- Use `Adam` optimizer instead of `RMSProp`.

## How to run

```bash
$ python3 main.py --do_train --do_eval
```

## Result

|                       | F1 (%)    |
| --------------------- | --------- |
| **Re-implementation** | **67.10** |
| Baseline (paper)      | 70.69     |

## References

- [Original Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16432)
- [Original Code Implementation](https://github.com/jlfu/NERmultimodal)
