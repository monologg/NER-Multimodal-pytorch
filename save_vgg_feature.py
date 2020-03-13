import os
import argparse

import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from PIL import Image
import numpy as np

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG16_NoTop(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG16_NoTop, self).__init__()
        self.features = features  # Only use this part
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        return self.features(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG16_NoTop(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg16_notop(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data", type=str, help="Path for data dir")
    parser.add_argument("--img_dir", default="img", type=str, help="Path for img dir")
    parser.add_argument("--feature_file", default="img_vgg_features.pt", type=str, help="Filename for preprocessed image features")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = vgg16_notop(pretrained=True)
    model.to(device)
    model.eval()

    # Only load the images that is in train/dev/test
    img_id_lst = []
    for text_filename in ['train', 'dev', 'test']:
        with open(os.path.join(args.data_dir, text_filename), 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("IMGID:"):
                    img_id_lst.append(int(line.replace("IMGID:", "").strip()))

    mean_pixel = [103.939, 116.779, 123.68]

    img_features = {}

    for idx, img_id in enumerate(img_id_lst):
        img_path = os.path.join(args.data_dir, args.img_dir, '{}.jpg'.format(img_id))
        try:
            im = Image.open(img_path)
            im = im.resize((224, 224))
            im = np.array(im)

            if im.shape == (224, 224):  # Check whether the channel of image is 1
                im = np.concatenate((np.expand_dims(im, axis=-1),) * 3, axis=-1)  # Change the channel 1 to 3

            im = im[:, :, :3]  # Some images have 4th channel, which is transparency value
        except Exception as inst:
            print("{} error!".format(img_id))
            print(inst)
            continue

        for c in range(3):
            im[:, :, c] = im[:, :, c] - mean_pixel[c]
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        im = torch.Tensor(im).to(device)
        with torch.no_grad():
            img_feature = model(im)

        img_feature = img_feature.squeeze(0).view(512, 7 * 7)
        img_feature = img_feature.transpose(1, 0)
        img_features[img_id] = img_feature

        if (idx + 1) % 100 == 0:
            print("{} done".format(idx + 1))

    # Save features with torch.save
    torch.save(img_features, os.path.join(args.data_dir, args.feature_file))
