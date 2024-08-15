import torch
import torchvision.models as models
from torchvision import transforms
from typing import Tuple, Optional, List
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torch.nn.parallel import DataParallel
from models import VAE_sty
import os
from tqdm import tqdm

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_path, transform=None):
        self.img_dir = img_dir
        self.attr_path = attr_path
        self.transform = transform
        self.attr = pd.read_csv(attr_path, sep="\s+", skiprows=2, header=None, dtype={i: int for i in range(1, 41)})
        self.attr.columns = ['image_name'] + ['attr' + str(i) for i in range(1, 41)]

        # 只保留训练集中的图片
        self.attr = self.attr[self.attr['image_name'].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))]
        self.img_names = self.attr['image_name'].tolist()

    def __len__(self):
        return len(self.attr)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_name)
        attributes = self.attr.iloc[idx, 1:].values.astype(int)
        attributes = (attributes + 1) // 2
        sample = {'image': image, 'attributes': torch.from_numpy(attributes)}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def get_custom_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming CelebA has RGB images
    ])

    # 训练数据集和数据加载器
    train_dataset = CelebADataset(img_dir="",
                                  attr_path="", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = CelebADataset(img_dir="",
                                 attr_path="", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=True, drop_last=True)

    return train_loader, test_loader

class Classifier(nn.Module):
    # 初始化函数，定义网络结构
    def __init__(self):
        super(Classifier, self).__init__()
        self.deconv = nn.ConvTranspose2d(512, 3, kernel_size=15, stride=7, padding=4, output_padding=1)
        self.resnet50 = resnet50(pretrained=False)
        pretrain_weights_path = ""
        state_dict = torch.load(pretrain_weights_path)
        self.resnet50.load_state_dict(state_dict)
        self.replace_bn_with_gn(self.resnet50)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 2)
        # 替换ResNet50网络的最后一层，使其输出大小为num_classes
        self.ReLu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.sigmoid = nn.Sigmoid()

    def replace_bn_with_gn(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_groups = 32  # You can adjust the number of groups based on your requirements
                gn = nn.GroupNorm(num_groups, module.num_features, affine=True)
                setattr(model, name, gn)
            else:
                self.replace_bn_with_gn(module)


    def forward(self, x):
        # 将输入tensor(batch_size, 512)转换为(batch_size, 3, 224, 224)
        x = x.view(-1, 512, 1, 1)
        x = self.deconv(x)
        # 将转换后的tensor输入到ResNet50网络中，得到分类结果
        x = self.resnet50(x)
        x = self.ReLu(x)
        x = self.fc1(x)
        x = self.ReLu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def adjust_state_dict_for_dataparallel(state_dict, is_dataparallel):
    """
    根据模型是否使用 DataParallel 包装来调整状态字典中的键名。

    参数：
    - state_dict (dict): 模型的状态字典。
    - is_dataparallel (bool): 模型是否使用 DataParallel 包装。

    返回：
    - new_state_dict (dict): 调整后的状态字典。
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if is_dataparallel:
            # 如果模型使用 DataParallel 包装，确保键名以 `module.` 开头
            new_key = f'module.{k}' if not k.startswith('module.') else k
        else:
            # 如果模型未使用 DataParallel 包装，去掉键名中的 `module.` 前缀
            new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:2')
train_loader , test_loader = get_custom_dataloader(32)
gen = VAE_sty().to(device)
if torch.cuda.device_count() > 1:
    gen = DataParallel(gen)
checkpoint_path = ""
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path)
    adjusted_state_dict = adjust_state_dict_for_dataparallel(checkpoint['gen_state_dict'],
                                                             torch.cuda.device_count() > 1)
    gen.load_state_dict(adjusted_state_dict)


attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                   'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                   'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                   'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
j=20
classifier = Classifier()
classifier = classifier.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    if not isinstance(classifier, nn.DataParallel):
        classifier = nn.DataParallel(classifier)
else:
    print("Using single GPU or CPU")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
criterion = nn.BCELoss().to(device)

checkpoint_dir = os.path.join("", f"{attribute_names[j]}.pth")
if os.path.exists(checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    # 修改模型键
    adjusted_state_dict = adjust_state_dict_for_dataparallel(checkpoint['model_state_dict'], torch.cuda.device_count() > 1)
    classifier.load_state_dict(adjusted_state_dict)
    # 加载优化器的状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    print(f"Resumed training from epoch {epoch}")
else:
    epoch = 0
    print("No checkpoint found.")
# Training loop
for epoch in range(epoch,1000):
    for i, data in tqdm(enumerate(train_loader, 0), desc="Processing", unit="batch"):
        images, labels = data['image'], data['attributes']
        images = images.to(device)
        labels = labels.to(device)
        _, _, _, w,_ = gen(images, None,None,w_space=True)
        one_hot_labels = F.one_hot(labels, num_classes=2).to(device)
        predictions = classifier(w)
        labelss = one_hot_labels[:, j]
        optimizer.zero_grad()
        logits = predictions
        logits = logits.squeeze(1)
        loss = criterion(logits.float(), one_hot_labels[:, j].float())
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            test_data = next(iter(test_loader))
            # test_data = data
            test_images, test_labels = test_data['image'], test_data['attributes']
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            _, _, _, w,_ = gen(test_images, None,None,w_space=True)
            test_one_hot_labels = F.one_hot(test_labels, num_classes=2).to(device)
            test_predictions = classifier(w)
            test_logits = test_predictions
            predicted_labels_test = (test_logits > 0.5).float()
            test_labels = test_labels[:, j].float().unsqueeze(1)
            correct_predictions_test = torch.sum((torch.all(predicted_labels_test == test_one_hot_labels[:, j], dim=1)))
            total_test = test_labels.size(0)
            test_accuracy = correct_predictions_test.item() / total_test
            print(f"[{epoch}/1000][{i}/{len(train_loader)}] Loss: {loss.item()}, Test Accuracy: {test_accuracy}")

    ckpt_dir = ""
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, f"{attribute_names[j]}.pth")
    if torch.cuda.device_count() > 1:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': classifier.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

    torch.save(checkpoint,checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path} for attribute {attribute_names[j]}')


















