import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize._linesearch import line_search
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
from torchvision.models import resnet152
import os
from typing import Tuple, Optional, List
import math
# device = torch.device('cuda:2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models import VAE_sty
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import numpy as np

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


    train_dataset = CelebADataset(img_dir="",
                        attr_path="", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = CelebADataset(img_dir="",
                        attr_path="", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=True, drop_last=True)

    return train_loader, test_loader

class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        # 加载预训练的ResNeXt模型
        self.resnext = resnet152(pretrained=True)
        self.replace_bn_with_gn(self.resnext)
        # 替换ResNeXt网络的最后一层，使其输出大小为num_classes
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, num_classes)
        # 上采样层，将空间维度调整到224x224
        self.deconv = nn.ConvTranspose2d(512, 3, kernel_size=15, stride=7, padding=4, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def replace_bn_with_gn(self, model):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_groups = 32
                gn = nn.GroupNorm(num_groups, module.num_features, affine=True)
                setattr(model, name, gn)
            else:
                self.replace_bn_with_gn(module)

    def forward(self, x):
        x = x.view(-1, 512, 1, 1)
        x = self.deconv(x)
        # 通过ResNeXt提取特征
        x = self.resnext(x)
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




#创建训练集和测试集
train_loader , test_loader = get_custom_dataloader(256)
#加载生成模型
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
attribute_name = attribute_names[j]
#加载分类器
classifier_dir=""
ckpt_path = os.path.join(classifier_dir, f"{attribute_names[j]}.pth")
print(ckpt_path)
classifier= Classifier()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    if not isinstance(classifier, nn.DataParallel):
        classifier = nn.DataParallel(classifier)
else:
    print("Using single GPU or CPU")
classifier = classifier.to(device)
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
checkpoint = torch.load(ckpt_path)

# 修改模型键
adjusted_state_dict = adjust_state_dict_for_dataparallel(checkpoint['model_state_dict'], torch.cuda.device_count() > 1)
classifier.load_state_dict(adjusted_state_dict)


def angle(v1, v2):
    if v1 is not None and v2 is not None:
        dot = np.dot(v1.cpu().detach().ravel(), v2.cpu().detach().ravel())
        norm1 = np.linalg.norm(v1.cpu().detach().ravel())
        norm2 = np.linalg.norm(v2.cpu().detach().ravel())
        cos = dot / (norm1 * norm2)
        return np.arccos(cos)
    else:
        return np.inf



def find_best_direction(directions):
    min_angle_sum = np.inf
    best_direction = None
    device = None
    for d in directions:
        # 如果方向是 None，则跳过
        if d is None:
            continue
        angle_sum = 0
        if device is None:
            device = d.device
        for other_d in directions:
            # 如果另一个方向是 None 或与当前方向相同，则跳过
            if other_d is None or d is other_d:
                continue
            angle_sum += angle(d, other_d)
        if angle_sum < min_angle_sum:
            min_angle_sum = angle_sum
            best_direction = d
    if best_direction is not None:
        best_direction = best_direction.cpu().detach()
        norm = np.linalg.norm(best_direction)
        if norm != 0:
            best_direction = best_direction / norm
        best_direction = best_direction.to(device)
    return best_direction

def find_direction(z, y, Classifier, f=10, max_iter=1000):
    # _,_,_, z = gen(image,None)
    z = z.float()
    x = z
    y = torch.tensor([y])
    y = F.one_hot(y, num_classes=2).float().to(device)
    batch_size = z.shape[0]
    print(batch_size)
    y = y.repeat(batch_size, 1)
    criterion_fn = nn.BCELoss()
    directions = []

    i = 0
    while i < max_iter:
        output = Classifier(z)
        pred = (output > 0.5).float()
        equal = torch.eq(y, pred)

        k = 0
        # while k < equal.shape[0]:
        while k < z.shape[0]:
            if equal[k].all():
                if i == 0:
                    print("分类器分类错误。。。。。")
                    # 删除当前的 z[j]
                    index = torch.nonzero(torch.arange(x.shape[0]).to(device) != k).squeeze()
                    x = torch.index_select(x, 0, index)
                    z = torch.index_select(z, 0, index)
                    equal = torch.index_select(equal, 0, index)
                    y = torch.index_select(y, 0, index)
                else:
                    print("找到了干预方向！！！", i)
                    directions.append(z[k] - x[k])
                    index = torch.nonzero(torch.arange(x.shape[0]).to(device) != k).squeeze()
                    x = torch.index_select(x, 0, index)
                    equal = torch.index_select(equal, 0, index)
                    z = torch.index_select(z, 0, index)
                    y = torch.index_select(y, 0, index)

            else:
                k += 1

        if z.shape[0] == 0:
            return directions
        output = Classifier(z)
        loss = criterion_fn(output, y)
        residual = -torch.autograd.grad(loss, z, create_graph=False)[0]
        z = z + f * residual
        z = z.float()
        i += 1

    print("没有找到干预方向。。。。。")
    return directions






dir_path = ''
os.makedirs(dir_path, exist_ok=True)
for i, data in tqdm(enumerate(train_loader, 0), desc="Processing", unit="batch"):
    images, labels = data['image'], data['attributes']
    images = images.to(device)
    labels = labels.to(device)
    print(labels[:,j])
    index0 = torch.nonzero(labels[:,j] == 0)
    images0 = torch.index_select(images, 0, index0.squeeze())
    _, _, _, z0,_ = gen(images0, None,None,w_space=True)
    index1 = torch.nonzero(labels[:,j] == 1)
    images1 = torch.index_select(images, 0, index1.squeeze())
    _, _, _, z1,_ = gen(images1, None,None,w_space=True)
    ds0 = find_direction(z0, 1, classifier)
    ds1 = find_direction(z1, 0, classifier)
    d0 = find_best_direction(ds0)
    d1 = find_best_direction(ds1)

    beat_dire0 = []
    beat_dire1 = []
    checkpoint_path = os.path.join(dir_path, f"{attribute_names[j]}.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        direction = checkpoint["d0"]
        if isinstance(direction, list):
            beat_dire0.extend(checkpoint['d0'])
        else:
            beat_dire0.append(checkpoint['d0'])
        direction = checkpoint["d1"]
        if isinstance(direction, list):
            beat_dire1.extend(checkpoint['d1'])
        else:
            beat_dire1.append(checkpoint['d1'])

    # 更新方向列表
    beat_dire0.append(d0)
    beat_dire1.append(d1)

    checkpoint = {
        'd0': beat_dire0,
        'd1': beat_dire1
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint updated at {checkpoint_path} for attribute {attribute_names[j]}')







