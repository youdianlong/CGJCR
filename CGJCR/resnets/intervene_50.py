import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:2')
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
from models import VAE_sty
import os
import numpy as np
from torch.nn.parallel import DataParallel


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


    # train_dataset = CelebADataset(img_dir="/data1/lc/celeba/eyeglass/",
    #                     attr_path="/data1/lc/celeba/list_attr_celeba.txt", transform=transform)
    train_dataset = CelebADataset(img_dir="",
                                  attr_path="", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = CelebADataset(img_dir="",
                        attr_path="", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False, drop_last=True)

    return train_loader, test_loader
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

def show_and_save(filename, image_tensor):
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    save_path = "" + filename
    if not os.path.exists(""):
        os.makedirs("")
    torchvision.utils.save_image(image_tensor, save_path)

train_loader,test_loader = get_custom_dataloader(1)
gen = VAE_sty().to(device)
# if torch.cuda.device_count() > 1:
#     gen = DataParallel(gen)

checkpoint_path = ""
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['gen_state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')  # 移除 'module.' 前缀
    new_state_dict[new_key] = value
gen.load_state_dict(new_state_dict)
# gen.load_state_dict(checkpoint['gen_state_dict'])
attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                   'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                   'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                   'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
j=[39]
intervene_attribute = [attribute_names[i] for i in j]
dir_path = ''

new_direction0 = []
new_direction1 = []
for attribute in intervene_attribute:
    checkpoint_path = os.path.join(dir_path, f"{attribute}.pth")
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    direction = checkpoint["d0"]
    if isinstance(direction, list):
        direction = find_best_direction(direction)
    new_direction0.append(direction)

    direction = checkpoint["d1"]
    if isinstance(direction, list):
        direction = find_best_direction(direction)
    new_direction1.append(direction)


data = next(iter(train_loader))
image,label = data['image'], data['attributes']
image=image.to(device)
label=label.to(device)
selected_directions = []
for i in range(len(j)):
    index = label[:,j[i]]
    if index == 0:
        selected_directions.append(new_direction0[i])
    elif index == 1:
        selected_directions.append(new_direction1[i])
intervene_factors = [0,5,10,15,20,25]
intervened_images = []
for factor in intervene_factors:
    new_direction0_intervened = [d * factor for d in selected_directions]
    _,_, rec_enc,_,_ = gen(image,new_direction0_intervened,None,w_space=True)
    intervened_images.append(rec_enc)

merged_images = torch.cat([image] + intervened_images, dim=0)
merged_images = merged_images.detach()
name = "_".join(intervene_attribute)
name = "Celeba_" + name + ".png"
show_and_save(name, make_grid((merged_images * 0.5 + 0.5).cpu(), nrow=10))








