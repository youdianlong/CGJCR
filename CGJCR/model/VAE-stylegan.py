import torch,gc
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.autograd import Variable
from loss_fn.ID_loss import IDLoss
# device = torch.device('cuda:2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm
import lpips
from dataloader import get_custom_dataloader
from models import VAE_sty, Discriminators
from utils import show_and_save, plot_loss
import os
from torch.nn.parallel import DataParallel
import torch.optim as optim


train_loader = get_custom_dataloader(32)
gen = VAE_sty().to(device)
discrim = Discriminators(conv_dim=32).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    if not isinstance(gen, nn.DataParallel):
        gen = nn.DataParallel(gen)
    if not isinstance(discrim, nn.DataParallel):
        discrim = nn.DataParallel(discrim)
else:
    print("Using single GPU or CPU")

print(f"Current device: {torch.cuda.current_device()}")
print(f"Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

real_batch = next(iter(train_loader))
real_batch = real_batch.to(device)

show_and_save("training", make_grid((real_batch * 0.5 + 0.5).cpu(), 8))

epochs = 1000
lr = 3e-4
alpha = 0.1
gamma = 15

ckpt_dir = ""
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

checkpoint_path = ""


def show_and_save(filename, image_tensor):
    # Display the image (optional)
    plt.imshow(image_tensor.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

    # Save the image
    save_path = "" + filename
    if not os.path.exists(""):
        os.makedirs("")
    torchvision.utils.save_image(image_tensor, save_path)


criterion = nn.BCELoss().to(device)
optim_E = optim.Adam(gen.module.encoder.parameters(), lr=lr)
optim_D = optim.Adam(gen.module.decoder.parameters(), lr=lr)
optim_Dis = optim.Adam(discrim.module.parameters(), lr=lr * alpha)
idloss_fn = IDLoss().eval().to(device)
z_fixed = Variable(torch.randn((32, 512))).to(device)
x_fixed = Variable(real_batch).to(device)
loss_fn = lpips.LPIPS(net="vgg").eval().to(device)

if checkpoint_path:
    checkpoint = torch.load(checkpoint_path)

    # # 39
    # new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['gen_state_dict'].items()}
    # gen.load_state_dict(new_state_dict)
    # new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['discrim_state_dict'].items()}
    # discrim.load_state_dict(new_state_dict)


    #114
    new_state_dict = {}
    for k, v in checkpoint['gen_state_dict'].items():
        if torch.cuda.device_count() > 1:
        #     new_state_dict[f'module.{k}'] = v
        # else:
            new_state_dict[k] = v
    gen.load_state_dict(new_state_dict)

    new_state_dict = {}
    for k, v in checkpoint['discrim_state_dict'].items():
        if torch.cuda.device_count() > 1:
        #     new_state_dict[f'module.{k}'] = v
        # else:
            new_state_dict[k] = v
    discrim.load_state_dict(new_state_dict)





    #discrim.load_state_dict(checkpoint['discrim_state_dict'])

    # 加载优化器的状态
    optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
    optim_Dis.load_state_dict(checkpoint['optim_Dis_state_dict'])
    optim_E.load_state_dict(checkpoint['optim_E_state_dict'])

    Enc_loss_list = checkpoint.get('Enc_loss_list', [])
    Dis_loss_list = checkpoint.get('Dis_loss_list', [])
    Dec_loss_list = checkpoint.get('Dec_loss_list', [])

    start_epoch = checkpoint['epoch'] + 1  # 加载的 epoch 值加1，从下一轮开始训练

else:
    print("No checkpoint path provided. Model will not be restored.")
    start_epoch = 0

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

best_loss = float('inf')
for epoch in range(start_epoch, epochs):

    Enc_loss_list, Dis_loss_list, Dec_loss_list = [], [], []

    for i, data in tqdm(enumerate(train_loader, 0), desc="Processing", unit="batch"):
        bs = data.size()[0]
        ones_label = torch.ones(bs, 1).to(device)
        zeros_label = torch.zeros(bs, 1).to(device)
        zeros_label1 = torch.zeros(bs, 1).to(device)
        datav = data.to(device)

        mean, logvar, rec_enc,_,_ = gen(datav,None,None,w_space=True)
        #判别器
        output = discrim(datav)
        errD_real = criterion(output, ones_label)
        output = discrim(rec_enc)
        errD_rec_enc = criterion(output, zeros_label)
        gan_loss = errD_real + errD_rec_enc
        Dis_loss_list.append(gan_loss.item())
        optim_Dis.zero_grad()
        optim_D.zero_grad()
        optim_E.zero_grad()
        gan_loss.backward(retain_graph=True)
        clip_grad_norm_(discrim.parameters(), max_norm=1.0)
        optim_Dis.step()
        #解码器
        n = 5
        for _ in range(n):
            mean, logvar, rec_enc,_,_ = gen(datav,None,None,w_space=True)
            output = discrim(datav)
            errD_real = criterion(output, ones_label)
            output = discrim(rec_enc)
            errD_rec_enc = criterion(output, zeros_label)
            gan_loss = errD_real + errD_rec_enc
            rec_loss = loss_fn(datav,rec_enc).mean()
            err_dec = gamma * rec_loss.mean() - gan_loss
            Dec_loss_list.append(err_dec.item())
            id_loss = idloss_fn(rec_enc, datav)
            mse_fn = nn.MSELoss()
            mse_loss = mse_fn(rec_enc, datav)
            prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data)
            err_enc = 5 * rec_loss + id_loss + 4 * mse_loss + prior_loss
            Enc_loss_list.append(err_enc.item())

            # 更新生成器和编码器的参数
            optim_D.zero_grad()
            optim_Dis.zero_grad()
            optim_E.zero_grad()
            err_dec.backward(retain_graph=True)
            clip_grad_norm_(gen.module.decoder.parameters(), max_norm=1.0)
            optim_D.step()
            err_enc.backward(retain_graph=False)
            clip_grad_norm_(gen.module.encoder.parameters(), max_norm=1.0)
            optim_E.step()

        if i % 100 == 0:
            b = gen(datav,None,None,w_space=True)[2]
            merged_images = torch.cat([datav[:8], b[:8]], dim=0)
            merged_images = merged_images.detach()
            show_and_save('Celebarec_epoch_%d.png' % epoch, make_grid((merged_images * 0.5 + 0.5).cpu(), nrow=8))
            print(
                '[%d/%d][%d/%d]\tLoss_mse: %.6f\tLoss_prior: %.6f\tRec_loss: %.6f\tID_loss: %.6f\tDis_loss: %0.6f\tEnc_loss: %.6f\tDec_loss: %.6f'
                % (epoch, epochs, i, len(train_loader),
                   mse_loss.item(), prior_loss, rec_loss.item(), id_loss.item(), gan_loss.item(), err_enc.item(),
                   err_dec.item()))

    checkpoint = {
        'gen_state_dict': gen.state_dict(),
        'discrim_state_dict': discrim.state_dict(),
        'optim_D_state_dict': optim_D.state_dict(),
        'optim_Dis_state_dict': optim_Dis.state_dict(),
        'optim_E_state_dict': optim_E.state_dict(),
        'Enc_loss_list': Enc_loss_list,
        'Dis_loss_list': Dis_loss_list,
        'Dec_loss_list': Dec_loss_list,
        'epoch': epoch
    }

    if epochs % 1 == 0:
        checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(ckpt_dir, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path} for epoch {epoch}')

    b = gen(x_fixed,None,None,w_space=True)[2]
    merged_images = torch.cat([real_batch[:8], b[:8]], dim=0)
    merged_images = merged_images.detach()
    print(merged_images.min(), merged_images.max())

    c,_ = gen.module.decoder(z_fixed,None,None,w_space=True)
    c = c.detach()
    print(c.min(), c.max())
    show_and_save('Celebarec_noise_epoch_%d.png' % epoch, make_grid((c * 0.5 + 0.5).cpu(), nrow=8))
    show_and_save('Celebarec_epoch_%d.png' % epoch, make_grid((merged_images * 0.5 + 0.5).cpu(), nrow=8))

plot_loss(Dis_loss_list)
plt.figure()
plot_loss(Dec_loss_list)
plt.figure()
plot_loss(Enc_loss_list)


