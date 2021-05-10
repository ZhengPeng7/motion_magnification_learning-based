import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from config import Config
from magnet import MagNet
from data import get_gen_ABC, cuda2numpy
from callbacks import save_model, gen_state_dict
from losses import criterion_mag


# Configurations
config = Config()
cudnn.benchmark = True

magnet = MagNet().cuda()
if config.pretrained_weights:
    magnet.load_state_dict(gen_state_dict(config.pretrained_weights))
if torch.cuda.device_count() > 1:
    magnet = nn.DataParallel(magnet)
criterion = nn.L1Loss().cuda()

optimizer = optim.Adam(magnet.parameters(), lr=config.lr, betas=config.betas)

if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)
print('Save_dir:', config.save_dir)

# Data generator
data_loader = get_gen_ABC(config, mode='train')
print('Number of training image couples:', data_loader.data_len)

# Training
for epoch in range(1, config.epochs+1):
    print('epoch:', epoch)
    losses, losses_y, losses_texture_AC, losses_texture_BM, losses_motion_BC = [], [], [], [], []
    for idx_load in range(0, data_loader.data_len, data_loader.batch_size):

        # Data Loading
        batch_A, batch_B, batch_C, batch_M, batch_amp = data_loader.gen()

        # G Train
        optimizer.zero_grad()
        y_hat, texture_AC, texture_BM, motion_BC = magnet(batch_A, batch_B, batch_C, batch_M, batch_amp, mode='train')
        loss_y, loss_texture_AC, loss_texture_BM, loss_motion_BC = criterion_mag(y_hat, batch_M, texture_AC, texture_BM, motion_BC, criterion)
        loss = loss_y + (loss_texture_AC + loss_texture_BM + loss_motion_BC) * 0.1
        loss.backward()
        optimizer.step()

        # Callbacks
        losses.append(loss.item())
        losses_y.append(loss_y.item())
        losses_texture_AC.append(loss_texture_AC.item())
        losses_texture_BM.append(loss_texture_BM.item())
        losses_motion_BC.append(loss_motion_BC.item())
        if (
                idx_load > 0 and
                ((idx_load // data_loader.batch_size) %
                 (data_loader.data_len // data_loader.batch_size // config.num_val_per_epoch)) == 0
        ):
            print(', {}%'.format(idx_load * 100 // data_loader.data_len), end='')

    # Collections
    save_model(magnet.state_dict(), losses, config.save_dir, epoch)
    print('\ntime: {}m, ep: {}, loss: {:.3e}, y: {:.3e}, tex_AC: {:.3e}, tex_BM: {:.3e}, mot_BC: {:.3e}'.format(
        int((time.time()-config.time_st)/60), epoch, np.mean(losses), np.mean(losses_y), np.mean(losses_texture_AC), np.mean(losses_texture_BM), np.mean(losses_motion_BC)
    ))
