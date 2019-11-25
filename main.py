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
from losses import criterion_mag_G


# Configurations
lambda_G_new, skip, videos_train = sys.argv[1:]
lambda_G_new = None if lambda_G_new == 'None' else float(lambda_G_new)
skip = None if skip == 'None' else int(skip)
videos_train = None if videos_train == 'None' else videos_train
config = Config(lambda_G_new, skip, videos_train)

# os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs
cudnn.benchmark = True

generator = MagNet()
if config.pretrained_weights:
    generator.load_state_dict(gen_state_dict(config.pretrained_weights))
if len(config.GPUs.split(',')) > 1:
    generator = nn.DataParallel(generator)
generator = generator.cuda()
criterion_G = nn.L1Loss().cuda()

optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=config.betas_G)

if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)
print('Save_dir:', config.save_dir)

# Data generator
data_loader = get_gen_ABC(config, mode='train')
print('Number of training image couples:', data_loader.data_len)

# Training
for epoch in range(1, config.epochs+1):
    print('epoch:', epoch)
    losses_G_old = []
    for idx_load in range(0, data_loader.data_len, data_loader.batch_size):

        # Data Loading
        batch_A, batch_B, batch_C, batch_M, batch_amp = data_loader.gen()

        # G Train
        optimizer_G.zero_grad()
        mag_A, appearance_AC, motion_BC = generator(batch_A, batch_B, batch_C, batch_amp, mode='train')
        loss_G = criterion_mag_G(mag_A, batch_M, appearance_AC, motion_BC, criterion_G)
        losses_G_old.append(loss_G.item())
        loss_G.backward()
        optimizer_G.step()
        if (
                idx_load > 0 and
                ((idx_load // data_loader.batch_size) %
                 (data_loader.data_len // data_loader.batch_size // config.num_val_per_epoch)) == 0
        ):
            print('ep{}, loss: old={:.2e}'.format(epoch, losses_G_old[-1]))

    # Collections
    save_model(generator.state_dict(), losses_G_old if losses_G_old != [] else [0],
                config.save_dir, epoch)
    print('epoch={}, loss_old={:.3e}, time={}m'.format(
        epoch, np.mean(losses_G_old) if losses_G_old != [] else 0,
        int((time.time()-config.time_st)/60)
    ))
