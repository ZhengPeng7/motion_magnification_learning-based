import os
import numpy as np
import torch


def save_model(weights_G, losses_G, weights_dir, epoch, weights_D=None, losses_D=None):
    loss_old = np.mean(losses_G)
    path_ckpt_G = os.path.join(
        weights_dir, 'generator_epoch{}_loss_old{:.2e}.pth'.format(epoch, loss_old)
    )
    torch.save(weights_G, path_ckpt_G)
    if weights_D is not None:
        loss_D = np.mean(losses_D)
        path_ckpt_D = os.path.join(
            weights_dir, 'discriminator_epoch{}_loss_D{:.2e}.pth'.format(epoch, loss_D)
        )
        torch.save(weights_D, path_ckpt_D)


def gen_state_dict(weights_path):
    st = torch.load(weights_path)
    st_ks = list(st.keys())
    st_vs = list(st.values())
    state_dict = {}
    for st_k, st_v in zip(st_ks, st_vs):
        state_dict[st_k.replace('module.', '')] = st_v
    return state_dict
