import os
import numpy as np
import torch


def save_model(weights, losses, weights_dir, epoch):
    loss = np.mean(losses)
    path_ckpt = os.path.join(
        weights_dir, 'magnet_epoch{}_loss{:.2e}.pth'.format(epoch, loss)
    )
    torch.save(weights, path_ckpt)


def gen_state_dict(weights_path):
    st = torch.load(weights_path)
    st_ks = list(st.keys())
    st_vs = list(st.values())
    state_dict = {}
    for st_k, st_v in zip(st_ks, st_vs):
        state_dict[st_k.replace('module.', '')] = st_v
    return state_dict
