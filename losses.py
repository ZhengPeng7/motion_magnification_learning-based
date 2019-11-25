import os
import numpy as np
import torch
import torch.autograd as ag


def criterion_mag_G(mag_A, batch_M, texture_AC, motion_BC, criterion):
    loss_M = criterion(mag_A, batch_M)
    loss_texture = criterion(*texture_AC) * 0.1
    loss_motion = criterion(*motion_BC) * 0.1
    # print("loss_M/app/mot: {:.2e}, {:.2e}, {:.2e}".format(loss_M, loss_texture, loss_motion))
    return loss_M + loss_texture + loss_motion
