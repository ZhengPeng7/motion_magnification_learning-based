import torch
import torch.nn as nn
from data import numpy2cuda


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class Conv2D_activa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride,
            padding=0, dilation=1, activation='relu'
    ):
        super(Conv2D_activa, self).__init__()
        self.padding = padding
        if self.padding:
            self.pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            dilation=dilation, bias=None
        )
        self.activation = activation
        if activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        if self.padding:
            x = self.pad(x)
        x = self.conv2d(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, dim_intermediate=32, ks=3, s=1):
        super(ResBlk, self).__init__()
        p = (ks - 1) // 2
        self.cba_1 = Conv2D_activa(dim_in, dim_intermediate, ks, s, p, activation='relu')
        self.cba_2 = Conv2D_activa(dim_intermediate, dim_out, ks, s, p, activation=None)

    def forward(self, x):
        y = self.cba_1(x)
        y = self.cba_2(y)
        return y + x


def _repeat_blocks(block, dim_in, dim_out, num_blocks, dim_intermediate=32, ks=3, s=1):
    blocks = []
    for idx_block in range(num_blocks):
        if idx_block == 0:
            blocks.append(block(dim_in, dim_out, dim_intermediate=dim_intermediate, ks=ks, s=s))
        else:
            blocks.append(block(dim_out, dim_out, dim_intermediate=dim_intermediate, ks=ks, s=s))
    return nn.Sequential(*blocks)


class Encoder(nn.Module):
    def __init__(
            self, dim_in=3, dim_out=32, num_resblk=3,
            use_texture_conv=True, use_motion_conv=True, texture_downsample=True,
            num_resblk_texture=2, num_resblk_motion=2
    ):
        super(Encoder, self).__init__()
        self.use_texture_conv, self.use_motion_conv = use_texture_conv, use_motion_conv

        self.cba_1 = Conv2D_activa(dim_in, 16, 7, 1, 3, activation='relu')
        self.cba_2 = Conv2D_activa(16, 32, 3, 2, 1, activation='relu')

        self.resblks = _repeat_blocks(ResBlk, 32, 32, num_resblk)

        # texture representation
        if self.use_texture_conv:
            self.texture_cba = Conv2D_activa(
                32, 32, 3, (2 if texture_downsample else 1), 1,
                activation='relu'
            )
        self.texture_resblks = _repeat_blocks(ResBlk, 32, dim_out, num_resblk_texture)

        # motion representation
        if self.use_motion_conv:
            self.motion_cba = Conv2D_activa(32, 32, 3, 1, 1, activation='relu')
        self.motion_resblks = _repeat_blocks(ResBlk, 32, dim_out, num_resblk_motion)

    def forward(self, x):
        x = self.cba_1(x)
        x = self.cba_2(x)
        x = self.resblks(x)

        if self.use_texture_conv:
            texture = self.texture_cba(x)
            texture = self.texture_resblks(texture)
        else:
            texture = self.texture_resblks(x)

        if self.use_motion_conv:
            motion = self.motion_cba(x)
            motion = self.motion_resblks(motion)
        else:
            motion = self.motion_resblks(x)

        return texture, motion


class Decoder(nn.Module):
    def __init__(self, dim_in=32, dim_out=3, num_resblk=9, texture_downsample=True):
        super(Decoder, self).__init__()
        self.texture_downsample = texture_downsample

        if self.texture_downsample:
            self.texture_up = nn.UpsamplingNearest2d(scale_factor=2)
            # self.texture_cba = Conv2D_activa(dim_in, 32, 3, 1, 1, activation='relu')

        self.resblks = _repeat_blocks(ResBlk, 64, 64, num_resblk, dim_intermediate=64)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.cba_1 = Conv2D_activa(64, 32, 3, 1, 1, activation='relu')
        self.cba_2 = Conv2D_activa(32, dim_out, 7, 1, 3, activation=None)

    def forward(self, texture, motion):
        if self.texture_downsample:
            texture = self.texture_up(texture)
        if motion.shape != texture.shape:
            texture = nn.functional.interpolate(texture, size=motion.shape[-2:])
        x = torch.cat([texture, motion], 1)

        x = self.resblks(x)

        x = self.up(x)
        x = self.cba_1(x)
        x = self.cba_2(x)

        return x


class Manipulator(nn.Module):
    def __init__(self):
        super(Manipulator, self).__init__()
        self.g = Conv2D_activa(32, 32, 3, 1, 1, activation='relu')
        self.h_conv = Conv2D_activa(32, 32, 3, 1, 1, activation=None)
        self.h_resblk = ResBlk(32, 32)

    def forward(self, motion_A, motion_B, amp_factor):
        motion = motion_B - motion_A
        motion_delta = self.g(motion) * amp_factor
        motion_delta = self.h_conv(motion_delta)
        motion_delta = self.h_resblk(motion_delta)
        motion_mag = motion_B + motion_delta
        return motion_mag


class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.encoder = Encoder(dim_in=3*1)
        self.manipulator = Manipulator()
        self.decoder = Decoder(dim_out=3*1)

    def forward(self, batch_A, batch_B, batch_C, batch_M, amp_factor, mode='train'):
        if mode == 'train':
            texture_A, motion_A = self.encoder(batch_A)
            texture_B, motion_B = self.encoder(batch_B)
            texture_C, motion_C = self.encoder(batch_C)
            texture_M, motion_M = self.encoder(batch_M)
            motion_mag = self.manipulator(motion_A, motion_B, amp_factor)
            y_hat = self.decoder(texture_B, motion_mag)
            texture_AC = [texture_A, texture_C]
            motion_BC = [motion_B, motion_C]
            texture_BM = [texture_B, texture_M]
            return y_hat, texture_AC, texture_BM, motion_BC
        elif mode == 'evaluate':
            texture_A, motion_A = self.encoder(batch_A)
            texture_B, motion_B = self.encoder(batch_B)
            motion_mag = self.manipulator(motion_A, motion_B, amp_factor)
            y_hat = self.decoder(texture_B, motion_mag)
            return y_hat


def main():
    model = MagNet()
    print('model:\n', model)


if __name__ == '__main__':
    main()
