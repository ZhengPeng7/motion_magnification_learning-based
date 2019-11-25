import os
import time
import numpy as np

class Config(object):
    def __init__(self, lambda_G_new=None, skip=None, videos_train=None):
        # General
        self.epochs = 200
        self.GPUs = '0'
        self.batch_size = 4 * len(self.GPUs.split(','))
        self.use_energy = False
        self.date = '2019-11-24'
        prefix = '..'
        self.dir_train = os.path.join(prefix, 'datasets/motion_mag_data/train')
        self.dir_test = os.path.join(prefix, 'datasets/motion_mag_data/test')
        self.dir_water = os.path.join(prefix, 'datasets/motion_mag_data/train/train_vid_frames/val_water')
        self.dir_baby = os.path.join(prefix, 'datasets/motion_mag_data/train/train_vid_frames/val_baby')
        self.dir_gun = os.path.join(prefix, 'datasets/motion_mag_data/train/train_vid_frames/val_gun')
        self.dir_drone = os.path.join(prefix, 'datasets/motion_mag_data/train/train_vid_frames/val_drone')
        self.dir_guitar = os.path.join(prefix, 'datasets/motion_mag_data/train/train_vid_frames/val_guitar')
        self.dir_cattoy = os.path.join(prefix, 'datasets/motion_mag_data/train/train_vid_frames/val_cattoy')
        # Training
        if videos_train is None:
            self.videos_train = []
        elif 'vals' == videos_train:
            bb_mode = videos_train[-1]
            bb_num = 100000
            self.videos_train = sorted([
                i for i in os.listdir(os.path.join(self.dir_train, 'train_vid_frames'))
                if i[:4] == 'val_'
                ]
            )[:bb_num]
        elif 'coco' in videos_train:
            self.videos_train = []
        else:
            self.videos_train = sorted(videos_train.split('-'))
        self.cursor_end = 0 if not 'coco' in videos_train else int(videos_train.split('coco')[-1])
        self.lambda_G_new = 1e-3 if lambda_G_new is None else lambda_G_new
        self.pretrained_weights = ''
        self.skip = 0 if not isinstance(skip, int) else skip
        self.batch_size_test = 1
        self.video_num = len(self.videos_train)
        self.preproc = ['resize']

        self.lr = 1e-4
        self.betas_G = (0.9, 0.999)

        # Callbacks
        self.time_st = time.time()
        self.losses_G_old = []

        self.num_val_per_epoch = 5

        if self.cursor_end > 0 and self.video_num == 0:
            dataset = 'coco-{}'.format(self.cursor_end)
        elif self.cursor_end == 0 and self.video_num > 0:
            if self.video_num < 10:
                dataset = 'video-{}'.format('-'.join(self.videos_train))
            else:
                dataset = 'video-{}'.format('-'.join([self.videos_train[0], self.videos_train[-1]]))
        elif self.cursor_end > 0 and self.video_num > 0:
            dataset = 'coco-{}'.format(self.cursor_end) + 'video-{}'.format(self.video_num)

        net = 'ECCV18'

        self.save_dir = os.path.join(prefix, 'weights/mm/{}_net{}_dataset{}_weights'.format(
            self.date, net, dataset
        ))

        self.coco_amp_lst = np.loadtxt(os.path.join(self.dir_train, 'train_mf.txt'))[:self.cursor_end]
