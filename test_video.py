import os
import sys
import cv2
import torch
import numpy as np
from config import Config
from magnet import MagNet
from data import get_gen_ABC, unit_postprocessing, numpy2cuda, resize2d
from callbacks import gen_state_dict


# config
lambda_G_new, skip, videos_train, testset = sys.argv[1:]
lambda_G_new = None if lambda_G_new == 'None' else float(lambda_G_new)
skip = None if skip == 'None' else int(skip)
videos_train = None if videos_train == 'None' else videos_train
config = Config(lambda_G_new, skip, videos_train)
# os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs
testset = testset if len(sys.argv) >= 5 else 'baby'
save_dir_test = config.save_dir.replace('weights', 'results') + '_' + testset

data_loader = get_gen_ABC(config, mode='test_on_'+testset)
print('Number of test image couples:', data_loader.data_len)
vid_size = cv2.imread(data_loader.paths[0]).shape[:2][::-1]

# Load weights
ep = ''
weights_file = sorted(
    [p for p in os.listdir(config.save_dir) if 'loss_old' in p and 'epoch{}'.format(ep) in p and '_old0.00e' not in p],
    key=lambda x: float(x.rstrip('.pth').split('_loss_old')[-1])
)[0]
weights_path = os.path.join(config.save_dir, weights_file)
ep = int(weights_path.split('epoch')[-1].split('_')[0])
state_dict = gen_state_dict(weights_path)

model_test = MagNet().cuda()
model_test.load_state_dict(state_dict)
model_test.eval()
print("Loading weights:", weights_file)

# Test
for amp in list(range(0, 10, 3)) + list(range(10, 51, 10)) + list(range(80, 121, 20)) + list(range(170, 271, 50)):
    print(amp)
    frames = []
    data_loader = get_gen_ABC(config, mode='test_on_'+testset)
    for idx_load in range(0, data_loader.data_len, data_loader.batch_size):
        if (idx_load+1) % 100 == 0:
            print('{}'.format(idx_load+1), end=', ')
        batch_A, batch_C = data_loader.gen_test()
        amp_factor = numpy2cuda(amp)
        for _ in range(len(batch_A.shape) - len(amp_factor.shape)):
            amp_factor = amp_factor.unsqueeze(-1)
        with torch.no_grad():
            mag_As = model_test(batch_A, 0, batch_C, amp_factor, mode='evaluate')
        for mag_A in mag_As:
            mag_A = unit_postprocessing(mag_A, vid_size=vid_size)
            frames.append(mag_A)
            if len(frames) >= data_loader.data_len:
                break
        if len(frames) >= data_loader.data_len:
            break
    data_loader = get_gen_ABC(config, mode='test_on_'+testset)
    frames = [unit_postprocessing(data_loader.gen_test()[0], vid_size=vid_size)] + frames

    # Make videos of framesMag
    video_dir = save_dir_test.rstrip('/') + '_videos_ep{}'.format(ep)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    FPS = 30
    out = cv2.VideoWriter(
        os.path.join(video_dir, '{}_amp{}.avi'.format(testset, amp)),
        cv2.VideoWriter_fourcc(*'DIVX'),
        FPS, frames[0].shape[-2::-1]
    )
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.bilateralFilter(frame, 9, 75, 75)
        cv2.putText(frame, 'amp_factor={}'.format(amp), (7, 37),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        out.write(frame)
    out.release()
    print('{} has been done.'.format(os.path.join(video_dir, '{}_amp{}.avi'.format(testset, amp))))
