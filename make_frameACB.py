"""
Put it into the corresponding datasets directory, e.g. `/datasets/motion_mag_data/train/train_vid_frames` for me.
Make the original frames into frameAs, frameBs, frameCs(same as frameBs here)
"""
import os
import sys


# Choose the dir you want
dirs = sorted([i for i in os.listdir('.') if i in 
sys.argv[1].split('+')
# and int(i.split('_')[-1].split('.')[0]) > 0
]
# , key=lambda x: int(x.split('_')[-1])
)[:]

for d in dirs:
    print('ACB-Processing on', d)
    os.chdir(d)
    os.mkdir('frameA')
    os.mkdir('frameC') 
    files = sorted([f for f in os.listdir('.') if f[-4:] == '.png'], key=lambda x: int(x.split('.')[0]))
    os.system('cp ./*png frameA && cp ./*png frameC') 
    os.remove(os.path.join('frameA', files[-1])) 
    os.remove(os.path.join('frameC', files[0])) 
    for f in sorted(os.listdir('frameC'), key=lambda x: int(x.split('.')[0])): 
        f_new = os.path.join('frameC', '%06d' % (int(f.split('.')[0])-1) + '.png') 
        f = os.path.join('frameC', f) 
        os.rename(f, f_new) 
    os.system('cp -r frameC frameB') 
    os.system('rm ./*.png')
    os.chdir('..')

