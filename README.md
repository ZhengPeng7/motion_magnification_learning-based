# motion_magnification_learning-based
This is an unofficial implementation of "[Learning-based Video Motion Magnification](https://arxiv.org/abs/1804.02684)" in Pytorch==1.3.0.
Please refer to [the official Tensorflow==1.3.0 version implementation](https://github.com/12dmodel/deep_motion_mag), which is a very nice work and helped me a lot.

# Data preparation

0. About the synthetic dataset for **training**, please refer to the official repository mentioned above.

1. Check the settings of val_dir in **config.py** and modify it if necessary.

2. To convert the **validation** video into frames:

    `mkdir VIDEO_NAME && ffmpeg -i VIDEO_NAME.mp4 -f image2 VIDEO_NAME/%06d.png`

3. Modify the frames into **frameA/frameB/frameC**:

    `python make_frameACB.py `(remember adapt the filter condition of the directory at the beginning of the program.)

# Little differences from the official codes

1. **Poisson noise** is not used here because I was a bit confused about that in official code. Although I coded it in data.py, and it works exactly the same as the official codes as I checked by examples.
2. About the **optimizer**, we kept it the same as that in the original paper -- Adam(lr=1e-4, betas=(0.9, 0.999)) with no weight decay, which is different from the official codes.
3. About the $ \lambda $ in loss, we also adhere to the original paper -- set to 0.1, which is different from the official codes.
4. The **temporal filter** is currently a bit confusing for me, so I haven't made the part of testing with temporal filter, sorry for that:(...

# One thing **important**

If you check the Fig.2-a in the original paper, you will find that the predicted magnified frame $\hat{Y}$ is actually $ texture(X_b) + motion(X_a->X_b) * \alpha $, although the former one is theoretically same as $ texture(X_a) + motion(X_a->X_b) * (\alpha + 1) $ with the same $ \alpha $.

<img src="materials/fig2-a.png" alt="fig2-a" style="zoom:60%;" div align=center />

However, what makes it matter is that the authors used perturbation for regularization, and the images in the dataset given has 4 parts:

1. frameA: $ X_a $, unperturbed;
2. frameB: perturbed frameC, is actually $ X_{b}^{'} $ in the paper,
3. frameC: the real $ X_b $, unperturbed;
4. **amplified**: represent both $ Y $ and $ Y^{'} $, perturbed.

Here is the first training sample, where you can see clear that **no perturbation** between **A-C** nor between **B-amp**, and no motion between B-C:

<img src="materials/dogs.png" alt="dog" style="zoom: 67%;" div align=center />

Given that, we don't have the unperturbed amplified frame, so **we can only use the former formula**(with $ texture(X_b) $). Besides, if you check the **loss** in the original paper, you will find the $ L_1(V_{b}^{'}, V_{Y}^{'}) $, where is the $ V_{Y}^{'}$?... I also referred to some third-party reproductions on this problem which confused me a lot, but none of them solve it. And some just gave 0 to $ L_1(V_{b}^{'}, V_{Y}^{'}) $ manually, so I think they noticed this problem too but didn't manage to understand it.

Here are some links to the issues about this problem in the official repository, [issue-1](https://github.com/12dmodel/deep_motion_mag/issues/3), [issue-2](https://github.com/12dmodel/deep_motion_mag/issues/5), [issue-3](https://github.com/12dmodel/deep_motion_mag/issues/4), if you want to check them.

# Run
`bash run.sh` to train and test.

It took me around 20 hours to train for 12 epochs on a single TITAN-Xp.

If you don't want to use all the 100,000 groups to train, you can modify the `frames_train='coco100000'` in config.py to coco30000 or some other number.

You can download the weights-ep12 from [the release](https://github.com/ZhengPeng7/motion_magnification_learning-based/releases/tag/v1.0), and `python test_videos.py baby-guitar-yourself-...` to do the test.

# Results

Here are some results generated from the model trained on the whole synthetic dataset for **12** epochs. 

Baby, amplification factor = 50

![baby](materials/baby_comp.gif)

Guitar, amplification factor = 20

![guitar](materials/guitar_comp.gif)

And I also took a video on the face of myself with amplification factor 20, which showed a Chinese idiom called '夺眶而出'😂.

![myself](materials/myself_comp.gif)

> Any question, all welcome:)
