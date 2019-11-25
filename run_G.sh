# Config parameters: lambda_G_new, skip, videos_train, cri_G [, testset]
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py 0 0 coco100000

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python test_video.py 0 0 coco100000 water
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python test_video.py 0 0 coco100000 baby
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python test_video.py 0 0 coco100000 guitar
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python test_video.py 0 0 coco100000 gun
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python test_video.py 0 0 coco100000 drone
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python test_video.py 0 0 coco100000 cattoy
