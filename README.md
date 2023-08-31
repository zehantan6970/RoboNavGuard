# RoboNavGuard
Robot Navigation Guardian solution
RoboNavGuard is an innovative hybrid system to assist indoor mobile robots in enhancing their visual grounding navigation accuracy through the utilization of the grid map.
## PyraBiNet++
PyraBiNet is an innovative hybrid model optimized for lightweight semantic segmentation tasks. This model ingeniously merges the merits of Convolutional Neural Networks (CNNs) and Transformers.
## PyraBiNet++ Installation
Our project has been tested on torch=1.7.0 torchvision=0.8.1 timm=0.3.2
## PyraBiNet++ Pre-Training
Train 300epochs on the ImageNet dataset with 8 GPUs:
sh dist_train.sh configs/sem_fpn/Pyrabinet/fpn_pyrabinet_ade20k_80k.py 8 --data-path /path/to/imagenet
## PyraBiNet++ Training
Use 4 GPUs to train pyrabinet+Semantic FPN on the ADE20K dataset, run:
dist_train.sh configs/sem_fpn/Pyrabinet/fpn_pyrabinet_ade20k_80k.py 4
## Evaluation
dist_test.sh configs/sem_fpn/Pyrabinet/fpn_pyrabinet_ade20k_80k.py /path/to/checkpoint_file 4 --out results.pkl --eval mIoU
## Calculating FLOPS & Params
python flops.py

## Acknowledgement
We appreciate the open-source of the following projects: [MVT-3DVG](https://github.com/sega-hsj/MVT-3DVG) ,  [OFA](https://github.com/OFA-Sys), [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [ScanRefer](https://github.com/daveredrum/ScanRefer), [EDA](https://github.com/yanmin-wu/EDA ),  [Mask_RCNN](https://github.com/matterport/Mask_RCNN ), [PVT](https://github.com/whai362/PVT),
and [ScanNet](https://github.com/ScanNet/ScanNet).
