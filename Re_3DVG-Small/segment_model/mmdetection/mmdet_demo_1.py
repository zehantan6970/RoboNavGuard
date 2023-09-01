import numpy as np
import os
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import mmcv
import cv2 as cv
import matplotlib.pyplot as plt
from mmdet.registry import VISUALIZERS
import time
config_file = '/media/light/light_t2/PROJECTS/D3VG_large/segment_model/mmdetection/configs/rtmdet/rtmdet-ins_x_8xb16-300e_coco.py'
checkpoint_file = '/media/light/light_t2/PROJECTS/model_checkpoints/rtmdet/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'

# config_file = '/media/light/light_t2/PROJECTS/D3VG_large/segment_model/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py'
# checkpoint_file = '/media/light/light_t2/PROJECTS/model_checkpoints/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth'

# config_file = 'configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py'
# checkpoint_file = '/media/light/light_t2/PROJECTS/model_checkpoints/mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic_20220407_104949-82f8d28d .pth'

# config_file = '/media/light/light_t2/PROJECTS/D3VG_large/segment_model/mmdetection/configs/mask_rcnn/mask-rcnn_x101-64x4d_fpn_1x_coco.py'
# checkpoint_file = '/media/light/light_t2/PROJECTS/model_checkpoints/maskrcnn/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth'




# register all modules in mmdet into the registries
register_all_modules()
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# envpath = '/home/light/anaconda3/envs/openmmlab/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
id_coco = np.array(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush'])


def generate_seg(img, output_dir):
    # init visualizer(run the block only once in jupyter notebook)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # # the dataset_meta is loaded from the checkpoint and
    # # then pass to the model in init_detector
    # visualizer.dataset_meta = model.dataset_meta
    # Use the detector to do inference
    start_time = time.time()
    image = mmcv.imread(img, channel_order='rgb')
    result = inference_detector(model, image)
    end_time = time.time()
    print("segment time cost:%fs" % (end_time - start_time))
    we_want_index = result.cpu().pred_instances["scores"] > 0.3
    we_want_masks = result.cpu().pred_instances["masks"][we_want_index]
    we_want_bboxes = result.cpu().pred_instances["bboxes"][we_want_index]
    we_want_labels = result.cpu().pred_instances["train_labels"][we_want_index]
    we_want_labels = id_coco[we_want_labels]
    # show the results
    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt = None,
        wait_time=0,
        pred_score_thr=0.3
    )
    visualizer.show()
    return we_want_labels, we_want_masks, we_want_bboxes


if __name__ == "__main__":
    img = '/home/light/文档/0731/image/rgb/1690775774.005657.png'
    output_dir = "result.jpg"
    generate_seg(img, output_dir)
