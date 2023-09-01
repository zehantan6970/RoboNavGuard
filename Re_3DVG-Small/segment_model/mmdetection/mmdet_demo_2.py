import os

import numpy as np
from mmdet.apis import init_detector, inference_detector

# import cv2 as cv

# 指定模型的配置文件和 checkpoint 文件路径
envpath = '/home/light/anaconda3/envs/openmmlab/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
# config_file = '/home/light/gree/slam/D3VG/mmdetection_master/configs/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco.py'
# checkpoint_file = '/media/light/light_t2/PROJECTS/model_checkpoints/checkpoints/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth'
config_file = '/home/light/gree/slam/D3VG/mmdetection_master/configs/mask_rcnn/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco-trash_a.py'
checkpoint_file = '/media/light/light_t2/PROJECTS/model_checkpoints/checkpoints/epoch_62.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')


def generate_seg(img, output_dir=None):
    """
    Args:
        img: 输入的图片
        output_dir: 分割结果保存文件夹路径

    Returns:
        boxes_class:目标框类别,segs:mask坐标,boxes:目标框坐标
    """
    # box类别
    boxes_class = []
    # mask掩模图
    segs = []
    # box
    boxes = []
    # 根据配置文件和 checkpoint 文件构建模型
    # 测试单张图片并展示结果
    result = inference_detector(model, img)
    # 或者将可视化结果保存为图片
    model.show_result(img, result, score_thr=0.5, out_file=output_dir, show=False)
    score = model.BOXES[:, -1]
    choose_index = np.where(score > 0.5)[0]
    label_id = model.LABEL[score > 0.5]
    for i, l in enumerate(label_id):
        if model.CLASSES[l] not in ["book", "picture"]:
            boxes_class.append(model.CLASSES[l])
            segs.append(model.SEGS[choose_index[i]])
            boxes.append(model.BOXES[:, :-1][choose_index[i]])
    print("box的类别:", boxes_class)
    return boxes_class, segs, boxes


#
if __name__ == "__main__":
    img = '/home/light/文档/0731/image/rgb/1690775774.005657.png'
    output_dir = None
    generate_seg(img, output_dir)
