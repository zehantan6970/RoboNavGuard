import argparse
import os
import time
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from dataload.dataloader_vit_small_patch16_224_in21k_with_vit_finetune import dataload, dataload_for_eval
from models.model_vit_en_small_patch16_224_in21k_with_finetune_BERT import Model
from segment_model.mmdetection.mmdet_demo_2 import generate_seg
from utils.utils import get_obb, del_file

envpath = '/home/light/anaconda3/envs/openmmlab/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
# -------- #
# 配置
# -------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--hidden_dim', default=512)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--nheads', default=4)
parser.add_argument('--dim_feedforward', default=2048)
parser.add_argument("--enc_layers", default=2)
parser.add_argument("--dec_layers", default=2)
parser.add_argument("--max_length", default=20)
parser.add_argument("--max_words_length", default=25)
parser.add_argument("--cls_num", default=20)
parser.add_argument("--pre_norm", default=True)
args = parser.parse_args()
vqa_model = Model(args).to("cuda")
# 需要说明是否模型测试
vqa_model.eval()
# 加载模型
d3vg_weights_file="/media/light/light_t2/PROJECTS/model_checkpoints/checkpoints/pytorch_model_19_loss_0.016065.pt"
train_vqa_para = torch.load(
    d3vg_weights_file)
vqa_model.load_state_dict(train_vqa_para)


def uv2xyz(camera_inter, scalingFactor, depth, uv):
    fx, fy, centerX, centerY = camera_inter
    # -----------------------------------------------------------------
    # 像素坐标u，v
    # -----------------------------------------------------------------
    u, v = uv
    # -----------------------------------------------------------------
    # 相机坐标X，Y，Z
    # -----------------------------------------------------------------
    Z = depth.getpixel((u, v)) / scalingFactor
    if Z == 0:
        return False
    else:
        X = (u - centerX) * Z / fx
        Y = (v - centerY) * Z / fy
        return [X, Y, Z]


def get_seg_info(rgb_path, depth_path, camera_inter, scale_factor):
    """

    Args:
        rgb_dir: 待分割的image文件夹路径
        depth_path: 深度图文件夹路径
        camera_inter: 相机内参
        scale_factor: 缩放因子

    Returns:
        dict: dtype=dict key: 相机id value: 第一个列表保存的segs分类,第二个列表保存mask在相机坐标系下的坐标,相机坐标系不是数组坐标系，好像是像素坐标系，第三个列表放入seg的box坐标
    """
    # 保存全部rgb的分割结果
    dict = {}
    # ---------------------------------------------------------------------
    # 第一个[]放入seg类别，第二个[]放入seg的mask坐标,第三个[]放入seg的box坐标
    # ---------------------------------------------------------------------
    dict["info"] = [[], [], []]

    boxes_class, segs, boxes = generate_seg(rgb_path, "res.jpg")
    depth = Image.open(depth_path)
    for i, box_class in enumerate(boxes_class):
        # where返回的是numpy坐标，numpy坐标转化为像素坐标需要调换
        v, u = np.where(segs[i] == True)
        u = np.array((u), dtype=np.int32)
        v = np.array((v), dtype=np.int32)
        uv = np.concatenate([u.reshape([u.shape[0], 1]), v.reshape([v.shape[0], 1])], axis=1)
        dict["info"][0].append(box_class)
        dict["info"][2].append(boxes[i])
        xyz_list = []
        for uv_ in uv:
            # ---------------------------------------------------------------------
            # 第二步 把mask的uv坐标转换为相机坐标系的xyz坐标
            # ---------------------------------------------------------------------
            xyz = uv2xyz(camera_inter, scale_factor, depth, uv_)
            if xyz:
                xyz_list.append(xyz)
        dict["info"][1].append(xyz_list)
    return dict


def data_info_extractor(rgb_image_path, depth_image_path):
    # kinect 1280*720
    CAMERA_INTER = [609.0, 609.1, 633.0, 366.9]
    # release 1280*720
    # CAMERA_INTER = [645.7, 645.1, 648.9, 367.5]
    # scaling factor
    SCALE_FACTOR = 1000

    color_img = Image.open(rgb_image_path).convert('RGB')
    color_img.save("resized_image.jpg")

    dict_info = get_seg_info(rgb_image_path, depth_image_path, CAMERA_INTER, SCALE_FACTOR)
    regions, centers, d3boxes, bboxes = get_proposals(dict_info, color_img)
    return regions, centers, d3boxes, bboxes


def get_proposals(dict_info, image):
    boxes = dict_info["info"][2]
    seg_points = dict_info["info"][1]
    # ----------------------- #
    # 截取region
    # ----------------------- #
    region_lst = []
    center_lst = []
    d3boxes = []
    bboxes = []
    del_file("region/")
    for i, box in enumerate(boxes):
        region = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

        center, vertex_set = get_obb(seg_points[i], i)
        if center != None:
            region.save("region/{}.jpg".format(i))
            region_lst.append(region)
            center_lst.append(center)
            d3boxes.append(vertex_set)
            bboxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
    return region_lst, center_lst, d3boxes, bboxes


def predict(region_lst, center_lst, eval_question):
    # 提取regions特征
    regions_feature = dataload_for_eval(region_lst)
    test_patch = [regions_feature]
    # regions中心坐标
    test_d3_patches = [center_lst]
    # 问题
    test_question = [eval_question[0]]
    logits = vqa_model(test_patch, np.array(test_d3_patches) / np.max(abs(np.array(test_d3_patches))),
                       test_question)
    # 显示预测的物体
    # region_lst[int(torch.argmax(logits, dim=1).cpu().detach().numpy())].show()
    # img = cv.cvtColor(np.asarray(region_lst[int(torch.argmax(logits, dim=1).cpu().detach().numpy())]), cv.COLOR_RGB2BGR)

    # cv.namedWindow("output",cv.WINDOW_NORMAL)
    # cv.imshow("output", img)
    # if cv.waitKey(0)==27:
    #     cv.destroyAllWindows()
    return int(torch.argmax(logits, dim=1).cpu().detach().numpy())


while True:
    print("please input rgb image:")
    rgb_image_path = input()
    print("please input depth image:")
    depth_image_path = input()
    region_lst, center_lst, d3boxes, bboxes = data_info_extractor(
        rgb_image_path,
        depth_image_path)
    while True:
        print("please input description:")
        description = input()
        if not description:
            break
        start_time = time.time()
        box_index = predict(region_lst, center_lst, [description])
        end_time = time.time()
        print("3dvg time cost:%fs" % (end_time - start_time))
        box_xyxy = bboxes[box_index]
        image = cv.imread(rgb_image_path)
        image = cv.rectangle(image, box_xyxy[:2], box_xyxy[2:], color=(0, 255, 0), thickness=2)
        cv.namedWindow("output", cv.WINDOW_NORMAL)
        cv.imshow("output", image)
        if cv.waitKey(0) == 27:
            cv.destroyAllWindows()
        cv.imwrite("vg.jpg", image)
