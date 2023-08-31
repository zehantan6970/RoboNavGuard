from models.model_vit_en_small_patch16_224_in21k_with_finetune_BERT import Model
import torch
from dataload.dataloader_vit_small_patch16_224_in21k_with_vit_finetune import dataload,dataload_for_eval
from utils.utils import getRt,transform,get_obb,get_iou_obb,get_box3d_min_max,box3d_iou
import numpy as np
import argparse
from PIL import Image
from segment_model.mmdetection.mmdet_demo_2 import generate_seg
import os
import cv2 as cv
envpath = '/home/light/anaconda3/envs/openmmlab/lib/python3.7/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
# -------- #
# 配置
# -------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--hidden_dim',default=512)
parser.add_argument('--dropout',default=0.1)
parser.add_argument('--nheads',default=4)
parser.add_argument('--dim_feedforward',default=2048)
parser.add_argument("--enc_layers",default=2)
parser.add_argument("--dec_layers",default=2)
parser.add_argument("--max_length",default=20)
parser.add_argument("--max_words_length",default=25)
parser.add_argument("--cls_num",default=20)
parser.add_argument("--pre_norm",default=True)
args=parser.parse_args()
vqa_model = Model(args)
# 需要说明是否模型测试
vqa_model.eval()
# 加载模型
train_vqa_para = torch.load( "/media/light/light_t2/PROJECTS/model_checkpoints/d3vg_small_vit_wF_e4_L2_BERT_use_datasets_train&val&test_vit19/pytorch_model_19_loss_0.016065.pt")
vqa_model.load_state_dict(train_vqa_para)
total = sum([param.nelement() for param in vqa_model.parameters()])

scannet="/media/light/light_t2/DATA/D3VG/对比试验数据集/scannet_v7"
scannet_frames_test="/media/light/light_t2/t2/DATA/scannet_frames_test"
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
        if len(uv) > 900:
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


def data_info_extractor(scannet_frames_25k, scan_frame_id):
    CAMERA_INTER = [577.5, 578.7, 318.9, 242.7]
    # scaling factor
    SCALE_FACTOR = 1000
    scene_id = scan_frame_id[:4]
    frame_id = scan_frame_id[-6:]
    an_id = scan_frame_id[4:6]
    color_img = Image.open(os.path.join(scannet_frames_25k, "scene{}_{}".format(scene_id, an_id), "color",
                             "{}.jpg".format(frame_id))).resize((640, 480))
    color_img.save("resized_image.jpg")
    depth_img = os.path.join(scannet_frames_25k, "scene{}_{}".format(scene_id, an_id), "depth",
                             "{}.png".format(frame_id))
    dict_info = get_seg_info("resized_image.jpg", depth_img, CAMERA_INTER, SCALE_FACTOR)
    region_lst, center_lst, d3boxes = get_proposals(dict_info, "resized_image.jpg")
    return region_lst, center_lst, d3boxes


def get_proposals(dict_info, color_img):
    boxes = dict_info["info"][2]
    seg_points = dict_info["info"][1]
    # ----------------------- #
    # 截取region
    # ----------------------- #
    region_lst = []
    center_lst = []
    d3boxes = []
    image = Image.open(color_img)
    for i, box in enumerate(boxes):
        region = image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        center, vertex_set = get_obb(seg_points[i],i)
        if center != None:
            region_lst.append(region)
            center_lst.append(center)
            d3boxes.append(vertex_set)
    return region_lst, center_lst, d3boxes

import time
def cal_iou(region_lst,center_lst,d3boxes,eval_question,scene):
    # 提取regions特征
    regions_feature=dataload_for_eval(region_lst)
    test_patch = [regions_feature]
    # regions中心坐标
    test_d3_patches = [center_lst]
    # 问题
    test_question = [eval_question[0]]
    Id=eval_question[1]
    logits = vqa_model(test_patch, np.array(test_d3_patches) / np.max(abs(np.array(test_d3_patches))),
                       test_question)
    # 获取预测的box的坐标
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1=get_box3d_min_max(d3boxes[int(torch.argmax(logits, dim=1).cpu().detach().numpy())])
    # 真实的物体的box的坐标
    img = cv.cvtColor(np.asarray(region_lst[int(torch.argmax(logits, dim=1).cpu().detach().numpy())]), cv.COLOR_RGB2BGR)
    cv.namedWindow("output", cv.WINDOW_NORMAL)
    cv.imshow("output", img)
    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()
    arr = np.load("{}/scans/{}/npy/{}_bbox.npy".format(scannet,scene,scene))
    x_min_2,y_min_2,z_min_2,x_max_2,y_max_2,z_max_2,label_id,obj_id= arr[Id]
    # 计算iou
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)
    print("iou",iou)
    return iou


with open("{}/scanrefer_en.txt".format(scannet),mode="r") as r:
    lines=r.readlines()
    total_description=len(lines)
    s=""
    num_025=0
    num_050=0
    t1=time.time()
    for l in lines:

        scene=l.split("|")[0]
        question=l.split("|")[1]
        Id=int(l.split("|")[2])
        if scene!=s:
            # 提取分割模型分割后的参数
            print(question)
            region_lst, center_lst, d3boxes = data_info_extractor(scannet_frames_test,
                                                                  scene[5:9]+scene[10:12]+scene[14:])
            iou=cal_iou(region_lst, center_lst, d3boxes, [question, Id],scene)
            s=scene
            if iou>0.25:
                num_025+=1
            if iou>0.5:
                num_050+=1
        else:
            print(question)
            iou=cal_iou(region_lst, center_lst, d3boxes, [question, Id],scene)
            if iou>0.25:
                num_025+=1
            if iou > 0.5:
                num_050 += 1
    t2=time.time()
    print("消耗总时间为:%fs"%(t2-t1))
    print("top0.25:",num_025/total_description)
    print("top0.5:",num_050/total_description)
