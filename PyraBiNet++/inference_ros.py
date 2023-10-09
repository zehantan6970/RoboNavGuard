#! /usr/bin/env python3
import rospy
import onnxruntime
import cv2
import numpy as np
import torch
import sys
from sensor_msgs.msg import Image
import message_filters
from sensor_msgs.msg import CameraInfo
#print(' - sensor_msgs.__file__ = ',sensor_msgs.__file__)
# 为了保证从我的编译文件加载模型，这块要自己改代码
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/home/zzw/segmentation/devel/lib/python3/dist-packages')
from cv_bridge import CvBridge, CvBridgeError
import cv_bridge
#print(' - cv_bridge.__file__ = ',cv_bridge.__file__)
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
from sensor_msgs.msg import PointCloud2, PointField



def Resize(img, size=(1024, 512)):
    img = cv2.resize(img, size, dst=None, interpolation=cv2.INTER_LINEAR)
    return img

def Normalize(img, mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375], to_rgb=True):
    img = img.astype(np.float32)
    mean = np.float64(np.array(mean).reshape(1, -1))
    stdinv = 1/np.float64(np.array(std).reshape(1,-1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img

def show_result(img, depth, header, seg, palette, opacity=0.5):
    global pub
    palette = np.array(palette)
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg[..., ::-1]
    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    #img = cv2.resize(img, (768, 512))

    depth[np.where(seg == 0)] = 0
    seg_depth = bridge.cv2_to_imgmsg(depth, "16UC1")#
    seg_depth.header = header
    #seg_img = bridge.cv2_to_imgmsg(img, "bgr8")#
    pub.publish(seg_depth)#
    rospy.loginfo("发布分割图片")#

    #cv2.imshow('img1', depth)
    cv2.imshow('img2', img)
    cv2.waitKey(1)


def load_onnx():
    global palette, ort_session
    palette = [[0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64]]
    ort_session = onnxruntime.InferenceSession('/home/zzw/segmentation/src/my_seg/scripts/tmp.onnx')
    rospy.loginfo("载入模型")


def callback(color, depth):
    global bridge
    try:
        cv_color = bridge.imgmsg_to_cv2(color, "bgr8")
        cv_depth = bridge.imgmsg_to_cv2(depth, "16UC1")
    except CvBridgeError as e:
        print(e)
        return
    ori_shape = cv_color.shape
    header = depth.header
    input_img = Resize(cv_color)
    input_img = Normalize(input_img)
    input_img = torch.from_numpy(input_img.transpose(2, 0, 1))
    input_img = input_img.unsqueeze(0)
    ort_input = {'input': input_img.numpy()}
    ort_output = ort_session.run(['output'], ort_input)[0][0]
    result = cv2.resize(ort_output[0].astype(np.uint8), (ori_shape[1], ori_shape[0]))
    show_result(cv_color, cv_depth, header, result, palette)


def img_subscriber():
    img_topic = "/camera/color/image_raw"
    depth_topic = '/camera/aligned_depth_to_color/image_raw'#'/camera/depth/image_rect_raw'#'/camera/aligned_depth_to_color/image_raw'
    global bridge, pub
    bridge = CvBridge()
    pub = rospy.Publisher("seg_result", Image, queue_size=1)
    color = message_filters.Subscriber(img_topic, Image)
    depth = message_filters.Subscriber(depth_topic, Image)
    color_depth = message_filters.TimeSynchronizer([color, depth], 1)
    color_depth.registerCallback(callback)
    rospy.spin()



if __name__ == '__main__':
    rospy.init_node("object_segmentation")
    load_onnx()
    img_subscriber()

    
