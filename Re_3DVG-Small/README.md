# Re_3DVG-Small
## 1. Environment
    torch==1.13.1+cuda==11.7 mmdet==2.25.2
    You need to port segment_model/mmdetection/mmdet to your built environment，because we added some necessary variables
## 2.Weights
    #### You can download the required weights from this link:https://drive.google.com/drive/folders/1plOlm15jFwyiE8LH_6qQPZ-cmIWE6LQ6?usp=drive_link
## 3.Configuration method
### You need to modify the following code
    1.demo_for_kinect.py     
        d3vg_weights_file -------------------------------> pytorch_model_19_loss_0.016065.pt
        CAMERA_INTER      -------------------------------> the camera inter parameter of your depth
        SCALE_FACTOR      -------------------------------> the scanle factor parameter of your depth
    2.mmdet_demo_2.py
        config_file       -------------------------------> mask_rcnn_r101_fpn_mstrain-poly_3x_coco.py
        checkpoint_file   -------------------------------> mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth
        you can choose other proper weights file
    3.dataloader_vit_small_patch16_224_in21k_with_vit_finetune.py
        vit_file          -------------------------------> pytorch_vit_19.pt
#### In order to be able to make the positioning more accurate, you can choose other proper weights file from [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v2.25.2)
## 4. Run demo
#### You can test your own data by running demo_for_kinect.py yourself,The size of your rgb data and depth data must be the same
## 5. comparative trial on reorganized scannet datas
### [scannet_v7 data link](https://drive.google.com/file/d/1hmSnEwCgXDu0gp5XcOqoMvuX91yjA7x5/view?usp=drive_link) This includes custom ScanNet frames 25k dataset

#### You need to point path of scannet to the scannet_v7 folder, and point the path of scannet_frames_test to the scannet_frames_test folder, before you run this code,please change the following code:
    1.mmdet_demo_2.py
        config_file       -------------------------------> mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco-trash_a.py
        checkpoint_file   -------------------------------> epoch_62.pth
    2.demo_comparative_iou.py
        scannet           -------------------------------> scannet_v7
        scannet_frames_test-------------------------------> scannet_frames_test
## 6. train datasets distribution
#### You can find train datasets and test datasets from this link: 
#### The data we trained included the following categories, each with a different number of categories
    'person': 20553, 'chair': 35868, 'dining table': 13364, 'umbrella': 7663, 'box': 945, 'cabinet': 5106, 'bed': 5499,
    'picture': 1810, 'refrigerator': 2926, 'oven': 3081, 'sink': 4623, 'floor mat': 281, 'sofa': 7178, 'counter': 964,
     'vase': 4556, 'television': 5639, 'table': 4251, 'handbag': 7323, 'potted plant': 7015, 'suitcase': 5181, 'bottle': 10959,
      'clock': 4279, 'backpack': 5845, 'door': 3036, 'laptop': 4420, 'plastic bag': 999, 'trash can': 1222, 'toilet': 4293,
      'pillow': 2576, 'paper': 391, 'towel': 466, 'curtain': 784, 'clothes': 682, 'shelves': 1285, 'lamp': 501, 'window': 1821,
      'blinds': 347, 'dresser': 480, 'desk': 2985, 'shower curtain': 271, 'whiteboard': 433, 'microwave': 1513, 'bookshelf': 863,
      'bag': 392, 'night stand': 397, 'bathtub': 316, 'mirror': 325, 'toaster': 195
