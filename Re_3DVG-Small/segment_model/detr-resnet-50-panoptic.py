import io
import requests
from PIL import Image
import torch
import numpy
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id
import time
import cv2 as cv
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
palette = itertools.cycle(sns.color_palette())
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("/media/light/light_t2/t2/DATA/scannet_frames_25k/scene0000_00/color/001700.jpg")

feature_extractor = DetrFeatureExtractor.from_pretrained("/media/light/light_t2/checkpoints/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("/media/light/light_t2/checkpoints/detr-resnet-50-panoptic").to("cuda")
t1=time.time()
# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt").to("cuda")

# forward pass
outputs = model(**inputs)

# use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

# # the segmentation is stored in a special-format png
# panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
# panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
# # retrieve the ids corresponding to each mask
# panoptic_seg_id = rgb_to_id(panoptic_seg)
# # Finally we color each mask individually
# panoptic_seg[:, :, :] = 0
# for id in range(panoptic_seg_id.max() + 1):
#   panoptic_seg[panoptic_seg_id == id] = numpy.asarray(next(palette)) * 255
#   coordinates=numpy.column_stack(numpy.where(panoptic_seg_id == id))
#   v_max,u_max=numpy.max(coordinates,axis=0)[0],numpy.max(coordinates,axis=0)[1]
#   v_min,u_min=numpy.min(coordinates,axis=0)[0],numpy.min(coordinates,axis=0)[1]
#   image_croped=image.crop((u_min,v_min,u_max,v_max))
#   image_croped.save("/media/light/light_t2/PROJECTS/D3VG_large/segment_model/crop_data/{}.jpg".format(id))
# t2=time.time()
# print("消耗时间%fs"%(t2-t1))
# plt.figure(figsize=(15,15))
# plt.imshow(panoptic_seg)
# plt.axis('off')
# plt.show()
from copy import deepcopy

# We extract the segments info and the panoptic result from DETR's prediction
segments_info = deepcopy(result["segments_info"])
# Panoptic predictions are stored in a special format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
final_w, final_h = panoptic_seg.size
# We convert the png into an segment id map
panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
panoptic_seg = torch.from_numpy(rgb_to_id(panoptic_seg))

# Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
for i in range(len(segments_info)):
  c = segments_info[i]["category_id"]
  segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else \
  meta.stuff_dataset_id_to_contiguous_id[c]
print(segments_info)
# Finally we visualize the prediction
v = Visualizer(numpy.array(image.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
v._default_font_size = 20
v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
cv.imshow("result",v.get_image())
cv.waitKey(0)
