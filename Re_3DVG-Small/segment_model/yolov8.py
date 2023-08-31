from ultralytics import YOLO

# Load a model
model = YOLO('/media/light/light_t2/PROJECTS/model_checkpoints/yolov8/yolov8x-seg.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['/media/light/light_t2/t2/DATA/scannet_frames_25k/scene0000_00/color/001700.jpg'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Class probabilities for classification outputs
    print(len(masks))