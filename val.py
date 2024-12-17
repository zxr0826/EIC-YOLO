import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt') 
    model.val(data='/dataset/data.yaml',
              split='test', 
              imgsz=640,
              batch=16,
              project='runs/val',
              name='exp',
              )