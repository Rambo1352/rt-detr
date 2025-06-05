import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/yaban/weights/best.pt')
    model.val(data=r'E:\soft\python_resource\DETR\RT-DETR\datasets\yaban\data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='yaban',
              )