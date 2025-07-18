import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('./ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=r'train_yaban.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                workers=0,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='yaban_resnet18_',
                # amp=True
                )