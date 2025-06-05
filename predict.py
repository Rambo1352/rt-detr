#
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#      yolo = YOLO("./runs/train/exp6/weights/best.pt", task="detect")
#      results = yolo(source="./tests/blct_img/blct_test03.png", save=True, conf=0.1)


import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    model = RTDETR('runs/train/yaban/weights/best.pt') # select your model.pt path
    model.predict(source='./datasets/yaban/valid/images',
                  conf=0.5,
                  project='runs/detect',
                  name='yaban',
                  save=True,
                  # visualize=True,  # visualize model features maps
                  line_width=2,  # line width of the bounding boxes
                  # show_conf=False,  # do not show prediction confidence
                  # show_labels=False,  # do not show prediction labels
                  save_txt=True,  # save results as .txt file
                  save_crop=True,  # save cropped images with results
                  )