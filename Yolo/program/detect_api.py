import time
from pathlib import Path
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import MyLoadImages
from utils.general import check_img_size,  non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier

class simulation_opt:
    def __init__(self,weights,img_size = 640,conf_thres = 0.15,
                 iou_thres = 0.45,device='',view_img=False,
                 classes = None,agnostic_nms = False,
                 augment = False,update = False,exist_ok = False):
        self.weights = weights
        self.source = None
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.exist_ok = exist_ok

        

class detectapi:
    def __init__(self,weights,img_size=416,):
        self.opt = simulation_opt(weights=weights,img_size=img_size)
        weights, imgsz = self.opt.weights,self.opt.img_size

        #Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = False

        #Load model
        self.model = attempt_load(weights,map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz,s=self.stride)

        if self.half:
            self.model.half()

        #Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt',map_location=self.device)['model']).to(self.device).eval()
        
        #read names and colors
        self.names = self.model.module.names if hasattr(self.model,'module') else self.model.names if hasattr(self.model,'module') else self.model.names
        self.colors = [[random.randint(0,255) for _ in range(3)]for _ in self.names]

    def detect(self,source):
        if type(source) != list:
            raise TypeError('source must be a list which contain pictures read by cv2')
        dataset = MyLoadImages(source,img_size=self.imgsz,stride=self.stride)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        #t0 = time.time()
        result = []

        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            #t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            #t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)
                #print time(inference + NMS)
                #print(f'{s}Done. ({t2 - t1:.3f}s)')

            det = pred[0]
            im0 = im0s.copy()
            #s += '%gx%g' %img.shape[2:] #print string
            #gn = torch.tensor(im0.shape)[[1,0,1,0]] #normalization gain whwh
            result_txt=[]

            if len(det):
                det[:,:4] = scale_coords(img.shape[2:],det[:,:4],im0.shape).round()
                #Write result

                for *xyxy, conf, cls in reversed(det):
                    #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4))/gn).view(-1).tolist() 
                    line = (int(cls.item()),[int(_.item())for _ in xyxy],conf.item())
                    result_txt.append(line)
                    label = f'{self.names[int(cls)]}{conf:.2f}'
                    plot_one_box(xyxy,im0,label=label,color=self.colors[int(cls)],line_thickness=3)
            result.append((im0,result_txt))
        return result,self.names