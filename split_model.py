from distutils.log import error
import torch
import torch.nn as nn
import copy
from models.experimental import attempt_load
from utils.torch_utils import select_device
from torchinfo import summary
import models.yolo as yolo
from models.yolo import Detect, IDetect, IAuxDetect, IBin
import sys
import numpy as np
from PIL import Image
import thop
from utils.torch_utils import time_synchronized

class SplitModel(yolo.Model):
    def __init__(self, cfg='cfg/training/yolov7.yaml', ch=3, nc=None, anchors=None, side = "edge", start_layer = 0, layers = 105, end_layer = 105 ) -> None:
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.layers = layers
        self.side = side
        super(SplitModel, self).__init__(cfg, ch, nc, anchors)

    def forward(self, x, augment=False, profile=False, inference = False, st = 0, en = 105, y_new = None):
        augment = False #lock these out for now, not prepared for split network 
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile, inference = inference, st = st, en = en, y_new = y_new)  # single-scale inference, train

    def forward_once(self, x, profile=False, st = 0, en = 105, inference = False, y_new = None):
        y, dt = [], []  # outputs
        if y_new:
            y = y_new
        enter_val = st
        exit_val = en
        counter = 0
        for m in self.model:
            if inference and exit_val < counter:
                break # exit early
            if not inference or enter_val <= counter:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if not hasattr(self, 'traced'):
                    self.traced=False

                if self.traced:
                    if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect):
                        break

                if profile:
                    c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                    o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
                x = m(x)  # run
                
                y.append(x if m.i in self.save else None)  # save output
            counter += 1

        if profile:
            print('%.1fms total' % sum(dt))
        if inference:
            return x, y # y must be available for reference if split, but we do not want to interfere with the source of yolov7
        else:
            return x



weights = 'yolov7.pt'
device = select_device('0')
edge_model = SplitModel(end_layer = 50)
cloud_model = SplitModel(start_layer = 51)
edge_model.to(device)
cloud_model.to(device)
edge_model.eval()
cloud_model.eval()
for i in range(10):
    edge_model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(edge_model.parameters())))  # run warmup blind
# x, y = edge_model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(edge_model.parameters())), inference = True)  # now profile
# print(time_synchronized() - t)
# torch.save(y, "yolov7_y_rand.pt")
# print(torch.tensor(y, dtype=torch.float64, device=device))
# y_mem = torch.load("yolov7_y_rand.pt")
d = 0
# summary(edge_model, input_size=(1, 3, 640, 640), depth = d, inference = True)
# summary(edge_model, input_size=(1, 3, 640, 640), depth = d, en = 50, inference = True)
# summary(cloud_model, input_size=(1, 1024, 20, 20), depth = d, st = 51, inference = True, y_new = y)
img = '../coco/images/test2017/000000000001.jpg'
t = time_synchronized()
input = torch.zeros(1, 3, 640, 640).to(device).type_as(next(edge_model.parameters()))
# input = torch.from_numpy(img).to(device)
t2 = time_synchronized()
for i in range(100):
    x2, _ = cloud_model(input, inference = True)
t3 = time_synchronized()
print(f"{(t3-t2)/100:.04f}")
# for i in range(106):
#     ts = time_synchronized()
#     x, y = edge_model(input, en = i, inference = True)
#     t3 = time_synchronized()
#     x2, _ = cloud_model(x, y_new = y, st = i + 1, inference = True)
#     t4 = time_synchronized()
#     print(f"{i}\t{sys.getsizeof(x)}\t{sys.getsizeof(y)}\t{t3-ts:.04f}\t{t4-t3:.04f}")