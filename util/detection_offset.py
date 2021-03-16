import numpy as np
import cv2
# c++ version pse based on opencv 3+
# from pse import decode as pse_decode
# from util.config import config as cfg
import torch
import time

class TextDetector(object):

    def __init__(self, model, cfg):
        # evaluation mode
        self.model = model
        model.eval()
        self.cfg=cfg
        # parameter
        self.scale = self.cfg.scale
        self.IN_SIZE = self.cfg.test_size


    def detect(self, image, img_show):
        # get model output
        b, c, h, w = image.shape
        if self.cfg.cuda:
            img = torch.ones((b, c, self.IN_SIZE[1], self.IN_SIZE[1]), dtype=torch.float32).cuda()
        else:
            img = torch.ones((b, c, self.IN_SIZE[1], self.IN_SIZE[1]), dtype=torch.float32)
        img[:,:, :h, :w]= image[:, :, :, :]
        tt = time.time()
        if self.cfg.exp_name != "Icdar2015":
            preds, pred_cls, pred_distance_norm, backbone_time, IM_time = self.model.forward(img, istrain=False)
        else:
            preds, pred_cls, pred_distance_norm, backbone_time, IM_time = self.model.forward(image, istrain=False)
        # preds = torch.sigmoid(preds[0, :, :h//self.scale, :w//self.scale])
        # preds = torch.sigmoid(preds)
        pred_cls = torch.sigmoid(pred_cls)
        print("pred_distance_norm", pred_distance_norm.shape)
        cv2.namedWindow("pred_cls", 0)
        cv2.imshow("pred_cls", (pred_cls[0,0].detach().cpu().numpy()*255).astype(np.uint8))
        cv2.namedWindow("pred_distance_norm", 0)
        cv2.imshow("pred_distance_norm", (pred_distance_norm[0].detach().cpu().numpy()*255).astype(np.uint8))
        t0 = time.time()
        preds = preds.detach().cpu().numpy()
        preds = preds[0]
        print("preds", preds.shape)
        for i in range(preds.shape[0]):
            cv2.namedWindow(str(i), 0)
            cv2.imshow(str(i), (preds[i]*255).astype(np.uint8))
        cv2.waitKey(0)

        return

        detach_time = time.time() - t0
        net_time = time.time() - tt
        # preds, boxes, contours, post_time = pse_decode(preds, self.scale, self.threshold)
        output = {
            'image': image,
            'tr': preds,
            #'bbox': boxes,
             "backbone_time": backbone_time,
             "IM_time": IM_time,
             "detach_time": detach_time
        }
        return contours, output, net_time, post_time






















