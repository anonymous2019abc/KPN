import numpy as np
import cv2
# c++ version pse based on opencv 3+
# from pse import decode as pse_decode
# from util.config import config as cfg
import torch
import time
from shapely.geometry import Polygon
import pyclipper

class TextDetector(object):

    def __init__(self, model, cfg):
        # evaluation mode
        self.model = model
        model.eval()
        self.cfg=cfg
        # parameter
        self.scale = self.cfg.scale
        self.IN_SIZE = self.cfg.test_size


    #@staticmethod
    def get_contours(self, preds_KPN, kernel_postions=None, floodfill=False):
        #https://blog.csdn.net/qq_37385726/article/details/82313004
        #https://blog.csdn.net/weixin_42296411/article/details/80966724
        t0 = time.time()
        contours_list = []
        bbox_list = []
        if preds_KPN is None:
            return contours_list, bbox_list, 0
        #preds_KPN_bin = torch.where(preds_KPN > thresh, torch.full_like(preds_KPN, 1), torch.full_like(preds_KPN, 0))
        preds_KPN_bin = np.where(preds_KPN > self.cfg.score_final_th, 1, 0).astype(np.uint8)
        # preds_KPN_bin = preds_KPN_bin[0].detach().cpu().numpy() #batch 0
        # kernel_postions = kernel_postions[0].detach().cpu().numpy() #batch 0
        for n in range(preds_KPN_bin.shape[0]):
            repeat_flag = False
            for dc in contours_list:
                if cv2.pointPolygonTest(dc // [self.scale, self.scale], (kernel_postions[0][n], kernel_postions[1][n]), False) >= 0:
                    print("repeat_flag")
                    repeat_flag = True
                    break
            if repeat_flag:
                continue
            
            if floodfill:
                if preds_KPN_bin[n][kernel_postions[1][n], kernel_postions[0][n]] == 0:
                    continue
                mask = np.zeros((preds_KPN_bin[n].shape[0]+2, preds_KPN_bin[n].shape[1]+2), dtype=np.uint8)
                seed_p = (kernel_postions[0][n], kernel_postions[1][n])
                cv2.floodFill(preds_KPN_bin[n], mask, seedPoint=seed_p, newVal=1, flags=cv2.FLOODFILL_MASK_ONLY | 8)
                mask = mask[1:-1, 1:-1]
            else:
                mask = preds_KPN_bin[n]


            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if len(contours):
                max_contour = None
                max_area = 0
                for i in range(len(contours)):
                    if cv2.pointPolygonTest(contours[i], (kernel_postions[0][n], kernel_postions[1][n]), False) >= 0:
                        local_area = cv2.contourArea(contours[i])
                        if local_area > max_area:
                            max_contour = contours[i]
                            max_area = local_area
                if max_contour is None:
                    continue


                #max_contour = contours[0]
                #max_area = cv2.contourArea(max_contour)
                #for i in range(1, len(contours)):
                    #if cv2.contourArea(contours[i]) > max_area:
                        #max_contour = contours[i]
                        #max_area = cv2.contourArea(max_contour)
                #TODO filter min area

                # epsilon = 0.01 * cv2.arcLength(max_contour, True)
                # approx = cv2.approxPolyDP(max_contour, epsilon, True)
                # mask_contours.append(approx)
                # print(points)
                # print(points*[scale, scale])
                # 轮廓近似，epsilon数值越小，越近似
                epsilon = 0.007 * cv2.arcLength(max_contour, True)
                approx = cv2.approxPolyDP(max_contour, epsilon, True)
                box = approx.reshape((-1, 2)) * [self.scale, self.scale]
                if box.shape[0] < 4:
                    continue
                unclip_ratio = 0.15
                poly = Polygon(box)
                distance = poly.area * unclip_ratio / poly.length
                offset = pyclipper.PyclipperOffset()
                offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                expanded = np.array(offset.Execute(distance))
                if expanded.shape[0]>1:
                    continue
                contours_list.append(expanded[0])


                rect = cv2.minAreaRect(max_contour)
                #rw, rh = rect[1][0], rect[1][1]
                #if min(rw, rh) < 10//self.scale:
                    #continue
                points = cv2.boxPoints(rect)
                points = np.int0(points)
                bbox_list.append(points * [self.scale, self.scale])

        post_time = time.time() - t0
        return contours_list, bbox_list, post_time

    def get_contours_full(self, preds_KPN, kernel_postions=None, is_with_postion=False):
        # https://blog.csdn.net/qq_37385726/article/details/82313004
        # https://blog.csdn.net/weixin_42296411/article/details/80966724
        t0 = time.time()
        contours_list = []
        bbox_list = []
        if preds_KPN is None:
            return contours_list, bbox_list, 0
        # preds_KPN_bin = torch.where(preds_KPN > thresh, torch.full_like(preds_KPN, 1), torch.full_like(preds_KPN, 0))
        preds_KPN_bin = np.where(preds_KPN > self.cfg.score_final_th, 1, 0)
        # preds_KPN_bin = preds_KPN_bin[0].detach().cpu().numpy() #batch 0
        # kernel_postions = kernel_postions[0].detach().cpu().numpy() #batch 0
        for n in range(preds_KPN_bin.shape[0]):
            if is_with_postion and kernel_postions:
                mask = np.zeros((preds_KPN_bin[n].shape[0] + 2, preds_KPN_bin[n].shape[1] + 2))
                seed_p = (kernel_postions[n][0], kernel_postions[n][1])
                cv2.floodFill(preds_KPN_bin[n], mask, seedPoint=seed_p, newVal=1, flags=cv2.FLOODFILL_MASK_ONLY | 8)
                mask = mask[1:-1, 1:-1]
            else:
                mask = preds_KPN_bin[n]

            contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours):
                # max_contour = contours[0]
                # max_area = cv2.contourArea(max_contour)
                # for i in range(1, len(contours)):
                #     if cv2.contourArea(contours[i]) > max_area:
                #         max_contour = contours[i]
                #         max_area = cv2.contourArea(max_contour)
                # TODO filter min area

                # epsilon = 0.01 * cv2.arcLength(max_contour, True)
                # approx = cv2.approxPolyDP(max_contour, epsilon, True)
                # mask_contours.append(approx)
                for max_contour in contours:
                    rect = cv2.minAreaRect(max_contour)
                    points = cv2.boxPoints(rect)
                    points = np.int0(points)
                    # print(points)
                    # print(points*[scale, scale])
                    # 轮廓近似，epsilon数值越小，越近似
                    epsilon = 0.007 * cv2.arcLength(max_contour, True)
                    approx = cv2.approxPolyDP(max_contour, epsilon, True)
                    if len(approx.reshape((-1, 2))) < 3:
                        continue
                    contours_list.append(approx.reshape((-1, 2)) * [self.scale, self.scale])
                    bbox_list.append(points * [self.scale, self.scale])

        post_time = time.time() - t0
        return contours_list, bbox_list, post_time

    def detect(self, image, img_show, vis=False):
        # get model output
        b, c, h, w = image.shape
        if self.cfg.cuda:
            img = torch.ones((b, c, self.IN_SIZE[1], self.IN_SIZE[1]), dtype=torch.float32).cuda()
        else:
            img = torch.ones((b, c, self.IN_SIZE[1], self.IN_SIZE[1]), dtype=torch.float32)
        img[:,:, :h, :w]= image[:, :, :, :]
        t0 = time.time()
        if self.cfg.exp_name != "Icdar2015":
            pred_cls, pred_centergauss, pred_KPN_cls, list_pred_xys, backbone_time, IM_time = self.model.forward(img, istrain=False)
        else:
            pred_cls, pred_centergauss, pred_KPN_cls, list_pred_xys, backbone_time, IM_time = self.model.forward(image, istrain=False)
        # preds = torch.sigmoid(preds[0, :, :h//self.scale, :w//self.scale])
        # preds = torch.sigmoid(preds)
        t1 = time.time()

        pred_centergauss = torch.sigmoid(pred_centergauss).detach().cpu().numpy()[0][0]
        # print("pred_xys",pred_xys,pred_xys.shape)

        if pred_KPN_cls is not None:
            if self.cfg.cls_branch:
                pred_cls = torch.sigmoid(pred_cls).detach().cpu().numpy()[0]
                preds = (torch.sigmoid(pred_KPN_cls) * torch.sigmoid(pred_cls)).detach().cpu().numpy()[0]
            else:
                preds = (torch.sigmoid(pred_KPN_cls)).detach().cpu().numpy()[0]

            # pred_mask, pred_mask_idx = torch.sigmoid(pred_KPN_cls)[0].max(dim=0)
            pred_mask = preds.max(axis=0)
            pred_xys = list_pred_xys[0].detach().cpu().numpy()
        else:
            preds = None
            pred_mask = None
            pred_xys = None
            # pred_mask = np.zeros_like(pred_centergauss)
        t2 = time.time()
        contours_list, bbox_list, post_time = self.get_contours(preds, kernel_postions=pred_xys)
        t3 = time.time()
        print("get contours time", (t3-t2)*1000, " ms")

        print("len(contours_list)", len(contours_list))
        if vis:
            print("time_forward",t1-t0)

            # compute time
            cv2.namedWindow("image", 0)
            cv2.imshow("image", img_show)

            if self.cfg.cls_branch:
                cv2.namedWindow("pred_cls", 0)
                cv2.imshow("pred_cls", (pred_cls[0]*255).astype(np.uint8))
            cv2.namedWindow("pred_centergauss", 0)
            cv2.imshow("pred_centergauss", (pred_centergauss*255).astype(np.uint8))
            if preds is not None:
                print("preds", preds.shape)
                for i in range(preds.shape[0]):
                    cv2.namedWindow(str(i), 0)
                    cv2.imshow(str(i), (preds[i]*255).astype(np.uint8))
                cv2.waitKey(0)

        net_time = t1 - t0
        detach_time = t2 - t1
        post_time = t3 - t2
        # preds, boxes, contours, post_time = pse_decode(preds, self.scale, self.threshold)
        output = {
            'image': image,
            'tr': pred_centergauss,
            'pred_mask': pred_mask,
            'bbox': bbox_list,
             "backbone_time": backbone_time,
             "IM_time": IM_time,
             "detach_time": detach_time,
             "net_time": net_time,
             "post_time": post_time
        }
        return contours_list, output






















