import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import TotalText, Ctw1500Text, Icdar15Text
from network.KPN import KPN_Net
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from util.visualize import visualize_detection, visualize_gt
from util.misc import to_device, mkdirs, rescale_result
from util.detection import TextDetector

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')

def data_transfer_ICDAR(contours):
    cnts = list()
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        if min(rect[1][0], rect[1][1]) <=8 :
            continue
        #rect=(rect[0],(rect[1][0]-3, rect[1][1]-3), rect[2])
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        # print(points.shape)
        # points = np.reshape(points, (4, 2))
        cnts.append(points)
    return cnts

def inference(detector, test_loader, output_dir_ori):
    output_dir = output_dir_ori + "_txt"

    total_time = 0.
    post_all_time =0.
    net_all_time = 0.
    backbone_all_time = 0.
    IM_all_time = 0.
    detach_all_time =0.
    if cfg.exp_name != "MLT2017":
        osmkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)

    # x = torch.ones(50, 1024*2, 1024*2).cuda().detach()
    # t1 = time.time()
    # b = x.cpu().cuda().cpu().cuda().cpu().cuda().cpu().cuda().cpu().cuda()
    # t2 = time.time()
    # print("first gpu2cpu time",t2 - t1)

    for i, (image, meta) in enumerate(test_loader):

        t_eval111 = time.time()
        x = to_device(torch.ones(1, 1, 1)).detach()
        x = x.detach().cpu().numpy().astype(np.uint8)
        print("-debug GPU2CPU time", time.time() - t_eval111)

        image= to_device(image)
        if cfg.cuda:
            torch.cuda.synchronize()
        idx = 0  # test mode can only run with batch_size == 1

        # visualization
        print("image",image.shape)
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        print("img_show",img_show.shape)
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        # get detection result
        time_start = time.time()
        contours, output = detector.detect(image, img_show, vis=cfg.eval_vis)
        time_end = time.time()
        if i > 0:
            total_time += time_end - time_start
            # total_time += (output["net_time"] + output["post_time"])
            post_all_time += output["post_time"]
            net_all_time += output["net_time"]
            backbone_all_time+= output["backbone_time"]
            IM_all_time += output["IM_time"]
            detach_all_time += output["detach_time"]
            # fps = (i + 1) / total_time
            # print('average time {} / {} images: {}. ({:.2f} fps); backbone-time:{:.2f}, IM-time:{:.2f}, post-time:{:0.2f}, Transfer-time:{:.2f}'.format(i + 1, len(test_loader), meta['image_id'][idx], fps, backbone_all_time*1000/(i+1), IM_all_time*1000/(i+1), post_all_time*1000/(i+1), detach_all_time*1000/(i+1)))
            fps = (i) / total_time
            print('average time {} / {} images: {}. ({:.2f} fps); backbone-time:{:.2f}, IM-time:{:.2f}, post-time:{:0.2f}, Transfer-time:{:.2f}'
                  .format(i + 1, len(test_loader), meta['image_id'][idx], fps, backbone_all_time*1000/(i), IM_all_time*1000/(i),
                          post_all_time*1000/(i), detach_all_time*1000/(i)))
            print('-single time {} / {} images: {}. ({:.2f} fps); backbone-time:{:.2f}, IM-time:{:.2f}, post-time:{:0.2f}, Transfer-time:{:.2f}'
                  .format(i + 1, len(test_loader), meta['image_id'][idx], fps, output["backbone_time"]*1000, output["IM_time"]*1000,
                          output["post_time"]*1000, output["detach_time"]*1000))


        if cfg.exp_name == "Icdar2015" or cfg.exp_name == "MLT2017" or cfg.exp_name == "TD500":
            pred_vis = visualize_detection(img_show, output['bbox'], output['tr'], output['pred_mask'])
        else:
            pred_vis = visualize_detection(img_show, contours, output['tr'], output['pred_mask'])
        if cfg.eval_vis:
            cv2.namedWindow("pred_vis",0)
            cv2.imshow("pred_vis",pred_vis)
            cv2.waitKey(0)

        # continue

        im_vis = pred_vis

        path = os.path.join(cfg.vis_dir, '{}_img'.format(cfg.exp_name), meta['image_id'][idx].split(".")[0]+".jpg")
        if cfg.store_img:
            cv2.imwrite(path, im_vis)

        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        # cv2.namedWindow("image_show_debug1", 0)
        # cv2.imshow("image_show_debug1",img_show)
        img_show, contours = rescale_result(img_show, contours, H, W)

        # cv2.drawContours(img_show, contours, -1, (0, 255, 0), 3)
        # cv2.namedWindow("image_show_debug", 0)
        # cv2.imshow("image_show_debug",img_show)
        # cv2.waitKey(0)

        # write to file
        if cfg.exp_name == "Icdar2015":
            fname = "res_" + meta['image_id'][idx].replace('jpg', 'txt')
            contours = data_transfer_ICDAR(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "MLT2017":
            out_dir = os.path.join(output_dir, str(cfg.checkepoch))
            if not os.path.exists(out_dir):
                mkdirs(out_dir)
            fname = meta['image_id'][idx].split("/")[-1].replace('ts', 'res')
            fname = fname.split(".")[0] + ".txt"
            data_transfer_MLT2017(contours, os.path.join(out_dir, fname))
        elif cfg.exp_name == "TD500":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            data_transfer_TD500(contours, os.path.join(output_dir, fname))

        else:
            fname = meta['image_id'][idx].replace('jpg', 'txt')
            write_to_file(contours, os.path.join(output_dir, fname))


def main(vis_dir_path):

    osmkdir(vis_dir_path)
    if cfg.exp_name == "Totaltext":
        path_ = "data/total-text-mat"
        if cfg.debug_data:
            path_ = 'data/total-text-mat_debug'
        testset = TotalText(
            data_root=cfg.data_model_path+path_,
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )

    elif cfg.exp_name == "Ctw1500":
        testset = Ctw1500Text(
            data_root=cfg.data_model_path+'data/ctw1500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "Icdar2015":
        testset = Icdar15Text(
            data_root=cfg.data_model_path+'data/Icdar2015',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "MLT2017":
        testset = Mlt2017Text(
            data_root=cfg.data_model_path+'data/MLT2017',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "TD500":
        testset = TD500Text(
            data_root=cfg.data_model_path+'data/TD500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    else:
        print("{} is not justify".format(cfg.exp_name))

    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers[0])

    # Model
    model = KPN_Net(backbone=cfg.net, is_training=False, cfg=cfg)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'TextGraph_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    model.load_model(model_path)

    # copy to cuda
    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    detector = TextDetector(model, cfg=cfg)

    print('Start testing TextGraph.')
    output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    inference(detector, test_loader, output_dir)




if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)
    cfg.debug_data = False


    vis_dir = os.path.join(cfg.vis_dir, '{}_img'.format(cfg.exp_name))

    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main(vis_dir)
