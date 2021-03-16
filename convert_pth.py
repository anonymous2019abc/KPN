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

import sys

def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    if cfg.cuda:
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    return state_dict
    
def save_model(model, epoch, lr, optimzer, iter=None):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
    backbone_name = model.backbone_name if not cfg.mgpu else model.module.backbone_name
    if type(backbone_name) != type("123"):
        backbone_name = backbone_name[0]
    if iter is not None:
        save_path = os.path.join(save_dir, 'TextGraph_{}_{}_{}.pth'.format(backbone_name, epoch, iter))
    else:
        save_path = os.path.join(save_dir, 'TextGraph_{}_{}.pth'.format(backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict() if not cfg.mgpu else model.module.state_dict(),
        'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)
    
if __name__ == "__main__":
    # parse arguments
    #model_path = sys.argv[1]
    #print(model_path)
    
    option = BaseOptions()
    option.parser.add_argument('--convert_model_path', default="none", type=str, help='Experiment name')
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)
    cfg.debug_data = False
    
    
    model = KPN_Net(backbone=cfg.net, is_training=False, cfg=cfg)
    #model.load_model(cfg.convert_model_path)
    state_dict = load_model(model, cfg.convert_model_path)

    # copy to cuda
    #model = model.to(cfg.device)
    #if cfg.cuda:
        #cudnn.benchmark = True
    print("model.state_dict()", state_dict['lr'])
    
    torch.save(state_dict['model'], cfg.convert_model_path+".cvt")#, _use_new_zipfile_serialization=False)
