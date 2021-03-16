from easydict import EasyDict
import torch
import os

config = EasyDict()

##################################
# xy_embedding
config.xy_embedding = []
config.torch_xy_embedding = {}

# path for data and pretrained models
config.data_model_path = "../../data_model/"

# min area size, don't care
config.min_area_DNC = 100
config.DNC_label = 255

# resize embedding, regression, pseudo_anchor_scale
config.pseudo_anchor_scale_score = 32
config.pseudo_anchor_scale = 512
#clip the offset [-0.999, 0.999] (1:config.pseudo_anchor_scale)
config.scale_clip_offset = False

# select TOPK kernel
config.TOPK_kernel = 50
config.TOPK_kernel_onetext = 5

# KPN debug
config.KPN_debug = False

#dataloder gt_kernel center point probability
config.center_probability = 0.33

# dataloder debug
config.dataloder_debug = False

#every k iter show the TOPK_kernel
config.iter_show_TOPK_kernel = 10
config.iter_show_TOPK_kernel_cnt = 0

#gauss radius
config.gauss_radius = -1

#dilate_KPN_gt_mask
config.dilate_KPN_gt_mask = True

#normal batch-wise KPN (in each batch, weight/=box_num)
# config.KPN_weight_batchwise = False


config.fuc_k = [1,4,7, 9]


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    # print(config.gpu)
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
