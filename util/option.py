import argparse
import torch
import os
import torch.backends.cudnn as cudnn

from datetime import datetime


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str


class BaseOptions(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--exp_name', default="Totaltext", type=str,
                                 choices=['Synthtext', 'Totaltext', 'Ctw1500',
                                          'Icdar2015'], help='Experiment name')
        self.parser.add_argument('--net', default='resnet50', type=str,
                                 choices=['vgg', 'vgg_bn', 'resnet50', 'resnet101'],
                                 help='Network architecture')
        self.parser.add_argument('--mgpu', action='store_true', help='Use multi-gpu to train model')
        # self.parser.add_argument('--mgpu', default=True, type=str2bool, help='Use multi-gpu to train model')

        self.parser.add_argument('--debug_data', default=False, type=str2bool, help='Use debug train data')
        self.parser.add_argument('--debug_dataloder', default=False, type=str2bool, help='Use debug train data')
        self.parser.add_argument("--gpu", default="3", help="set gpu id", type=str)
        self.parser.add_argument('--num_workers', default=[4, 2, 1], type=int, nargs='+', help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--batch_size', default=[4, 2, 1], type=int, nargs='+', help='Batch size for training')
        self.parser.add_argument('--input_size', default=[640, 832, 1024], type=int, nargs='+', help='model input size')
        self.parser.add_argument('--max_epoch', default=1000, type=int, help='Max epochs')
        self.parser.add_argument('--save_freq', default=10, type=int, help='save weights every # epoch')

        self.parser.add_argument('--scale', default=2,type=int, help='prediction on 1/scale feature map')
        self.parser.add_argument('--cls_branch', default=False, type=str2bool, help='Use full cls_branch')
        self.parser.add_argument('--add_focal_loss', default=True, type=str2bool, help='Use full cls_branch')

        self.parser.add_argument('--score_kernel_th', default=0.175,type=float, help='threshold for kernel proposal')
        self.parser.add_argument('--score_final_th', default=0.4,type=float, help='threshold for final mask')
        self.parser.add_argument('--test_size', default=[720, 1024], type=int, nargs='+', help='model input size')
        self.parser.add_argument('--eval_vis', default=False, type=str2bool, help='imshow in eval')
        self.parser.add_argument('--store_img', default=True, type=str2bool, help='store result img')
        self.parser.add_argument('--checkepoch', default=300, type=int, help='Load checkpoint number 850, 860, 870')

        # self.parser.add_argument('--debug_synth', default=False, type=str2bool, help='train_gt_kernels')
        self.parser.add_argument('--train_gt_kernels', default=True, type=str2bool, help='train_gt_kernels')
        self.parser.add_argument('--eval_topk', default=1,type=int, help='threshold for final mask')
        #####################KPN paramers###################
        #in config.py

        #####################fixed paramers###################
        # basic opts
        self.parser.add_argument('--resume', default=None, type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--save_dir', default='./model/', help='Path to save checkpoint models')
        self.parser.add_argument('--vis_dir', default='./vis/', help='Path to save visualization images')
        self.parser.add_argument('--log_dir', default='./logs/', help='Path to tensorboard log')
        self.parser.add_argument('--output_dir', default='./vis/', help='Path to output')
        self.parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='Training Loss')
        # self.parser.add_argument('--input_channel', default=1, type=int, help='number of input channels' )
        self.parser.add_argument('--pretrain', default=False, type=str2bool, help='Pretrained AutoEncoder model')
        self.parser.add_argument('--verbose', '-v', default=True, type=str2bool, help='Whether to output debug info')
        self.parser.add_argument('--viz', action='store_true', help='Whether to output debug info')



        # # train opts
        # self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
        # self.parser.add_argument('--lr_adjust', default='fix',
        #                          choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        # self.parser.add_argument('--stepvalues', default=[], nargs='+', type=int, help='# of iter to change lr')
        # self.parser.add_argument('--weight_decay', '--wd', default=0., type=float, help='Weight decay for SGD')
        # self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        # self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        # self.parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        # self.parser.add_argument('--display_freq', default=10, type=int, help='display training metrics every # iter')
        # self.parser.add_argument('--viz_freq', default=150, type=int, help='visualize training process every # iter')
        # self.parser.add_argument('--log_freq', default=10000, type=int, help='log to tensorboard every # iterations')
        # self.parser.add_argument('--val_freq', default=1000, type=int, help='do validation every # iterations')

        # data args
        self.parser.add_argument('--rescale', type=float, default=255.0, help='rescale factor')
        self.parser.add_argument('--means', type=int, default=(0.485, 0.456, 0.406), nargs='+', help='mean')
        self.parser.add_argument('--stds', type=int, default=(0.229, 0.224, 0.225), nargs='+', help='std')


        # self.parser.add_argument('--max_annotation', default=200, type=int, help='max polygon per image')
        # self.parser.add_argument('--max_points', default=20, type=int, help='max point per polygon')
        # self.parser.add_argument('--use_hard', default=True, type=str2bool, help='use hard examples (annotated as #)')


        # eval args
        # self.parser.add_argument('--start_epoch', default=0,type=int, help='start epoch number')
        #
        #
        # # demo args
        # self.parser.add_argument('--img_root', default=None, type=str, help='Path to deploy images')

    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu

        # Setting default torch Tensor type
        if self.args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Create weights saving directory
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_dir, self.args.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

    def update(self, args, extra_options):

        for k, v in extra_options.items():
            setattr(args, k, v)
