import argparse
import os
import datetime


class Options(object):

    def __init__(self):
        super(Options, self).__init__()

    def initialize(self):

        parser = argparse.ArgumentParser()

        # basic settings
        parser.add_argument('--root', type=str, default='./DP-dataset', help='Root directory for datasets')
        parser.add_argument('--mode', type=str, default="train")
        parser.add_argument('--val', type=bool, default=True, help='Whether to use validation (default: True)')
        parser.add_argument('--audio', type=bool, default=False, help='Whether to use audio (default: False)')
        parser.add_argument('--center_loss', type=bool, default=True, help='Whether to use center loss (default: False)')
        parser.add_argument('--focal_loss', type=bool, default=False, help='Whether to use focal loss (default: False)')
        parser.add_argument('--AUs', type=bool, default=False, help='Whether to use Action Units (default: False)')
        parser.add_argument('--fusionmodel', type=str, default="Base")
        parser.add_argument('--dataset', type=str, default="Palsy330")
        parser.add_argument('--savename', type=str, default="")
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids, eg. 0,1,2; -1 for cpu.')
        parser.add_argument('--use_test', type=bool, default=False, help='Whether to use test set (default: True)')
        parser.add_argument('--resume', default=None, type=str,
                            metavar='PATH', help='path to latest checkpoint')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--fold', default='1', type=str)
        parser.add_argument('--seed', default='42', type=int)
        # numeric settings
        parser.add_argument('--workers', default=4, type=int,
                            metavar='N', help='number of data loading workers')
        parser.add_argument('--epochs', default=100, type=int,
                            metavar='N', help='number of total epochs to run')
        
        parser.add_argument('-b', '--batch_size',
                            default=32, type=int, metavar='N')
        parser.add_argument('--num_classes', default=2, type=int)

        # model settings
        parser.add_argument('--num_frames', default=16,
                            type=int, help='number of frames')
        parser.add_argument('--instance_length', default=4,
                            type=int, metavar='N', help='instance length')
        parser.add_argument('--crop_size', default=112,
                            type=int, metavar='N', help='crop size')
        parser.add_argument('--model', default='r3d',
                            type=str, help='Backbone')
        # training hyperparameters
        parser.add_argument('--label_smoothing', default=0.1,
                            type=float, help='ratio of label smoothing')
        # augmentation
        parser.add_argument('--random_sample', default=True, type=bool)
        parser.add_argument('--color_jitter', default=0.4, type=float)
        # optimizer
        parser.add_argument('-o', '--optimizer',
                            default="AdamW", type=str, metavar='Opti')
        parser.add_argument('--lr', '--learning_rate',
                            default=5e-4, type=float, metavar='LR', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
        parser.add_argument('--wd', '--weight_decay', default=0.05,
                            type=float, metavar='W', dest='weight_decay')
        parser.add_argument('--eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')

        # scheduler
        parser.add_argument('--lr_scheduler', default="cosine", type=str)
        parser.add_argument('--warmup_epochs', default=30, type=int)
        parser.add_argument('--min_lr', default=5e-6, type=float)
        parser.add_argument('--warmup_lr', default=0, type=float)

        return parser

    def parse(self):

        parser = self.initialize()
        args = parser.parse_args()

        # change the format of gpu_ids for set_device
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            cur_id = int(str_id)
            if cur_id >= 0:
                args.gpu_ids.append(cur_id)

        # use the current time as the name of the directory
        now = datetime.datetime.now()
        time_str = now.strftime("-[%m-%d]-[%H-%M]")
        if args.savename!="":
            args.name = args.savename + time_str
        else:
            args.name = args.dataset + time_str
        args.output_path = "outputs/" + args.name + "/"
        os.makedirs(args.output_path, exist_ok=True)

        # init the csv file path
        args.train_dataset = "DataSplit/train(single-labeled)/set_X.csv"
        args.val_dataset = "DataSplit/val(single-labeled)/set_X.csv"
        args.test_dataset = "DataSplit/test(single-labeled)/set_X.csv"
        # set the fold
        args.train_dataset = args.train_dataset.replace('X', str(args.fold))
        args.test_dataset = args.test_dataset.replace('X', str(args.fold))
        args.val_dataset = args.val_dataset.replace('X', str(args.fold))
        return args
