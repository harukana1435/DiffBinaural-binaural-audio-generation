##### for train #####
import argparse


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--id', default='realBinaural',
                            help="a name for identifying the model")
        parser.add_argument('--arch_frame', default='resnet18',
                            help="architecture of net_frame")
        parser.add_argument('--weights_frame', default='',
                            help="weights to finetune net_frame")
        parser.add_argument('--weights_unet', default='',
                            help="weights to finetune unet")
        parser.add_argument('--num_channels', default=32, type=int,
                            help='number of channels')
        parser.add_argument('--num_frames', default=1, type=int,
                            help='number of frames')
        parser.add_argument('--img_pool', default='maxpool',
                            help="avg or max pool image features")
        parser.add_argument('--loss', default='l1',
                            help="loss function to use")
        parser.add_argument('--weighted_loss', default=1, type=int,
                            help="weighted loss")
        parser.add_argument('--split', default='val',
                            help="val or test")
        
        parser.add_argument('--decay_factor', default=0.94, type=float)
        parser.add_argument('--learning_rate_decrease_itr', default=50, type=int)
        
        parser.add_argument('--max_sources', default=4, type=int)

        # Data related arguments
        parser.add_argument('--num_gpus', default=1, type=int,
                            help='number of gpus to use')
        parser.add_argument('--gpu_ids', default="0", type=str,
                            help='which gpus to use')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=32, type=int,
                            help='number of data loading workers')
        parser.add_argument('--num_val', default=-1, type=int,
                            help='number of images to evalutate')
        parser.add_argument('--num_vis', default=40, type=int,
                            help='number of images to evalutate')
        
        parser.add_argument('--audLen', default=16384, type=int,
                            help='sound length')
        parser.add_argument('--audRate', default=16000, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1024, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")
        parser.add_argument('--num_mels', default=80, type=int,
                            help="nums of mel")

        parser.add_argument('--imgSize', default=224, type=int,
                            help='size of input frame')
        parser.add_argument('--vidRate', default=8, type=float,
                            help='video frame sampling rate')
        
        parser.add_argument('--pos_type', default="3D", type=str,
                            help='pos type')

        # Misc arguments
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='/home/h-okano/DiffBinaural/checkpoints',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=10,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')
        parser.add_argument('--num_train_timesteps', type=int, default=300,
                            help='steps for forward process')
        parser.add_argument('--num_sample_timesteps', type=int, default=300,
                            help='steps for backward process')
        
        parser.add_argument('--dir_frames', default='/home/h-okano/DiffBinaural/FairPlay/frames',
                            help='dir of frames')
        parser.add_argument('--dir_det_pos', default='/home/h-okano/DiffBinaural/processed_data/det_pos_npy',
                            help='folder to detection and 3d position')        


        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--list_train',
                            default='/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration/data/solo_train.csv')
        parser.add_argument('--list_val',
                            default='/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration/data/solo_val.csv')
        parser.add_argument('--dup_trainset', default=1, type=int,
                            help='duplicate so that one epoch has more iters')

        # optimization related arguments
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')

        parser.add_argument('--lr_frame', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_unet',
                            default=1e-3, type=float, help='LR')
        parser.add_argument('--lr_steps',
                            nargs='+', type=int, default=[40, 60],
                            help='steps to drop LR in epochs')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')
        self.parser = parser

    def add_test_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='eval', help="train/eval")
        parser.add_argument('--list_test', default='/home/h-okano/real_dataset/split/split_00/test.csv')
        parser.add_argument('--output_dir_left', default='./generated_files')
        parser.add_argument('--output_dir_right', default='./generated_files')
        self.parser = parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

    def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
        self.print_arguments(args)
        return args
    
    def parse_test_arguments(self):
        self.add_test_arguments()
        args = self.parser.parse_args()
        self.print_arguments(args)
        return args
