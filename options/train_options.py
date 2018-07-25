# Xi Peng, May 2017
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lr', type=float, default=2.5e-4,
                    help='initial learning rate')
        self.parser.add_argument('--bs', type=int, default=24,
                    help='mini-batch size')
        self.parser.add_argument('--load_checkpoint', type=bool, default=False,
                    help='use checkpoint model')
        self.parser.add_argument('--adjust_lr', type=bool, default=False,
                    help='adjust learning rate')
        self.parser.add_argument('--resume_prefix', type=str, default='',
                    help='checkpoint name for resuming')
        self.parser.add_argument('--nEpochs', type=int, default=200,
                    help='number of total training epochs to run')
        self.parser.add_argument('--best_pckh', type=float, default=0.,
                    help='best result until now')
        self.parser.add_argument('--train_list', type=str, default='train_list.txt',
                    help='train image list')
        self.parser.add_argument('--val_list', type=str, default='val_list.txt',
                    help='validation image list')
        self.parser.add_argument('--print_freq', type=int, default=10,
                    help='print log every n iterations')
        self.parser.add_argument('--display_freq', type=int, default=10,
                    help='display figures every n iterations')

        # self.parser.add_argument('--momentum', type=float, default=0.90,
        #             help='momentum term of sgd')
        # self.parser.add_argument('--weight_decay', type=float, default=1e-4,
        #             help='weight decay term of sgd')
        # self.parser.add_argument('--beta1', type=float, default=0.5,
        #             help='momentum term of adam')
