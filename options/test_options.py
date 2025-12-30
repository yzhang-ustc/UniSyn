from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='/home1/yuezhang/code/UniSyn_git/results/', help='saves results here.')
        self.parser.add_argument('--prefix', type=str, default='oldnet')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--condition', type=str, default='0,1,1,1', help='available condition')
        self.parser.add_argument('--target_modal', type=int, default=0, help='target modal')
        self.isTrain = False
