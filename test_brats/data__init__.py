from test_brats.dataload_missing import BratsDataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def create_dataset(opt):
    
    data_loader = CustomDatasetDataLoader(opt)
    
    return data_loader



class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        dataset_class = BratsDataset
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
            # transforms.RandomHorizontalFlip()
        ])
        
        self.dataset = dataset_class(opt, transform)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = opt.batch_size  ,
            # shuffle = not opt.unshuffle,
            shuffle = False,
            num_workers = opt.num_workers)

    def load_data(self):
        print('===TT===')
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
