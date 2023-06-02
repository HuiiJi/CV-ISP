import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
from pdb import set_trace as stx
import random
# import mmengine
# import mmcv
import glob
import torchvision

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

DATA_FORMAT = 'resize'
NORMALIZE = False

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir =  '', img_options=None, train_stage = DATA_FORMAT,normalize = NORMALIZE):
        super(DataLoaderTrain, self).__init__()

        self.train_stage = train_stage
        id_ = os.listdir('')
        inp_files = [os.path.join('', i) for i in id_]
        tar_files = [os.path.join('', i) for i in id_]
        self.normalize = normalize
        self.inp_filenames = inp_files
        self.tar_filenames = tar_files
        self.train_stage = train_stage
        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        ps = self.ps
        inp_path = self.inp_filenames[index]
        tar_path = self.tar_filenames[index]

        assert os.path.basename(inp_path) == os.path.basename(tar_path), 'inp and tar is not match'
        # inp_name = str(inp_path) + np.random.choice(beta)
        inp_name = str(inp_path)

        inp_img = Image.open(inp_name).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        if self.normalize:
            transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
            transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
            inp_img =  transform_input(inp_img)
            tar_img = transform_gt(tar_img)
        else:
            inp_img = TF.to_tensor(inp_img)
            tar_img = TF.to_tensor(tar_img)

        if self.train_stage == 'crop':
            assert ps is not None, f'please set patch_size for train, {ps} is not used'
            hh, ww = tar_img.shape[1], tar_img.shape[2]
            rr     = random.randint(0, hh-ps)
            cc     = random.randint(0, ww-ps)
            inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
            tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        elif self.train_stage == 'resize':
            inp_img = torchvision.transforms.Resize([ps, ps])(inp_img)
            tar_img = torchvision.transforms.Resize([ps, ps])(tar_img)
        else:
            raise VaiseError('your train_stage should be in [crop, resize]')

        aug  = random.randint(0, 3)
        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        return tar_img, inp_img

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir= '', img_options=None, val_stage = DATA_FORMAT,normalize=NORMALIZE):
        super(DataLoaderVal, self).__init__()    
        self.val_stage = val_stage
        id_ = os.listdir('')
        inp_files = [os.path.join('', i) for i in id_]
        tar_files = [os.path.join('', i) for i in id_]
        # inp_files = [os.path.join(rgb_dir, f"{int(name) + 1}.png") for name in names]
        self.normalize = normalize
        self.inp_filenames = inp_files
        self.tar_filenames = tar_files
        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        ps = self.ps

        inp_path = self.inp_filenames[index]
        tar_path = self.tar_filenames[index]

        # inp_name = str(inp_path) + np.random.choice(beta)
        inp_name = str(inp_path)

        inp_img = Image.open(inp_name).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # Validate on center crop
        if self.val_stage == 'crop':
            assert ps is not None, f'please set patch_size for train, {ps} is not used'
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))
        elif self.val_stage == 'resize':
            inp_img = torchvision.transforms.Resize([ps, ps])(inp_img)
            tar_img = torchvision.transforms.Resize([ps, ps])(tar_img)
        else:
            raise VaiseError('your train_stage should be in [crop, resize]')

        if self.normalize:
            transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
            transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
            inp_img =  transform_input(inp_img)
            tar_img = transform_gt(tar_img)
        else:
            inp_img = TF.to_tensor(inp_img)
            tar_img = TF.to_tensor(tar_img)
  
        return tar_img, inp_img

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename
        
        
def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)


