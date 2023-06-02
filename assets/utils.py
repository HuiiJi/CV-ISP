import glob
import os
import numpy as np
import rawpy
import cv2
import torch
from tqdm import tqdm
import torch.nn as nn

class PRE_PROCESS():
    def __init__(self):
        super(CVISP_IP, self).__init__()
    
    def simple_dm(self, input_data : np.ndarray, half_resolution : bool = False)->np.ndarray:
        assert len(input_data.shape) == 2 or input_data.shape[2] == 1, 'Input data shape does not match metadata, input data should be 2D or 3D with 1 channel, but got {}'.format(input_data.shape)
        raw_data = self.pack_img(input_data)
        raw_data = np.stack((raw_data[:, :, 0],
                             np.mean(raw_data[:, :, 1:3], -1),
                             raw_data[:, :, 3]), -1)
        raw_data = cv2.resize(raw_data, (self.w, self.h), interpolation=cv2.INTER_LINEAR) if not half_resolution else raw_data
        return raw_data
      
    def mkfolder(self, path : str):
        if not os.path.exists(path):
            os.makedirs(path)
            print({path}, 'is created for saving the results.')
    
    def check_raw_image(self, image_path:str):
        for id in tqdm(os.listdir(image_path)):
            if not id.endswith('.dng'):
                continue
            image = rawpy.imread(os.path.join(image_path, id)).raw_image_visible
            assert image is not  None, 'image is not exist, please check the image path.'
            
    def load_ckp(self, checkpoint_path : str = None, model : nn.Module = None):
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
        assert checkpoint is not None, 'checkpoint is None, please check the checkpoint_path.'
        try :
            # model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint)
            print('load single-GPU ckp, path = {}'.format(checkpoint_path))
        except:
            new_ckp = {}
            for k,v in checkpoint['state_dict'].items():
                new_ckp[k[7:]] = v
            model.load_state_dict(new_ckp)
            print('load multiple-GPU ckp, path = {}'.format(checkpoint_path))
           
    def get_rgb(self, input_path : str):
        assert input_path is not None, f'rgb_input_path is not exists, please check your path'
        assert input_path.endswith(('jpg', 'png', 'JPG', 'PNG', 'JPEG', 'bmp', 'BMP')), 'Input path should be an image such as jpg, png, bmp, etc., but got {}'.format(input_path)
        image = cv2.imread(input_path)[..., ::-1]
        return image
    
    def get_metadata_with_json(self, input_path : str):
        import json
        assert input_path is not None, f'metadata_input_path is not exists, please check your path'
        assert input_path.endswith(('json', 'JSON')), 'Input path should be a json file, but got {}'.format(input_path)
        with open(input_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def get_metadata(self, input_path : str)->np.ndarray:
        import rawpy
        assert os.path.exists(input_path), 'Input path does not exist!, please check your input path'
        assert not input_path.endswith(('jpg', 'png', 'raw')), 'Input path should be raw file such as [.dng, .ARW] or other camera sensor dtype, ' \
                                                               'not jpg/png/raw file, please check your input path'
        raw = rawpy.imread(input_path)
        assert raw is not None, 'Raw data is None, please check your input path'
        rawdata = raw.raw_image_visible
        metadata = {}
        metadata['black_level'] = raw.black_level_per_channel[0]
        metadata['white_level'] = raw.white_level
        metadata['white_balance_gain'] = np.asarray(raw.camera_whitebalance[:3]) / np.asarray(raw.camera_whitebalance[1])
        metadata['ccm'] = np.asarray(raw.color_matrix[:, :3])
        metadata['color_desc'] = raw.color_desc
        metadata['height'] = raw.sizes.height if raw.sizes.height is not None else rawdata.shape[0]
        metadata['width'] = raw.sizes.width if raw.sizes.width is not None else raw.data.shape[1]
        metadata['bayer_pattern_matrix'] = raw.raw_pattern if raw.raw_pattern is not None else None
        metadata['bayer_pattern'] = self.get_bayer_pattern(metadata['bayer_pattern_matrix'])
        metadata['cam2rgb'] = np.asarray(raw.rgb_xyz_matrix) if raw.rgb_xyz_matrix is not None else None
        
        assert rawdata.shape[0] == metadata['height'] and rawdata.shape[1] == metadata['width'], 'Raw data shape does not match metadata, rawdata should be {}x{}, but got {}x{}'.format(metadata['height'], metadata['width'], rawdata.shape[0], rawdata.shape[1])
        assert len(rawdata.shape) == 2 or rawdata.shape[2] == 1, 'Raw data shape does not match metadata, rawdata should be 2D or 3D with 1 channel, but got {}'.format(rawdata.shape)
        assert rawdata.dtype == np.uint16, 'Raw data type should be uint16, but got {}'.format(rawdata.dtype)
        return metadata, rawdata
    
class POST_PROCESS():
    def __init__(self):
        super(CVISP_IP, self).__init__()
    
    def show_demo_image(self, img : np.ndarray):
        cv2.imshow('demo_img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_demo_image(self, save_path : str, img : np.ndarray):
        assert isinstance(save_path, str), 'save_path is not string, please check the save_path.'
        assert img.shape[0] > 0 and img.shape[1] > 0, 'img is empty, please check the img.'
        cv2.imwrite(save_path, img)
        print('save the result to {}'.format(save_path))
        
class CVISP_IP():
    def __init__(self):
        super(CVISP_IP, self).__init__()
        
    def normalize(self, input_data : np.ndarray, black_level : int, white_level : int) -> np.ndarray:
        output_data = np.maximum(input_data.astype(float) - black_level, 0) / (white_level - black_level)
        return output_data
    
    def invnormalization(self, input_data : np.ndarray, black_level : int, white_level : int) -> np.ndarray:
        output_data = input_data * (white_level - black_level) + black_level
        return output_data
    
    def f32_to_u8(self, input_data : np.ndarray)->np.ndarray:
        assert input_data.min() >= 0 and input_data.max() <= 1, 'input_data is not in range [0, 1], please check the input_data.'
        output_data = np.clip(input_data * 255, 0, 255).astype(np.uint8)
        return output_data
    
    def pack_img(self, input_data : np.ndarray)->np.ndarray:
        h, w = input_data.shape[:2]
        output_data = np.zeros((h//2, w//2, 4), dtype=np.float32)
        output_data[:, :, 0] = input_data[0:h:2, 0:w:2]
        output_data[:, :, 1] = input_data[0:h:2, 1:w:2]
        output_data[:, :, 2] = input_data[1:h:2, 0:w:2]
        output_data[:, :, 3] = input_data[1:h:2, 1:w:2]
        return output_data
    
    def gamma_correction(self, input_data : np.ndarray, gamma : float)->np.ndarray:
        assert input_data.max() <= 1, 'input_data should be normalized.'
        output_data = np.power(np.clip(input_data, 1e-8, 1), 1 / gamma).clip(0, 1)
        return output_data

    def apply_ccm(self, input_data : np.ndarray, ccm : np.ndarray)->np.ndarray:
        assert input_data.max() <= 1, 'input_data should be normalized.'
        if np.zeros_like(ccm).all() == ccm.all():
            ccm = np.array([[1.631906, -0.381807, -0.250099], [-0.298296, 1.614734, -0.316438],[0.023770, -0.538501, 1.514732 ]])
        output_data = input_data.dot(np.array(ccm).T).clip(0, 1)
        return output_data

    def apply_wb(self, input_data : np.ndarray, wb : np.ndarray)->np.ndarray:
        assert input_data.max() <= 1, 'input_data should be normalized.'
        if np.zeros_like(wb).all() == wb.all():
            wb = np.array([1.9075353622436523, 1.0,  1.7717266607284546])
        output_data = (input_data * wb).clip(0, 1)
        return output_data

    def apply_lut(self, input_data : np.ndarray, lut : np.ndarray)->np.ndarray:
        output_data = lut[input_data]
        return output_data

    def apply_usm(self, input_data : np.ndarray, strength : float, radius : int)->np.ndarray:
        output_data = cv2.GaussianBlur(input_data, (radius, radius), 0)
        output_data = input_data + strength * (input_data - output_data)
        return output_data
    
    def apply_gain(self, input_data: np.ndarray)->np.ndarray:
        assert input_data.max() <= 1, 'input_data should be normalized.'
        gain = 1. / input_data.max()
        output_data = input_data * (gain  + 1.8)
        output_data = output_data.clip(0, 1)
        return output_data
    
    def bayer_unify(self, raw: np.ndarray, input_pattern: str, target_pattern: str, mode: str) -> np.ndarray:
        BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG", "BGRG", "RGBG", "GRGB", "GBGR"]
        if input_pattern not in BAYER_PATTERNS:
            raise ValueError('Unknown input bayer pattern!')
        if target_pattern not in BAYER_PATTERNS:
            raise ValueError('Unknown target bayer pattern!')
        if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
            raise ValueError('raw should be a 2-dimensional numpy.ndarray!')
        if input_pattern == target_pattern:
            h_offset, w_offset = 0, 0
        elif input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]:
            h_offset, w_offset = 1, 0
        elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
            h_offset, w_offset = 0, 1
        elif input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]:
            h_offset, w_offset = 1, 1
        else:
            # raise RuntimeError('Unexpected pair of input and target bayer pattern!')
            h_offset, w_offset = 0, 0
            print('bayer_patten is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]', 'so we use the original bayer pattern')
        if mode == "pad":
            out = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
        elif mode == "crop":
            h, w, c = raw.shape
            out = raw[h_offset:h - h_offset, w_offset:w - w_offset]
        else:
            raise ValueError('Unknown normalization mode!')
            
    # def bilinear_dm(self, input_data : np.ndarray)->np.ndarray:
    #     raw_data = demosaicing_CFA_Bayer_bilinear(raw_data, 'RGGB')
    #     raw_data = raw_data.astype(np.float32).clip(0., 1.)
    #     retrun raw_data
        
    # def mal04_dm(self, input_data : np.ndarray)->np.ndarray:
    #     raw_data = demosaicing_CFA_Bayer_Malvar2004(raw_data, 'RGGB')
    #     raw_data = raw_data.astype(np.float32).clip(0., 1.)
    #     return raw_data
    # def men07_dm(self, input_data : np.ndarray)->np.ndarray:
    #     raw_data = demosaicing_CFA_Bayer_Menon2007(raw_data, 'RGGB')
    #     raw_data = raw_data.astype(np.float32).clip(0., 1.)
    #     return raw_data
    
    def get_bayer_pattern(self, bayer_pattern_matrix : str)->str:
        bayer_desc = 'RGBG'
        input_bayer_pattern = ''
        if bayer_pattern_matrix is not None:
            for i in range(2):
                for k in range(2):
                    input_bayer_pattern += (bayer_desc[bayer_pattern_matrix[i][k]])
        else:
            input_bayer_pattern = 'RGGB'
        return input_bayer_pattern
            
    def run_isp_with_meta(self)->np.ndarray:
        raw_data = self.normalization(self.raw_data, self.black_level, self.white_level)
        raw_data = self.bayer_unify(raw_data, self.bayer_pattern, self.target_bayer_pattern, 'pad')
        raw_data = self.simple_dm(raw_data)
        raw_data = self.apply_wb(raw_data, self.white_balance_gain)
        raw_data = self.apply_ccm(raw_data, self.ccm)
        raw_data = self.gamma_correction(raw_data, 2.2)
        # raw_data = self.apply_lut(raw_data, self.lut)
        # raw_data = self.apply_usm(raw_data, strength=0.5, radius=7)
        raw_data = self.f32_to_u8(raw_data)
        # raw_data = self.inv_normalization(raw_data, self.black_level, self.white_level)

        self.show_demo_image(raw_data) if self.show_image else None
        self.save_demo_image(os.path.join(self.result_output_dir,
                                        os.path.basename(self.input_data_path)[:-4]+'.png'),                                                  
                             raw_data[..., ::-1]) if self.save_image else None
        return raw_data
      
class CVISP_simulator():
    def __init__(self):
        super(CVISP_IP, self).__init__()
        IP_config = {'normalize', self.normalize, 'demosaic', self.simple_dm}
    
    def run():
        pass
          
          

        


        

        

