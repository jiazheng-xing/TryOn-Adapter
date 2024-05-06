import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import argparse

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
    
def get_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)

    parser.add_argument('--warped_path', type=str, default='/home/ock/aigc/GP-VTON-main/sample/viton_hd/train_paired/warped')
    parser.add_argument('--mask_path', type=str, default='/home/ock/aigc/GP-VTON-main/sample/viton_hd/train_paired/mask')
    parser.add_argument('--output_folder', type=str, default='/home/ock/aigc/Try-On-old/highlight/train')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_opt()
    warped_path = opt.warped_path
    warped_mask_path = opt.mask_path
    output_folder =  opt.output_folder
    os.makedirs(output_folder,exist_ok=True)

    for filename in tqdm(os.listdir(warped_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            image_path = os.path.join(warped_path, filename)
            mask_path = os.path.join(warped_mask_path, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask =  cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(mask, kernel, iterations=3)

            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            gradient = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0) #* (1-mask)
            gradient[eroded_mask == 0] = 0
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path,gradient)
