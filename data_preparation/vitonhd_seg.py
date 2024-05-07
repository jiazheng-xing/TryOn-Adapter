import json
import os.path
from os import path as osp

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
import argparse
import time 
from tqdm import tqdm

def get_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=15)
    parser.add_argument('--load_height', type=int, default=512)
    parser.add_argument('--load_width', type=int, default=384)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--dataset_dir', type=str, default='/data/extern/vition-HD')
    parser.add_argument('--dataset_mode', type=str, default='train')
    parser.add_argument('--paired', type=str, default='paired')
    # parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--dataset_list', type=str, default='train_pairs_1018new.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--semantic_nc', type=int, default=18)

    parser.add_argument('--warp_cloth_list', type=str, default='/data/extern/vition-HD/sample')

    opt = parser.parse_args()
    return opt

def show(title, array):
    plt.title(title)
    plt.imshow(array)
    plt.show()

def remap_image_vitionhd(img):
    if isinstance(img,torch.Tensor):
        img = np.array(img)
        if img.ndim == 3:
            img = (img[0] if img.shape[0] == 1 else img.transpose(1,2,0))
    unique_image = np.unique(img)
    if 9 in unique_image:
        color_label_map = [
            [0,0,0],
            [254,0,0],
            [0,0,254],
            [254,85,0],
            [0, 127, 0],
            [51,169,220],
            [0,254,254],
            [85, 254, 169],
            [169, 254, 85],
            [254, 254, 0],
            [254, 169, 0],
            [85, 85, 0],
            [52, 86, 127],
         ]
    else:
        color_label_map = [
             [0,0,0],
             [254,0,0],
             [0,0,254],
             [254,85,0],
             [0,85,85],
             [51,169,220],
             [0,254,254],
             [85, 254, 169],
             [169, 254, 85],
             [254, 254, 0],
             [254, 169, 0],
             [85, 85, 0],
             [52, 86, 127],
        ]
    rgb = remap_colors(img, color_label_map)
    return rgb

def get_unique_rgb_values(img):
    if isinstance(img,torch.Tensor):
        img = np.array(img).transpose(1,2,0)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reshaped_img = img_rgb.reshape(-1, img_rgb.shape[-1])
    unique_rgb_values = np.unique(reshaped_img, axis=0)

    return unique_rgb_values

class VITONDataset(data.Dataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = osp.join(opt.dataset_dir, opt.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset_mode = opt.dataset_mode
        self.warp_cloth_list = opt.warp_cloth_list
        # load data list
        img_names = []
        c_names = []
        self.dataset_list = opt.dataset_list
        self.paired = opt.paired
        with open(osp.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                try:
                    img_name, c_name, _ = line.strip().split()
                except:
                    img_name, c_name = line.strip().split()
                img_names.append(img_name)
                c_names.append(c_name)

        self.img_names = img_names
        self.c_names = dict()
        if self.paired == 'paired':
            self.c_names['unpaired'] = img_names
        else:
            self.c_names['unpaired'] = c_names

    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        img_name = self.img_names[index]
        if '.png' in img_name:
            img_name = img_name.replace('.png','.jpg')
        c_name = {}
        for key in self.c_names:
            c_name[key] = self.c_names[key][index]
            if '.png' in c_name[key]:
                c_name[key] = c_name[key].replace('.png', '.jpg')


        # load densepose image
        densepose_name = img_name.replace('.jpg', '.png')
        densepose_img = cv2.imread(os.path.join(self.data_path, 'densepose', densepose_name),cv2.IMREAD_GRAYSCALE)
        # breakpoint()
        densepose_img = cv2.resize(densepose_img, ( self.load_width, self.load_height))

        dense_label_map = {
            0: ['background', [0]],
            1: ['body', [2]],
            2: ['left_hand', [4]],
            3: ['right_hand', [3]],
            4: ['head', [23, 24]],
            5: ['left_arm', [4, 15, 17, 19, 21]],
            6: ['right_arm', [3, 16, 18, 20, 22]],
            7: ['left_leg', [10, 14]],
            8: ['right_leg', [9, 13]],
            9: ['left_shoe', [5]],
            10: ['right_shoe', [6]],
            11: ['unknow', [1, 7, 8, 11, 12]]
        }
        label_dense_new = torch.zeros(len(dense_label_map), self.load_height, self.load_width, dtype=torch.float)
        label_dense_list = torch.FloatTensor(25, self.load_height, self.load_width).zero_()
        labels_new_temp = torch.from_numpy(np.array(densepose_img)[None]).long()
        label_dense_list = label_dense_list.scatter_(0, labels_new_temp, 1.0)
        for i in range(len(dense_label_map)):
            for j in dense_label_map[i][1]:
                label_dense_new[i] += label_dense_list[j]

        # warped_mask_name = list(c_name.values())[0].replace('.jpg', '.png')
        warped_mask_name = img_name
        if self.dataset_mode == 'train':
            warped_mask = cv2.imread(osp.join(self.warp_cloth_list, 'train_paired/mask', warped_mask_name.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
        elif self.dataset_mode == 'test' :
            if self.paired=='paired':
                warped_mask = cv2.imread(osp.join(self.warp_cloth_list, 'test_paired/mask', warped_mask_name.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
            elif self.paired=='unpaired':
                warped_mask = cv2.imread(osp.join(self.warp_cloth_list, 'test_unpaired/mask', warped_mask_name.replace('.jpg', '.png')),
                                         cv2.IMREAD_GRAYSCALE)
        else:
            warped_mask = cv2.imread(osp.join(self.warp_cloth_list, 'mask', warped_mask_name),cv2.IMREAD_GRAYSCALE)
        # warped_mask = cv2.resize(warped_mask,(self.load_width, self.load_height),cv2.INTER_NEAREST)
        # breakpoint()
        try:
            warped_mask[warped_mask > 0] = 1
            warped_mask = erode_image(warped_mask)
            warped_mask[warped_mask > 0] = 1
            warped_mask = torch.from_numpy(np.array(warped_mask)[None]).long()
        except:
            print(osp.join(self.warp_cloth_list, 'train_paired/mask', warped_mask_name.replace('.jpg', '.png')))
            return -1

        parse_os_agnostic_name = img_name.replace('.jpg', '.png')
        parse_os_agnostic = Image.open(osp.join(self.data_path, 'image-parse-agnostic-v3.2', parse_os_agnostic_name))
        parse_os_agnostic = transforms.Resize(self.load_width, interpolation=0)(parse_os_agnostic)
        parse_os_agnostic = torch.from_numpy(np.array(parse_os_agnostic)[None]).long()

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        # parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        parse_agnostic_map.scatter_(0, parse_os_agnostic, 1.0)
        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        parse_agnostic_map_copy = copy.deepcopy(new_parse_agnostic_map)
        label_warped_raw = torch.zeros(warped_mask.shape)
        label_warped_erode = torch.zeros(warped_mask.shape)
        label_warped_seg = torch.zeros(warped_mask.shape)
        label_warped_fix = torch.zeros(warped_mask.shape)
        
        label_dense_head = copy.deepcopy(label_dense_new[4])
        label_dense_left_arm = copy.deepcopy(label_dense_new[5])
        label_dense_right_arm = copy.deepcopy(label_dense_new[6])
        label_dense_body = copy.deepcopy(label_dense_new[1])
        warped_mask_new = copy.deepcopy(warped_mask)
        for i in range(1,len(labels)):
            warped_mask_new = warped_mask_new - parse_agnostic_map_copy[i]
        warped_mask_new[warped_mask_new<0] = 0
        parse_agnostic_map_copy[3] = warped_mask_new
        parse_agnostic_map_fix = copy.deepcopy(parse_agnostic_map_copy)
        for i in range(1,len(labels)):
            label_dense_body = label_dense_body - parse_agnostic_map_fix[i]
            label_dense_left_arm = label_dense_left_arm - parse_agnostic_map_fix[i]
            label_dense_right_arm = label_dense_right_arm - parse_agnostic_map_fix[i]
            label_dense_head = label_dense_head - parse_agnostic_map_fix[i]
        label_dense_body[label_dense_body<0]=0
        label_dense_left_arm[label_dense_left_arm<0]=0
        label_dense_right_arm[label_dense_right_arm<0]=0
        label_dense_head[label_dense_head<0]=0

        label_dense_left_arm_erode = detect_filter_connected_components(label_dense_left_arm)
        label_dense_right_arm_erode = detect_filter_connected_components(label_dense_right_arm)

        parse_agnostic_map_fix[2] += label_dense_body
        parse_agnostic_map_fix[2] += label_dense_head
        
        parse_agnostic_map_fix_erode = copy.deepcopy(parse_agnostic_map_fix)
        parse_agnostic_map_fix_erode[5] += label_dense_left_arm_erode
        parse_agnostic_map_fix_erode[6] += label_dense_right_arm_erode
        Id_ = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        Id_[2] += label_dense_body + label_dense_head
        Id_[5] += label_dense_left_arm_erode
        Id_[6] += label_dense_right_arm_erode
        
        parse_agnostic_map_fix[5] += label_dense_left_arm
        parse_agnostic_map_fix[6] += label_dense_right_arm
        Id_label = torch.zeros(warped_mask.shape)
        for i in range(len(labels)):
            label_warped_fix += parse_agnostic_map_fix[i] * i
            label_warped_seg += parse_agnostic_map_copy[i] * i
            label_warped_raw += new_parse_agnostic_map[i] * i
            label_warped_erode += parse_agnostic_map_fix_erode[i] * i
            Id_label += Id_[i] * i
            

        result = {
            'img_name': img_name,
            'c_name': c_name,
            'parse_agnostic': new_parse_agnostic_map,  # [13,1024,768]
            'seg':label_warped_erode,
        }
        return result

    def __len__(self):
        return len(self.img_names)

def view_image(img):
    if isinstance(img,torch.Tensor):
        img = np.array(img)
        if img.ndim == 3:
            img = (img[0] if img.shape[0] == 1 else img.transpose(1,2,0))
    # img[img == 8] = 20
    temp = np.asarray(img * 30, dtype=np.uint8)
    # np.where((image_rgb[..., 0] == rgb[2]) & (image_rgb[..., 1] == rgb[1]) & (image_rgb[..., 2] == rgb[0]))
    # temp[temp==32] = 222
    # print("np.unique(temp)",np.unique(temp), np.unique(img))
    img = cv2.applyColorMap(temp, cv2.COLORMAP_TURBO)
    idx = np.where((img[..., 0] == 59) & (img[..., 1] == 18) & (img[..., 2] == 48))
    img[idx] = [0,0,0]
    return img

def remap_colors(gray,remap_color_map):
    rgb_img = np.zeros(shape=(gray.shape[0],gray.shape[1],3), dtype=np.uint8)
    for map_idx, rgb in enumerate(remap_color_map):
        idx = np.where(gray[...] == map_idx)
        rgb_img[(idx[0],idx[1],np.ones(idx[0].shape,dtype=np.uint8)*0)] = rgb[2]
        rgb_img[(idx[0],idx[1],np.ones(idx[0].shape,dtype=np.uint8)*1)] = rgb[1]
        rgb_img[(idx[0],idx[1],np.ones(idx[0].shape,dtype=np.uint8)*2)] = rgb[0]
    return rgb_img


def detect_filter_connected_components(image, area_threshold=12):
    image[image < 0] = 0
    if isinstance(image, torch.Tensor):
        image = np.asarray(image, dtype=np.uint8)
    image[image > 0] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image)

    output_image = np.zeros_like(image)
    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            output_image[labels == i] = 1

    return output_image

def erode_image(img):
    if isinstance(img,torch.Tensor):
        img = np.array(img)
    kernel = np.ones((3,3), dtype=np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    dige_dilate = cv2.dilate(eroded, kernel, iterations=1)
    return torch.from_numpy(dige_dilate)


class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=0, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == '__main__':
    opt = get_opt()
    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)
    for i, inputs in tqdm(enumerate(test_loader.data_loader), total=len(test_loader.data_loader)):
        for j in range(opt.batch_size):
            save_path = os.path.join(opt.save_dir,opt.dataset_mode,f'warped_{opt.paired}')
            os.makedirs(save_path,exist_ok=True)
            # breakpoint()
            cv2.imwrite(os.path.join(save_path, inputs['img_name'][j]), remap_image_vitionhd(inputs['seg'][j]))
