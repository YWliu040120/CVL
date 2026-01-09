from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
transform_grd = transforms.Compose([transforms.Resize((350, 350)), 
                                    transforms.ToTensor()])
transform_sat = transforms.Compose([
            transforms.Resize((630, 630)),
            transforms.ToTensor()
        ])

class MatchingDataset(Dataset):
    def __init__(self, root = '/zssd/dataset/liuyaowei/vigor',split='samearea',train=True):
        self.root = root
        self.split = split
        self.train = train
        self.transform_sat = transform_sat
        self.transform_grd = transform_grd  
        self.sat_paths_list,self.grd_paths_list,self.gt_u, self.gt_v = self.get_gt_data()
    def __len__(self):
        return len(self.sat_paths_list)
    def get_city_list(self):
        if self.split == 'samearea':
            return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        if self.split == 'crossarea':
            return ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']
        return []
    def get_gt_files(self,city):
        if self.split == 'samearea':
            return os.path.join(self.root,city, 'same_area_balanced_train__corrected.txt' if self.train else 'same_area_balanced_test__corrected.txt')
        return os.path.join(self.root, city, 'pano_label_balanced__corrected.txt')

    def get_gt_data(self):
        sat_paths_list = []
        grd_paths_list = []
        gt_u_list = []
        gt_v_list = []
        city_list = self.get_city_list()
        for city in city_list:
            gt_file = self.get_gt_files(city)
            sat_dir = os.path.join(self.root,city,'satellite')
            grd_dir = os.path.join(self.root,city,'curved_bev_output','down_image')
            with open(gt_file, 'r') as f:
                for line in f:
                    data = np.array(line.split())
                    sat_paths_list.append(os.path.join(sat_dir, data[1]))
                    gt_u_list.append(-float(data[3]))
                    gt_v_list.append(float(data[2]))
                    base_name = os.path.splitext(os.path.basename(data[0]))[0]
                    grd_paths_list.append(os.path.join(grd_dir, base_name + '.png'))

            assert len(sat_paths_list) == len(grd_paths_list) == len(gt_u_list) == len(gt_v_list)

            print(f"加载了 {len(grd_paths_list)} 个数据")
        return sat_paths_list,grd_paths_list,gt_u_list,gt_v_list

    def __getitem__(self, idx):
        sate_img = Image.open(self.sat_paths_list[idx]).convert('RGB')
        sate_img_tensor = self.transform_sat(sate_img)
        grd_img = Image.open(self.grd_paths_list[idx]).convert('RGB')
        grd_img_tensor = self.transform_grd(grd_img)
        gt_u = self.gt_u[idx]
        gt_v = self.gt_v[idx]
        city = self._get_city_name(self.sat_paths_list[idx])
        return sate_img_tensor, grd_img_tensor, gt_u, gt_v,city

    def _get_city_name(self, path):
        for city in ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']:
            if city in path:
                return city
        return 'Unknown'
