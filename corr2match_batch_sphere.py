# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from models.modules import DinoExtractor # 假设这个模块是正确的
# from PIL import Image
# import numpy as np
# import os
# from tqdm import tqdm # 引入 tqdm 进度条
# from PIL import Image, ImageDraw

# def auto_crop_white_border(image, threshold=240):
#     """自动裁剪白色边框"""
#     if isinstance(image, str):
#         image = Image.open(image)
    
#     img_array = np.array(image)
#     mask = np.mean(img_array, axis=2) < threshold
#     coords = np.argwhere(mask)
    
#     if len(coords) > 0:
#         y0, x0 = coords.min(axis=0)
#         y1, x1 = coords.max(axis=0) + 1
#         cropped = image.crop((x0, y0, x1, y1))
#         return cropped
#     return image

# def mark_positions_on_satellite(sate_img_path, gt_shift_u, gt_shift_v, pred_u, pred_v, 
#                                save_path='assets/output/pair_0001/marked_satellite.png'):
#     """
#     在卫星图像上标记真实位置和预测位置
#     """
#     # 加载原始卫星图像
#     sate_img_original = Image.open(sate_img_path).convert('RGB')
#     img_width, img_height = sate_img_original.size
    
#     # 创建图像副本用于绘制
#     marked_img = sate_img_original.copy()
#     draw = ImageDraw.Draw(marked_img)
    
#     # 计算图像中心点
#     center_x, center_y = img_width // 2, img_height // 2
    
#     # 将偏移量转换为图像坐标
#     def offset_to_pixel(offset_u, offset_v):
#         pixel_x = center_x + offset_u * (img_width / 640.0)
#         pixel_y = center_y + offset_v * (img_height / 640.0)
#         return int(pixel_x), int(pixel_y)
    
#     # 计算真实位置和预测位置的像素坐标
#     gt_x, gt_y = offset_to_pixel(gt_shift_u, gt_shift_v)
#     pred_x, pred_y = offset_to_pixel(pred_u.item() if torch.is_tensor(pred_u) else pred_u, 
#                                    pred_v.item() if torch.is_tensor(pred_v) else pred_v)
    
#     # 绘制简单的点标记
#     point_radius = 8
    
#     # 真实位置 - 红色实心圆点
#     draw.ellipse([gt_x - point_radius, gt_y - point_radius, 
#                   gt_x + point_radius, gt_y + point_radius], 
#                  fill='red', outline='red')
    
#     # 预测位置 - 绿色实心圆点
#     draw.ellipse([pred_x - point_radius, pred_y - point_radius, 
#                   pred_x + point_radius, pred_y + point_radius], 
#                  fill='green', outline='green')
    
#     # 绘制连接线
#     draw.line([gt_x, gt_y, pred_x, pred_y], fill='yellow', width=2)
    
#     # 添加图例
#     legend_x, legend_y = 20, 20
#     legend_spacing = 30
    
#     # 真实位置图例
#     draw.ellipse([legend_x, legend_y, legend_x + 8, legend_y + 8], 
#                  fill='red', outline='red')
#     draw.text((legend_x + 15, legend_y - 3), "GT Position", fill='red')
    
#     # 预测位置图例
#     draw.ellipse([legend_x, legend_y + legend_spacing, legend_x + 8, legend_y + legend_spacing + 8], 
#                  fill='green', outline='green')
#     draw.text((legend_x + 15, legend_y + legend_spacing - 3), "Pred Position", fill='green')
    
#     # 添加误差信息
#     error_distance = torch.sqrt((pred_u - gt_shift_u)**2 + (pred_v - gt_shift_v)**2).item()
#     meter_per_pixel = 0.118141
#     distance_m = error_distance * meter_per_pixel
    
#     info_text = f"Error: {distance_m:.2f}m"
#     draw.text((legend_x, legend_y + 2*legend_spacing), info_text, fill='white')
    
#     # 保存图像
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     marked_img.save(save_path)
#     print(f"标记图像已保存到: {save_path}")
    
#     return marked_img

# class MatchingDataset(Dataset):
#     def __init__(self, sat_paths, grd_paths, gt_u_list, gt_v_list):
#         self.sat_paths = sat_paths
#         self.grd_paths = grd_paths
#         self.gt_u = gt_u_list
#         self.gt_v = gt_v_list

#         # 定义变换
#         self.transform_grd = transforms.Compose([transforms.ToTensor()])
#         self.transform_sat = transforms.Compose([
#             transforms.Resize((630, 630)),
#             transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.sat_paths)

#     def __getitem__(self, idx):
#         # 1. 加载和处理卫星图像
#         sate_img = Image.open(self.sat_paths[idx]).convert('RGB')
#         sate_img_tensor = self.transform_sat(sate_img)

#         # 2. 加载和处理地面图像（自动裁剪白边）
#         grd_img = Image.open(self.grd_paths[idx]).convert('RGB')
#         grd_img_cropped = auto_crop_white_border(grd_img)
#         grd_img_resized = grd_img_cropped.resize((224, 224))
#         grd_img_tensor = self.transform_grd(grd_img_resized)

#         # 3. 真实位置和原始路径信息
#         gt_u = self.gt_u[idx]
#         gt_v = self.gt_v[idx]
        
#         # 返回张量和原始信息
#         return sate_img_tensor, grd_img_tensor, gt_u, gt_v, self.sat_paths[idx], self.grd_paths[idx]

# def batch_matching(sat_dir, grd_dir, gt_file, output_dir, batch_size=32):
#     # 1. 数据准备
#     grd_img_path_list = []
#     sat_img_path_list = []
#     gt_shift_u_list = []
#     gt_shift_v_list = []
    
#     # 从 GT 文件加载所有数据路径和标签
#     with open(gt_file, 'r') as f:
#         for line in f:
#             data = np.array(line.split())
#             sat_img_path_list.append(os.path.join(sat_dir, data[1]))
#             gt_shift_u_list.append(-float(data[3]))
#             gt_shift_v_list.append(float(data[2]))
#             base_name = os.path.splitext(os.path.basename(data[0]))[0]
#             grd_img_path_list.append(os.path.join(grd_dir, base_name + '.png'))
    
#     assert len(sat_img_path_list) == len(grd_img_path_list) == len(gt_shift_u_list) == len(gt_shift_v_list)
    
#     print(f"加载了 {len(grd_img_path_list)} 个数据")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 初始化数据集和数据加载器
#     dataset = MatchingDataset(sat_img_path_list, grd_img_path_list, gt_shift_u_list, gt_shift_v_list)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
#     # 设备设置和特征提取器初始化（只进行一次）
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     shared_feature_extractor = DinoExtractor().to(device)
    
#     all_errors = []
    
#     # 2. 批次处理循环 (带 tqdm)
#     with torch.no_grad():
#         for batch_idx, (sat_tensors, grd_tensors, gt_u_batch, gt_v_batch, sat_paths_batch, grd_paths_batch) in enumerate(tqdm(dataloader, desc="Batch Matching")):
        
#             # 移动到设备
#             sat_tensors = sat_tensors.to(device)
#             grd_tensors = grd_tensors.to(device)
            
#             B = sat_tensors.shape[0] # 当前批次大小
            
#             # --- 特征提取 (批处理) ---
#             grd_feat = shared_feature_extractor(grd_tensors)
#             sat_feat = shared_feature_extractor(sat_tensors)
            
#             # --- 特征匹配 (批处理) ---
            
#             # 假设 grd_feat 是 B x C x H x W
#             _, C, H, W = grd_feat.shape
#             _, _, H_sat, W_sat = sat_feat.shape
            
#             # 归一化地面特征 (每个样本独立归一化)
#             # 注意：这里的归一化应该是在 C 维度上进行，即 dim=1
#             grd_feat_normalized = F.normalize(grd_feat, dim=1) # 形状: B x C x H x W
            
#             # 由于无法直接用 grouped conv 实现 B 个独立的 C x H_sat x W_sat 与 C x H x W 卷积
#             # 并且要保持原有的余弦距离计算逻辑，我们保留 for 循环进行相关性计算
            
#             for i in range(B):
#                 single_sat_feat = sat_feat[i:i+1] # 1 x C x H_sat x W_sat
#                 single_grd_feat = grd_feat_normalized[i:i+1] # 1 x C x H x W
                
#                 # 重新进行相关性计算 (原代码的逻辑)
#                 s_feat = single_sat_feat.reshape(1, -1, H_sat, W_sat) # 1 x C x H_sat x W_sat
                
#                 # F.conv2d 的输出是 [1, 1, H_out, W_out]
#                 # [0] 消除 Batch 维度，结果是 [1, H_out, W_out]
#                 corr = F.conv2d(s_feat, single_grd_feat, groups=1)[0]
                
#                 # --- 核心修正代码段 ---
#                 # 增加一个索引 [0] 或使用 .squeeze(0) 来消除维度为 1 的通道维度
#                 corr = corr[0] 
#                 # 修正后 corr 的形状是 [H_out, W_out]，现在 corr.shape 只有两个值
#                 # ---------------------

#                 # 归一化分母计算 (用于余弦距离)
#                 denominator = F.avg_pool2d(single_sat_feat.pow(2), (H, W), stride=1, divisor_override=1)
#                 denominator = torch.sum(denominator, dim=1) # 形状: [1, H_out, W_out]
#                 denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
                
#                 # denominator[0] 形状是 [H_out, W_out]
#                 corr = 2 - 2 * corr / denominator[0] 
                
#                 corr_H, corr_W = corr.shape # 修正后，这里不会报错
                
#                 # 计算预测位置
#                 max_index = torch.argmin(corr.reshape(-1))
#                 pred_u_tensor = (max_index % corr_W - (corr_W / 2 + 0.5))
#                 pred_v_tensor = (max_index // corr_W - (corr_H / 2 + 0.5))
                
#                 # 坐标转换 (原代码的缩放因子)
#                 pred_u_final = pred_u_tensor * 14.0/630 * 640
#                 pred_v_final = pred_v_tensor * 14.0/630 * 640
                
#                 gt_shift_u = gt_u_batch[i]
#                 gt_shift_v = gt_v_batch[i]

#                 # 计算误差
#                 meter_per_pixel = 0.118141
#                 distance_error = torch.sqrt((pred_u_final - gt_shift_u)**2 + (pred_v_final - gt_shift_v)**2) * meter_per_pixel
                
#                 all_errors.append(distance_error.item())
                
#                 # --- 保存标记图像 (仍然需要单样本操作) ---
#                 sat_path = sat_paths_batch[i]
#                 grd_path = grd_paths_batch[i]
                
#                 base_name = os.path.splitext(os.path.basename(grd_path))[0]
#                 marked_img_path = os.path.join(output_dir, f'{base_name}_marked.png')
                
#                 # 注意：mark_positions_on_satellite 函数需要 PILLOW 图像路径
#                 mark_positions_on_satellite(
#                     sate_img_path=sat_path, 
#                     gt_shift_u=gt_shift_u, 
#                     gt_shift_v=gt_shift_v, 
#                     pred_u=pred_u_final, 
#                     pred_v=pred_v_final, 
#                     save_path=marked_img_path
#                 )
#     # 3. 结果汇总
#     if all_errors:
#         avg_error = sum(all_errors) / len(all_errors)
#         print(f"\n处理完成！共处理 {len(all_errors)} 张图片")
#         print(f"平均误差: {avg_error:.4f} 米")
        
#         # 保存结果
#         with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
#             f.write(f"总图片数: {len(all_errors)}\n")
#             f.write(f"平均误差: {avg_error:.4f} 米\n")
#             f.write(f"最大误差: {max(all_errors):.4f} 米\n")
#             f.write(f"最小误差: {min(all_errors):.4f} 米\n")


# if __name__ == "__main__":
#     sat_directory = '/ssd/liuyaowei/FG2/vigor_datasets/NewYork/satellite'
#     grd_directory = '/zssd/dataset/liuyaowei/vigor/NewYork_bev/3dpc'
#     gt_file_path = '/ssd/liuyaowei/FG2/vigor_datasets/splits/NewYork/pano_label_balanced__corrected.txt'
#     output_directory = '/zssd/dataset/liuyaowei/vigor/NewYork_bev/output_batch'

#     # 批次大小设置
#     BATCH_SIZE = 300
    
#     batch_matching(
#         sat_dir=sat_directory,
#         grd_dir=grd_directory,
#         gt_file=gt_file_path,
#         output_dir=output_directory,
#         batch_size=BATCH_SIZE
#     )
#     print("\n批次匹配完成！")


import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models.modules import DinoExtractor_large,DinoExtractor_base,DinoExtractor_large_reg,DinoExtractor_base_reg
from PIL import Image, ImageDraw
import numpy as np
import os
from tqdm import tqdm # 引入 tqdm 进度条

def mark_positions_on_satellite(sate_img_path, gt_shift_u, gt_shift_v, pred_u, pred_v, 
                                 save_path='assets/output/pair_0001/marked_satellite.png'):
    # ... (原有实现)
    sate_img_original = Image.open(sate_img_path).convert('RGB')
    img_width, img_height = sate_img_original.size
    
    marked_img = sate_img_original.copy()
    draw = ImageDraw.Draw(marked_img)
    
    center_x, center_y = img_width // 2, img_height // 2
    
    def offset_to_pixel(offset_u, offset_v):
        pixel_x = center_x + offset_u * (img_width / 640.0)
        pixel_y = center_y + offset_v * (img_height / 640.0)
        return int(pixel_x), int(pixel_y)
    
    gt_x, gt_y = offset_to_pixel(gt_shift_u, gt_shift_v)
    pred_x, pred_y = offset_to_pixel(pred_u.item() if torch.is_tensor(pred_u) else pred_u, 
                                     pred_v.item() if torch.is_tensor(pred_v) else pred_v)
    
    point_radius = 8
    
    draw.ellipse([gt_x - point_radius, gt_y - point_radius, gt_x + point_radius, gt_y + point_radius], 
                 fill='red', outline='red')
    
    draw.ellipse([pred_x - point_radius, pred_y - point_radius, pred_x + point_radius, pred_y + point_radius], 
                 fill='green', outline='green')
    
    draw.line([gt_x, gt_y, pred_x, pred_y], fill='yellow', width=2)
    
    legend_x, legend_y = 20, 20
    legend_spacing = 30
    
    draw.ellipse([legend_x, legend_y, legend_x + 8, legend_y + 8], fill='red', outline='red')
    draw.text((legend_x + 15, legend_y - 3), "GT Position", fill='red')
    
    draw.ellipse([legend_x, legend_y + legend_spacing, legend_x + 8, legend_y + legend_spacing + 8], fill='green', outline='green')
    draw.text((legend_x + 15, legend_y + legend_spacing - 3), "Pred Position", fill='green')
    
    error_distance = torch.sqrt((pred_u - gt_shift_u)**2 + (pred_v - gt_shift_v)**2).item()
    meter_per_pixel = 0.118141
    distance_m = error_distance * meter_per_pixel
    
    info_text = f"Error: {distance_m:.2f}m"
    draw.text((legend_x, legend_y + 2*legend_spacing), info_text, fill='white')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    marked_img.save(save_path)
    print(f"标记图像已保存到: {save_path}")
    
    return marked_img

class MatchingDataset(Dataset):
    def __init__(self, sat_paths, grd_paths, gt_u_list, gt_v_list):
        self.sat_paths = sat_paths
        self.grd_paths = grd_paths
        self.gt_u = gt_u_list
        self.gt_v = gt_v_list

        self.transform_grd = transforms.Compose([transforms.ToTensor()])
        self.transform_sat = transforms.Compose([
            transforms.Resize((630, 630)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sat_paths)

    def __getitem__(self, idx):
        sate_img = Image.open(self.sat_paths[idx]).convert('RGB')
        sate_img_tensor = self.transform_sat(sate_img)

        grd_img = Image.open(self.grd_paths[idx]).convert('RGB')
        grd_img = transforms.CenterCrop(size=500)(grd_img)
        grd_img_resized = transforms.Resize((350,350))(grd_img)
        grd_img_tensor = self.transform_grd(grd_img_resized)

        gt_u = self.gt_u[idx]
        gt_v = self.gt_v[idx]
        
        return sate_img_tensor, grd_img_tensor, gt_u, gt_v, self.sat_paths[idx], self.grd_paths[idx]

def batch_matching(sat_dir, grd_dir, gt_file, output_dir,meter_per_pixel, batch_size=32):
    # 1. 数据准备
    grd_img_path_list = []
    sat_img_path_list = []
    gt_shift_u_list = []
    gt_shift_v_list = []
    
    # 从 GT 文件加载所有数据路径和标签
    with open(gt_file, 'r') as f:
        for line in f:
            data = np.array(line.split())
            sat_img_path_list.append(os.path.join(sat_dir, data[1]))
            gt_shift_u_list.append(-float(data[3]))
            gt_shift_v_list.append(float(data[2]))
            base_name = os.path.splitext(os.path.basename(data[0]))[0]
            grd_img_path_list.append(os.path.join(grd_dir, base_name + '.png'))
    
    assert len(sat_img_path_list) == len(grd_img_path_list) == len(gt_shift_u_list) == len(gt_shift_v_list)
    
    print(f"加载了 {len(grd_img_path_list)} 个数据")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化数据集和数据加载器
    dataset = MatchingDataset(sat_img_path_list, grd_img_path_list, gt_shift_u_list, gt_shift_v_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 设备设置和特征提取器初始化（只进行一次）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared_feature_extractor = DinoExtractor_large().to(device)
    
    all_errors = []
    detailed_results = []
    
    with torch.no_grad():
        for batch_idx, (sat_tensors, grd_tensors, gt_u_batch, gt_v_batch, sat_paths_batch, grd_paths_batch) in enumerate(tqdm(dataloader, desc="Batch Matching")):
        
            # 移动到设备
            sat_tensors = sat_tensors.to(device)
            grd_tensors = grd_tensors.to(device)
            
            B = sat_tensors.shape[0] # 当前批次大小
            
            # --- 特征提取 (批处理) ---
            grd_feat = shared_feature_extractor(grd_tensors)
            sat_feat = shared_feature_extractor(sat_tensors)
            
            # --- 特征匹配 (批处理) ---
            
            # 假设 grd_feat 是 B x C x H x W
            _, C, H, W = grd_feat.shape
            _, _, H_sat, W_sat = sat_feat.shape
            
            # 归一化地面特征 (每个样本独立归一化)
            grd_feat_normalized = F.normalize(grd_feat, dim=1) # 形状: B x C x H x W
            
            for i in range(B):
                single_sat_feat = sat_feat[i:i+1] # 1 x C x H_sat x W_sat
                single_grd_feat = grd_feat_normalized[i:i+1] # 1 x C x H x W
                
                # 重新进行相关性计算 (原代码的逻辑)
                s_feat = single_sat_feat.reshape(1, -1, H_sat, W_sat) # 1 x C x H_sat x W_sat
                
                # F.conv2d 的输出是 [1, 1, H_out, W_out]
                corr = F.conv2d(s_feat, single_grd_feat, groups=1)[0]
                
                # --- 核心修正代码段 ---
                corr = corr[0] # 消除维度为 1 的通道维度，形状为 [H_out, W_out]
                # ---------------------

                # 归一化分母计算 (用于余弦距离)
                denominator = F.avg_pool2d(single_sat_feat.pow(2), (H, W), stride=1, divisor_override=1)
                denominator = torch.sum(denominator, dim=1) # 形状: [1, H_out, W_out]
                denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
                
                # denominator[0] 形状是 [H_out, W_out]
                corr = 2 - 2 * corr / denominator[0] 
                
                corr_H, corr_W = corr.shape 
                # 计算预测位置
                max_index = torch.argmin(corr.reshape(-1))
                pred_u_tensor = (max_index % corr_W - (corr_W / 2 + 0.5))
                pred_v_tensor = (max_index // corr_W - (corr_H / 2 + 0.5))
                # 坐标转换 (原代码的缩放因子)
                pred_u_final = pred_u_tensor * 14.0/630 * 640
                pred_v_final = pred_v_tensor * 14.0/630 * 640
                
                gt_shift_u = gt_u_batch[i]
                gt_shift_v = gt_v_batch[i]

                distance_error = torch.sqrt((pred_u_final - gt_shift_u)**2 + (pred_v_final - gt_shift_v)**2) * meter_per_pixel
                
                error_m = distance_error.item()
                all_errors.append(error_m)
                
                # --- 收集详细结果 ---
                grd_path = grd_paths_batch[i]
                grd_filename = os.path.basename(grd_path)
                
                detailed_results.append({
                    'filename': grd_filename,
                    'error_m': error_m,
                    'pred_u': pred_u_final.item(),
                    'pred_v': pred_v_final.item(),
                    'gt_u': gt_shift_u.item() if torch.is_tensor(gt_shift_u) else gt_shift_u,
                    'gt_v': gt_shift_v.item() if torch.is_tensor(gt_shift_v) else gt_shift_v
                })
                
                # --- 保存标记图像 ---
                sat_path = sat_paths_batch[i]
                
                base_name = os.path.splitext(grd_filename)[0]
                marked_img_path = os.path.join(output_dir, f'{base_name}_marked.png')
                
                mark_positions_on_satellite(
                    sate_img_path=sat_path, 
                    gt_shift_u=gt_shift_u, 
                    gt_shift_v=gt_shift_v, 
                    pred_u=pred_u_final, 
                    pred_v=pred_v_final, 
                    save_path=marked_img_path
                )

    # 3. 结果汇总与保存
    if all_errors:
        avg_error = sum(all_errors) / len(all_errors)
        count = len(all_errors)
        
        print(f"\n处理完成！共处理 {count} 张图片")
        print(f"平均误差: {avg_error:.4f} 米")
        
        # --- 保存详细结果到 TXT 文件 ---
        detailed_path = os.path.join(output_dir, 'detailed_errors.txt')
        with open(detailed_path, 'w') as f:
            # 写入标题行
            f.write("Filename,Error_m,Pred_U,Pred_V,GT_U,GT_V\n")
            # 写入每个配对的详细结果
            for res in detailed_results:
                f.write(f"{res['filename']},{res['error_m']:.4f},{res['pred_u']:.2f},{res['pred_v']:.2f},{res['gt_u']:.2f},{res['gt_v']:.2f}\n")
        print(f"详细误差结果已保存到: {detailed_path}")

        # 保存概要结果
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"总图片数: {count}\n")
            f.write(f"平均误差: {avg_error:.4f} 米\n")
            f.write(f"最大误差: {max(all_errors):.4f} 米\n")
            f.write(f"最小误差: {min(all_errors):.4f} 米\n")

def batch_matching_new(sat_dir, grd_dir, gt_file, output_dir, meter_per_pixel, batch_size=32):
    # 1. 数据准备
    grd_img_path_list = []
    sat_img_path_list = []
    gt_shift_u_list = []
    gt_shift_v_list = []
    
    # 从 GT 文件加载所有数据路径和标签
    with open(gt_file, 'r') as f:
        for line in f:
            data = np.array(line.split())
            sat_img_path_list.append(os.path.join(sat_dir, data[1]))
            gt_shift_u_list.append(-float(data[3]))
            gt_shift_v_list.append(float(data[2]))
            base_name = os.path.splitext(os.path.basename(data[0]))[0]
            grd_img_path_list.append(os.path.join(grd_dir, base_name + '.png'))
    
    assert len(sat_img_path_list) == len(grd_img_path_list) == len(gt_shift_u_list) == len(gt_shift_v_list)
    
    print(f"加载了 {len(grd_img_path_list)} 个数据")
    os.makedirs(output_dir, exist_ok=True)
    print(len(grd_img_path_list))
    print(len(sat_img_path_list))
    print(len(gt_shift_u_list))
    print(len(gt_shift_v_list))
    # 初始化数据集和数据加载器
    dataset = MatchingDataset(sat_img_path_list, grd_img_path_list, gt_shift_u_list, gt_shift_v_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 设备设置和特征提取器初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared_feature_extractor = DinoExtractor_large().to(device)
    
    all_errors = []
    detailed_results = []
    
    # 2. 批处理循环
    with torch.no_grad():
        for batch_idx, (sat_tensors, grd_tensors, gt_u_batch, gt_v_batch, sat_paths_batch, grd_paths_batch) in enumerate(tqdm(dataloader, desc="Batch Matching")):
            
            # 移动到设备
            sat_tensors = sat_tensors.to(device)
            grd_tensors = grd_tensors.to(device)
            gt_u_batch = gt_u_batch.to(device)
            gt_v_batch = gt_v_batch.to(device)
            
            B = sat_tensors.shape[0]
            
            grd_feat = shared_feature_extractor(grd_tensors)
            sat_feat = shared_feature_extractor(sat_tensors)
            
            B, C, H, W = grd_feat.shape
            _, _, H_sat, W_sat = sat_feat.shape
            
            # 归一化特征
            grd_feat_normalized = F.normalize(grd_feat, dim=1)  # 形状: B x C x H x W
            sat_feat_normalized = F.normalize(sat_feat, dim=1)  # 形状: B x C x H_sat x W_sat
            
            # 批处理卷积 (关键优化)
            sat_input = sat_feat_normalized.reshape(1, B*C, H_sat, W_sat)
            corr = F.conv2d(sat_input, grd_feat_normalized, groups=B)  # 形状: [1, B, H_out, W_out]
            corr = corr[0]  # 移除第一个维度 -> [B, H_out, W_out]
            
            H_out = H_sat - H + 1
            W_out = W_sat - W + 1
            
            # 计算分母 (批处理)
            denominator = F.avg_pool2d(sat_feat_normalized.pow(2), (H, W), stride=1, divisor_override=1)
            denominator = torch.sum(denominator, dim=1)  # 形状: [B, H_out, W_out]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            
            # 计算距离
            corr = 2 - 2 * corr / denominator
            
            # 找到最小距离位置
            corr_flat = corr.reshape(B, -1)
            max_indices = torch.argmin(corr_flat, dim=1)
            
            # 计算预测坐标
            pred_u_tensor = (max_indices % W_out - (W_out / 2 + 0.5))
            pred_v_tensor = (max_indices // W_out - (H_out / 2 + 0.5))
            
            # 坐标转换
            pred_u_final = pred_u_tensor * 14.0/630 * 640
            pred_v_final = pred_v_tensor * 14.0/630 * 640
            
            # 计算误差
            distance_error = torch.sqrt((pred_u_final - gt_u_batch)**2 + (pred_v_final - gt_v_batch)**2) * meter_per_pixel
            
            # 保存结果
            for i in range(B):
                error_m = distance_error[i].item()
                all_errors.append(error_m)
                
                # 收集详细结果
                grd_path = grd_paths_batch[i]
                grd_filename = os.path.basename(grd_path)
                
                detailed_results.append({
                    'filename': grd_filename,
                    'error_m': error_m,
                    'pred_u': pred_u_final[i].item(),
                    'pred_v': pred_v_final[i].item(),
                    'gt_u': gt_u_batch[i].item(),
                    'gt_v': gt_v_batch[i].item()
                })
                
                # 保存标记图像
                sat_path = sat_paths_batch[i]
                base_name = os.path.splitext(grd_filename)[0]
                marked_img_path = os.path.join(output_dir, f'{base_name}_marked.png')
                
                mark_positions_on_satellite(
                    sate_img_path=sat_path, 
                    gt_shift_u=gt_u_batch[i], 
                    gt_shift_v=gt_v_batch[i].item(), 
                    pred_u=pred_u_final[i].item(), 
                    pred_v=pred_v_final[i].item(), 
                    save_path=marked_img_path
                )

    # 3. 结果汇总与保存
    if all_errors:
        avg_error = sum(all_errors) / len(all_errors)
        count = len(all_errors)
        
        print(f"\n处理完成！共处理 {count} 张图片")
        print(f"平均误差: {avg_error:.4f} 米")
        
        # 保存详细结果
        detailed_path = os.path.join(output_dir, 'detailed_errors.txt')
        with open(detailed_path, 'w') as f:
            f.write("Filename,Error_m,Pred_U,Pred_V,GT_U,GT_V\n")
            for res in detailed_results:
                f.write(f"{res['filename']},{res['error_m']:.4f},{res['pred_u']:.2f},{res['pred_v']:.2f},{res['gt_u']:.2f},{res['gt_v']:.2f}\n")
        print(f"详细误差结果已保存到: {detailed_path}")

        # 保存概要结果
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"总图片数: {count}\n")
            f.write(f"平均误差: {avg_error:.4f} 米\n")
            f.write(f"最大误差: {max(all_errors):.4f} 米\n")
            f.write(f"最小误差: {min(all_errors):.4f} 米\n")

if __name__ == "__main__":
    # sat_directory = '/ssd/liuyaowei/FG2/vigor_datasets/Chicago/satellite'
    # grd_directory = '/zssd/dataset/liuyaowei/vigor/Chicago_bev/curved_bev_output/down_image'
    # gt_file_path = '/ssd/liuyaowei/FG2/vigor_datasets/splits/Chicago/pano_label_balanced__corrected.txt'
    # output_directory = '/zssd/dataset/liuyaowei/vigor/Chicago_bev/curved_bev_output/marked'
    # meter_per_pixel = 0.111262
    # BATCH_SIZE = 300
    # 'NewYork': 0.113248
    # 'Seattle': 0.100817
    # 'SanFrancisco': 0.118141
    # 'Chicago': 0.111262
    # 批次大小设置
    sat_directory = 'lalala/sat'
    grd_directory = 'lalala/curved_bev_output'
    gt_file_path = 'a.txt'
    output_directory = 'lalala/output_curved_bev'
    meter_per_pixel = 0.118141
    BATCH_SIZE = 19
    
    batch_matching_new(
        sat_dir=sat_directory,
        grd_dir=grd_directory,
        gt_file=gt_file_path,
        output_dir=output_directory,
        meter_per_pixel=meter_per_pixel,
        batch_size=BATCH_SIZE
    )
    print("\n批次匹配完成！")