
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
from models.modules import DinoV2VigorModel
from dataloaders.dataloader import MatchingDataset

NewYork_res = 0.113248 
Seattle_res = 0.100817
SanFrancisco_res = 0.118141 
Chicago_res = 0.111262 

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="DPT Heatmap Testing")
    parser.add_argument('--split', type=str, choices=('samearea', 'crossarea'), default='samearea')
    # 默认为 False，表示加载测试集
    parser.add_argument('--train', type=int, default=0, help="1 for training set, 0 for test set")
    parser.add_argument('--batch_size', type=int, default=24)
    # 必须指定模型权重路径
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pth)")
    return parser.parse_args()

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(42)
    
    print("Initializing Model...")
    model = DinoV2VigorModel(patch_size=14)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    model.to(device)
    model.eval()

    # 数据集加载逻辑
    print(f"Loading Data (Split: {args.split})...")
    train = args.train
    test_set = MatchingDataset(split=args.split, train=train)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Start Testing: {len(test_set)} samples.")
    
    translation_error = []

    with torch.no_grad():
        for sat, grd, gt_u, gt_v, city in tqdm(test_dataloader, desc="Testing"):
            sat, grd = sat.to(device), grd.to(device)
            gt_u, gt_v = gt_u.to(device), gt_v.to(device)

            # 1. 特征提取
            sat_feat = model(sat)
            grd_feat = model(grd) 
            
            # 2. L2 归一化
            sat_feat = F.normalize(sat_feat, dim=1)
            grd_feat = F.normalize(grd_feat, dim=1)

            B, C, Hs, Ws = sat_feat.shape
            _, _, Hg, Wg = grd_feat.shape

            # 3. 计算相关性 (Cross Correlation)
            sat_input = sat_feat.view(1, B*C, Hs, Ws)
            corr = F.conv2d(sat_input, grd_feat, groups=B)[0]
            
            # 计算输出尺寸
            H_out = Hs - Hg + 1
            W_out = Ws - Wg + 1

            # 4. 计算 Cosine Similarity (修正后的逻辑)
            # 分母 A: 卫星图局部窗口的模长
            sat_sq = sat_feat.pow(2)
            sat_window_norm_sq = F.avg_pool2d(sat_sq, (Hg, Wg), stride=1, divisor_override=1)
            sat_window_norm_sq = torch.sum(sat_window_norm_sq, dim=1) 
            sat_window_norm = torch.sqrt(sat_window_norm_sq + 1e-6)
            
            # 分母 B: 地面图核的模长 (常数)
            grd_kernel_norm = np.sqrt(Hg * Wg)
            
            # 相似度 [-1, 1]
            cosine_sim = corr / (sat_window_norm * grd_kernel_norm + 1e-6)

            # 5. 寻找最大相似度位置 (Argmax)
            cosine_sim_flat = cosine_sim.reshape(B, -1)
            max_indices = torch.argmax(cosine_sim_flat, dim=1)
            
            # 6. 计算预测坐标 (特征图尺度)
            # 这里的坐标原点是特征图中心
            pred_u_tensor = (max_indices % W_out - (W_out / 2 + 0.5))
            pred_v_tensor = (max_indices // W_out - (H_out / 2 + 0.5))
            
            # 7. 坐标映射回米 (Meters)
            # 动态计算 scale_factor (每个 batch 可能略有不同，取决于 Ws)
            scale_factor = 640.0 / Ws
            
            # 这里的 /630*640 是因为原始图片大小为640，而地面图输入模型前被 resize 到630（考虑到dino模型patch大小是14）
            pred_u_final = pred_u_tensor * scale_factor / 630 * 640
            pred_v_final = pred_v_tensor * scale_factor / 630 * 640
            
            # 8. 计算欧氏距离误差
            distance_error = torch.sqrt((pred_u_final - gt_u)**2 + (pred_v_final - gt_v)**2)
            for b in range(B):
                if city[b] == 'NewYork':
                    translation_error.append(distance_error[b].cpu().item() * NewYork_res)
                elif city[b] == 'Seattle':
                    translation_error.append(distance_error[b].cpu().item() * Seattle_res)
                elif city[b] == 'SanFrancisco':
                    translation_error.append(distance_error[b].cpu().item() * SanFrancisco_res)
                elif city[b] == 'Chicago':
                    translation_error.append(distance_error[b].cpu().item() * Chicago_res)
    # --- 统计结果 ---
    mean_error = np.mean(translation_error)
    median_error = np.median(translation_error)
    min_error = np.min(translation_error)
    max_error = np.max(translation_error)

    print("\n" + "="*30)
    print(f"  TEST RESULTS (Split: {args.split})")
    print("="*30)
    print(f"Mean Error   : {mean_error:.4f} meters")
    print(f"Median Error : {median_error:.4f} meters")
    print(f"Min Error    : {min_error:.4f} meters")
    print(f"Max Error    : {max_error:.4f} meters")
    print("-" * 30)
    print("="*30)

if __name__ == "__main__":
    args = parse_args()
    test(args)
    # python test.py --split samearea --checkpoint ./runs/checkpoints/dpt_vigor_samearea_epoch_1.pth --train 0