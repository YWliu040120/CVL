import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
import numpy as np
import os
from tqdm import tqdm
from models.modules import DinoV2VigorModel
from dataloaders.dataloader import MatchingDataset
from utils.loss import HeatmapLoss
from torch.utils.tensorboard import SummaryWriter

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DPT Heatmap Training")
    parser.add_argument('--split', type=str, choices=('samearea','crossarea'), default='samearea')
    parser.add_argument('--train', type=int, default=1, help="1 for training set, 0 for test set")
    # Output directory
    parser.add_argument('--save_dir', type=str, default='checkpoints/dpt_heatmap_train')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=24)  # Adjust based on GPU memory
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--epoch_to_resume', type=int, help='from which epoch to continue training', default=0) #从哪一帧开始恢复

    return parser.parse_args()
def train(args):
    writer = SummaryWriter(log_dir=f'runs/')
    device = torch.device('cuda')
    os.makedirs(args.save_dir, exist_ok=True)
    setup_seed(42)  # 设置随机种子，确保结果可复现
    split = args.split
    train = args.train
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    epoch_to_resume = args.epoch_to_resume

    print("Initializing Model...")
    model = DinoV2VigorModel(patch_size=14)
    if epoch_to_resume > 0:
        model.load_state_dict(torch.load(os.path.join('runs/checkpoints', str(epoch_to_resume-1), 'model.pt')))
    model.to(device)

    optimizer = torch.optim.AdamW(model.dpt_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)

    vigor_dataset = MatchingDataset(split=split, train=train)
    dataset_size = vigor_dataset.__len__()
    print(f"Dataset Size: {dataset_size} samples.")
    index_list = np.arange(vigor_dataset.__len__())
    np.random.shuffle(index_list)
    train_indices = index_list[0: int(len(index_list)*0.8)]
    val_indices = index_list[int(len(index_list)*0.8):]
    train_set = Subset(vigor_dataset, train_indices)
    val_set = Subset(vigor_dataset, val_indices)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    loss_fn = HeatmapLoss(sigma=2.0)
    print(f"Start Training (Float32): {len(train_set)} train samples, {args.epochs} epochs.")

    for epoch in range(epochs):
        model.train() # 开启 DPT Head 的 Dropout/BN
        epoch_loss = []
        batch_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for sat, grd, gt_u, gt_v,city in pbar:
            sat, grd = sat.to(device), grd.to(device)
            gt_u, gt_v = gt_u.to(device), gt_v.to(device)

            # 1. 特征提取
            sat_feat = model(sat)
            grd_feat = model(grd) 
            print(sat_feat.shape, grd_feat.shape)
            sat_feat = F.normalize(sat_feat, dim=1)
            grd_feat = F.normalize(grd_feat, dim=1)
            
            B, C, Hs, Ws = sat_feat.shape
            _, _, Hg, Wg = grd_feat.shape
            
            # 将 Sat 作为 Input, Grd 作为 Kernel
            sat_input = sat_feat.view(1, B*C, Hs, Ws)
            
            corr = F.conv2d(sat_input, grd_feat, groups=B)
            corr = corr[0]

            sat_sq = sat_feat.pow(2)
            sat_window_norm_sq = F.avg_pool2d(sat_sq, (Hg, Wg), stride=1, divisor_override=1)
            sat_window_norm_sq = torch.sum(sat_window_norm_sq, dim=1) 
            sat_window_norm = torch.sqrt(sat_window_norm_sq + 1e-6)
            grd_kernel_norm = np.sqrt(Hg * Wg)
            cosine_sim = corr / (sat_window_norm * grd_kernel_norm + 1e-6)    
            scale_factor = 640.0 / Ws
            batch_loss = loss_fn(cosine_sim, gt_u, gt_v, scale_factor)

            writer.add_scalar('Loss/Batch', batch_loss.item(), epoch * len(train_dataloader) + pbar.n)

            optimizer.zero_grad()
            batch_loss.backward() 
            optimizer.step() 
            epoch_loss.append(batch_loss.item())
            pbar.set_postfix({'loss': f"{batch_loss.item():.4f}"})

        scheduler.step()
        avg_loss = np.mean(epoch_loss)
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        # 保存模型
        save_path = os.path.join(args.save_dir, f"dpt_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

        # Evaluation
        print('Evaluating on validation set...')
        model.eval()
        translation_error = []

        with torch.no_grad():
            for sat, grd, gt_u, gt_v in tqdm(val_dataloader, desc="Testing"):
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
                
                pred_u_tensor = (max_indices % W_out - (W_out / 2 + 0.5))
                pred_v_tensor = (max_indices // W_out - (H_out / 2 + 0.5))
            
                scale_factor = 640.0 / Ws
                
                pred_u_final = pred_u_tensor * scale_factor / 630 * 640
                pred_v_final = pred_v_tensor * scale_factor / 630 * 640
                
                # 8. 计算欧氏距离误差
                distance_error = torch.sqrt((pred_u_final - gt_u)**2 + (pred_v_final - gt_v)**2)
                translation_error.extend(distance_error.cpu().numpy().tolist())

        trans_mean = np.mean(translation_error)
        writer.add_scalar('Validation/Translation_Mean', trans_mean, epoch)
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}, Translation Mean Error: {trans_mean:.4f}")
    writer.close()
if __name__ == "__main__":
    args = parse_args()
    train(args)

# python train.py --split samearea --batch_size 96 --lr 1e-4 --epochs 25 --save_dir ./runs/checkpoints