import torch
import torch.nn as nn
import torch.nn.functional as F
class HeatmapLoss(nn.Module):
    def __init__(self, sigma=2.0, temperature=20.0): # 新增 temperature
        super().__init__()
        self.sigma = sigma
        self.temperature = temperature # 控制分布的尖锐程度
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def generate_gaussian_target(self, H, W, center_u, center_v):
        # ... (这也部分保持不变) ...
        cx_base = (W - 1) / 2.0
        cy_base = (H - 1) / 2.0
        cx = cx_base + center_u
        cy = cy_base + center_v
        
        x = torch.arange(0, W, 1, dtype=torch.float32)
        y = torch.arange(0, H, 1, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        heatmap = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * self.sigma**2))
        return heatmap / (heatmap.sum() + 1e-6)

    def forward(self, corr_map, gt_u, gt_v, scale_factor):
        """
        corr_map: [B, H, W] (Cosine Similarity, 范围 -1 到 1)
        gt_u, gt_v: [B] (真实偏移量)
        scale_factor: float 或 scalar tensor
        """
        B, H, W = corr_map.shape
        device = corr_map.device
        
        # 1. 创建网格
        # yy, xx 形状: [H, W]
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        
        # [关键修正] 增加 Batch 维度: [H, W] -> [1, H, W]
        xx = xx.unsqueeze(0) 
        yy = yy.unsqueeze(0)

        # 2. 计算中心点坐标 [B, 1, 1]
        cx_base = (W - 1) / 2.0
        cy_base = (H - 1) / 2.0
        
        # 确保 scale_factor 在正确设备上或转为 float
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(device)

        # gt_u, gt_v 形状 [B] -> [B, 1, 1]
        # 这里的假设是：gt_u 是相对于中心的偏移量 (shift)
        cx = cx_base + (gt_u / scale_factor).view(B, 1, 1)
        cy = cy_base + (gt_v / scale_factor).view(B, 1, 1)
        
        # 3. 广播计算高斯分布 [B, H, W]
        # 现在: [1,H,W] - [B,1,1] = [B,H,W] -> 广播成功
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        target_maps = torch.exp(-dist_sq / (2 * self.sigma**2))
        
        # 4. 归一化 (让每个样本的 Heatmap 概率和为 1)
        # 这一步对 KLDivLoss 至关重要
        target_maps = target_maps / (target_maps.sum(dim=(1,2), keepdim=True) + 1e-6)
        
        # 5. 计算 Loss
        # 乘 Temperature (例如 20.0) 拉伸分布
        scaled_corr = corr_map * self.temperature
        
        # LogSoftmax 
        log_prob = F.log_softmax(scaled_corr.view(B, -1), dim=1).view(B, H, W)
        
        return self.criterion(log_prob, target_maps)