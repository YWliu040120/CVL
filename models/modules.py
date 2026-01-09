import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from DINO_modules.dinov2 import vit_large,vit_base
from DINO_modules.dinov2_reg import vit_large_reg, vit_base_reg
import torch.nn.functional as F
class DinoExtractor_large(nn.Module):
    """
    DINOv2 Feature Extractor using a ViT-L/14 backbone.
    """

    def __init__(self, dinov2_weights=None):
        super().__init__()

        # Define DINOv2 extractor parameters
        self.dino_channels = 1024
        self.dino_downfactor = 14
        self.amp_dtype = torch.float16  # Define float precision

        if dinov2_weights is None:
            dinov2_weights = load_state_dict_from_url(
                "/ssd/liuyaowei/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth",
                map_location="cpu"
            )

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )

        self.dinov2_vitl14 = vit_large(**vit_kwargs)
        self.dinov2_vitl14.load_state_dict(dinov2_weights)
        self.dinov2_vitl14.requires_grad_(False)
        self.dinov2_vitl14.eval()
        self.dinov2_vitl14.to(self.amp_dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure spatial dimensions are divisible by dino_downfactor
        x = x[:, :, : self.dino_downfactor * (H // self.dino_downfactor),
                 : self.dino_downfactor * (W // self.dino_downfactor)]

        with torch.no_grad():
            features = self.dinov2_vitl14.forward_features(x.to(self.amp_dtype))
            features = features["x_norm_patchtokens"].permute(0, 2, 1).reshape(
                B, self.dino_channels, H // self.dino_downfactor, W // self.dino_downfactor
            ).float()

        return features
   
class DinoExtractor_base(nn.Module):
    """
    DINOv2 Feature Extractor using a ViT-base/14 backbone.
    """

    def __init__(self, dinov2_weights=None):
        super().__init__()

        # Define DINOv2 extractor parameters
        self.dino_channels = 768
        self.dino_downfactor = 14
        self.amp_dtype = torch.float16  # Define float precision

        if dinov2_weights is None:
            dinov2_weights = load_state_dict_from_url(
                "/ssd/liuyaowei/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
                map_location="cpu"
            )

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )

        self.dinov2_vitb14 = vit_base(**vit_kwargs)
        self.dinov2_vitb14.load_state_dict(dinov2_weights)
        self.dinov2_vitb14.requires_grad_(False)
        self.dinov2_vitb14.eval()
        self.dinov2_vitb14.to(self.amp_dtype)
    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure spatial dimensions are divisible by dino_downfactor
        x = x[:, :, : self.dino_downfactor * (H // self.dino_downfactor),
                 : self.dino_downfactor * (W // self.dino_downfactor)]

        with torch.no_grad():
            features = self.dinov2_vitb14.forward_features(x.to(self.amp_dtype))
            features = features["x_norm_patchtokens"].permute(0, 2, 1).reshape(
                B, self.dino_channels, H // self.dino_downfactor, W // self.dino_downfactor
            ).float()

        return features
    
class DinoExtractor_large_reg(nn.Module):
    """
    DINOv2 Feature Extractor using a ViT-L/14 backbone with Registers.
    """

    def __init__(self, dinov2_weights=None):
        super().__init__()

        # 1. 定义参数
        self.dino_channels = 1024
        self.dino_downfactor = 14
        self.amp_dtype = torch.float16

        if dinov2_weights is None:
            dinov2_weights = load_state_dict_from_url(
                "/ssd/liuyaowei/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
                map_location="cpu"
            )
        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=4,  
        )

        # 初始化模型
        self.dinov2_vitl14 = vit_large_reg(**vit_kwargs)
        
        self.dinov2_vitl14.load_state_dict(dinov2_weights)
        self.dinov2_vitl14.requires_grad_(False)
        self.dinov2_vitl14.eval()
        self.dinov2_vitl14.to(self.amp_dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x[:, :, : self.dino_downfactor * (H // self.dino_downfactor),
                  : self.dino_downfactor * (W // self.dino_downfactor)]

        with torch.no_grad():
            
            output_dict = self.dinov2_vitl14.forward_features(x.to(self.amp_dtype))
            features = output_dict["x_norm_patchtokens"]
            
            features = features.permute(0, 2, 1).reshape(
                B, self.dino_channels, H // self.dino_downfactor, W // self.dino_downfactor
            ).float()

        return features

class DinoExtractor_base_reg(nn.Module):
    
    def __init__(self, dinov2_weights=None):
        super().__init__()

        self.dino_channels = 768
        self.dino_downfactor = 14
        self.amp_dtype = torch.float16

        if dinov2_weights is None:
            dinov2_weights = load_state_dict_from_url(
                "/ssd/liuyaowei/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth",
                map_location="cpu"
            )

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            num_register_tokens=4, # 如果加载的是普通权重，这里设为 0
        )

        self.dinov2_vitb14 = vit_base_reg(**vit_kwargs)
        self.dinov2_vitb14.load_state_dict(dinov2_weights)
        self.dinov2_vitb14.requires_grad_(False)
        self.dinov2_vitb14.eval()
        self.dinov2_vitb14.to(self.amp_dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x[:, :, : self.dino_downfactor * (H // self.dino_downfactor),
                  : self.dino_downfactor * (W // self.dino_downfactor)]

        with torch.no_grad():
            output_dict = self.dinov2_vitb14.forward_features(x.to(self.amp_dtype))
            features = output_dict["x_norm_patchtokens"]
            
            features = features.permute(0, 2, 1).reshape(
                B, self.dino_channels, H // self.dino_downfactor, W // self.dino_downfactor
            ).float()

        return features

class DPTHead(nn.Module):
    def __init__(self, in_channels=1024, feature_dim=256, use_bn=False):
        super().__init__()
        
        self.projectors = nn.ModuleList([
            nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False)
            for _ in range(4)
        ])
        
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
        )

        self.upsample_head = nn.Sequential(
            # Upsample 2x
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
            
            # Upsample 2x (Total 4x) - 此时分辨率为原图的 4/14 ≈ 1/3.5
            nn.Conv2d(feature_dim // 2, feature_dim // 4, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_dim // 4, 32, kernel_size=1) 
        )

    def forward(self, features):
        x = self.projectors[3](features[3])

        x = x + self.projectors[2](features[2])
        
        x = x + self.projectors[1](features[1])

        x = x + self.projectors[0](features[0])

        x = self.refine(x)

        high_res_feature = self.upsample_head(x)
        
        return high_res_feature

class DinoV2VigorModel(nn.Module):
    def __init__(self, patch_size=14):
        super().__init__()
        self.dino_downfactor = 14  # DINOv2 特征下采样因子
        dinov2_weights = torch.load("/ssd/liuyaowei/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth", map_location="cpu")
        self.backbone = vit_large(
            patch_size=patch_size, 
            img_size=518,
            init_values=1.0, 
            block_chunks=0
        )
        self.backbone.load_state_dict(dinov2_weights)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        
        self.dpt_head = DPTHead(in_channels=1024, feature_dim=256)
        self.out_indices = [4, 11, 17, 23]

    def forward(self, x):
        
        B, C, H, W = x.shape
        x = x[:, :, : self.dino_downfactor * (H // self.dino_downfactor),
                 : self.dino_downfactor * (W // self.dino_downfactor)]
        
        features = self.backbone.get_intermediate_layers(
            x, 
            n=self.out_indices, 
            reshape=True,
            norm=True 
        )
        
       
        features = list(features)
        
        high_res_output = self.dpt_head(features)
        
        return high_res_output
