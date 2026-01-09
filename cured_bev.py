# import os
# import torch
# import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# from torchvision.utils import save_image
# from tqdm import tqdm
# from PIL import ImageOps

# def grid_sample(image, optical, jac=None):
#     # Interpolation function 

#     N, C, IH, IW = image.shape  # Extracting dimensions from the image tensor
#     _, H, W, _ = optical.shape  # Extracting dimensions from the optical tensor

#     ix = optical[..., 0].view(N, 1, H, W)  
#     iy = optical[..., 1].view(N, 1, H, W)  

#     with torch.no_grad():
#         ix_nw = torch.floor(ix)  
#         iy_nw = torch.floor(iy)  
#         ix_ne = ix_nw + 1        
#         iy_ne = iy_nw            
#         ix_sw = ix_nw            
#         iy_sw = iy_nw + 1        
#         ix_se = ix_nw + 1        
#         iy_se = iy_nw + 1       

#         # Clamp coordinates to be within valid range
#         torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
#         torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

#         torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
#         torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

#         torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
#         torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

#         torch.clamp(ix_se, 0, IW - 1, out=ix_se)
#         torch.clamp(iy_se, 0, IH - 1, out=iy_se)

#     # Create masks for valid coordinates
#     mask_x = (ix >= 0) & (ix <= IW - 1)
#     mask_y = (iy >= 0) & (iy <= IH - 1)
#     mask = mask_x * mask_y

#     assert torch.sum(mask) > 0  # Ensure that there are valid coordinates

#     # Calculate the weights for interpolation
#     nw = (ix_se - ix) * (iy_se - iy) * mask
#     ne = (ix - ix_sw) * (iy_sw - iy) * mask
#     sw = (ix_ne - ix) * (iy - iy_ne) * mask
#     se = (ix - ix_nw) * (iy - iy_nw) * mask

#     # Flatten the image for easier indexing
#     image = image.view(N, C, IH * IW)

#     # Gather the values at the four corners
#     nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
#     ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
#     sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
#     se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

#     # Perform bilinear interpolation
#     out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

#     if jac is not None:
#         # Calculate the gradients with respect to x and y
#         dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
#                     sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
#         dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
#                     sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
#         dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)  # [N, C, H, W, 2]

#         # Combine with the jacobian if provided
#         jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
#         jac_new1 = torch.sum(jac_new, dim=-1)

#         return out_val, jac_new1  # Return the interpolated values and updated jacobian
#     else:
#         return out_val, None  # Return only the interpolated values if no jacobian is provided

# def BEV_transform(rot, B, S, H, W, meter_per_pixel, Camera_height):

#     # This function performs BEV conversion and establishes the mapping relationships between different coordinates.
#     # Create a meshgrid for coordinates
#     ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=rot.device), 
#                             torch.arange(0, S, dtype=torch.float32, device=rot.device), indexing='ij')
#     ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # Expand and repeat for batch size
#     jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # Expand and repeat for batch size
    
#     # Height extension
#     # This is the key factor that needs to be modified
#     max_dist = torch.sqrt(torch.tensor(2*(S/2)**2, dtype=torch.float32, device=rot.device))
#     dist_to_center = torch.sqrt((ii - S/2)**2 + (jj - S/2)**2)
#     normalized_dist = dist_to_center / max_dist
#     h = 3 * (normalized_dist**4)

#     # Calculate the radius from the center of the grid
#     radius = torch.sqrt((ii - (S / 2 - 0.5)) ** 2 + (jj - (S / 2 - 0.5)) ** 2)
    
#     # Calculate the angle (theta) in radians
#     theta = torch.atan2(ii - (S / 2 - 0.5), jj - (S / 2 - 0.5))
    
#     # Normalize theta to be within the range [0, 2π)
#     theta = (-np.pi / 2 + theta % (2 * np.pi)) % (2 * np.pi)
    
#     # Adjust theta by the rotation and map to image width (W) 
#     theta = (theta + rot[:, None, None] * np.pi) % (2 * np.pi)
#     theta = theta / (2 * np.pi) * W
    
#     # Calculate the minimum elevation angle (phimin)
#     meter_per_pixel_tensor = torch.full((B, 1, 1), meter_per_pixel, device=radius.device)
#     phimin = torch.atan2(radius * meter_per_pixel_tensor, torch.tensor(Camera_height, device=radius.device) + h)
#     phimin = phimin / np.pi * H  # Map to image height (H)

#     # Stack theta and phimin to create UV coordinates
#     uv = torch.stack([theta, phimin], dim=-1)
    
#     return uv  # Return the UV coordinates

# def resize_and_pad_image(image, target_height, target_width):

#     # For CVUSA
#     # Get the current dimensions of the image
#     current_width, current_height = image.size

#     # Calculate the padding needed for height
#     if current_height == target_height:
#         padded_image = image
#     else:
#         # Calculate the padding needed for the top and bottom
#         top_padding = (target_height - current_height) // 2
#         bottom_padding = target_height - current_height - top_padding
#         # Use Pillow's ImageOps.expand to add padding
#         padded_image = ImageOps.expand(image, (0, top_padding, 0, bottom_padding), fill='black')

#     # Resize the image to the target dimensions
#     resized_image = padded_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
#     return resized_image

# def process_images(input_folder, output_folder, uv, H, W, B, resize_and_pad=False):
#     # This function performs BEV conversion in batch mode.

#     # Define transformation: resize and convert to tensor
#     transform = transforms.Compose([
#         transforms.Resize((H, W)),
#         transforms.ToTensor(),
#     ])
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # List all JPG files in the input directory
#     files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

#     # Process images in batches
#     for i in tqdm(range(0, len(files), B), desc="Processing images"):
#         # Adjust B to the remaining file count if necessary
#         current_batch_size = min(B, len(files) - i)
#         batch_files = files[i:i + current_batch_size]
#         batch_images = []
#         batch_output_paths = []

#         # Load and transform images
#         for file in batch_files:
#             file_path = os.path.join(input_folder, file)
            
#             # --- 修改开始 ---
#             # 1. 获取不带后缀的文件名
#             file_name_no_ext = os.path.splitext(file)[0]
#             # 2. 拼接输出路径，强制指定后缀为 .png
#             output_path = os.path.join(output_folder, file_name_no_ext + '.png')
#             # --- 修改结束 ---

#             image = Image.open(file_path)

#             # Apply resize and pad preprocessing if the flag is set to True
#             if resize_and_pad:
#                 image = resize_and_pad_image(image, H, W)

#             image_tensor = transform(image)
#             batch_images.append(image_tensor)
#             batch_output_paths.append(output_path)

#         # Stack images into a batch tensor
#         batch_images_tensor = torch.stack(batch_images)

#         # Apply grid sampling transformation
#         transformed_images, _ = grid_sample(batch_images_tensor, uv)

#         # Save transformed images
#         for j, transformed_image in enumerate(transformed_images):
#             # torchvision.utils.save_image 会根据 output_path 的后缀自动保存为 PNG 格式
#             save_image(transformed_image, batch_output_paths[j])

# def main():
#     input_folder = '/ssd/liuyaowei/FG2/vigor_datasets/Chicago/panorama'
#     output_folder = '/zssd/dataset/liuyaowei/vigor/Chicago_bev/curved_bev_output/down_image'

#     B = 100           
#     S = 500          
#     H = 1024          
#     W = 2048
#     Camera_height = -1.5 

#     # Create a rotation tensor with all values set to 90 degrees (If North is in the center of Street View)
#     rot = torch.tensor([90] * B, dtype=torch.float32)

#     # Define the scale of meters per pixel （CVACT）
#     meter_per_pixel = 0.06

#     # Compute UV coordinates for the satellite to ground transformation
#     uv = BEV_transform(rot, B, S, H, W, meter_per_pixel, Camera_height)

#     # Process images using the computed UV coordinates
#     process_images(input_folder, output_folder, uv, H, W, B)

# if __name__ == "__main__":
#     main()

# version 2 :we can use dataloader

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

def grid_sample(image, optical, jac=None):
    # Interpolation function 

    N, C, IH, IW = image.shape  # Extracting dimensions from the image tensor
    _, H, W, _ = optical.shape  # Extracting dimensions from the optical tensor

    ix = optical[..., 0].view(N, 1, H, W)  
    iy = optical[..., 1].view(N, 1, H, W)  

    with torch.no_grad():
        ix_nw = torch.floor(ix)  
        iy_nw = torch.floor(iy)  
        ix_ne = ix_nw + 1        
        iy_ne = iy_nw            
        ix_sw = ix_nw            
        iy_sw = iy_nw + 1        
        ix_se = ix_nw + 1        
        iy_se = iy_nw + 1       

        # Clamp coordinates to be within valid range
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    # Create masks for valid coordinates
    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask = mask_x * mask_y

    assert torch.sum(mask) > 0  # Ensure that there are valid coordinates

    # Calculate the weights for interpolation
    nw = (ix_se - ix) * (iy_se - iy) * mask
    ne = (ix - ix_sw) * (iy_sw - iy) * mask
    sw = (ix_ne - ix) * (iy - iy_ne) * mask
    se = (ix - ix_nw) * (iy - iy_nw) * mask

    # Flatten the image for easier indexing
    image = image.view(N, C, IH * IW)

    # Gather the values at the four corners
    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

    # Perform bilinear interpolation
    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:
        # Calculate the gradients with respect to x and y
        dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
                    sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
        dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
                    sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
        dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)  # [N, C, H, W, 2]

        # Combine with the jacobian if provided
        jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
        jac_new1 = torch.sum(jac_new, dim=-1)

        return out_val, jac_new1  # Return the interpolated values and updated jacobian
    else:
        return out_val, None  # Return only the interpolated values if no jacobian is provided

def BEV_transform(rot, B, S, H, W, meter_per_pixel, Camera_height):

    # This function performs BEV conversion and establishes the mapping relationships between different coordinates.
    # Create a meshgrid for coordinates
    ii, jj = torch.meshgrid(torch.arange(0, S, dtype=torch.float32, device=rot.device), 
                            torch.arange(0, S, dtype=torch.float32, device=rot.device), indexing='ij')
    ii = ii.unsqueeze(dim=0).repeat(B, 1, 1)  # Expand and repeat for batch size
    jj = jj.unsqueeze(dim=0).repeat(B, 1, 1)  # Expand and repeat for batch size
    
    # Height extension
    # This is the key factor that needs to be modified
    max_dist = torch.sqrt(torch.tensor(2*(S/2)**2, dtype=torch.float32, device=rot.device))
    dist_to_center = torch.sqrt((ii - S/2)**2 + (jj - S/2)**2)
    normalized_dist = dist_to_center / max_dist
    h = 3 * (normalized_dist**4)

    # Calculate the radius from the center of the grid
    radius = torch.sqrt((ii - (S / 2 - 0.5)) ** 2 + (jj - (S / 2 - 0.5)) ** 2)
    
    # Calculate the angle (theta) in radians
    theta = torch.atan2(ii - (S / 2 - 0.5), jj - (S / 2 - 0.5))
    
    # Normalize theta to be within the range [0, 2π)
    theta = (-np.pi / 2 + theta % (2 * np.pi)) % (2 * np.pi)
    
    # Adjust theta by the rotation and map to image width (W) 
    theta = (theta + rot[:, None, None] * np.pi) % (2 * np.pi)
    theta = theta / (2 * np.pi) * W
    
    # Calculate the minimum elevation angle (phimin)
    meter_per_pixel_tensor = torch.full((B, 1, 1), meter_per_pixel, device=radius.device)
    phimin = torch.atan2(radius * meter_per_pixel_tensor, torch.tensor(Camera_height, device=radius.device) + h)
    phimin = phimin / np.pi * H  # Map to image height (H)

    # Stack theta and phimin to create UV coordinates
    uv = torch.stack([theta, phimin], dim=-1)
    
    return uv  # Return the UV coordinates

class BEVDataset(Dataset):
    def __init__(self, input_folder, output_folder, H, W):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.input_folder, file_name)
        
        # 生成保存路径
        file_name_no_ext = os.path.splitext(file_name)[0]
        output_path = os.path.join(self.output_folder, file_name_no_ext + '.png')
        
        # 读取并预处理
        image = Image.open(file_path)
        image_tensor = self.transform(image)
        
        return image_tensor, output_path

def main():
    # 路径设置
    input_folder = '/ssd/liuyaowei/FG2/vigor_datasets/Chicago/panorama'
    output_folder = '/zssd/dataset/liuyaowei/vigor/Chicago_bev/curved_bev_output/down_image'
    os.makedirs(output_folder, exist_ok=True)

    # 参数设置
    B = 200              # Batch Size 可以设置大一点
    num_workers = 0  
    S = 500
    H = 1024
    W = 2048
    Camera_height = -1.5
    meter_per_pixel = 0.06
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 准备 UV 坐标 (只需计算一次)
    # 注意：这里我们生成一个 B=1 的 uv，后面在循环里 repeat 或者广播即可，
    # 或者直接生成 batch 大小的 uv 也可以，只要 device 对就行。
    # 为了简单，我们让 BEV_transform 生成 batch 为 B 的 uv
    rot = torch.tensor([90] * B, dtype=torch.float32).to(device)
    uv = BEV_transform(rot, B, S, H, W, meter_per_pixel, Camera_height) # 确保你的函数里 device 处理是对的
    # 如果你的 BEV_transform 内部创建 tensor 没指定 device，可能需要修改一下，或者手动 .to(device)
    uv = uv.to(device)

    # 2. 构建 DataLoader
    dataset = BEVDataset(input_folder, output_folder, H, W)
    
    # pin_memory=True 可以加快 CPU 到 GPU 的传输
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"开始处理，使用 Batch Size: {B}, Workers: {num_workers}")

    # 3. 批次处理
    for batch_images, batch_paths in tqdm(dataloader):
        # 将数据搬到 GPU
        batch_images = batch_images.to(device)
        
        # 动态处理最后一个 batch 大小可能不足 B 的情况
        current_batch_size = batch_images.size(0)
        if current_batch_size != B:
            # 重新生成对应大小的 uv，或者切片
            rot_temp = torch.tensor([90] * current_batch_size, dtype=torch.float32).to(device)
            uv_temp = BEV_transform(rot_temp, current_batch_size, S, H, W, meter_per_pixel, Camera_height).to(device)
            transformed_images, _ = grid_sample(batch_images, uv_temp)
        else:
            transformed_images, _ = grid_sample(batch_images, uv)

        # 保存图片 (这一步依然是 CPU 串行，但因为读取变快了，整体会变快)
        # 即使这里慢，前面多进程已经开始读下一批数据了
        for j, img_tensor in enumerate(transformed_images):
             save_image(img_tensor, batch_paths[j])

if __name__ == "__main__":
    # Windows 下多进程需要这个保护
    main()