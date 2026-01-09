import torch
import os
from PIL import Image, ImageDraw


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
