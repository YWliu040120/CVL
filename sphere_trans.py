import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from sphere_transform.utils import get_BEV_tensor, get_BEV_projection

# 设置当前工作目录
current_directory = os.getcwd()
if current_directory.split('/')[-1] == 'demo':
    os.chdir('..')

# 配置参数
input_folder = "assets/input_images"  # 输入图片文件夹路径
output_folder = "lalala/sphere_output"  # BEV输出文件夹路径

image_extensions = ['jpg','png']
os.makedirs(output_folder, exist_ok=True)
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_folder, f"*.{ext}")))
    image_files.extend(glob.glob(os.path.join(input_folder, f"*.{ext.upper()}")))

print(f"找到 {len(image_files)} 张图片")

# 初始化统计数据
success_count = 0
failed_files = []

# 使用tqdm创建进度条
with tqdm(total=len(image_files), desc="处理进度", unit="张", 
          bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
    
    for i, img_path in enumerate(image_files):
        try:
            # 更新进度条描述显示当前处理的文件名
            pbar.set_postfix({
                "当前文件": os.path.basename(img_path)[:20] + "..." if len(os.path.basename(img_path)) > 20 else os.path.basename(img_path)
            })
            
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"\n警告: 无法读取图片 {os.path.basename(img_path)}, 跳过")
                failed_files.append((img_path, "无法读取"))
                pbar.update(1)
                continue
            
            # 转换颜色空间: BGR -> RGB
            img = img[:, :, ::-1]
            
            # BEV处理
            out = get_BEV_projection(img, 500, 500, Fov=160, dty=0, dx=0, dy=-10)
            BEV = get_BEV_tensor(img, 500, 500, Fov=80 * 2, dty=0, dx=0, dy=0, out=out).cpu().numpy().astype(np.uint8)
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}.png")
            
            # 保存BEV图像
            if len(BEV.shape) == 3 and BEV.shape[2] == 3:
                BEV_bgr = BEV[:, :, ::-1]  # RGB -> BGR
                cv2.imwrite(output_path, BEV_bgr)
            else:
                cv2.imwrite(output_path, BEV)
                
            success_count += 1
            
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 50:
                error_msg = error_msg[:50] + "..."
            failed_files.append((img_path, error_msg))
            pbar.set_postfix({"错误": error_msg})
        
        # 更新进度条
        pbar.update(1)

# 输出总结报告
print("\n" + "="*60)
print(f"✅ BEV转换完成!")
print(f"📊 处理统计:")
print(f"   总计图片: {len(image_files)} 张")
print(f"   成功转换: {success_count} 张")
print(f"   处理失败: {len(failed_files)} 张")

if failed_files:
    print(f"\n❌ 失败文件列表 (前5个):")
    for i, (file_path, error) in enumerate(failed_files[:5]):
        print(f"   {i+1}. {os.path.basename(file_path)}: {error}")
    if len(failed_files) > 5:
        print(f"   ... 还有 {len(failed_files)-5} 个失败项")

print(f"\n📁 输出目录: {os.path.abspath(output_folder)}")
if success_count > 0:
    print("🎉 已成功生成BEV图像!")
