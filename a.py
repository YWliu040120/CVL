
from PIL import Image
def get_image_size(image_path):
    """获取图像的宽度和高度"""
    with Image.open(image_path) as img:
        return img.size  # 返回 (宽度, 高度)
    
# 示例用法
image_path = "lalala/curved_bev_output/__EPszp5486MewfwSMqmSQ,37.742475,-122.404157,.png"  # 替换为你的图像路径
width, height = get_image_size(image_path)
print(f"图像宽度: {width}, 图像高度: {height}")

# import os
# import shutil
# txt_path = '/zssd/dataset/liuyaowei/vigor/SanFrancisco_bev/curved_bev_output/marked/detailed_errors.txt'
# source_folder = '/zssd/dataset/liuyaowei/vigor/SanFrancisco_bev/curved_bev_output/down_image' 
# output_best_dir = '/ssd/liuyaowei/DA-2-main/error_analyze/good'   
# output_worst_dir = '/ssd/liuyaowei/DA-2-main/error_analyze/bad' 
# os.mkdir(output_best_dir)
# os.mkdir(output_worst_dir)
# marked_source_folder = '/zssd/dataset/liuyaowei/vigor/SanFrancisco_bev/curved_bev_output/marked'

# def process_errors_and_copy():
#     data_entries = []
#     if not os.path.exists(txt_path):
#         print(f"错误：找不到 txt 文件 {txt_path}")
#         return

#     print("正在读取详细错误文件...")
#     with open(txt_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#     content_lines = lines[1:]

#     for line in content_lines:
#         line = line.strip()
#         if not line:
#             continue
        
#         parts = line.split(',')
#         try:
#             error_val = float(parts[-5])
#             filename = ",".join(parts[:-5]) 
            
#             data_entries.append({
#                 'filename': filename,
#                 'error': error_val
#             })
#         except (ValueError, IndexError):
#             continue
#     sorted_data = sorted(data_entries, key=lambda x: x['error'])
#     best_10 = sorted_data[:10]
#     worst_10 = sorted_data[-10:]
#     print(best_10)
#     print(worst_10)
#     def copy_file_pair(file_list, target_dir, label):
#         if not os.path.exists(target_dir):
#             os.makedirs(target_dir)
        
#         print(f"\n=== 开始复制 {label} (包含 Original 和 Marked) ===")
#         count = 0
#         for item in file_list:
#             fname = item['filename']     # 原始文件名，如: xxx,37.1,122.1,.png
#             err = item['error']
            
#             # 构造 Marked 文件名
#             # 逻辑：将 ".png" 替换为 "_marked.png"
#             # 原始: "...,.png" -> 替换后: "...,_marked.png" (符合截图中的格式)
#             fname_marked = fname.replace('.png', '_marked.png')

#             # 源路径
#             src_origin = os.path.join(source_folder, fname)
#             src_marked = os.path.join(marked_source_folder, fname_marked)

#             # 目标路径 (保持原名)
#             dst_origin = os.path.join(target_dir, fname)
#             dst_marked = os.path.join(target_dir, fname_marked)

#             # 执行复制 - 原始图
#             try:
#                 shutil.copy2(src_origin, dst_origin)
#                 print(f"[OK] 原图: {fname} (Err: {err:.4f})")
#             except FileNotFoundError:
#                 print(f"[!!] 原图未找到: {src_origin}")

#             # 执行复制 - Marked 图
#             try:
#                 shutil.copy2(src_marked, dst_marked)
#                 print(f"     -> Marked图: {fname_marked}")
#             except FileNotFoundError:
#                 print(f"     [!!] Marked图未找到: {src_marked}")
            
#             count += 1
#         print(f"完成 {label} 处理。")
#     copy_file_pair(best_10, output_best_dir, "最小误差 Top 10")
#     copy_file_pair(worst_10, output_worst_dir, "最大误差 Top 10")

# if __name__ == "__main__":
#     process_errors_and_copy()



# import cv2
# import numpy as np

# # 读取原图和mask
# image = cv2.imread('assets/input_images/_0Ud0q5JrXcYKIpS8GZ0KA,37.776165,-122.403530,.jpg')
# mask = cv2.imread('assets/input_images_masked_car/_0Ud0q5JrXcYKIpS8GZ0KA,37.776165,-122.403530,.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度读取

# # 确保mask是二值或0-1范围的
# if mask.max() > 1:
#     mask = mask / 255.0  # 归一化到0-1

# # 扩展mask维度以匹配图片通道
# mask_3d = cv2.merge([mask, mask, mask])

# # 相乘
# result = image * mask_3d

# # 保存结果
# cv2.imwrite('masked_image.jpg', result)

#判断一个文件夹下多少图片
# import os

# def count_png_files(folder_path):
#     # 1. 检查文件夹是否存在
#     if not os.path.exists(folder_path):
#         print(f"❌ 错误：文件夹 '{folder_path}' 不存在")
#         return

#     count = 0
#     # 2. 遍历文件夹
#     # os.listdir 只列出当前文件夹下的文件，不包含子文件夹
#     try:
#         files = os.listdir(folder_path)
#         for filename in files:
#             # 3. 判断后缀 (转为小写比较，忽略大小写差异)
#             if filename.lower().endswith(".png"):
#                 count += 1
        
#         print(f"📂 文件夹: {folder_path}")
#         print(f"📊 PNG图片数量: {count} 张")
        
#     except Exception as e:
#         print(f"⚠️ 读取出错: {e}")

# # --- 使用方式 ---
# target_folder = "/zssd/dataset/liuyaowei/vigor/Chicago_bev/curved_bev_output/down_image/"  # 替换成你的文件夹路径
# count_png_files(target_folder)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from muon import MuonWithAuxAdam  # 假设这是您的自定义优化器

# class CompleteModel(nn.Module):
#     def __init__(self, num_classes=10, vocab_size=1000, embed_dim=64):
#         """
#         完整的神经网络模型，包含embedding、卷积、线性等层
        
#         参数:
#             num_classes: 分类类别数
#             vocab_size: 词汇表大小
#             embed_dim: embedding维度
#         """
#         super(CompleteModel, self).__init__()
        
#         # ========== Embedding 部分 ==========
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
        
#         # ========== 卷积部分 ==========
#         # 第一个卷积块
#         self.conv1 = nn.Conv2d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
        
#         # 第二个卷积块
#         self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
        
#         # 第三个卷积块
#         self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(512)
        
#         # 第四个卷积块（深度可分离卷积）
#         self.depthwise_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512)
#         self.pointwise_conv = nn.Conv2d(512, 256, kernel_size=1)
#         self.bn4 = nn.BatchNorm2d(256)
        
#         # 空间注意力机制
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(256, 1, kernel_size=7, padding=3),
#             nn.Sigmoid()
#         )
        
#         # ========== 池化层 ==========
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # ========== 全连接部分 ==========
#         # 线性层1
#         self.fc1 = nn.Linear(256, 128)
#         self.dropout1 = nn.Dropout(0.5)
        
#         # 线性层2
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout2 = nn.Dropout(0.3)
        
#         # 输出层
#         self.classifier = nn.Linear(64, num_classes)
        
#         # ========== 其他层 ==========
#         # 残差连接的1x1卷积
#         self.residual_conv = nn.Conv2d(embed_dim, 128, kernel_size=1)
        
#         # Layer Normalization
#         self.ln1 = nn.LayerNorm(128)
#         self.ln2 = nn.LayerNorm(256)
        
#         # 初始化权重
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         """初始化模型权重"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Embedding):
#                 nn.init.normal_(m.weight, 0, 0.01)
    
#     def forward(self, x, input_shape=(32, 32)):
#         """
#         前向传播
        
#         参数:
#             x: 输入tensor, shape: (batch_size, seq_len) 或 (batch_size,)
#             input_shape: 输入图像形状 (H, W)，默认32x32
            
#         返回:
#             output: 模型输出
#         """
#         batch_size = x.size(0)
        
#         # ========== Embedding ==========
#         # 假设输入是序列，先embedding
#         if x.dim() == 2:  # (batch_size, seq_len)
#             embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
#             # 转换为2D: 假设seq_len对应高度，embed_dim对应通道
#             # 这里需要根据实际情况调整
#             h, w = input_shape
#             seq_len = embedded.size(1)
#             embedded = embedded.view(batch_size, -1, h, w)
#         else:  # 如果已经是2D输入
#             embedded = self.embedding(x)  # 假设x是索引
            
#         # ========== 卷积块1 ==========
#         # 残差连接
#         residual = self.residual_conv(embedded)
        
#         x = self.conv1(embedded)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = x + residual  # 残差连接
#         x = self.max_pool(x)  # 下采样
        
#         # ========== 卷积块2 ==========
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.ln1(x.view(batch_size, 128, -1).transpose(1, 2)).transpose(1, 2).view(x.shape)
#         x = self.max_pool(x)
        
#         # ========== 卷积块3 ==========
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = self.max_pool(x)
        
#         # ========== 深度可分离卷积 ==========
#         x = self.depthwise_conv(x)
#         x = self.pointwise_conv(x)
#         x = self.bn4(x)
#         x = F.relu(x)
        
#         # ========== 空间注意力 ==========
#         attention = self.spatial_attention(x)
#         x = x * attention  # 应用注意力
        
#         # ========== 全局池化 ==========
#         x = self.pool(x)  # (batch, 256, 1, 1)
#         x = x.view(batch_size, -1)  # (batch, 256)
        
#         # ========== 全连接层 ==========
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout1(x)
        
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
        
#         # ========== 输出层 ==========
#         output = self.classifier(x)
        
#         return output


# def create_model_and_optimizer(num_classes=10, vocab_size=1000, embed_dim=64):
#     """
#     创建模型和优化器
    
#     返回:
#         model: 模型实例
#         optimizer: 优化器实例
#     """
#     # 创建模型
#     model = CompleteModel(num_classes=num_classes, vocab_size=vocab_size, embed_dim=embed_dim)
    
#     # 定义参数分组
#     # 1. 二维权重参数（使用Muon）
#     hidden_weights = [
#         p for name, p in model.named_parameters() 
#         if p.ndim >= 2 and not name.startswith('classifier')
#     ]
    
#     # 2. 一维参数（偏置、gain等，不使用Muon）
#     hidden_gains_biases = [
#         p for name, p in model.named_parameters()
#         if p.ndim < 2 and not name.startswith('classifier')
#     ]
    
#     # 3. 分类头和embedding参数（不使用Muon）
#     classifier_params = [p for name, p in model.named_parameters() if name.startswith('classifier')]
#     embedding_params = [p for name, p in model.named_parameters() if 'embedding' in name]
#     nonhidden_params = classifier_params + embedding_params
    
#     # 创建参数组
#     param_groups = [
#         # 卷积和全连接的权重使用Muon
#         {
#             'params': hidden_weights,
#             'use_muon': True,
#             'lr': 0.02,
#             'weight_decay': 0.01,
#             'betas': (0.9, 0.999)  # 如果需要覆盖默认值
#         },
#         # 偏置、BN参数、分类头、embedding不使用Muon
#         {
#             'params': hidden_gains_biases + nonhidden_params,
#             'use_muon': False,
#             'lr': 3e-4,
#             'betas': (0.9, 0.95),
#             'weight_decay': 0.01
#         }
#     ]
    
#     # 创建优化器
#     optimizer = MuonWithAuxAdam(param_groups)
    
#     return model, optimizer

# # 使用示例
# if __name__ == "__main__":
#     model, optimizer = create_model_and_optimizer(
#         num_classes=10,
#         vocab_size=1000,
#         embed_dim=64
#     )
#     # 打印模型结构
#     print("模型结构:")
#     print(model)
#     print("\n模型参数量:", sum(p.numel() for p in model.parameters()))
    
#     # 打印参数分组信息
#     print("\n优化器参数分组:")
#     for i, group in enumerate(optimizer.param_groups):
#         num_params = sum(p.numel() for p in group['params'])
#         print(f"组 {i}:")
#         print(f"  使用Muon: {group.get('use_muon', False)}")
#         print(f"  学习率: {group.get('lr', 'N/A')}")
#         print(f"  参数数量: {num_params}")
#         print(f"  参数示例: {[p.shape for p in group['params'][:2]] if group['params'] else '无参数'}")
    
#     # 测试前向传播
#     batch_size = 4
#     seq_len = 100
#     dummy_input = torch.randint(0, 1000, (batch_size, seq_len))
    
#     print("\n测试前向传播:")
#     output = model(dummy_input, input_shape=(10, 10))  # 假设10x10特征图
#     print(f"输入形状: {dummy_input.shape}")
#     print(f"输出形状: {output.shape}")
#     print(f"输出示例:\n{output}")
    
#     # 测试训练步骤
#     print("\n测试训练步骤:")
#     # 模拟损失
#     target = torch.randint(0, 10, (batch_size,))
#     criterion = nn.CrossEntropyLoss()
#     loss = criterion(output, target)
    
#     # 反向传播
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     print(f"损失值: {loss.item():.4f}")
#     print("训练步骤完成!")
