import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import shap
from MobileViT_v3_ref_CBAM import MobileViTv3_v2
import matplotlib.pyplot as plt


# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 加载PyTorch模型
model_path = "C:/Users/DELL/Desktop/project/learn-pytorch/美度模型/AesViT-Net/2cbam_refbackbone_best.pt"
checkpoint = torch.load(model_path, map_location=device)
model = MobileViTv3_v2((224, 224), 1, 3)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 加载类标签
idx_to_labels = np.load('class_labels.npy', allow_pickle=True).item()
print("Loaded class labels:", idx_to_labels)
class_names = list(idx_to_labels.keys())
print("Class names:", class_names)

# 载入一张测试图像，整理维度
img_path = r"C:\Users\DELL\Desktop\小论文\1 实验记录\11实验\grad-cam\mid\样本\7_chevrolet_malibu_2021.jpg"
img_pil = Image.open(img_path).resize((224, 224)).convert('RGB')
X = torch.Tensor(np.array(img_pil)).unsqueeze(0)
print(f"Original shape: {X.shape}")

# 预处理
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    print(f"Original shape (NHWC): {x.shape}")  # 打印原始形状
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    print(f"Converted shape (NCHW): {x.shape}")  # 打印转换后的形状
    return x

def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    print(f"Original shape (NCHW): {x.shape}")  # 打印原始形状
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    print(f"Converted shape (NHWC): {x.shape}")  # 打印转换后的形状
    return x

transform = transforms.Compose([
    transforms.Lambda(nhwc_to_nchw),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x * (1 / 255.0)),
    transforms.Normalize(mean=mean, std=std),
    transforms.Lambda(nchw_to_nhwc),
])

# 创建逆转换以便验证
inv_transform = transforms.Compose([
    transforms.Lambda(nhwc_to_nchw),
    transforms.Normalize(mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                         std=(1 / np.array(std)).tolist()),
    transforms.Lambda(nchw_to_nhwc),
])

# 应用变换
transformed_img = transform(X)
print(f"Transformed shape: {transformed_img.shape}")

# 应用逆变换以验证
inv_transformed_img = inv_transform(transformed_img)
print(f"Inverse Transformed shape: {inv_transformed_img.shape}")

# 构建模型预测函数
def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img)).to(device)
    output = model(img)
    return output

# 测试整个工作流正常
out = predict(transformed_img)
print("Output shape:", out.shape)  # 期望输出：torch.Size([1, 3])
classes = torch.argmax(out, axis=1).detach().cpu().numpy()
print(f'Predicted class index: {classes}, Class name: {np.array(class_names)[classes]}')

# 设置shap可解释性分析算法
input_img = transformed_img.numpy()
print(f"Input image shape: {input_img.shape}")  # 确认输入图像的形状
batch_size = 50
n_evals = 10000  # 迭代次数越大，显著性分析粒度越精细，计算消耗时间越长

# 定义 mask，遮盖输入图像上的局部区域
masker_blur = shap.maskers.Image("blur(64, 64)", input_img.shape[1:])
print(f"SHAP Explainer created with provided prediction function and masker")

explainer = shap.Explainer(predict, masker_blur, output_names=class_names)
print(f"Explainer Model Output Names: {explainer.output_names}")

# 指定多个预测类别
shap_values = explainer(input_img, max_evals=n_evals, batch_size=batch_size, outputs=[0, 1, 2])

# 整理张量维度
shap_values.data = inv_transform(torch.tensor(shap_values.data)).cpu().numpy()[0]  # 原图
shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]  # shap值热力图

# 打印shap值和原图形状以确认
print(f"shap_values.data shape: {shap_values.data.shape}")
print(f"shap_values.values shape: {shap_values.values[0].shape}")

# 可视化shap值热力图
shap.image_plot(shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names)

