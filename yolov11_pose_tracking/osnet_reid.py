# osnet_reid.py
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from collections import OrderedDict
import pickle
from functools import partial

# 导入OSNet模型定义
from .OSNet import osnet_x0_25

class OSNetReID:
    """OSNet ReID特征提取器 - 完整版本"""
    def __init__(self, model_path, device='cuda'):
        """
        OSNet ReID特征提取器
        Args:
            model_path: osnet_x0_25.pth 模型路径
            device: 运行设备 ('cuda', 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        start_time = time.time()
        self.model = self._load_model_engine_style(model_path)
        model_load_time = (time.time() - start_time) * 1000
        print(f"模型加载耗时: {model_load_time:.2f}ms")
        
        # 预处理参数
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 模型设置为评估模式
        self.model.eval()
        print(f"模型加载成功: {model_path}")

    def _load_checkpoint(self, fpath):
        """加载checkpoint"""
        if fpath is None:
            raise ValueError('File path is None')
        fpath = os.path.abspath(os.path.expanduser(fpath))
        if not os.path.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))
        map_location = None if torch.cuda.is_available() else 'cpu'
        try:
            checkpoint = torch.load(fpath, map_location=map_location)
        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )
        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise
        return checkpoint

    def _load_model_engine_style(self, model_path):
        """使用与engine.py完全一致的模型加载方式"""
        # 创建标准OSNet模型
        model = osnet_x0_25(num_classes=1, pretrained=False)
        
        if os.path.exists(model_path):
            checkpoint = self._load_checkpoint(model_path)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            model_dict = model.state_dict()
            new_state_dict = OrderedDict()
            matched_layers, discarded_layers = [], []

            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # discard module.

                if k in model_dict and model_dict[k].size() == v.size():
                    new_state_dict[k] = v
                    matched_layers.append(k)
                else:
                    discarded_layers.append(k)

            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)

            if len(matched_layers) == 0:
                print(
                    '警告: 预训练权重"{}"无法加载，请手动检查键名'.format(model_path)
                )
            else:
                print('成功加载预训练权重从"{}"'.format(model_path))
                if len(discarded_layers) > 0:
                    print('** 以下层因不匹配的键或层大小而被丢弃: {}'.format(discarded_layers))
        else:
            print("警告: 未找到预训练权重，使用随机初始化")
        
        # 移除分类头，只保留特征提取部分
        model.classifier = nn.Identity()
        
        return model.to(self.device)

    def preprocess_image(self, image_path):
        """
        预处理图像
        """
        # 使用PIL读取图像
        img = Image.open(image_path).convert('RGB')
        
        # 应用转换
        img_tensor = self.transform(img)
        
        # 添加batch维度
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)

    def preprocess_cv2_image(self, cv2_img):
        """
        处理OpenCV图像
        """
        # 转换BGR到RGB
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 应用转换
        img_tensor = self.transform(img_pil)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device)

    def extract_features(self, image_path):
        """
        从图像文件提取特征向量
        """
        with torch.no_grad():
            # 预处理
            input_tensor = self.preprocess_image(image_path)
            
            # 前向传播
            features = self.model(input_tensor)
            
            # 特征归一化
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            # 转换为numpy数组
            features = features.cpu().numpy().squeeze()
            
        return features

    def extract_features_from_cv2(self, cv2_img):
        """
        从OpenCV图像提取特征
        """
        with torch.no_grad():
            input_tensor = self.preprocess_cv2_image(cv2_img)
            features = self.model(input_tensor)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            features = features.cpu().numpy().squeeze()
        return features

    def extract_feature(self, cv2_img):
        """
        从OpenCV图像提取特征 (简化接口)
        """
        return self.extract_features_from_cv2(cv2_img)

    def extract_features_batch(self, image_paths):
        """
        批量提取特征
        """
        features_list = []
        
        for img_path in image_paths:
            features = self.extract_features(img_path)
            features_list.append(features)
            
        return np.array(features_list)

    def calculate_similarity(self, features1, features2):
        """
        计算余弦相似度
        """
        # 确保是1D数组
        features1 = features1.flatten()
        features2 = features2.flatten()
        
        # 计算余弦相似度
        similarity = 1 - cosine(features1, features2)
        return similarity

    def compare_features(self, feat1, feat2):
        """
        比较两个特征向量的相似度
        """
        if np.linalg.norm(feat1) == 0 or np.linalg.norm(feat2) == 0:
            return 0.0
        return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))