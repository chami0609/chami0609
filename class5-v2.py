#MLP-mixerの多クラス分類をワンストップで行う（改良版）
# 混同行列、fold毎の評価、閾値最適化、ハイパーパラメータ探索機能を追加

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from tqdm import tqdm
import warnings
import itertools
import joblib
import gc
from datetime import datetime
from glob import glob
from math import floor
from pathlib import Path
from shapely.geometry import box, Point
import geopandas as gpd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (mean_squared_error, accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, 
                           classification_report, roc_auc_score, balanced_accuracy_score)
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import json
# Machine Learning ---------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import optuna
from optuna.samplers import TPESampler
# import h3pandas
import rasterio
import rasterio.features as rasioftr
warnings.filterwarnings('ignore')


import geopandas as gpd
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import hashlib
from datetime import datetime
from shapely.geometry import Point
warnings.filterwarnings('ignore')


# *******************************************************************
# Data Processing Class for Landslide Image Chips (tip-based)
# *******************************************************************
class LandslideTipProcessor:
    def __init__(self, rgb_paths, landslide_path, legacy_path, epsg_code=6675, resolution=0.2):
        self.rgb_paths = rgb_paths if isinstance(rgb_paths, list) else [rgb_paths]
        self.epsg_code = epsg_code
        self.resolution = resolution # m

        # Load polygons
        self.landslide_gdf = gpd.read_file(landslide_path)
        if self.landslide_gdf.crs != f"EPSG:{epsg_code}":
            self.landslide_gdf.to_crs(epsg=epsg_code, inplace=True)
        self.landslide_gdf['landslide_id'] = range(len(self.landslide_gdf))

        self.legacy_gdf = gpd.read_file(legacy_path)
        if self.legacy_gdf.crs != f"EPSG:{epsg_code}":
            self.legacy_gdf.to_crs(epsg=epsg_code, inplace=True)

        assert self.legacy_gdf.crs == self.landslide_gdf.crs


    def _load_raster_data(self, filepath):
        with rasterio.open(filepath) as src:
            data = src.read()
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data.squeeze()
        return data, transform, crs, (bounds.left, bounds.bottom, bounds.right, bounds.top)

    def _rasterize_polygons(self, gdf, id_col, rgb_ras, rgb_transform):
        shapes = [(geom, i) for i, geom in zip(gdf[id_col], gdf.geometry)]
        return rasioftr.rasterize(shapes, out_shape=rgb_ras.shape[:2], transform=rgb_transform, fill=np.nan, all_touched=True)

    def _pixel_to_coord(self, col, row, rgb_transform):
        x, y = rgb_transform * (col + 0.5, row + 0.5)
        return x, y

    def extract_tips(self, tip_size_m=5, slide_ratio=0.5):
        tip_size_grids = int(tip_size_m / self.resolution)
        l_buf = tip_size_grids // 2
        u_buf = tip_size_grids - l_buf - 1
        n_slide = max(1, int(tip_size_grids * slide_ratio))

        tips = []
        labels = []
        pixel_locations = []
        geo_locations = []
        landslide_ids = []

        for rgb_path in self.rgb_paths:
            print(f"Processing {rgb_path}...")
            # Load raster
            rgb_ras, rgb_transform, rgb_crs, rgb_bounds = self._load_raster_data(rgb_path)
            aoi_size = rgb_ras.shape[:2]
            i_range = range(l_buf, aoi_size[1] - u_buf, n_slide)
            j_range = range(l_buf, aoi_size[0] - u_buf, n_slide)
            # Clip polygons
            extent_gdf = gpd.GeoDataFrame(geometry=[box(*rgb_bounds)], crs=rgb_crs)
            landslide_aoi_gdf = gpd.clip(self.landslide_gdf, extent_gdf)
            legacy_aoi_gdf = gpd.clip(self.legacy_gdf, extent_gdf)
            # Rasterize
            landslide_ras = self._rasterize_polygons(landslide_aoi_gdf, 'landslide_id', rgb_ras, rgb_transform)
            legacy_ras = self._rasterize_polygons(legacy_aoi_gdf, 'id', rgb_ras, rgb_transform)
            legacy_ras = np.where(np.isnan(landslide_ras), np.nan, legacy_ras)
            # Create tips
            with tqdm(total=len(i_range) * len(j_range), desc="Extracting tips") as pbar:
                for i in i_range:
                    for j in j_range:
                        if np.isnan(landslide_ras[j, i]):
                            pbar.update(1)
                            continue

                        landslide_id = int(landslide_ras[j, i])
                        legacy_type = 0 if np.isnan(legacy_ras[j, i]) else int(legacy_ras[j, i])

                        rgb_tip = rgb_ras[j - l_buf:j + u_buf + 1, i - l_buf:i + u_buf + 1, :]
                        legacy_tip = legacy_ras[j - l_buf:j + u_buf + 1, i - l_buf:i + u_buf + 1]
                        is_legacy = (legacy_tip > 0) & ~((rgb_tip[:, :, 0] == 0) & (rgb_tip[:, :, 1] == 0) & (rgb_tip[:, :, 2] == 0))
                        legacy_ratio = round(np.sum(is_legacy) / (tip_size_grids ** 2), 5)

                        if rgb_tip.shape[0] != tip_size_grids or rgb_tip.shape[1] != tip_size_grids:
                            continue

                        tips.append(rgb_tip.astype(np.uint8))
                        labels.append(legacy_ratio)
                        pixel_locations.append((i, j))
                        geo_locations.append(self._pixel_to_coord(i, j, rgb_transform))
                        landslide_ids.append(landslide_id)

                        pbar.update(1)

        return np.array(tips), np.array(labels), np.array(pixel_locations), np.array(geo_locations), np.array(landslide_ids)


# *******************************************************************
# 閾値最適化クラス
# *******************************************************************
class ThresholdOptimizer:
    """クラス分類の閾値を最適化するクラス"""
    
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.thresholds = None
        self.best_thresholds = None
        self.best_score = -np.inf
    
    def label_to_class(self, labels, thresholds):
        """連続値ラベルをクラスラベルに変換"""
        classes = np.zeros(len(labels), dtype=int)
        for i, label in enumerate(labels):
            for j, threshold in enumerate(thresholds):
                if label <= threshold:
                    classes[i] = j
                    break
            else:
                classes[i] = len(thresholds)
        return classes
    
    def optimize_thresholds(self, y_true_continuous, y_pred_probs, metric='f1_macro'):
        """
        閾値を最適化
        y_true_continuous: 連続値の真のラベル（0.0～1.0）
        y_pred_probs: モデルの予測確率 (n_samples, n_classes)
        metric: 最適化する指標 ('f1_macro', 'accuracy', 'balanced_accuracy')
        """
        from scipy.optimize import differential_evolution
        
        def objective(thresholds):
            # 閾値を昇順にソート
            thresholds = sorted(thresholds)
            
            # 連続値ラベルをクラスラベルに変換
            y_true_classes = self.label_to_class(y_true_continuous, thresholds)
            y_pred_classes = np.argmax(y_pred_probs, axis=1)
            
            # 評価指標を計算
            if metric == 'f1_macro':
                score = f1_score(y_true_classes, y_pred_classes, average='macro')
            elif metric == 'accuracy':
                score = accuracy_score(y_true_classes, y_pred_classes)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true_classes, y_pred_classes)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            return -score  # 最小化問題に変換
        
        # 初期閾値の範囲を設定
        bounds = [(0.0, 1.0) for _ in range(self.n_classes - 1)]
        
        # 差分進化法で最適化
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            seed=42,
            disp=False
        )
        
        self.best_thresholds = sorted(result.x)
        self.best_score = -result.fun
        
        return self.best_thresholds, self.best_score


# *******************************************************************
# PyTorch Dataset（閾値可変対応版）
# *******************************************************************
class LandslideDataset(Dataset):
    """PyTorch用のデータセットクラス（閾値可変対応）"""
    
    def __init__(self, images, labels, pixel_locations, geo_locations, 
                 region_idx=None, transform=None, patch_size=25, 
                 thresholds=None, n_classes=5):
        self.images = images
        self.labels = labels  # 連続値ラベル（0.0～1.0）
        self.pixel_locations = pixel_locations
        self.geo_locations = geo_locations
        self.region_idx = region_idx
        self.transform = transform
        self.patch_size = patch_size
        self.n_classes = n_classes
        
        # デフォルトの閾値設定
        if thresholds is None:
            if n_classes == 5:
                self.thresholds = [0.0, 0.25, 0.5, 0.75]
            elif n_classes == 3:
                self.thresholds = [0.0, 0.5]
            elif n_classes == 2:
                self.thresholds = [0.5]
            else:
                # 等間隔で分割
                self.thresholds = [i / n_classes for i in range(1, n_classes)]
        else:
            self.thresholds = thresholds

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # 画像を必ずpatch_size x patch_sizeにリサイズ
        if image.shape[0] != self.patch_size or image.shape[1] != self.patch_size:
            image = Image.fromarray(image)
            image = image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            image = np.array(image)
        if len(image.shape) == 3:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        else:
            image = np.stack([image, image, image], axis=2)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if self.transform:
            image = self.transform(image)
        
        # 閾値に基づいてクラスラベルを生成
        raw_label = self.labels[idx]
        class_label = 0
        for i, threshold in enumerate(self.thresholds):
            if raw_label <= threshold:
                class_label = i
                break
        else:
            class_label = len(self.thresholds)
        
        result = {
            'image': image,
            'label': torch.tensor(class_label, dtype=torch.long),
            'raw_label': torch.tensor(raw_label, dtype=torch.float32),  # 連続値ラベルも保持
            'i': self.pixel_locations[idx][0],
            'j': self.pixel_locations[idx][1],
            'lon': self.geo_locations[idx][0],
            'lat': self.geo_locations[idx][1],
        }
        if self.region_idx is not None:
            result['region_idx'] = self.region_idx[idx]
        return result


# *******************************************************************
# MLP-Mixer Model
# *******************************************************************
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, num_classes, dim, depth, token_dim, channel_dim, task='classification'):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size // patch_size) ** 2
        self.task = task

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)


# *******************************************************************
# CNN Model
# *******************************************************************
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, hidden_dim=128, depth=4, kernel_size=3):
        super(CNN, self).__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(depth):
            out_channels = hidden_dim * (2 ** i)
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            if i % 2 == 1:  # プーリングを隔層に適用
                layers.append(nn.MaxPool2d(2))
            current_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# *******************************************************************
# Vision Transformer Model
# *******************************************************************
class ViT(nn.Module):
    def __init__(self, image_size=25, patch_size=5, in_channels=3, num_classes=5,
                 dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1):
        super(ViT, self).__init__()
        
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size.'
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
    
    def forward(self, img):
        p = self.patch_size
        
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


# *******************************************************************
# 混同行列の可視化関数
# *******************************************************************
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', 
                         save_path=None, normalize=False):
    """
    混同行列を可視化する関数
    
    Parameters:
    - cm: 混同行列
    - classes: クラス名のリスト
    - title: グラフのタイトル
    - save_path: 保存先パス（Noneの場合は保存しない）
    - normalize: 正規化するかどうか
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# *******************************************************************
# 評価メトリクスを計算する関数
# *******************************************************************
def calculate_metrics(y_true, y_pred, y_prob=None, classes=None):
    """
    各種評価メトリクスを計算
    
    Returns:
    - metrics: 評価指標の辞書
    - cm: 混同行列
    """
    metrics = {}
    
    # 基本的な精度指標
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # クラス別の指標
    metrics['precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # マクロ平均
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 加重平均
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # AUC（確率が利用可能な場合）
    if y_prob is not None and len(np.unique(y_true)) > 2:
        try:
            metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            metrics['auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
        except:
            metrics['auc_ovr'] = None
            metrics['auc_ovo'] = None
    
    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm


# *******************************************************************
# モデル初期化関数
# *******************************************************************
def init_model(model_name, hyperparams, n_classes=5):
    """
    モデルを初期化する関数
    
    Parameters:
    - model_name: 'MLP-Mixer', 'CNN', 'ViT'
    - hyperparams: ハイパーパラメータの辞書
    - n_classes: クラス数
    """
    if model_name == 'MLP-Mixer':
        model = MLPMixer(
            in_channels=3,
            image_size=25,
            patch_size=hyperparams.get('patch_size', 5),
            num_classes=n_classes,
            dim=hyperparams.get('dim', 512),
            depth=hyperparams.get('depth', 12),
            token_dim=hyperparams.get('token_dim', 256),
            channel_dim=hyperparams.get('channel_dim', 2048)
        )
    elif model_name == 'CNN':
        model = CNN(
            in_channels=3,
            num_classes=n_classes,
            hidden_dim=hyperparams.get('hidden_dim', 128),
            depth=hyperparams.get('depth', 4),
            kernel_size=hyperparams.get('kernel_size', 3)
        )
    elif model_name == 'ViT':
        model = ViT(
            image_size=25,
            patch_size=hyperparams.get('patch_size', 5),
            in_channels=3,
            num_classes=n_classes,
            dim=hyperparams.get('dim', 256),
            depth=hyperparams.get('depth', 6),
            heads=hyperparams.get('heads', 8),
            mlp_dim=hyperparams.get('mlp_dim', 512),
            dropout=hyperparams.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


# *******************************************************************
# ハイパーパラメータ探索のためのObjective関数
# *******************************************************************
def create_objective(model_name, train_dataset, val_dataset, n_classes, device, n_epochs=20):
    """Optunaのobjective関数を作成"""
    
    def objective(trial):
        # モデル別のハイパーパラメータ探索空間
        if model_name == 'MLP-Mixer':
            hyperparams = {
                'patch_size': trial.suggest_categorical('patch_size', [5]),
                'dim': trial.suggest_categorical('dim', [256, 512, 768]),
                'depth': trial.suggest_int('depth', 6, 16),
                'token_dim': trial.suggest_categorical('token_dim', [128, 256, 512]),
                'channel_dim': trial.suggest_categorical('channel_dim', [1024, 2048, 3072]),
            }
        elif model_name == 'CNN':
            hyperparams = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
                'depth': trial.suggest_int('depth', 3, 6),
                'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
            }
        elif model_name == 'ViT':
            hyperparams = {
                'patch_size': trial.suggest_categorical('patch_size', [5]),
                'dim': trial.suggest_categorical('dim', [128, 256, 384]),
                'depth': trial.suggest_int('depth', 4, 8),
                'heads': trial.suggest_categorical('heads', [4, 8]),
                'mlp_dim': trial.suggest_categorical('mlp_dim', [256, 512, 1024]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 共通のハイパーパラメータ
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # モデルの初期化
        model = init_model(model_name, hyperparams, n_classes)
        model = model.to(device)
        
        # データローダーの作成
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 損失関数とオプティマイザ
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        # 訓練
        best_val_f1 = 0
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['image'].to(device)
                    labels = batch['label']
                    
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu()
                    
                    val_preds.extend(preds.numpy())
                    val_labels.extend(labels.numpy())
            
            # F1スコアの計算
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
            
            scheduler.step()
            
            # Pruning（早期終了）
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # メモリクリア
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return best_val_f1
    
    return objective


# *******************************************************************
# Cross-Validation with Confusion Matrix per Fold
# *******************************************************************
def cross_validate_with_confusion_matrix(model_name, hyperparams, images, labels, 
                                        pixel_locations, geo_locations, n_classes=5, 
                                        n_folds=5, device='cuda', save_dir='results',
                                        optimize_threshold=False):
    """
    クロスバリデーションを実行し、各foldの混同行列を生成
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # クラスラベルの生成（層化分割用）
    class_labels = []
    default_thresholds = [i / n_classes for i in range(1, n_classes)]
    for label in labels:
        for i, threshold in enumerate(default_thresholds):
            if label <= threshold:
                class_labels.append(i)
                break
        else:
            class_labels.append(n_classes - 1)
    
    fold_results = []
    all_cms = []
    best_thresholds_per_fold = []
    
    # 結果保存用ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(images, class_labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # データセットの作成
        train_dataset = LandslideDataset(
            images[train_idx], labels[train_idx],
            pixel_locations[train_idx], geo_locations[train_idx],
            n_classes=n_classes
        )
        val_dataset = LandslideDataset(
            images[val_idx], labels[val_idx],
            pixel_locations[val_idx], geo_locations[val_idx],
            n_classes=n_classes
        )
        
        # データローダーの作成
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # モデルの初期化と訓練
        model = init_model(model_name, hyperparams, n_classes)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        # 訓練
        print("Training...")
        best_model_state = None
        best_val_f1 = 0
        
        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                inputs = batch['image'].to(device)
                labels_batch = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if (epoch + 1) % 5 == 0:
                model.eval()
                val_preds = []
                val_labels = []
                val_probs = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        inputs = batch['image'].to(device)
                        labels_batch = batch['label']
                        
                        outputs = model(inputs)
                        probs = torch.softmax(outputs, dim=1).cpu()
                        preds = torch.argmax(outputs, dim=1).cpu()
                        
                        val_probs.append(probs.numpy())
                        val_preds.extend(preds.numpy())
                        val_labels.extend(labels_batch.numpy())
                
                val_f1 = f1_score(val_labels, val_preds, average='macro')
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
                          f"Val F1 = {val_f1:.4f}")
            
            scheduler.step()
        
        # 最良モデルをロード
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最終評価
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        val_raw_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                labels_batch = batch['label']
                raw_labels_batch = batch['raw_label']
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu()
                preds = torch.argmax(outputs, dim=1).cpu()
                
                val_probs.append(probs.numpy())
                val_preds.extend(preds.numpy())
                val_labels.extend(labels_batch.numpy())
                val_raw_labels.extend(raw_labels_batch.numpy())
        
        val_probs = np.vstack(val_probs)
        
        # 閾値の最適化（オプション）
        if optimize_threshold and fold == 0:  # 最初のfoldでのみ最適化
            optimizer = ThresholdOptimizer(n_classes=n_classes)
            best_thresholds, best_score = optimizer.optimize_thresholds(
                np.array(val_raw_labels), val_probs, metric='f1_macro'
            )
            best_thresholds_per_fold.append(best_thresholds)
            print(f"  Optimized thresholds: {best_thresholds}")
            print(f"  Optimized F1 score: {best_score:.4f}")
        
        # メトリクスと混同行列の計算
        metrics, cm = calculate_metrics(val_labels, val_preds, val_probs)
        
        # 結果の保存
        fold_result = {
            'fold': fold + 1,
            'metrics': metrics,
            'confusion_matrix': cm
        }
        fold_results.append(fold_result)
        all_cms.append(cm)
        
        # 混同行列の可視化と保存
        class_names = [f'Class {i}' for i in range(n_classes)]
        plot_confusion_matrix(
            cm, class_names,
            title=f'{model_name} - Fold {fold + 1} Confusion Matrix',
            save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}.png'),
            normalize=False
        )
        
        # 正規化版も保存
        plot_confusion_matrix(
            cm, class_names,
            title=f'{model_name} - Fold {fold + 1} Normalized Confusion Matrix',
            save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}_normalized.png'),
            normalize=True
        )
        
        # foldごとの結果を表示
        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  F1 per class: {metrics['f1']}")
        
        # メモリクリア
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # 全体の混同行列（平均）
    avg_cm = np.mean(all_cms, axis=0).astype(int)
    plot_confusion_matrix(
        avg_cm, class_names,
        title=f'{model_name} - Average Confusion Matrix across all folds',
        save_path=os.path.join(model_save_dir, 'confusion_matrix_average.png'),
        normalize=False
    )
    
    # 結果のサマリーを作成
    summary = {
        'model': model_name,
        'n_folds': n_folds,
        'n_classes': n_classes,
        'hyperparams': hyperparams,
        'fold_results': fold_results,
        'average_confusion_matrix': avg_cm.tolist(),
        'best_thresholds': best_thresholds_per_fold if optimize_threshold else None
    }
    
    # JSONファイルに保存
    with open(os.path.join(model_save_dir, 'cv_results.json'), 'w') as f:
        # NumPy配列をリストに変換してJSON保存
        json_summary = summary.copy()
        for fold_result in json_summary['fold_results']:
            fold_result['confusion_matrix'] = fold_result['confusion_matrix'].tolist()
            for key, value in fold_result['metrics'].items():
                if isinstance(value, np.ndarray):
                    fold_result['metrics'][key] = value.tolist()
        json.dump(json_summary, f, indent=2)
    
    # 平均メトリクスの計算と表示
    print(f"\n{'='*60}")
    print(f"{model_name} - Cross-Validation Summary")
    print(f"{'='*60}")
    
    avg_metrics = {}
    for key in fold_results[0]['metrics'].keys():
        if key in ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            values = [fold['metrics'][key] for fold in fold_results]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            print(f"{key}: {avg_metrics[key]['mean']:.4f} ± {avg_metrics[key]['std']:.4f}")
    
    return summary


# *******************************************************************
# Main Training Pipeline with Hyperparameter Search
# *******************************************************************
def train_with_hyperparameter_search(model_name, images, labels, pixel_locations, 
                                    geo_locations, n_classes=5, n_trials=50, 
                                    device='cuda', save_dir='results'):
    """
    ハイパーパラメータ探索を含む訓練パイプライン
    """
    print(f"\n{'='*80}")
    print(f"Hyperparameter Search for {model_name}")
    print(f"{'='*80}")
    
    # データの分割（訓練用と検証用）
    from sklearn.model_selection import train_test_split
    
    # クラスラベルの生成（層化分割用）
    class_labels = []
    default_thresholds = [i / n_classes for i in range(1, n_classes)]
    for label in labels:
        for i, threshold in enumerate(default_thresholds):
            if label <= threshold:
                class_labels.append(i)
                break
        else:
            class_labels.append(n_classes - 1)
    
    train_idx, val_idx = train_test_split(
        np.arange(len(images)), 
        test_size=0.2, 
        stratify=class_labels, 
        random_state=42
    )
    
    # データセットの作成
    train_dataset = LandslideDataset(
        images[train_idx], labels[train_idx],
        pixel_locations[train_idx], geo_locations[train_idx],
        n_classes=n_classes
    )
    val_dataset = LandslideDataset(
        images[val_idx], labels[val_idx],
        pixel_locations[val_idx], geo_locations[val_idx],
        n_classes=n_classes
    )
    
    # Optunaでハイパーパラメータ探索
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    objective = create_objective(
        model_name, train_dataset, val_dataset, 
        n_classes, device, n_epochs=30
    )
    
    print(f"Starting {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 最良のハイパーパラメータを取得
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\nBest trial:")
    print(f"  Value (F1 macro): {best_value:.4f}")
    print(f"  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    
    # 結果を保存
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # ハイパーパラメータをJSONファイルに保存
    with open(os.path.join(model_save_dir, 'best_hyperparams.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Optunaの可視化
    try:
        import optuna.visualization as vis
        
        # 最適化履歴
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(model_save_dir, 'optimization_history.html'))
        
        # パラメータの重要度
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(model_save_dir, 'param_importances.html'))
        
        # パラメータの関係
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(model_save_dir, 'parallel_coordinate.html'))
        
    except Exception as e:
        print(f"Could not create Optuna visualizations: {e}")
    
    return best_params


# *******************************************************************
# Process Multiple Datasets
# *******************************************************************
def process_multiple_datasets(dataset_configs, landslide_path, epsg_code=6675, 
                             tip_size_m=5, slide_ratio=0.5):
    """
    複数のデータセットを処理して統合
    """
    all_tips = []
    all_labels = []
    all_pixel_locs = []
    all_geo_locs = []
    all_landslide_ids = []
    all_dataset_names = []
    
    for config in dataset_configs:
        print(f"\nProcessing dataset: {config['dataset_name']}...")
        
        # 各データセットごとにLegacy pathを使用
        processor = LandslideTipProcessor(
            rgb_paths=str(config['rgb_path']),
            landslide_path=str(landslide_path),
            legacy_path=str(config['legacy_path']),
            epsg_code=epsg_code,
            resolution=0.2
        )
        
        tips, labels, pixel_locs, geo_locs, landslide_ids = processor.extract_tips(
            tip_size_m=tip_size_m,
            slide_ratio=slide_ratio
        )
        
        print(f"  Extracted {len(tips)} tips from {config['dataset_name']}")
        print(f"  Label distribution: min={labels.min():.3f}, max={labels.max():.3f}, mean={labels.mean():.3f}")
        
        all_tips.append(tips)
        all_labels.append(labels)
        all_pixel_locs.append(pixel_locs)
        all_geo_locs.append(geo_locs)
        all_landslide_ids.append(landslide_ids)
        all_dataset_names.extend([config['dataset_name']] * len(tips))
    
    # 全データを結合
    all_tips = np.concatenate(all_tips, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_pixel_locs = np.concatenate(all_pixel_locs, axis=0)
    all_geo_locs = np.concatenate(all_geo_locs, axis=0)
    all_landslide_ids = np.concatenate(all_landslide_ids, axis=0)
    all_dataset_names = np.array(all_dataset_names)
    
    return all_tips, all_labels, all_pixel_locs, all_geo_locs, all_landslide_ids, all_dataset_names


# *******************************************************************
# Main Function
# *******************************************************************
def main():
    """メイン実行関数"""
    
    # pyprojrootを使用してプロジェクトルートを取得
    try:
        from pyprojroot import here
        root_dir = here()
    except ImportError:
        print("pyprojroot not found, using current directory as root")
        root_dir = Path(".")
    
    # トライアル名の設定
    trial_name = "noto-v0-class5-trial"
    
    # ディレクトリの設定
    data_dir = root_dir / 'data'
    data_annotation_dir = data_dir / "annotation"
    data_Noto_April_dir = data_dir / "Noto_April"
    data_sample_dir = data_Noto_April_dir / "annotation_sample"
    output_dir = root_dir / "output" / trial_name
    fig_dir = output_dir / "figs"
    
    # ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # データセット設定
    dataset_configs = [
        {
            'rgb_path': data_sample_dir / '07ed874.jpg',
            'dataset_name': '07ed874',
            'legacy_path': data_dir / 'annotation' / '07ed874' / '07ed874.shp'
        },
        {
            'rgb_path': data_sample_dir / '07ed882.jpg',
            'dataset_name': '07ed882',
            'legacy_path': data_dir / 'annotation' / '07ed882' / '07ed882.shp'
        },
        {
            'rgb_path': data_sample_dir / '07ed921.jpg',
            'dataset_name': '07ed921',
            'legacy_path': data_dir / 'annotation' / '07ed921' / '07ed921.geojson'
        },
        {
            'rgb_path': data_sample_dir / '07ed922.jpg',
            'dataset_name': '07ed922',
            'legacy_path': data_dir / 'annotation' / '07ed922' / '07ed922.shp'
        },
        {
            'rgb_path': data_sample_dir / '07ee703.jpg',
            'dataset_name': '07ee703',
            'legacy_path': data_dir / 'annotation' / '07ee703' / '07ee703.geojson'
        },
        {
            'rgb_path': data_sample_dir / '07fd041.jpg',
            'dataset_name': '07fd041',
            'legacy_path': data_dir / 'annotation' / 'Porigon_april_fix.geojson'
        },
        {
            'rgb_path': data_sample_dir / '07fd111.jpg',
            'dataset_name': '07fd111',
            'legacy_path': data_dir / 'annotation' / '07fd111' / '07fd111.geojson'
        },
        {
            'rgb_path': data_sample_dir / '07ed684.jpg',
            'dataset_name': '07ed684',
            'legacy_path': data_dir / 'annotation' / '07ed684' / '07ed684.geojson'
        },
    ]
    
    # 設定パラメータ
    config = {
        'dataset_configs': dataset_configs,
        'landslide_path': data_dir / 'noto_landslide' / 'hokai-all.geojson',
        'output_dir': output_dir,
        'fig_dir': fig_dir,
        'epsg_code': 6675,
        'tip_size_m': 5,
        'slide_ratio': 0.5,
        'n_outer_cv': 10,  # クロスバリデーションのfold数
        'n_epochs': 50,  # エポック数（デフォルトは50）
        'batch_size': 64,
        'model_names': ['MLP-Mixer'],
        'n_trials': 32,  # ハイパーパラメータ探索のトライアル数
        'n_classes': 5,  # クラス数
        'optimize_threshold': True,  # 閾値最適化を行うか
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"{'='*80}")
    print(f"Landslide Classification Pipeline")
    print(f"{'='*80}")
    print(f"Trial name: {trial_name}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {config['device']}")
    print(f"Number of datasets: {len(dataset_configs)}")
    print(f"Models to train: {config['model_names']}")
    print(f"Number of classes: {config['n_classes']}")
    print(f"{'='*80}")
    
    # 複数データセットの処理と統合
    print("\nProcessing multiple datasets...")
    all_tips, all_labels, all_pixel_locs, all_geo_locs, all_landslide_ids, all_dataset_names = \
        process_multiple_datasets(
            dataset_configs=config['dataset_configs'],
            landslide_path=config['landslide_path'],
            epsg_code=config['epsg_code'],
            tip_size_m=config['tip_size_m'],
            slide_ratio=config['slide_ratio']
        )
    
    print(f"\n{'='*60}")
    print(f"Total extracted tips: {len(all_tips)}")
    print(f"Overall label distribution:")
    print(f"  Min: {all_labels.min():.3f}")
    print(f"  Max: {all_labels.max():.3f}")
    print(f"  Mean: {all_labels.mean():.3f}")
    print(f"  Std: {all_labels.std():.3f}")
    
    # データセットごとの統計情報
    print(f"\nDataset-wise statistics:")
    for dataset_name in np.unique(all_dataset_names):
        mask = all_dataset_names == dataset_name
        dataset_labels = all_labels[mask]
        print(f"  {dataset_name}: n={len(dataset_labels)}, "
              f"mean={dataset_labels.mean():.3f}, std={dataset_labels.std():.3f}")
    
    # 各モデルで訓練と評価
    results = {}
    
    for model_name in config['model_names']:
        print(f"\n{'='*80}")
        print(f"Processing {model_name}")
        print(f"{'='*80}")
        
        # 結果保存ディレクトリの作成
        model_output_dir = config['output_dir'] / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ハイパーパラメータ探索
        print(f"\nStarting hyperparameter search with {config['n_trials']} trials...")
        best_params = train_with_hyperparameter_search(
            model_name=model_name,
            images=all_tips,
            labels=all_labels,
            pixel_locations=all_pixel_locs,
            geo_locations=all_geo_locs,
            n_classes=config['n_classes'],
            n_trials=config['n_trials'],
            device=config['device'],
            save_dir=str(model_output_dir)
        )
        
        # 最適化されたハイパーパラメータでクロスバリデーション
        print(f"\nPerforming {config['n_outer_cv']}-fold cross-validation...")
        cv_results = cross_validate_with_confusion_matrix(
            model_name=model_name,
            hyperparams=best_params,
            images=all_tips,
            labels=all_labels,
            pixel_locations=all_pixel_locs,
            geo_locations=all_geo_locs,
            n_classes=config['n_classes'],
            n_folds=config['n_outer_cv'],
            device=config['device'],
            save_dir=str(model_output_dir),
            optimize_threshold=config['optimize_threshold']
        )
        
        results[model_name] = {
            'cv_results': cv_results,
            'best_params': best_params,
            'dataset_names': all_dataset_names
        }
    
    # 全体の結果サマリーを作成
    print(f"\n{'='*80}")
    print("Final Summary - All Models")
    print(f"{'='*80}")
    
    summary_data = []
    for model_name, result in results.items():
        cv_result = result['cv_results']
        avg_metrics = {}
        
        for fold_result in cv_result['fold_results']:
            for key in ['accuracy', 'balanced_accuracy', 'f1_macro', 
                       'precision_macro', 'recall_macro']:
                if key not in avg_metrics:
                    avg_metrics[key] = []
                avg_metrics[key].append(fold_result['metrics'][key])
        
        summary_data.append({
            'Model': model_name,
            'N_Samples': len(all_tips),
            'N_Datasets': len(dataset_configs),
            'Accuracy': f"{np.mean(avg_metrics['accuracy']):.4f} ± {np.std(avg_metrics['accuracy']):.4f}",
            'Balanced_Acc': f"{np.mean(avg_metrics['balanced_accuracy']):.4f} ± {np.std(avg_metrics['balanced_accuracy']):.4f}",
            'F1_Macro': f"{np.mean(avg_metrics['f1_macro']):.4f} ± {np.std(avg_metrics['f1_macro']):.4f}",
            'Precision_Macro': f"{np.mean(avg_metrics['precision_macro']):.4f} ± {np.std(avg_metrics['precision_macro']):.4f}",
            'Recall_Macro': f"{np.mean(avg_metrics['recall_macro']):.4f} ± {np.std(avg_metrics['recall_macro']):.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # サマリーをCSVとして保存
    summary_path = config['output_dir'] / 'all_models_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # 詳細な結果をJSONとして保存
    detailed_results_path = config['output_dir'] / 'detailed_results.json'
    with open(detailed_results_path, 'w') as f:
        # NumPy配列をリストに変換してJSON保存
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'best_params': result['best_params'],
                'n_samples': len(all_tips),
                'n_datasets': len(dataset_configs),
                'dataset_names': list(np.unique(result['dataset_names']))
            }
        json.dump(json_results, f, indent=2)
    
    print(f"Detailed results saved to: {detailed_results_path}")
    print(f"\n{'='*80}")
    print(f"All results saved to {config['output_dir']}")
    print("Pipeline completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()