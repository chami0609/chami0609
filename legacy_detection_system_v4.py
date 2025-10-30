"""
土砂崩れ地域のレガシー検出システム - 複数データセット対応版
複数のJPG+SHP形式の航空写真セットに対応
"""

# Utils -----------------------
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
from shapely.geometry import Point
import geopandas as gpd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import json
# Machine Learning ---------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from einops import repeat
from einops.layers.torch import Rearrange
import optuna
from timm.optim import create_optimizer
import rasterio
import rasterio.features as rasioftr

import geopandas as gpd
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import hashlib
from datetime import datetime
from shapely.geometry import Point
warnings.filterwarnings('ignore')

from legacy_detection_system_v3 import (
    LandslideTipProcessor, LandslideDataset, 
    init_model, train_one_epoch, validate_one_epoch,
    ViT, LeNet, MLPMixer
)
# === new_v1 models and ensemble import ===
from new_v1 import EfficientNetB0, ResNet18Custom, MobileNetV3Small, ImprovedViT, WeightedEnsembleModel

# *******************************************************************
# Data Processing Class for Multiple Datasets
# *******************************************************************
class MultiDatasetLandslideProcessor:
    """複数データセットに対応したランドスライド処理クラス"""
    
    def __init__(self, dataset_configs, epsg_code=6675, resolution=0.2):
        """
        dataset_configs: リスト形式で複数のデータセット設定を受け取る
        各設定は以下の形式:
        {
            'rgb_path': 'path/to/image.jpg',
            'shp_path': 'path/to/shape.shp',  # オプション
            'dataset_name': 'dataset_id'
        }
        """
        self.dataset_configs = dataset_configs
        self.epsg_code = epsg_code
        self.resolution = resolution  # m
        self.datasets_metadata = []
        
    def process_all_datasets(self, landslide_path, legacy_path, tip_size_m=5, slide_ratio=0.5):
        """全データセットを処理してチップを生成"""
        
        # ポリゴンデータの読み込み（全データセット共通）
        landslide_gdf = gpd.read_file(landslide_path)
        if landslide_gdf.crs != f"EPSG:{self.epsg_code}":
            landslide_gdf.to_crs(epsg=self.epsg_code, inplace=True)
        landslide_gdf['landslide_id'] = range(len(landslide_gdf))
        
        legacy_gdf = gpd.read_file(legacy_path)
        if legacy_gdf.crs != f"EPSG:{self.epsg_code}":
            legacy_gdf.to_crs(epsg=self.epsg_code, inplace=True)
        
        # 全データセットからチップを収集
        all_tips = []
        all_labels = []
        all_pixel_locations = []
        all_geo_locations = []
        all_landslide_ids = []
        all_dataset_ids = []
        
        for idx, config in enumerate(self.dataset_configs):
            print(f"\n{'='*60}")
            print(f"Processing dataset {idx+1}/{len(self.dataset_configs)}: {config.get('dataset_name', config['rgb_path'])}")
            print(f"{'='*60}")
            
            # 単一データセット用のプロセッサを作成
            processor = LandslideTipProcessor(
                rgb_paths=config['rgb_path'],
                landslide_path=landslide_path,
                legacy_path=config.get('legacy_path', legacy_path),  # 個別指定
                epsg_code=self.epsg_code,
                resolution=self.resolution
            )
            
            # チップを抽出
            tips, labels, pixel_locs, geo_locs, landslide_ids = processor.extract_tips(
                tip_size_m=tip_size_m,
                slide_ratio=slide_ratio
            )
            
            # データセットメタデータを保存
            self.datasets_metadata.append({
                'dataset_id': idx,
                'dataset_name': config.get('dataset_name', f'dataset_{idx}'),
                'rgb_path': config['rgb_path'],
                'n_tips': len(tips),
                'n_landslides': len(np.unique(landslide_ids))
            })
            
            print(f"Extracted {len(tips)} tips from dataset {config.get('dataset_name', idx)}")
            
            # 結果を集約
            all_tips.append(tips)
            all_labels.append(labels)
            all_pixel_locations.append(pixel_locs)
            all_geo_locations.append(geo_locs)
            all_landslide_ids.append(landslide_ids)
            all_dataset_ids.append(np.full(len(tips), idx))
        
        # 全データを結合
        combined_tips = np.concatenate(all_tips, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        combined_pixel_locs = np.concatenate(all_pixel_locations, axis=0)
        combined_geo_locs = np.concatenate(all_geo_locations, axis=0)
        combined_landslide_ids = np.concatenate(all_landslide_ids, axis=0)
        combined_dataset_ids = np.concatenate(all_dataset_ids, axis=0)
        
        print(f"\n{'='*60}")
        print(f"Total tips extracted: {len(combined_tips)}")
        print(f"Total datasets processed: {len(self.dataset_configs)}")
        print(f"Total unique landslides: {len(np.unique(combined_landslide_ids))}")
        print(f"{'='*60}")
        
        return (combined_tips, combined_labels, combined_pixel_locs, 
                combined_geo_locs, combined_landslide_ids, combined_dataset_ids)
    
    def save_metadata(self, output_path):
        """データセットのメタデータを保存"""
        # Path型をstrに変換
        serializable_metadata = []
        for meta in self.datasets_metadata:
            meta_copy = meta.copy()
            for k, v in meta_copy.items():
                if isinstance(v, Path):
                    meta_copy[k] = str(v)
            serializable_metadata.append(meta_copy)
        with open(output_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        print(f"Metadata saved to {output_path}")


# *******************************************************************
# Enhanced Dataset Class with Dataset ID
# *******************************************************************
class MultiDatasetLandslideDataset(Dataset):
    """複数データセット対応のPyTorchデータセットクラス"""
    
    def __init__(self, images, labels, pixel_locations, geo_locations, 
                 landslide_ids, dataset_ids, transform=None, patch_size=25):
        self.images = images
        self.labels = labels
        self.pixel_locations = pixel_locations
        self.geo_locations = geo_locations
        self.landslide_ids = landslide_ids
        self.dataset_ids = dataset_ids
        self.transform = transform
        self.patch_size = patch_size

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
        
        return {
            'image': image,
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'i': self.pixel_locations[idx][0],
            'j': self.pixel_locations[idx][1],
            'lon': self.geo_locations[idx][0],
            'lat': self.geo_locations[idx][1],
            'landslide_id': self.landslide_ids[idx],
            'dataset_id': self.dataset_ids[idx]
        }


# *******************************************************************
# Spatial Cross-Validator considering multiple datasets
# *******************************************************************
class MultiDatasetSpatialCV:
    """複数データセットを考慮した空間的クロスバリデーション"""
    
    def __init__(self, n_splits=10, strategy='landslide', random_state=42):
        self.n_splits = n_splits
        self.strategy = strategy
        self.random_state = random_state
        
    def split(self, landslide_ids, dataset_ids):
        np.random.seed(self.random_state)
        
        if self.strategy == 'balanced-dataset':
            # 各データセットのサンプル数がfoldごとに均等になるよう分割
            all_indices = np.arange(len(dataset_ids))
            folds = [[] for _ in range(self.n_splits)]
            for d_id in np.unique(dataset_ids):
                idxs = all_indices[dataset_ids == d_id]
                np.random.shuffle(idxs)
                fold_sizes = [len(idxs) // self.n_splits] * self.n_splits
                for i in range(len(idxs) % self.n_splits):
                    fold_sizes[i] += 1
                start = 0
                for fold, size in enumerate(fold_sizes):
                    folds[fold].extend(idxs[start:start+size])
                    start += size
            for fold in range(self.n_splits):
                test_idx = np.array(folds[fold])
                train_idx = np.setdiff1d(all_indices, test_idx)
                yield train_idx, test_idx
        elif self.strategy == 'landslide':
            # 土砂崩れIDベースの分割
            unique_landslides = np.unique(landslide_ids)
            np.random.shuffle(unique_landslides)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
            for train_ls_idx, test_ls_idx in kf.split(unique_landslides):
                train_landslides = unique_landslides[train_ls_idx]
                test_landslides = unique_landslides[test_ls_idx]
                
                train_mask = np.isin(landslide_ids, train_landslides)
                test_mask = np.isin(landslide_ids, test_landslides)
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                yield train_idx, test_idx
                
        elif self.strategy == 'dataset':
            # データセットベースの分割
            unique_datasets = np.unique(dataset_ids)
            kf = KFold(n_splits=min(self.n_splits, len(unique_datasets)), 
                      shuffle=True, random_state=self.random_state)
            
            for train_ds_idx, test_ds_idx in kf.split(unique_datasets):
                train_datasets = unique_datasets[train_ds_idx]
                test_datasets = unique_datasets[test_ds_idx]
                
                train_mask = np.isin(dataset_ids, train_datasets)
                test_mask = np.isin(dataset_ids, test_datasets)
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                yield train_idx, test_idx
                
        else:  # mixed strategy
            # データセットと土砂崩れIDの両方を考慮
            combined_ids = list(zip(dataset_ids, landslide_ids))
            unique_combined = list(set(combined_ids))
            np.random.shuffle(unique_combined)
            
            fold_size = len(unique_combined) // self.n_splits
            
            for i in range(self.n_splits):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else len(unique_combined)
                test_combined = unique_combined[start_idx:end_idx]
                
                test_mask = np.array([cid in test_combined for cid in combined_ids])
                train_mask = ~test_mask
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                yield train_idx, test_idx


# *******************************************************************
# Enhanced Pipeline for Multiple Datasets
# *******************************************************************
from legacy_detection_system_v3 import LegacyDetectionPipeline
class EnhancedLegacyDetectionPipeline(LegacyDetectionPipeline):
    """複数データセット対応の拡張パイプライン"""
    
    def __init__(self, output_dir, device='cuda', random_state=42):
        super().__init__(output_dir, device, random_state)
        
    def train_with_multiple_datasets(self, dataset, landslide_ids, dataset_ids, 
                                    model_names, n_outer_cv=10, n_trials=32, 
                                    n_epochs=64, batch_size=64, cv_strategy='landslide'):
        """複数データセットでの交差検証付き訓練"""
        
        cv_results = {}
        
        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"Training {model_name} with multiple datasets")
            print(f"CV Strategy: {cv_strategy}")
            print(f"{'='*50}")
            
            # ハイパーパラメータ最適化
            print("Optimizing hyperparameters...")
            objective = self._create_multi_dataset_objective(
                model_name=model_name,
                train_data=dataset,
                train_landslide_ids=landslide_ids,
                train_dataset_ids=dataset_ids,
                device=self.device,
                n_inner_cv=5,
                n_inner_epochs=8,
                batch_size=batch_size
            )
            
            study = optuna.create_study(direction='minimize')
            try:
                study.optimize(objective, n_trials=n_trials)
            except Exception as e:
                print(f"Optuna optimization failed for {model_name}: {e}")
                continue
            best_hyperparams = study.best_params
            print(f"Best hyperparameters: {best_hyperparams}")
            
            # 外部交差検証
            print("\nStarting outer cross-validation...")
            spatial_cv = MultiDatasetSpatialCV(
                n_splits=n_outer_cv, 
                strategy=cv_strategy,
                random_state=self.random_state
            )
            
            cv_scores = []
            cv_predictions = []
            dataset_specific_scores = {d_id: [] for d_id in np.unique(dataset_ids)}
            
            for fold, (train_idx, test_idx) in enumerate(spatial_cv.split(landslide_ids, dataset_ids)):
                print(f"\nFold {fold + 1}/{n_outer_cv}")
                print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
                
                # データセット別の分布を表示
                test_datasets = dataset_ids[test_idx]
                unique_test_datasets, counts = np.unique(test_datasets, return_counts=True)
                print(f"Test datasets distribution: {dict(zip(unique_test_datasets, counts))}")
                
                # データローダーの作成
                train_subset = torch.utils.data.Subset(dataset, train_idx)
                test_subset = torch.utils.data.Subset(dataset, test_idx)
                
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
                
                # モデルの初期化と訓練
                try:
                    model = init_model(model_name, self._convert_hyperparams(model_name, best_hyperparams))
                    if model is None:
                        print(f"Model initialization failed for {model_name}. Skipping fold.")
                        continue
                    model = model.to(self.device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss_fn = nn.MSELoss()
                    
                    train_losses = []
                    test_losses = []
                    best_test_loss = float('inf')
                    best_model_state = None
                    
                    for epoch in range(n_epochs):
                        train_loss, _, _ = train_one_epoch(
                            epoch, model, train_loader, optimizer, loss_fn, 
                            self.transform, self.device
                        )
                        test_loss, test_outputs, test_labels = validate_one_epoch(
                            epoch, model, test_loader, loss_fn, self.device
                        )
                        
                        train_losses.append(train_loss)
                        test_losses.append(test_loss)
                        
                        # ベストモデルを保存
                        if test_loss < best_test_loss:
                            best_test_loss = test_loss
                            best_model_state = model.state_dict()
                    
                    # ベストモデルをロード
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                except Exception as e:
                    print(f"Error in training fold {fold} for {model_name}: {e}")
                    continue
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 学習曲線の保存
                self._save_learning_curve(train_losses, test_losses, model_name, fold)
                
                # 最終的な評価（データセット別も含む）
                model.eval()
                all_preds = []
                all_labels = []
                all_dataset_ids_test = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        inputs = batch['image'].to(self.device)
                        labels = batch['label'].cpu().numpy()
                        dataset_ids_batch = batch['dataset_id'].cpu().numpy()
                        outputs = model(inputs).cpu().numpy()[:, 0]
                        
                        all_preds.extend(outputs.tolist())
                        all_labels.extend(labels.tolist())
                        all_dataset_ids_test.extend(dataset_ids_batch.tolist())
                
                # 全体のメトリクス
                rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
                correlation, _ = pearsonr(all_labels, all_preds)
                
                cv_scores.append({
                    'fold': fold,
                    'rmse': rmse,
                    'correlation': correlation,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx)
                })
                
                # データセット別のメトリクス
                for d_id in np.unique(all_dataset_ids_test):
                    mask = np.array(all_dataset_ids_test) == d_id
                    if np.sum(mask) > 0:
                        d_labels = np.array(all_labels)[mask]
                        d_preds = np.array(all_preds)[mask]
                        d_rmse = np.sqrt(mean_squared_error(d_labels, d_preds))
                        d_corr, _ = pearsonr(d_labels, d_preds) if len(d_labels) > 1 else (0, 1)
                        
                        dataset_specific_scores[d_id].append({
                            'fold': fold,
                            'rmse': d_rmse,
                            'correlation': d_corr,
                            'n_samples': np.sum(mask)
                        })
                
                cv_predictions.append({
                    'fold': fold,
                    'predictions': all_preds,
                    'labels': all_labels,
                    'dataset_ids': all_dataset_ids_test
                })
            
            # 結果の保存
            cv_results[model_name] = {
                'best_hyperparams': best_hyperparams,
                'cv_scores': cv_scores,
                'cv_predictions': cv_predictions,
                'dataset_specific_scores': dataset_specific_scores
            }
            
            self._save_multi_dataset_results(model_name, cv_results[model_name])
        
        return cv_results
    
    def _create_multi_dataset_objective(self, model_name, train_data, train_landslide_ids, 
                                       train_dataset_ids, device, n_inner_cv=5, 
                                       n_inner_epochs=8, batch_size=64):
        """複数データセット用の目的関数"""
        
        def objective(trial):
            hyperparams = {}
            # 画像サイズを取得（データセットのpatch_size属性を参照、なければ25）
            image_size = getattr(train_data, 'patch_size', 25)
            patch_size_candidates = [i for i in range(1, image_size + 1) if image_size % i == 0]
            # ハイパーパラメータ範囲を現実的に修正
            if model_name == 'ViT':
                hyperparams['PATCH_SIZE'] = trial.suggest_categorical('PATCH_SIZE', patch_size_candidates)
                hyperparams['N_DIM'] = trial.suggest_int('N_DIM', 4, 7)  # 2^4=16 ~ 2^7=128
                hyperparams['N_HEADS'] = trial.suggest_int('N_HEADS', 1, 3)  # 2^1=2 ~ 2^3=8
                hyperparams['MLP_DIM'] = trial.suggest_int('MLP_DIM', 4, 7)
                hyperparams['DROPOUT'] = trial.suggest_float('DROPOUT', 0.0, 0.5, step=0.05)
                hyperparams['DEPTH'] = trial.suggest_int('DEPTH', 2, 6)
                hyperparams['USE_CLS_TOKEN'] = trial.suggest_categorical('USE_CLS_TOKEN', [True, False])
            elif model_name == 'CNN':
                hyperparams['NUM_FILTER1'] = trial.suggest_int('NUM_FILTER1', 3, 6)
                hyperparams['KERNEL_SIZE1'] = trial.suggest_int('KERNEL_SIZE1', 3, 7, step=2)
                hyperparams['NUM_FILTER2'] = trial.suggest_int('NUM_FILTER2', 3, 6)
                hyperparams['KERNEL_SIZE2'] = trial.suggest_int('KERNEL_SIZE2', 3, 7, step=2)
                hyperparams['DROPOUT'] = trial.suggest_float('DROPOUT', 0.0, 0.5, step=0.05)
            elif model_name == 'MLP-Mixer':
                hyperparams['PATCH_SIZE'] = trial.suggest_categorical('PATCH_SIZE', patch_size_candidates)
                hyperparams['TOKEN_DIM'] = trial.suggest_int('TOKEN_DIM', 4, 7)
                hyperparams['CHANNEL_DIM'] = trial.suggest_int('CHANNEL_DIM', 4, 7)
                hyperparams['MLP_DIM'] = trial.suggest_int('MLP_DIM', 4, 7)
                hyperparams['N_BLOCK'] = trial.suggest_int('N_BLOCK', 1, 4)
            elif model_name == 'EfficientNet':
                hyperparams['PRETRAINED'] = trial.suggest_categorical('PRETRAINED', [True, False])
                hyperparams['DROPOUT'] = trial.suggest_float('DROPOUT', 0.0, 0.5, step=0.05)
                hyperparams['FREEZE_BACKBONE'] = trial.suggest_categorical('FREEZE_BACKBONE', [True, False])
            elif model_name == 'ResNet18':
                hyperparams['PRETRAINED'] = trial.suggest_categorical('PRETRAINED', [True, False])
                hyperparams['DROPOUT'] = trial.suggest_float('DROPOUT', 0.0, 0.5, step=0.05)
            elif model_name == 'MobileNetV3':
                hyperparams['PRETRAINED'] = trial.suggest_categorical('PRETRAINED', [True, False])
                hyperparams['DROPOUT'] = trial.suggest_float('DROPOUT', 0.0, 0.5, step=0.05)
            spatial_cv = MultiDatasetSpatialCV(
                n_splits=n_inner_cv, 
                strategy='landslide',
                random_state=42
            )
            val_losses = []
            for fold, (train_idx, val_idx) in enumerate(spatial_cv.split(train_landslide_ids, train_dataset_ids)):
                if len(val_idx) == 0:
                    continue
                try:
                    train_subset = torch.utils.data.Subset(train_data, train_idx)
                    val_subset = torch.utils.data.Subset(train_data, val_idx)
                    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
                    model = init_model(model_name, self._convert_hyperparams(model_name, hyperparams))
                    if model is None:
                        print(f"Model initialization failed for {model_name} in objective. Skipping fold.")
                        continue
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss_fn = nn.MSELoss()
                    best_val_loss = float('inf')
                    for epoch in range(n_inner_epochs):
                        train_loss, _, _ = train_one_epoch(
                            epoch, model, train_loader, optimizer, loss_fn, 
                            transforms.Compose([]), device
                        )
                        val_loss, _, _ = validate_one_epoch(
                            epoch, model, val_loader, loss_fn, device
                        )
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                except Exception as e:
                    print(f"Error in objective fold {fold} for {model_name}: {e}")
                    continue
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            return np.mean(val_losses) if val_losses else float('inf')
        
        return objective
    
    def _convert_hyperparams(self, model_name, hyperparams):
        """ハイパーパラメータを適切な形式に変換"""
        converted = {}
        
        if model_name == 'ViT':
            converted['PATCH_SIZE'] = hyperparams['PATCH_SIZE']
            converted['N_DIM'] = 2**hyperparams['N_DIM']
            converted['N_HEADS'] = 2**hyperparams['N_HEADS']
            converted['MLP_DIM'] = 2**hyperparams['MLP_DIM']
            converted['DROPOUT'] = hyperparams['DROPOUT']
            converted['DEPTH'] = hyperparams['DEPTH']
            converted['USE_CLS_TOKEN'] = hyperparams['USE_CLS_TOKEN']
        
        elif model_name == 'CNN':
            converted['NUM_FILTER1'] = 2**hyperparams['NUM_FILTER1']
            converted['KERNEL_SIZE1'] = hyperparams['KERNEL_SIZE1']
            converted['NUM_FILTER2'] = 2**hyperparams['NUM_FILTER2']
            converted['KERNEL_SIZE2'] = hyperparams['KERNEL_SIZE2']
            converted['DROPOUT'] = hyperparams['DROPOUT']
        
        elif model_name == 'MLP-Mixer':
            converted['PATCH_SIZE'] = hyperparams['PATCH_SIZE']
            converted['TOKEN_DIM'] = 2**hyperparams['TOKEN_DIM']
            converted['CHANNEL_DIM'] = 2**hyperparams['CHANNEL_DIM']
            converted['MLP_DIM'] = 2**hyperparams['MLP_DIM']
            converted['N_BLOCK'] = 2**hyperparams['N_BLOCK']
        
        elif model_name == 'EfficientNet':
            converted['PRETRAINED'] = hyperparams['PRETRAINED']
            converted['DROPOUT'] = hyperparams['DROPOUT']
            converted['FREEZE_BACKBONE'] = hyperparams['FREEZE_BACKBONE']
        
        elif model_name == 'ResNet18':
            converted['PRETRAINED'] = hyperparams['PRETRAINED']
            converted['DROPOUT'] = hyperparams['DROPOUT']
        
        elif model_name == 'MobileNetV3':
            converted['PRETRAINED'] = hyperparams['PRETRAINED']
            converted['DROPOUT'] = hyperparams['DROPOUT']
        
        return converted
    
    def _save_multi_dataset_results(self, model_name, results):
        """複数データセットの結果を保存"""
        # 通常の結果保存
        self._save_cv_results(model_name, results)
        
        # データセット別の結果も保存
        dataset_scores_df = []
        for d_id, scores in results['dataset_specific_scores'].items():
            for score in scores:
                score['dataset_id'] = d_id
                dataset_scores_df.append(score)
        
        if dataset_scores_df:
            df = pd.DataFrame(dataset_scores_df)
            df.to_csv(self.output_dir / f'dataset_specific_scores_{model_name}.csv', index=False)
            
            # データセット別の平均スコアも計算
            avg_scores = df.groupby('dataset_id').agg({
                'rmse': 'mean',
                'correlation': 'mean',
                'n_samples': 'sum'
            }).round(4)
            avg_scores.to_csv(self.output_dir / f'dataset_average_scores_{model_name}.csv')


# *******************************************************************
# Main execution function
# *******************************************************************
from pyprojroot import here
def main():
    """メイン実行関数"""
    
    trial_name = "noto-v9-april"
    root_dir = here()
    data_dir = root_dir / 'data'
    data_annotation_dir = data_dir / "annotation"
    data_Noto_April_dir = data_dir / "Noto_April"
    data_sample_dir = data_Noto_April_dir / "annotation_sample"
    output_dir = root_dir / "output" / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 複数データセットの設定
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
        # 他のデータセットを追加
        # {
        #     'rgb_path': data_dir / 'images' / 'another_image.jpg',
        #     'dataset_name': 'another_dataset'
        # },
    ]
    
    # 設定
    config = {
        'dataset_configs': dataset_configs,
        'landslide_path': data_dir / 'noto_landslide' / 'hokai-all.geojson',
        'legacy_path': data_dir / 'annotation' / 'Porigon_april_fix.geojson',  # ←不要なら削除
        'output_dir': output_dir,
        'fig_dir': fig_dir,
        'epsg_code': 6675,
        'tip_size_m': 5,
        'slide_ratio': 0.5,
        'n_outer_cv': 10,
        'n_epochs': 64,
        'batch_size': 64,
        'model_names': ['CNN', 'ViT', 'MLP-Mixer', 'EfficientNet', 'ResNet18', 'MobileNetV3'],
        'n_trials': 32,
        'cv_strategy': 'balanced-dataset',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*80)
    print("Legacy Detection System - Multiple Datasets Version")
    print("="*80)
    print(f"Number of datasets: {len(dataset_configs)}")
    print(f"Device: {config['device']}")
    print(f"Models to train: {', '.join(config['model_names'])}")
    print(f"CV strategy: {config['cv_strategy']}")
    print("="*80)
    
    # Step 1: データ処理
    print("\n" + "="*80)
    print("Step 1: Processing multiple datasets")
    print("="*80)
    
    processor = MultiDatasetLandslideProcessor(
        dataset_configs=config['dataset_configs'],
        epsg_code=config['epsg_code'],
        resolution=0.2
    )
    
    tips, labels, pixel_locs, geo_locs, landslide_ids, dataset_ids = processor.process_all_datasets(
        landslide_path=config['landslide_path'],
        legacy_path=config['legacy_path'],
        tip_size_m=config['tip_size_m'],
        slide_ratio=config['slide_ratio']
    )
    
    # メタデータを保存
    processor.save_metadata(output_dir / 'dataset_metadata.json')
    
    # データ統計を表示
    print("\n" + "="*80)
    print("Data Statistics:")
    print("="*80)
    print(f"Total tips: {len(tips)}")
    print(f"Label distribution:")
    print(f"  Min: {np.min(labels):.4f}")
    print(f"  Max: {np.max(labels):.4f}")
    print(f"  Mean: {np.mean(labels):.4f}")
    print(f"  Std: {np.std(labels):.4f}")
    
    # データセット別の統計
    for d_id in np.unique(dataset_ids):
        mask = dataset_ids == d_id
        print(f"\nDataset {d_id}:")
        print(f"  Tips: {np.sum(mask)}")
        print(f"  Landslides: {len(np.unique(landslide_ids[mask]))}")
        print(f"  Mean label: {np.mean(labels[mask]):.4f}")
    
    # ラベル分布ヒストグラム
    plt.figure(figsize=(8,4))
    plt.hist(labels, bins=30, alpha=0.7)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution (All Data)')
    plt.tight_layout()
    plt.savefig(config['fig_dir'] / 'label_distribution.png', dpi=200)
    plt.close()

    # データセットごとのサンプル数
    plt.figure(figsize=(8,4))
    pd.Series(dataset_ids).value_counts().sort_index().plot(kind='bar')
    plt.xlabel('Dataset ID')
    plt.ylabel('Sample Count')
    plt.title('Samples per Dataset')
    plt.tight_layout()
    plt.savefig(config['fig_dir'] / 'samples_per_dataset.png', dpi=200)
    plt.close()
    
    # Step 2: データセットの作成
    print("\n" + "="*80)
    print("Step 2: Creating PyTorch dataset")
    print("="*80)
    
    # データ拡張transformを定義
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # 必要に応じて他のtransformも追加
    ])

    dataset = MultiDatasetLandslideDataset(
        images=tips,
        labels=labels,
        pixel_locations=pixel_locs,
        geo_locations=geo_locs,
        landslide_ids=landslide_ids,
        dataset_ids=dataset_ids,
        transform=train_transform,  # ここでtransformを渡す
        patch_size=25
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Step 3: モデルの訓練
    print("\n" + "="*80)
    print("Step 3: Training models with cross-validation")
    print("="*80)
    
    pipeline = EnhancedLegacyDetectionPipeline(
        output_dir=config['output_dir'],
        device=config['device'],
        random_state=42
    )
    
    cv_results = pipeline.train_with_multiple_datasets(
        dataset=dataset,
        landslide_ids=landslide_ids,
        dataset_ids=dataset_ids,
        model_names=config['model_names'],
        n_outer_cv=config['n_outer_cv'],
        n_trials=config['n_trials'],
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        cv_strategy=config['cv_strategy']
    )
    
    # Step 4: 結果のサマリー
    print("\n" + "="*80)
    print("Step 4: Results Summary")
    print("="*80)
    
    for model_name in config['model_names']:
        print(f"\n{model_name} Results:")
        scores = cv_results[model_name]['cv_scores']
        rmse_scores = [s['rmse'] for s in scores]
        corr_scores = [s['correlation'] for s in scores]
        
        print(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        print(f"  Correlation: {np.mean(corr_scores):.4f} ± {np.std(corr_scores):.4f}")
    
    # Step 5: 最終モデルの訓練
    print("\n" + "="*80)
    print("Step 5: Training final models on all data")
    print("="*80)
    
    final_models = pipeline.train_final_models(
        dataset=dataset,
        model_names=config['model_names'],
        cv_results=cv_results,
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size']
    )
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    print(f"Results saved to: {config['output_dir']}")
    print(f"Figures saved to: {config['fig_dir']}")
    
    # Step 2.5: Fold割り当ての可視化・分析用出力
    # fold割り当てを取得
    spatial_cv = MultiDatasetSpatialCV(
        n_splits=config['n_outer_cv'],
        strategy=config['cv_strategy'],
        random_state=42
    )
    # 最初のfold割り当てを取得
    fold_assignments = np.zeros(len(labels), dtype=int)
    for fold, (_, test_idx) in enumerate(spatial_cv.split(landslide_ids, dataset_ids)):
        fold_assignments[test_idx] = fold

    # GeoDataFrame作成
    geometry = [Point(lon, lat) for lon, lat in geo_locs]
    gdf = gpd.GeoDataFrame({
        'id': range(len(labels)),
        'label': labels,
        'landslide_id': landslide_ids,
        'dataset_id': dataset_ids,
        'fold': fold_assignments,
        'geometry': geometry
    }, crs=f"EPSG:{config['epsg_code']}")

    # UTM座標系に変換
    gdf_prj = gdf.to_crs(gdf.estimate_utm_crs())

    # 可視化
    n_folds = config['n_outer_cv']
    cols_tab = cm.get_cmap('tab20', n_folds)
    color_ramp = ListedColormap([cols_tab(i) for i in range(n_folds)])

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_prj.plot(column='fold', ax=ax, cmap=color_ramp, markersize=2, legend=True, legend_kwds={'label': 'Fold'})
    ax.set_title('Spatial K-Fold Assignment')
    plt.tight_layout()
    plt.savefig(config['fig_dir'] / 'spatial_kfold_assignment.png', dpi=300, bbox_inches='tight')
    plt.close()

    # fold割り当てcsvも出力
    gdf[['id', 'fold', 'landslide_id', 'dataset_id', 'label']].to_csv(config['output_dir'] / 'fold_assignments.csv', index=False)
    
    # foldごとのサンプル数・ラベル分布などをcsv出力
    fold_stats = []
    for fold in range(n_folds):
        mask = fold_assignments == fold
        fold_stats.append({
            'fold': fold,
            'n_samples': np.sum(mask),
            'mean_label': np.mean(labels[mask]),
            'std_label': np.std(labels[mask]),
            'n_landslides': len(np.unique(landslide_ids[mask])),
            'n_datasets': len(np.unique(dataset_ids[mask]))
        })
    pd.DataFrame(fold_stats).to_csv(config['output_dir'] / 'fold_statistics.csv', index=False)
    
    # アンサンブルモデルの評価
    print("\n" + "="*50)
    print("Creating Ensemble Model")
    print("="*50)
    model_weights = []
    for model_name in config['model_names']:
        scores_df = pd.DataFrame(cv_results[model_name]['cv_scores'])
        avg_corr = scores_df['correlation'].mean()
        model_weights.append(avg_corr)
    model_weights = np.array(model_weights) / np.sum(model_weights)
    print(f"Model weights: {dict(zip(config['model_names'], model_weights))}")
    ensemble_models = [cv_results[m]['models'][0] for m in config['model_names'] if cv_results[m]['models']]
    ensemble = WeightedEnsembleModel(
        models=ensemble_models,
        weights=model_weights,
        method='weighted_average'
    ).to(config['device'])
    # アンサンブルモデルの評価
    ensemble.eval()
    ensemble_preds = []
    ensemble_labels = []
    test_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(config['device'])
            labels_batch = batch['label'].cpu().numpy()
            outputs = ensemble(inputs).cpu().numpy().squeeze()
            ensemble_preds.extend(outputs)
            ensemble_labels.extend(labels_batch)
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr
    ensemble_rmse = np.sqrt(mean_squared_error(ensemble_labels, ensemble_preds))
    ensemble_corr, _ = pearsonr(ensemble_labels, ensemble_preds)
    print(f"\nEnsemble Results:")
    print(f"RMSE: {ensemble_rmse:.4f}")
    print(f"Correlation: {ensemble_corr:.4f}")

    # === 可視化とサマリー保存 ===
    from resume_training_v4 import plot_cv_scores, plot_ensemble_results
    # CVスコア可視化
    plot_cv_scores(cv_results, fig_dir)
    # アンサンブル結果可視化
    ensemble_results_data = {
        'ensemble_predictions': ensemble_preds,
        'labels': ensemble_labels,
        'weights': model_weights.tolist()
    }
    plot_ensemble_results(ensemble_results_data, fig_dir)
    # サマリー保存
    summary = {
        'individual_models': {
            model: {
                'mean_rmse': pd.DataFrame(cv_results[model]['cv_scores'])['rmse'].mean(),
                'std_rmse': pd.DataFrame(cv_results[model]['cv_scores'])['rmse'].std(),
                'mean_correlation': pd.DataFrame(cv_results[model]['cv_scores'])['correlation'].mean(),
                'std_correlation': pd.DataFrame(cv_results[model]['cv_scores'])['correlation'].std(),
            }
            for model in config['model_names']
        },
        'ensemble': {
            'rmse': ensemble_rmse,
            'correlation': ensemble_corr,
            'weights': dict(zip(config['model_names'], model_weights.tolist()))
        }
    }
    with open(output_dir / 'training_summary_extended.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return cv_results, final_models


# *******************************************************************
# Inference function for new data
# *******************************************************************
def inference_on_new_data(model_path, new_image_path, model_name, 
                         landslide_path, legacy_path, output_path,
                         tip_size_m=5, slide_ratio=0.5, batch_size=64,
                         device='cuda'):
    """新しいデータに対する推論実行"""
    
    print("\n" + "="*80)
    print("Inference on New Data")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Image: {new_image_path}")
    
    # データ処理
    try:
        processor = LandslideTipProcessor(
            rgb_paths=new_image_path,
            landslide_path=landslide_path,
            legacy_path=legacy_path,
            epsg_code=6675,
            resolution=0.2
        )
        tips, labels, pixel_locs, geo_locs, landslide_ids = processor.extract_tips(
            tip_size_m=tip_size_m,
            slide_ratio=slide_ratio
        )
    except Exception as e:
        print(f"Error in tip extraction: {e}")
        return None
    
    print(f"Extracted {len(tips)} tips for inference")
    
    # データセット作成
    try:
        dataset = LandslideDataset(
            images=tips,
            labels=labels,
            pixel_locations=pixel_locs,
            geo_locations=geo_locs,
            region_idx=None,
            transform=None,
            patch_size=25
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        import json
        with open(f'best_hyperparams_{model_name}.json', 'r') as f:
            best_hyperparams = json.load(f)
        model = init_model(model_name, best_hyperparams)
        if model is None:
            print(f"Model initialization failed for {model_name}.")
            return None
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        all_predictions = []
        all_locations = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running inference"):
                inputs = batch['image'].to(device)
                outputs = model(inputs).cpu().numpy()[:, 0]
                all_predictions.extend(outputs.tolist())
                all_locations.extend(list(zip(
                    batch['lon'].numpy(),
                    batch['lat'].numpy(),
                    batch['i'].numpy(),
                    batch['j'].numpy()
                )))
        results_df = pd.DataFrame({
            'prediction': all_predictions,
            'label': labels,
            'longitude': [loc[0] for loc in all_locations],
            'latitude': [loc[1] for loc in all_locations],
            'pixel_i': [loc[2] for loc in all_locations],
            'pixel_j': [loc[3] for loc in all_locations],
            'landslide_id': landslide_ids
        })
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # 統計を表示
        print("\nInference Statistics:")
        print(f"  Mean prediction: {np.mean(all_predictions):.4f}")
        print(f"  Std prediction: {np.std(all_predictions):.4f}")
        print(f"  Min prediction: {np.min(all_predictions):.4f}")
        print(f"  Max prediction: {np.max(all_predictions):.4f}")
        
        # 精度評価（ラベルが利用可能な場合）
        if not all(l == 0 for l in labels):
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            rmse = np.sqrt(mean_squared_error(labels, all_predictions))
            mae = mean_absolute_error(labels, all_predictions)
            correlation, _ = pearsonr(labels, all_predictions)
            
            print("\nAccuracy Metrics:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Correlation: {correlation:.4f}")
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return None
    finally:
        torch.cuda.empty_cache()
        gc.collect()
    
    return results_df


# *******************************************************************
# Batch inference for multiple new datasets
# *******************************************************************
def batch_inference(model_paths, new_dataset_configs, model_names,
                   landslide_path, legacy_path, output_dir,
                   tip_size_m=5, slide_ratio=0.5, batch_size=64):
    """複数の新しいデータセットに対してバッチ推論を実行"""
    
    print("\n" + "="*80)
    print("Batch Inference on Multiple Datasets")
    print("="*80)
    
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        model_results = {}
        model_path = model_paths[model_name]
        
        for config in new_dataset_configs:
            dataset_name = config.get('dataset_name', config['rgb_path'])
            print(f"\nProcessing {dataset_name}...")
            
            output_path = Path(output_dir) / f'predictions_{model_name}_{dataset_name}.csv'
            
            results_df = inference_on_new_data(
                model_path=model_path,
                new_image_path=config['rgb_path'],
                model_name=model_name,
                landslide_path=landslide_path,
                legacy_path=legacy_path,
                output_path=output_path,
                tip_size_m=tip_size_m,
                slide_ratio=slide_ratio,
                batch_size=batch_size
            )
            
            model_results[dataset_name] = results_df
        
        all_results[model_name] = model_results
    
    # 全体のサマリーを作成
    print("\n" + "="*80)
    print("Batch Inference Summary")
    print("="*80)
    
    summary_data = []
    for model_name, model_results in all_results.items():
        for dataset_name, df in model_results.items():
            summary_data.append({
                'model': model_name,
                'dataset': dataset_name,
                'n_samples': len(df),
                'mean_prediction': df['prediction'].mean(),
                'std_prediction': df['prediction'].std(),
                'min_prediction': df['prediction'].min(),
                'max_prediction': df['prediction'].max()
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(Path(output_dir) / 'batch_inference_summary.csv', index=False)
    print(summary_df)
    
    return all_results


# Original model classes (ViT, CNN, MLP-Mixer) remain the same
# Include all the model definitions from the original code here...
# [Previous model definitions would be included here]

# Also include the original LandslideTipProcessor class and other necessary components
# [Original utility classes would be included here]


# モデル初期化関数の拡張
def init_model_extended(model_name, hyperparams_dict, image_size=25):
    model = None
    try:
        if model_name == 'ViT':
            model = ImprovedViT(
                image_size=image_size,
                n_classes=1,
                depth=hyperparams_dict.get('DEPTH', 3),
                dim=hyperparams_dict['N_DIM'],
                patch_size=hyperparams_dict['PATCH_SIZE'],
                n_heads=hyperparams_dict['N_HEADS'],
                mlp_dim=hyperparams_dict['MLP_DIM'],
                dropout=hyperparams_dict['DROPOUT'],
                use_cls_token=hyperparams_dict.get('USE_CLS_TOKEN', True)
            )
        elif model_name == 'CNN':
            from legacy_detection_system_v3 import LeNet
            model = LeNet(
                num_filter1=hyperparams_dict['NUM_FILTER1'],
                kernel_size1=hyperparams_dict['KERNEL_SIZE1'],
                num_filter2=hyperparams_dict['NUM_FILTER2'],
                kernel_size2=hyperparams_dict['KERNEL_SIZE2'],
                dropout=hyperparams_dict['DROPOUT'],
                image_size=image_size
            )
        elif model_name == 'MLP-Mixer':
            from legacy_detection_system_v3 import MLPMixer
            model = MLPMixer(
                in_channels=3,
                image_size=image_size,
                num_classes=1,
                patch_size=hyperparams_dict['PATCH_SIZE'],
                dim=hyperparams_dict['MLP_DIM'],
                depth=hyperparams_dict['N_BLOCK'],
                token_dim=hyperparams_dict['TOKEN_DIM'],
                channel_dim=hyperparams_dict['CHANNEL_DIM']
            )
        elif model_name == 'EfficientNet':
            model = EfficientNetB0(
                num_classes=1,
                pretrained=hyperparams_dict.get('PRETRAINED', True),
                dropout=hyperparams_dict['DROPOUT'],
                freeze_backbone=hyperparams_dict.get('FREEZE_BACKBONE', False)
            )
        elif model_name == 'ResNet18':
            model = ResNet18Custom(
                num_classes=1,
                pretrained=hyperparams_dict.get('PRETRAINED', True),
                dropout=hyperparams_dict['DROPOUT'],
                input_size=image_size
            )
        elif model_name == 'MobileNetV3':
            model = MobileNetV3Small(
                num_classes=1,
                pretrained=hyperparams_dict.get('PRETRAINED', True),
                dropout=hyperparams_dict['DROPOUT']
            )
        else:
            print(f"Unknown model: {model_name}")
            return None
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}")
        return None
    return model

if __name__ == "__main__":
    main()