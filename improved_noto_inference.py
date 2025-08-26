# Utils -----------------------
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, zipfile, io
from pyprojroot import here
from PIL import Image
import random
from tqdm import tqdm
import warnings
import seaborn as sns
from einops import repeat
from einops.layers.torch import Rearrange
import itertools
import joblib
import gc
from datetime import datetime
from glob import glob
from math import floor
from pathlib import Path
import logging
import json

# Machine Learning ---------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import optuna
import timm
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.optim import create_optimizer
from timm.utils import AverageMeter, accuracy
from timm.utils.summary import update_summary
from scipy.stats import pearsonr

# Geospatial ----------------------------
import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
from rasterio.crs import CRS
# import h3pandas  # コメントアウト（使用されていない）

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    # encoding='utf-8'  # ← 削除
)
logger = logging.getLogger(__name__)

# *******************************************************************
# Define Functions
# *******************************************************************
# IO Functions ------------------------------
def jbl_saver(object, filename, compress=3):
    """joblibでオブジェクトを保存"""
    try:
        with open(filename, 'wb') as web:
            joblib.dump(object, web, compress=compress)
        logger.info(f"Saved: {filename}")
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")
        raise


def jbl_loader(filename):
    """joblibでオブジェクトを読み込み"""
    try:
        with open(filename, 'rb') as web:
            data = joblib.load(web)
        logger.info(f"Loaded: {filename}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        raise


# Create data loader for pytorch
def custum_loader(data, batch_size, shuffle=False, num_workers=0):
    """カスタムデータローダーの作成"""
    try:
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader
    except Exception as e:
        logger.error(f"Failed to create data loader: {e}")
        raise


# *******************************************************************
# Define models
# *******************************************************************
# ViT --------------------------------
class Patching(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=patch_size, pw=patch_size)

    def forward(self, x):
        x = self.net(x)
        return x


class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)

    def forward(self, x):
        x = self.net(x)
        return x


class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))

    def forward(self, x):
        batch_size, _, __ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.concat([cls_tokens, x], dim=1)
        x += self.pos_embedding
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim // n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h=self.n_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.concat = Rearrange("b h n d -> b n (h d)", h=self.n_heads)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        logit = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_heads ** -0.5)
        attention_weight = self.softmax(logit)
        attention_weight = self.dropout(attention_weight)

        output = torch.matmul(attention_weight, v)
        output = self.concat(output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, depth, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttention(dim=dim, n_heads=n_heads, dropout=dropout),
                MLP(dim=dim, hidden_dim=mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(self.norm(x)) + x
            x = ff(self.norm(x)) + x
        return x


class MLPHead(nn.Module):
    def __init__(self, dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.net(x)
        return torch.sigmoid(x), x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_classes, dim, depth, n_heads, channels=3, mlp_dim=256, dropout=0.):
        super().__init__()
        
        n_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size
        self.depth = depth

        self.patching = Patching(patch_size=patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim=patch_dim, dim=dim)
        self.embedding = Embedding(dim=dim, n_patches=n_patches)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(dim=dim, n_heads=n_heads, mlp_dim=mlp_dim, depth=depth, dropout=dropout)
        self.mlp_head = MLPHead(dim=dim, out_dim=n_classes, dropout=dropout)

    def forward(self, img):
        x = img
        x = self.patching(x)
        x = self.linear_projection_of_flattened_patches(x)
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = x[:, 0]
        x_out, _ = self.mlp_head(x)
        return x_out


# LeNet (CNN) --------------------------------
class LeNet(nn.Module):
    def __init__(self, num_filter1, kernel_size1, num_filter2, kernel_size2, dropout, image_size=25):
        super(LeNet, self).__init__()
        self.num_filter2 = num_filter2
        boader1 = int((kernel_size1 - 1)/2)
        boader2 = int((kernel_size2 - 1)/2)
        px_size1 = floor((image_size - 2*boader1)/2)
        self.px_size2 = floor((px_size1 - 2*boader2)/2)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_filter1, kernel_size=kernel_size1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=num_filter1, out_channels=num_filter2, kernel_size=kernel_size2, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=num_filter2 * self.px_size2 * self.px_size2, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=1)
        self.dropout = dropout

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_filter2 * self.px_size2 * self.px_size2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sigmoid(self.fc3(x))
        return x


# MLP-Mixer -------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )
    
    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
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
        return torch.sigmoid(self.mlp_head(x))


# *******************************************************************
# Model initialization and prediction functions
# *******************************************************************
def init_model(model_name, hyperparams_dict, image_size=25):
    """モデルの初期化"""
    try:
        if model_name == 'ViT':
            model = ViT(
                image_size=image_size,
                n_classes=1,
                depth=3,
                dim=hyperparams_dict['N_DIM'],
                patch_size=hyperparams_dict['PATCH_SIZE'],
                n_heads=hyperparams_dict['N_HEADS'],
                mlp_dim=hyperparams_dict['MLP_DIM'],
                dropout=hyperparams_dict['DROPOUT']
            ).to(device)
        elif model_name == 'CNN':
            model = LeNet(
                num_filter1=hyperparams_dict['NUM_FILTER1'],
                kernel_size1=hyperparams_dict['KERNEL_SIZE1'],
                num_filter2=hyperparams_dict['NUM_FILTER2'],
                kernel_size2=hyperparams_dict['KERNEL_SIZE2'],
                dropout=hyperparams_dict['DROPOUT'],
                image_size=image_size
            ).to(device)
        elif model_name == 'MLP-Mixer':
            model = MLPMixer(
                in_channels=3,
                image_size=image_size,
                num_classes=1,
                patch_size=hyperparams_dict['PATCH_SIZE'],
                dim=hyperparams_dict['MLP_DIM'],
                depth=hyperparams_dict['N_BLOCK'],
                token_dim=hyperparams_dict['TOKEN_DIM'],
                channel_dim=hyperparams_dict['CHANNEL_DIM']
            ).to(device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        logger.info(f"Initialized {model_name} model")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize {model_name}: {e}")
        raise


def predict_all(model, loader):
    """全データに対する予測"""
    model.eval()
    all_outputs, all_lon, all_lat = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            try:
                inputs = batch['image'].float().to(device)
                lon, lat = batch['lon'], batch['lat']
                outputs = model(inputs)
                
                all_outputs.extend(outputs.cpu().numpy()[:, 0].tolist())
                all_lon.extend(lon.numpy().tolist())
                all_lat.extend(lat.numpy().tolist())
            except Exception as e:
                logger.error(f"Error in prediction batch: {e}")
                continue
    
    return all_outputs, all_lon, all_lat


# *******************************************************************
# Main Inference Script
# *******************************************************************
def main():
    # Env settings ------------------------------
    warnings.filterwarnings('ignore')
    
    # パス設定
    abs_path = here()
    path_to_data = abs_path / 'data'
    path_to_alldata = path_to_data / "Noto_April"
    path_to_model = abs_path / "output" / "noto-v07-april-balanced"#モデル変更で変えるパス
    
    # 出力ディレクトリ
    trial_name = "noto-april-v07-april-inference"
    path_to_output = abs_path / "output" / trial_name
    path_to_prediction = path_to_output / "prediction"
    path_to_output.mkdir(parents=True, exist_ok=True)
    path_to_prediction.mkdir(parents=True, exist_ok=True)
    
    # ログファイル設定
    log_file = path_to_output / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("="*60)
    logger.info("推論処理開始")
    logger.info(f"出力先: {path_to_output}")
    logger.info("="*60)
    
    # パラメータ設定
    IMAGE_SIZE = 25
    BATCH_SIZE = 64
    epsg_wgs84 = 4326
    model_names = ['CNN', 'ViT', 'MLP-Mixer']
    
    # デバイス設定
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cuda")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # pickleファイルの検索
    pickle_files = list(path_to_alldata.glob("*-subset-*.pickle"))
    if not pickle_files:
        logger.error("pickleファイルが見つかりません")
        raise FileNotFoundError("No pickle files found")
    
    logger.info(f"見つかったpickleファイル数: {len(pickle_files)}")
    
    # 地区ごとに分類
    area_files = {}
    for pf in pickle_files:
        if "Noto_1_place" in pf.name:
            area_name = "noto1"
        elif "Noto_2_place" in pf.name:
            area_name = "noto2"
        else:
            # ファイル名から地区を推定
            area_name = pf.name.split('-')[0]
        
        if area_name not in area_files:
            area_files[area_name] = []
        area_files[area_name].append(pf)
    
    logger.info(f"地区別ファイル数: {[(k, len(v)) for k, v in area_files.items()]}")
    
    # モデルの読み込み
    models = {}
    for model_name in model_names:
        try:
            param_file = path_to_model / f'best_hyperparams_{model_name}.json'
            model_file = path_to_model / f'final_model_{model_name}.pth'

            if not param_file.exists() or not model_file.exists():
                logger.warning(f"{model_name}のモデルファイルが見つかりません")
                continue

            # JSONファイルはjoblibではなくjsonで読み込む
            with open(param_file, 'r', encoding='utf-8') as f:
                final_params = json.load(f)

            model = init_model(model_name, final_params, IMAGE_SIZE)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            models[model_name] = model
            logger.info(f"{model_name}モデルを読み込みました")
        except Exception as e:
            logger.error(f"{model_name}モデルの読み込みに失敗: {e}")
            continue
    
    if not models:
        logger.error("モデルが一つも読み込めませんでした")
        raise RuntimeError("No models loaded successfully")
    
    # 地区ごとに推論
    all_predictions = []
    
    for area_name, area_pickle_files in area_files.items():
        logger.info(f"\n{area_name}の処理開始 ({len(area_pickle_files)}ファイル)")
        
        for i, data_file in enumerate(sorted(area_pickle_files)):
            subset_name = data_file.stem
            logger.info(f"  処理中: {subset_name} ({i+1}/{len(area_pickle_files)})")
            
            try:
                # データ読み込み
                # import pdb; pdb.set_trace()  # デバッグ用
                data_all = jbl_loader(data_file)
                # imageは最初の3チャンネルを使用
                for i in tqdm(range(len(data_all))):
                    if 'image' in data_all[i]:
                        data_all[i]['image'] = data_all[i]['image'][:3]
                if len(data_all) == 0:
                    logger.warning(f"    {subset_name}は空のデータです（スキップ）")
                    continue
                
                logger.info(f"    データ数: {len(data_all)}")
                
                # 各モデルで推論
                predictions = {}
                for model_name, model in models.items():
                    loader_all = custum_loader(data_all, BATCH_SIZE, shuffle=False)
                    y_pred, lon, lat = predict_all(model, loader_all)
                    
                    if model_name == list(models.keys())[0]:
                        # 最初のモデルで座標情報を保存
                        predictions['lon'] = lon
                        predictions['lat'] = lat
                        predictions['area'] = area_name
                        predictions['subset'] = subset_name
                    
                    predictions[model_name] = y_pred
                    logger.info(f"    {model_name}の推論完了")
                
                # データフレーム作成
                pred_df = pd.DataFrame(predictions)
                
                # サブセットごとに保存
                subset_output = path_to_prediction / f'prediction-{area_name}-{subset_name}.parquet'
                pred_df.to_parquet(subset_output, index=False)
                all_predictions.append(pred_df)
                
                # メモリ解放
                del data_all, pred_df
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"    {subset_name}の処理でエラー: {e}")
                continue
    
    # 全データの統合
    if all_predictions:
        logger.info("\n全データの統合開始")
        df_all = pd.concat(all_predictions, ignore_index=True)
        logger.info(f"総データ数: {len(df_all):,}")
        
        # GeoDataFrame作成
        pred_all_gdf = gpd.GeoDataFrame(
            df_all,
            geometry=gpd.points_from_xy(df_all.lon, df_all.lat, crs=f'EPSG:{epsg_wgs84}')
        )
        
        # アンサンブル統計量の計算
        model_cols = [col for col in df_all.columns if col in model_names]
        if len(model_cols) > 1:
            pred_all_gdf['ensemble_mean'] = pred_all_gdf[model_cols].mean(axis=1)
            pred_all_gdf['ensemble_std'] = pred_all_gdf[model_cols].std(axis=1)
            pred_all_gdf['ensemble_cv'] = pred_all_gdf['ensemble_std'] / (pred_all_gdf['ensemble_mean'] + 1e-8)
        
        # 統計情報の出力
        logger.info("\n予測値の統計:")
        for col in model_cols + ['ensemble_mean']:
            if col in pred_all_gdf.columns:
                logger.info(f"  {col}: mean={pred_all_gdf[col].mean():.4f}, std={pred_all_gdf[col].std():.4f}")
        
        # 保存
        output_base = f'prediction-noto-all-epsg{epsg_wgs84}'
        pred_all_gdf.to_parquet(path_to_prediction / f'{output_base}.parquet', index=False)
        pred_all_gdf.to_file(path_to_prediction / f'{output_base}.geojson', driver='GeoJSON')
        
        # 地区別の統計
        logger.info("\n地区別の統計:")
        for area in pred_all_gdf['area'].unique():
            area_data = pred_all_gdf[pred_all_gdf['area'] == area]
            logger.info(f"  {area}: {len(area_data):,}点")
        
        logger.info("\n推論処理完了")
        logger.info(f"結果保存先: {path_to_prediction}")
    else:
        logger.error("予測結果が一つもありません")


if __name__ == "__main__":
    main()