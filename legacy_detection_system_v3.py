"""
土砂崩れ地域のレガシー検出システム - 修正版
JPG+JGW形式の航空写真に対応
ジオメトリエラーを修正
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
from shapely.geometry import box, Point
import geopandas as gpd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
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
# import h3pandas
import rasterio
import rasterio.features as rasioftr
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
# PyTorch Dataset
# *******************************************************************
class LandslideDataset(Dataset):
    """PyTorch用のデータセットクラス"""
    
    def __init__(self, images, labels, pixel_locations, geo_locations, region_idx=None, transform=None, patch_size=25):
        self.images = images
        self.labels = labels
        self.pixel_locations = pixel_locations
        self.geo_locations = geo_locations
        self.region_idx = region_idx
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
        result = {
            'image': image,
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'i': self.pixel_locations[idx][0],
            'j': self.pixel_locations[idx][1],
            'lon': self.geo_locations[idx][0],
            'lat': self.geo_locations[idx][1],
        }
        if self.region_idx is not None:
            result['region_idx'] = self.region_idx[idx]
        return result


# *******************************************************************
# Define models (ViT, CNN, MLP-Mixer)
# *******************************************************************
# ViT Components --------------------------------
class Patching(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_size, pw = patch_size)

    def forward(self, x):
        return self.net(x)


class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)

    def forward(self, x):
        return self.net(x)


class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))

    def forward(self, x):
        batch_size, _, __ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b = batch_size)
        x = torch.concat([cls_tokens, x], dim = 1)
        x += self.pos_embedding
        return x


class MLP(nn.Module):
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


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout = 0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim // n_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h = self.n_heads)
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.concat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)

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
    def __init__(self, dim, n_heads, mlp_dim, depth, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttention(dim = dim, n_heads = n_heads, dropout = dropout),
                MLP(dim = dim, hidden_dim = mlp_dim, dropout = dropout)
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
        return F.sigmoid(x), x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, n_classes, dim, depth, n_heads, 
                 channels = 3, mlp_dim = 256, dropout = 0.):
        super().__init__()
        n_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size
        self.depth = depth
        self.patching = Patching(patch_size = patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, dim = dim)
        self.embedding = Embedding(dim = dim, n_patches = n_patches)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, 
                                                     mlp_dim = mlp_dim, depth = depth, 
                                                     dropout=dropout)
        self.mlp_head = MLPHead(dim = dim, out_dim = n_classes, dropout = dropout)

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


# CNN (LeNet) --------------------------------
class LeNet(nn.Module):
    def __init__(self, num_filter1, kernel_size1, num_filter2, kernel_size2, dropout, image_size=25):
        super(LeNet, self).__init__()
        self.num_filter2 = num_filter2
        boader1 = int((kernel_size1 - 1)/2)
        boader2 = int((kernel_size2 - 1)/2)
        px_size1 = floor((image_size - 2*boader1)/2)
        self.px_size2 = floor((px_size1 - 2*boader2)/2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_filter1, 
                              kernel_size=kernel_size1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=num_filter1, out_channels=num_filter2,
                              kernel_size=kernel_size2, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=num_filter2 * self.px_size2 * self.px_size2, 
                            out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=1)
        self.dropout = dropout

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_filter2 * self.px_size2 * self.px_size2)
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout)
        x = F.sigmoid(self.fc3(x))
        return x


# MLP-Mixer --------------------------------
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
            FeedForward(dim, channel_dim, dropout),
        )
    
    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, 
                 depth, token_dim, channel_dim):
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
        return F.sigmoid(self.mlp_head(x))


# *******************************************************************
# Training & Validation Functions
# *******************************************************************
def init_model(model_name, hyperparams_dict, image_size=25):
    """モデルの初期化"""
    if model_name == 'ViT':
        model = ViT(image_size=image_size,
                    n_classes=1,
                    depth=3,
                    dim=hyperparams_dict['N_DIM'],
                    patch_size=hyperparams_dict['PATCH_SIZE'],
                    n_heads=hyperparams_dict['N_HEADS'],
                    mlp_dim=hyperparams_dict['MLP_DIM'],
                    dropout=hyperparams_dict['DROPOUT'])
    elif model_name == 'CNN':
        model = LeNet(num_filter1=hyperparams_dict['NUM_FILTER1'], 
                     kernel_size1=hyperparams_dict['KERNEL_SIZE1'], 
                     num_filter2=hyperparams_dict['NUM_FILTER2'], 
                     kernel_size2=hyperparams_dict['KERNEL_SIZE2'], 
                     dropout=hyperparams_dict['DROPOUT'],
                     image_size=image_size)
    elif model_name == 'MLP-Mixer':
        model = MLPMixer(in_channels=3, 
                        image_size=image_size, 
                        num_classes=1,
                        patch_size=hyperparams_dict['PATCH_SIZE'], 
                        dim=hyperparams_dict['MLP_DIM'], 
                        depth=hyperparams_dict['N_BLOCK'], 
                        token_dim=hyperparams_dict['TOKEN_DIM'], 
                        channel_dim=hyperparams_dict['CHANNEL_DIM'])
    return model


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, transform, device):
    """1エポックの訓練"""
    model.train()
    epoch_loss = 0
    all_outputs, all_labels = [], []
    
    for batch in tqdm(loader, desc=f'Training epoch {epoch}'):
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(transform(inputs))
        loss = loss_fn(outputs, labels.reshape(-1, 1))
        
        all_outputs.extend(outputs.cpu().detach().numpy()[:, 0].tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() / len(loader)
    
    return epoch_loss, all_outputs, all_labels


def validate_one_epoch(epoch, model, loader, loss_fn, device):
    """1エポックの検証"""
    model.eval()
    epoch_loss = 0
    all_outputs, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Validating epoch {epoch}'):
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.reshape(-1, 1))
            
            all_outputs.extend(outputs.cpu().numpy()[:, 0].tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            epoch_loss += loss.item() / len(loader)
    
    return epoch_loss, all_outputs, all_labels


# *******************************************************************
# Hyperparameter Optimization
# *******************************************************************

# copied from atsuma legacy_detection_v04_ensemble.py //////////////////////////////////////////
# Bayesian optimizer ---------------------------
def opt_model(trial, model_name, data, point, train_index, config):
    inner_hyperparams_dict = hyperparams_generator(model_name)
    if model_name == 'ViT':
        inner_hyperparams_dict['PATCH_SIZE'] = trial.suggest_int('PATCH_SIZE', 5, 5)
        inner_hyperparams_dict['N_DIM'] = 2**trial.suggest_int('N_DIM', 5, 8)
        inner_hyperparams_dict['N_HEADS'] = 2**trial.suggest_int('N_HEADS', 1, 4)
        inner_hyperparams_dict['MLP_DIM'] = 2**trial.suggest_int('MLP_DIM', 5, 8)
        inner_hyperparams_dict['DROPOUT'] = trial.suggest_discrete_uniform('DROPOUT', 0.0, 0.5, 0.1)
    elif model_name == 'CNN':
    # CNN hyperparams -------------------------
        inner_hyperparams_dict['NUM_FILTER1'] = 2**trial.suggest_int('NUM_FILTER1', 3, 6)
        inner_hyperparams_dict['KERNEL_SIZE1'] = trial.suggest_int('KERNEL_SIZE1', 3, 5, 2)
        inner_hyperparams_dict['NUM_FILTER2'] = 2**trial.suggest_int('NUM_FILTER2', 3, 6)
        inner_hyperparams_dict['KERNEL_SIZE2'] = trial.suggest_int('KERNEL_SIZE2', 3, 5, 2)
        inner_hyperparams_dict['DROPOUT'] = trial.suggest_discrete_uniform('DROPOUT', 0.0, 0.5, 0.1)
    elif model_name == 'MLP-Mixer':
        # MLP-Mixer hyperparams -------------------------
        inner_hyperparams_dict['PATCH_SIZE'] = trial.suggest_int('PATCH_SIZE', 5, 5)
        inner_hyperparams_dict['TOKEN_DIM'] = 2**trial.suggest_int('TOKEN_DIM', 5, 8)
        inner_hyperparams_dict['CHANNEL_DIM'] = 2**trial.suggest_int('CHANNEL_DIM', 5, 8)
        inner_hyperparams_dict['MLP_DIM'] = 2**trial.suggest_int('MLP_DIM', 5, 8)
        inner_hyperparams_dict['N_BLOCK'] = 2**trial.suggest_int('N_BLOCK', 1, 4)
    # params inner CV 
    inner_data = data[train_index]
    inner_point = point[train_index]
    inner_region = KMeans(n_clusters = N_INNER_CV, random_state=0).fit(inner_point)
    inner_test_losses_list = []
    for inner_cv in range(N_INNER_CV):
        # timer 
        print(f'{model_name}: outer_cv-{outer_cv} inner_cv-{inner_cv} processing....')
        MAX_TIMES = N_OUTER_CV*N_INNER_CV*N_TRIAL*len(model_names)
        global counter, previous_time
        counter += 1
        previous_time = datetime.now()
        # data loader
        inner_train_index, inner_test_index = np.where(inner_region.labels_!=inner_cv)[0], np.where(inner_region.labels_==inner_cv)[0]
        inner_data_train, inner_data_val = data_splitter_cv(inner_data, inner_train_index, inner_test_index)
        inner_loader_train, inner_loader_val = custum_loader(inner_data_train, INNER_BATCH_SIZE, shuffle=True), custum_loader(inner_data_val, INNER_BATCH_SIZE, shuffle=True)
        # Initialize model
        model = init_model(model_name, inner_hyperparams_dict)
        # Initialize loss & optimizer
        loss_fn = nn.MSELoss().to(device)
        optimizer = create_optimizer(args, model)
        inner_train_losses, inner_test_losses = [], []
        for epoch in range(N_INNER_EPOCHS):
            # Train
            inner_epoch_train_loss = train_one_epoc(inner_cv, epoch, N_INNER_EPOCHS, model, inner_loader_train, optimizer, loss_fn, transform, args)
            inner_train_losses.append(inner_epoch_train_loss)
            # Validation
            inner_epoch_test_loss = validate_one_epoc(outer_cv, epoch, N_INNER_EPOCHS, model, inner_loader_val, loss_fn)
            inner_test_losses.append(inner_epoch_test_loss)
        inner_test_losses_list.append(np.mean(inner_test_losses))
        print((MAX_TIMES - counter), '/', MAX_TIMES, ' finish on ', (MAX_TIMES - counter)*(datetime.now()-previous_time)+datetime.now())
    return np.mean(inner_test_losses_list)


def hyperparams_generator(model_name):
    hyperparams_dict = {}
    if model_name == 'ViT':
        hyperparams_dict['PATCH_SIZE'] = 0 
        hyperparams_dict['N_DIM'] = 0
        hyperparams_dict['N_HEADS'] = 0
        hyperparams_dict['MLP_DIM'] = 0
        hyperparams_dict['DROPOUT'] = 0
    elif model_name == 'CNN':
    # CNN hyperparams -------------------------
        hyperparams_dict['NUM_FILTER1'] = 0
        hyperparams_dict['KERNEL_SIZE1'] = 0
        hyperparams_dict['NUM_FILTER2'] = 0
        hyperparams_dict['KERNEL_SIZE2'] = 0
        hyperparams_dict['DROPOUT'] = 0
    elif model_name == 'MLP-Mixer':
        # MLP-Mixer hyperparams -------------------------
        hyperparams_dict['PATCH_SIZE'] = 0
        hyperparams_dict['TOKEN_DIM'] = 0
        hyperparams_dict['CHANNEL_DIM'] = 0
        hyperparams_dict['MLP_DIM'] = 0
        hyperparams_dict['N_BLOCK'] = 0
    return hyperparams_dict


def best_hyperparams_converter(model_name, hyperparams_dict, study):
    if model_name == 'ViT':
        hyperparams_dict['PATCH_SIZE'] = study.best_params['PATCH_SIZE']
        hyperparams_dict['N_DIM'] = 2**study.best_params['N_DIM']
        hyperparams_dict['N_HEADS'] = 2**study.best_params['N_HEADS']
        hyperparams_dict['MLP_DIM'] = 2**study.best_params['MLP_DIM']
        hyperparams_dict['DROPOUT'] = study.best_params['DROPOUT']
    elif model_name == 'CNN':
    # CNN hyperparams -------------------------
        hyperparams_dict['NUM_FILTER1'] = 2**study.best_params['NUM_FILTER1']
        hyperparams_dict['KERNEL_SIZE1'] = study.best_params['KERNEL_SIZE1']
        hyperparams_dict['NUM_FILTER2'] = 2**study.best_params['NUM_FILTER2']
        hyperparams_dict['KERNEL_SIZE2'] = study.best_params['KERNEL_SIZE2']
        hyperparams_dict['DROPOUT'] = study.best_params['DROPOUT']
    elif model_name == 'MLP-Mixer':
        # MLP-Mixer hyperparams -------------------------
        hyperparams_dict['PATCH_SIZE'] = study.best_params['PATCH_SIZE']
        hyperparams_dict['TOKEN_DIM'] = 2**study.best_params['TOKEN_DIM']
        hyperparams_dict['CHANNEL_DIM'] = 2**study.best_params['CHANNEL_DIM']
        hyperparams_dict['MLP_DIM'] = 2**study.best_params['MLP_DIM']
        hyperparams_dict['N_BLOCK'] = 2**study.best_params['N_BLOCK']
    return hyperparams_dict

# /////////////////////////////////////////////////////////////////////////////////////////////







































def create_objective(model_name, train_data, train_landslide_ids, device, args, 
                    n_inner_cv=5, n_inner_epochs=8, batch_size=64):
    """Optuna用の目的関数を作成"""
    
    def objective(trial):
        # ハイパーパラメータの提案
        hyperparams = {}
        
        if model_name == 'ViT':
            hyperparams['PATCH_SIZE'] = trial.suggest_int('PATCH_SIZE', 5, 5)
            hyperparams['N_DIM'] = 2**trial.suggest_int('N_DIM', 5, 8)
            hyperparams['N_HEADS'] = 2**trial.suggest_int('N_HEADS', 1, 4)
            hyperparams['MLP_DIM'] = 2**trial.suggest_int('MLP_DIM', 5, 8)
            hyperparams['DROPOUT'] = trial.suggest_float('DROPOUT', 0.0, 0.5, step=0.1)
        
        elif model_name == 'CNN':
            hyperparams['NUM_FILTER1'] = 2**trial.suggest_int('NUM_FILTER1', 3, 6)
            hyperparams['KERNEL_SIZE1'] = trial.suggest_int('KERNEL_SIZE1', 3, 5, step=2)
            hyperparams['NUM_FILTER2'] = 2**trial.suggest_int('NUM_FILTER2', 3, 6)
            hyperparams['KERNEL_SIZE2'] = trial.suggest_int('KERNEL_SIZE2', 3, 5, step=2)
            hyperparams['DROPOUT'] = trial.suggest_float('DROPOUT', 0.0, 0.5, step=0.1)
        
        elif model_name == 'MLP-Mixer':
            hyperparams['PATCH_SIZE'] = trial.suggest_int('PATCH_SIZE', 5, 5)
            hyperparams['TOKEN_DIM'] = 2**trial.suggest_int('TOKEN_DIM', 5, 8)
            hyperparams['CHANNEL_DIM'] = 2**trial.suggest_int('CHANNEL_DIM', 5, 8)
            hyperparams['MLP_DIM'] = 2**trial.suggest_int('MLP_DIM', 5, 8)
            hyperparams['N_BLOCK'] = 2**trial.suggest_int('N_BLOCK', 1, 4)
        
        # 内部交差検証
        unique_landslides = np.unique(train_landslide_ids)
        kf = KFold(n_splits=n_inner_cv, shuffle=True, random_state=42)
        val_losses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(unique_landslides)):
            # 土砂崩れIDベースでデータを分割
            train_landslides = unique_landslides[train_idx]
            val_landslides = unique_landslides[val_idx]
            
            train_mask = np.isin(train_landslide_ids, train_landslides)
            val_mask = np.isin(train_landslide_ids, val_landslides)
            
            train_subset = torch.utils.data.Subset(train_data, np.where(train_mask)[0])
            val_subset = torch.utils.data.Subset(train_data, np.where(val_mask)[0])
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            
            # モデルの訓練
            model = init_model(model_name, hyperparams).to(device)
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
            
            val_losses.append(best_val_loss)
        
        return np.mean(val_losses)
    
    return objective








# *******************************************************************
# Main Training Pipeline
# *******************************************************************
class LegacyDetectionPipeline:
    """レガシー検出のメインパイプライン"""
    
    def __init__(self, output_dir, device='cuda', random_state=42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        
        # データ拡張の定義
        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
        ])
    
    def _save_learning_curve(self, train_losses, test_losses, model_name, fold):
        """学習曲線を保存"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'{model_name} - Fold {fold + 1}')
        plt.legend()
        plt.savefig(self.output_dir / f'learning_curve_{model_name}_fold{fold}.png')
        plt.close()
    
    def _save_cv_results(self, model_name, results):
        """交差検証の結果を保存"""
        import pandas as pd
        import json
        scores_df = pd.DataFrame(results['cv_scores'])
        scores_df.to_csv(self.output_dir / f'cv_scores_{model_name}.csv', index=False)
        
        predictions_data = []
        for pred_dict in results['cv_predictions']:
            fold = pred_dict['fold']
            for pred, label in zip(pred_dict['predictions'], pred_dict['labels']):
                predictions_data.append({
                    'fold': fold,
                    'prediction': pred,
                    'label': label
                })
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(self.output_dir / f'cv_predictions_{model_name}.csv', index=False)
        
        with open(self.output_dir / f'best_hyperparams_{model_name}.json', 'w') as f:
            json.dump(results['best_hyperparams'], f, indent=2)

    def train_with_cv(self, dataset, landslide_ids, model_names, n_outer_cv=10, 
                      n_trials=32, n_epochs=64, batch_size=64):
        """交差検証による訓練"""
        cv_results = {}
        
        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            # ハイパーパラメータ最適化
            print("Optimizing hyperparameters...")
            objective = create_objective(
                model_name=model_name,
                train_data=dataset,
                train_landslide_ids=landslide_ids,
                device=self.device,
                args=None,
                n_inner_cv=5,
                n_inner_epochs=8,
                batch_size=batch_size
            )
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            best_hyperparams = study.best_params
            print(f"Best hyperparameters: {best_hyperparams}")
            
            # 外部交差検証
            print("\nStarting outer cross-validation...")
            spatial_cv = SpatialCrossValidator(n_splits=n_outer_cv, random_state=self.random_state)
            cv_scores = []
            cv_predictions = []
            
            for fold, (train_idx, test_idx) in enumerate(spatial_cv.split(landslide_ids)):
                print(f"\nFold {fold + 1}/{n_outer_cv}")
                
                # データローダーの作成
                train_subset = torch.utils.data.Subset(dataset, train_idx)
                test_subset = torch.utils.data.Subset(dataset, test_idx)
                
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
                
                # モデルの初期化と訓練
                model = init_model(model_name, best_hyperparams).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                loss_fn = nn.MSELoss()
                
                train_losses = []
                test_losses = []
                
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
                
                # 学習曲線の保存
                self._save_learning_curve(train_losses, test_losses, model_name, fold)
                
                # 最終的な評価
                model.eval()
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        inputs = batch['image'].to(self.device)
                        labels = batch['label'].cpu().numpy()
                        outputs = model(inputs).cpu().numpy()[:, 0]
                        
                        all_preds.extend(outputs.tolist())
                        all_labels.extend(labels.tolist())
                
                # メトリクスの計算
                rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
                correlation, _ = pearsonr(all_labels, all_preds)
                
                cv_scores.append({
                    'fold': fold,
                    'rmse': rmse,
                    'correlation': correlation
                })
                
                cv_predictions.append({
                    'fold': fold,
                    'predictions': all_preds,
                    'labels': all_labels
                })
            
            # 結果の保存
            cv_results[model_name] = {
                'best_hyperparams': best_hyperparams,
                'cv_scores': cv_scores,
                'cv_predictions': cv_predictions
            }
            
            self._save_cv_results(model_name, cv_results[model_name])
        
        return cv_results

    def save_kfold_indices(self, all_indices, model_name):
        """k-fold分割のインデックスを保存"""
        import json
        output_path = self.output_dir / f'kfold_indices_{model_name}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_indices, f, indent=2)
        print(f"k-fold分割インデックスを保存しました: {output_path}")


    def train_final_models(self, dataset, model_names, cv_results, n_epochs=64, batch_size=64):
        """最終モデルの訓練（全データを使用）"""
        final_models = {}
        
        for model_name in model_names:
            print(f"\nTraining final {model_name} model...")
            
            # ベストハイパーパラメータでモデルを初期化
            best_hyperparams = cv_results[model_name]['best_hyperparams']
            model = init_model(model_name, best_hyperparams).to(self.device)
            
            # 全データで訓練
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            
            for epoch in range(n_epochs):
                train_loss, _, _ = train_one_epoch(
                    epoch, model, train_loader, optimizer, loss_fn,
                    self.transform, self.device
                )
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
            
            # モデルの保存
            torch.save(model.state_dict(), self.output_dir / f'final_model_{model_name}.pth')
            final_models[model_name] = model
        
        return final_models
