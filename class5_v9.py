# 5クラス分類をワンストップで行う
# 混同行列、fold毎の評価、閾値最適化、ハイパーパラメータ探索機能を追加
# アンサンブル学習対応

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import re
from tqdm import tqdm
import warnings
import itertools
import joblib
import gc
from datetime import datetime
from glob import glob
from math import floor, isclose
from pathlib import Path
import pickle
from shapely.geometry import box, Point
import geopandas as gpd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (mean_squared_error, accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, 
                           classification_report, roc_auc_score, balanced_accuracy_score)
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, gaussian_kde
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
# Machine Learning ---------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models  # ResNet用に追加
from torch.utils.data import Dataset, DataLoader
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import optuna
from optuna.samplers import TPESampler
# import h3pandas
def _configure_dll_search_paths() -> List[Path]:
    """WindowsでGDAL/PROJ系DLLを確実に見つけられるようDLL検索パスを追加する。"""
    configured_paths: List[Path] = []
    if os.name != 'nt':
        return configured_paths

    candidates: List[Path] = []
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        base = Path(conda_prefix)
        candidates.extend([
            base / 'Library' / 'bin',
            base / 'Library' / 'lib',
            base / 'Library',
        ])

    env_hint = os.environ.get('RASTERIO_DLL_PATH')
    if env_hint:
        candidates.append(Path(env_hint))

    for candidate in candidates:
        if not candidate or not candidate.exists():
            continue
        try:
            os.add_dll_directory(str(candidate))  # type: ignore[attr-defined]
        except (AttributeError, FileNotFoundError, OSError):
            # Fallback: prepend to PATH if add_dll_directory is unavailable
            current_path = os.environ.get('PATH', '')
            candidate_str = str(candidate)
            if candidate_str not in current_path.split(os.pathsep):
                os.environ['PATH'] = candidate_str + os.pathsep + current_path
        configured_paths.append(candidate)

    return configured_paths
_CONFIGURED_DLL_DIRS = _configure_dll_search_paths()
try:
    import rasterio
    import rasterio.features as rasioftr
    from rasterio.warp import reproject, Resampling
    from rasterio.merge import merge as rio_merge
    from rasterio.vrt import WarpedVRT
except OSError as exc:
    conda_prefix = os.environ.get('CONDA_PREFIX', '（未検出）')
    hint = (
        "rasterio / GDAL の依存DLLを読み込めませんでした。"
        "Windows環境の場合は、使用中のConda環境の 'Library\\bin' および 'Library\\lib' を"
        " PATH へ追加するか、同フォルダへのフルパスを `RASTERIO_DLL_PATH` として指定してください。"
        f" 現在検知したConda環境: {conda_prefix}. エラーメッセージ: {exc}"
    )
    raise RuntimeError(hint) from exc
from scipy.ndimage import zoom
warnings.filterwarnings('ignore')

FIVE_CLASS_FIXED_THRESHOLDS = [0.0, 0.25, 0.5, 0.75]
# 日本語フォントの設定（利用可能な場合）
try:
    import matplotlib.font_manager as fm
    # 日本語フォントを探す
    jp_fonts = [f.name for f in fm.fontManager.ttflist if 'Japan' in f.name or 'Noto' in f.name]
    if jp_fonts:
        plt.rcParams['font.sans-serif'] = [jp_fonts[0], 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import geopandas as gpd
from matplotlib import cm, colormaps
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from affine import Affine
import hashlib
from datetime import datetime
from shapely.geometry import Point
warnings.filterwarnings('ignore')

try:
    from osgeo import gdal, osr
except ImportError:
    gdal = None
    osr = None

# 閾値定義-------------------------------------------------------
FIVE_CLASS_FIXED_THRESHOLDS = [0.0, 0.25, 0.5, 0.75]


# *******************************************************************
# Overview Utilities (GDAL overviews builder)
# *******************************************************************
def _has_overviews(path: Path) -> bool:
    """Check if the raster has overviews (any band)."""
    try:
        if gdal is not None:
            ds = gdal.Open(str(path), gdal.GA_ReadOnly)
            if ds is None:
                return False
            try:
                band = ds.GetRasterBand(1)
                return bool(band and band.GetOverviewCount() > 0)
            finally:
                ds = None
        else:
            with rasterio.open(path) as src:
                ovs = src.overviews(1) if src.count else []
                return bool(ovs and len(ovs) > 0)
    except Exception:
        return False


def _default_overview_levels(width: int, height: int, min_dim: int = 512) -> List[int]:
    """Suggest overview factors (2x pyramid) until the smallest dimension drops below min_dim."""
    levels: List[int] = []
    level = 2
    smallest = min(width, height)
    while smallest / level >= max(1, min_dim):
        levels.append(level)
        # avoid extremely long chains
        if level >= 1024:
            break
        level *= 2
    # Ensure we have at least one level for very large rasters
    if not levels and smallest > min_dim:
        levels = [2, 4, 8]
    return levels


def ensure_raster_overviews(path: Path, levels: Optional[List[int]] = None, resampling: str = "average") -> bool:
    """Ensure a raster has GDAL overviews (.ovr) on disk.

    Returns True if overviews exist or were created successfully.
    If GDAL is unavailable, returns False without raising.
    """
    p = Path(path)
    if not p.exists():
        return False

    # Fast path: already present
    if _has_overviews(p):
        return True

    # Try to compute sensible default levels if not provided
    try:
        with rasterio.open(p) as src:
            if levels is None:
                levels = _default_overview_levels(src.width, src.height)
    except Exception:
        # If we can't open via rasterio, leave levels as-is and let GDAL attempt
        levels = levels or [2, 4, 8, 16, 32, 64]

    if not levels:
        # Nothing to build
        return False

    if gdal is None:
        # Rasterio fallback (build external overviews via GDAL bindings not available)
        try:
            with rasterio.open(p, 'r+') as dst:
                dst.build_overviews(levels, Resampling.average)
                # annotate overview resampling for GDAL tooling
                dst.update_tags(ns='rio_overview', resampling=resampling)
            return _has_overviews(p)
        except Exception:
            return False

    # Use GDAL directly (preferred and more robust for VRT/GTiff)
    try:
        # Compression and BigTIFF options for external .ovr
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
        gdal.SetConfigOption('INTERLEAVE_OVERVIEW', 'PIXEL')
        gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'IF_SAFER')

        ds = gdal.Open(str(p), gdal.GA_ReadOnly)
        if ds is None:
            return False
        try:
            band = ds.GetRasterBand(1)
            if band is None:
                return False
            # Map resampling string to GDAL enum
            resampling_map = {
                'nearest': gdal.GRA_NearestNeighbour,
                'bilinear': gdal.GRA_Bilinear,
                'cubic': gdal.GRA_Cubic,
                'cubicspline': gdal.GRA_CubicSpline,
                'lanczos': gdal.GRA_Lanczos,
                'average': gdal.GRA_Average,
                'mode': gdal.GRA_Mode,
                'gauss': gdal.GRA_Gauss,
            }
            gdal_resampling = resampling_map.get(resampling.lower(), gdal.GRA_Average)
            # Build overviews for all bands
            for bidx in range(1, ds.RasterCount + 1):
                b = ds.GetRasterBand(bidx)
                if b is None:
                    continue
                b.BuildOverviews(gdal_resampling, levels)
        finally:
            ds = None
        return _has_overviews(p)
    except Exception:
        return False


# *******************************************************************
# Balanced Spatial K-Fold Implementation
# *******************************************************************
class BalancedSpatialKFold:
    """
    空間的K分割を行いながら、各フォールドのサンプル数を可能な限り均等にするクラス
    """
    
    def __init__(self, n_splits=5, min_samples_per_fold=1000, random_state=42):
        self.n_splits = n_splits
        self.min_samples_per_fold = min_samples_per_fold
        self.random_state = random_state
        
    def split(self, X_coords, y=None, groups=None):
        """
        X_coords: (n_samples, 2) array of coordinates [lon, lat] or [x, y]
        y: labels (optional)
        groups: group labels like landslide_ids (optional)
        
        Returns: generator of (train_idx, test_idx) tuples
        """
        n_samples = len(X_coords)
        
        # 初期クラスタリング
        kmeans = KMeans(n_clusters=self.n_splits * 3, random_state=self.random_state)
        initial_clusters = kmeans.fit_predict(X_coords)
        
        # クラスタを再グループ化して均等な分割を作成
        fold_assignments = self._balance_clusters(
            X_coords, initial_clusters, self.n_splits, groups
        )
        
        # 各フォールドをyieldする
        for fold in range(self.n_splits):
            test_idx = np.where(fold_assignments == fold)[0]
            train_idx = np.where(fold_assignments != fold)[0]
            yield train_idx, test_idx
    
    def get_fold_assignments(self, X_coords, y=None, groups=None):
        """
        各サンプルのフォールド番号を返す
        """
        n_samples = len(X_coords)
        
        # Step 1: 初期の細かいクラスタリング（目標フォールド数の3-5倍）
        n_initial_clusters = min(self.n_splits * 4, n_samples // 100)  # サンプル数に応じて調整
        n_initial_clusters = max(self.n_splits, n_initial_clusters)  # 最低でもn_splits個
        
        kmeans = KMeans(n_clusters=n_initial_clusters, random_state=self.random_state, n_init=10)
        initial_clusters = kmeans.fit_predict(X_coords)
        
        # Step 2: クラスタのバランシング
        fold_assignments = self._balance_clusters(
            X_coords, initial_clusters, self.n_splits, groups
        )
        
        return fold_assignments
    
    def _balance_clusters(self, X_coords, initial_clusters, n_folds, groups=None):
        """
        初期クラスタを再グループ化して、均等なフォールドを作成
        """
        n_samples = len(X_coords)
        target_size = n_samples // n_folds
        
        # クラスタごとの統計情報を計算
        cluster_info = []
        unique_clusters = np.unique(initial_clusters)
        
        for cluster_id in unique_clusters:
            mask = initial_clusters == cluster_id
            cluster_points = X_coords[mask]
            
            # グループ（地すべりID）の情報も考慮
            if groups is not None:
                cluster_groups = groups[mask]
                unique_groups = np.unique(cluster_groups)
            else:
                unique_groups = []
            
            cluster_info.append({
                'id': cluster_id,
                'size': len(cluster_points),
                'centroid': np.mean(cluster_points, axis=0),
                'points_idx': np.where(mask)[0],
                'groups': unique_groups
            })
        
        # クラスタをサイズでソート（大きい順）
        cluster_info.sort(key=lambda x: x['size'], reverse=True)
        
        # フォールドの初期化
        folds = [[] for _ in range(n_folds)]
        fold_sizes = [0] * n_folds
        fold_groups = [set() for _ in range(n_folds)]
        
        # 貪欲法でクラスタをフォールドに割り当て
        for cluster in cluster_info:
            # グループの分割を避けるため、既にそのグループを含むフォールドを優先
            best_fold = None
            
            if groups is not None and len(cluster['groups']) > 0:
                # このクラスタのグループを既に含むフォールドを探す
                for fold_idx in range(n_folds):
                    if any(g in fold_groups[fold_idx] for g in cluster['groups']):
                        if fold_sizes[fold_idx] + cluster['size'] <= target_size * 1.2:  # 20%の余裕を持たせる
                            best_fold = fold_idx
                            break
            
            # 適切なフォールドが見つからない場合、最小のフォールドに割り当て
            if best_fold is None:
                # 空間的な近さも考慮
                fold_scores = []
                for fold_idx in range(n_folds):
                    size_score = -fold_sizes[fold_idx]  # サイズが小さいほど良い
                    
                    # 既存のポイントとの距離
                    if len(folds[fold_idx]) > 0:
                        existing_points = []
                        for c in folds[fold_idx]:
                            existing_points.extend(c['points_idx'])
                        existing_coords = X_coords[existing_points]
                        dist = np.mean(np.linalg.norm(
                            existing_coords - cluster['centroid'], axis=1
                        ))
                        spatial_score = -dist  # 距離が近いほど良い
                    else:
                        spatial_score = 0
                    
                    fold_scores.append(size_score + 0.1 * spatial_score)
                
                best_fold = np.argmax(fold_scores)
            
            # クラスタをフォールドに追加
            folds[best_fold].append(cluster)
            fold_sizes[best_fold] += cluster['size']
            if groups is not None:
                fold_groups[best_fold].update(cluster['groups'])
        
        # 最終的なフォールド割り当てを作成
        fold_assignments = np.zeros(n_samples, dtype=int)
        for fold_idx, fold_clusters in enumerate(folds):
            for cluster in fold_clusters:
                fold_assignments[cluster['points_idx']] = fold_idx
        
        return fold_assignments


def apply_balanced_spatial_kfold(images, labels, geo_locations, landslide_ids, config):
    """
    バランスの取れた空間的K分割を適用する関数
    """
    n_samples = len(labels)
    print(f"\n📍 Applying Balanced Spatial K-Fold")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Number of folds: {config['n_outer_cv']}")
    
    # GeoDataFrameの作成
    geometry = [Point(lon, lat) for lon, lat in geo_locations]
    
    data = {
        'id': list(range(n_samples)),
        'label': list(labels),
        'landslide_id': list(landslide_ids),
        'geometry': geometry
    }
    
    gdf = gpd.GeoDataFrame(data, crs=f'EPSG:{config["epsg_code"]}')
    
    # UTM座標系に変換
    gdf_prj = gdf.to_crs(gdf.estimate_utm_crs())
    coords = np.array([[geom.x, geom.y] for geom in gdf_prj.geometry])
    
    # パラメータの設定
    n_splits = config['n_outer_cv']
    min_samples = config.get('min_points_per_fold', 1000)
    
    # 自動的にフォールド数を調整
    total_samples = len(gdf_prj)
    max_possible_folds = total_samples // min_samples
    
    if n_splits > max_possible_folds:
        print(f"\n⚠️  Warning: Requested {n_splits} folds, but with {min_samples} min samples per fold,")
        print(f"    maximum possible is {max_possible_folds}. Adjusting to {max_possible_folds}.")
        n_splits = max(1, max_possible_folds)
    
    # BalancedSpatialKFoldの適用
    best_fold_assignments = None
    best_balance_score = float('inf')
    
    # 複数の試行で最適な分割を探す
    for attempt in range(3):
        random_state = config.get('base_seed', 42) + attempt * 1000
        
        bskf = BalancedSpatialKFold(
            n_splits=n_splits,
            min_samples_per_fold=min_samples,
            random_state=random_state
        )
        
        try:
            fold_assignments = bskf.get_fold_assignments(
                coords, 
                y=labels, 
                groups=landslide_ids
            )
            
            # バランスを評価（変動係数）
            fold_counts = np.bincount(fold_assignments)
            cv = np.std(fold_counts) / np.mean(fold_counts)
            
            print(f"  Attempt {attempt + 1}: CV = {cv:.3f}, fold sizes = {fold_counts}")
            
            if cv < best_balance_score:
                best_balance_score = cv
                best_fold_assignments = fold_assignments
                
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            continue
    
    if best_fold_assignments is None:
        raise ValueError("Could not create valid spatial folds")
    
    # 最良の分割を使用
    gdf_prj['folds'] = best_fold_assignments
    spatial_folds = best_fold_assignments
    
    print(f"\n✅ Final configuration: {n_splits} folds with CV = {best_balance_score:.3f}")
    
    # 統計情報を表示
    print_fold_statistics_improved(spatial_folds, landslide_ids, labels)
    
    # 空間分布の可視化
    if config.get('visualize_folds', True):
        visualize_balanced_spatial_kfold(gdf_prj, config)
    
    return spatial_folds, gdf_prj


def print_fold_statistics_improved(fold_assignments, landslide_ids, labels):
    """改善された統計情報の表示"""
    n_folds = len(np.unique(fold_assignments))
    fold_counts = np.bincount(fold_assignments)
    
    print("\n📊 Balanced Spatial K-Fold Statistics:")
    print(f"  Total samples: {len(fold_assignments):,}")
    print(f"  Number of folds: {n_folds}")
    print(f"  Target samples per fold: {len(fold_assignments) // n_folds:,}")
    
    print("\n  Samples per fold:")
    for fold in range(n_folds):
        count = fold_counts[fold]
        percentage = count / len(fold_assignments) * 100
        deviation = (count - len(fold_assignments) / n_folds) / (len(fold_assignments) / n_folds) * 100
        print(f"    Fold {fold}: {count:6,} ({percentage:5.1f}%) [{deviation:+5.1f}% from target]")
    
    # バランス評価指標
    mean_count = np.mean(fold_counts)
    std_count = np.std(fold_counts)
    cv = std_count / mean_count
    max_min_ratio = np.max(fold_counts) / np.min(fold_counts) if np.min(fold_counts) > 0 else float('inf')
    
    print(f"\n  Balance metrics:")
    print(f"    Mean: {mean_count:.1f}, Std: {std_count:.1f}")
    print(f"    Coefficient of Variation: {cv:.3f}")
    print(f"    Max/Min ratio: {max_min_ratio:.2f}")
    
    # 評価
    if cv < 0.1:
        print("    ✅ Excellent balance!")
    elif cv < 0.2:
        print("    ✅ Good balance")
    else:
        print("    ⚠️  Consider adjusting parameters for better balance")


def visualize_balanced_spatial_kfold(gdf_with_folds, config):
    """バランスの取れた空間K分割の可視化"""
    n_folds = len(gdf_with_folds['folds'].unique())
    # 8色パレット（視認性の高い色を選定）
    palette = [
        '#e41a1c',  # 赤
        '#377eb8',  # 青
        '#4daf4a',  # 緑
        '#984ea3',  # 紫
        '#ff7f00',  # オレンジ
        '#ffff33',  # 黄
        '#a65628',  # 茶
        '#f781bf',  # ピンク
    ]
    color_cycle = [palette[i % len(palette)] for i in range(n_folds)]
    color_ramp = ListedColormap(color_cycle)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # 空間分布の表示
    gdf_with_folds.plot(
        column='folds',
        ax=ax,
        cmap=color_ramp,
        markersize=2,
        legend=False
    )
    ax.set_title('Balanced Spatial K-Fold Distribution', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Easting (m)', fontsize=12, color='white')
    ax.set_ylabel('Northing (m)', fontsize=12, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.grid(color='#bec8d1', alpha=0.2, linewidth=0.6)

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor('#137a7f')
        legend.get_frame().set_edgecolor('#86cecb')
        legend.get_title().set_color('#fffeec')
        for text in legend.get_texts():
            text.set_color('#fffeec')

    # 統計情報をテキストボックスに追加
    fold_counts = gdf_with_folds['folds'].value_counts().sort_index()
    stats_text = 'Samples per fold:\n'
    stats_text += '\n'.join([f'Fold {f}: {c:,}' for f, c in fold_counts.items()])
    stats_text += f'\n\nCV: {np.std(fold_counts) / np.mean(fold_counts):.3f}'

    ax.text(
        1.02,
        0.5,
        stats_text,
        transform=ax.transAxes,
        verticalalignment='center',
        fontsize=10,
        color='#fffeec',
        bbox=dict(boxstyle='round', facecolor='#137a7f', edgecolor='#e12885', alpha=0.9)
    )

    plt.tight_layout()
    save_path = config['fig_dir'] / 'balanced_spatial_kfold.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"  Spatial fold visualization saved to: {save_path}")
    plt.close()


# *******************************************************************
# DEM Mosaic Utilities
# *******************************************************************
# NOTE: 垂直基準（ジオイド/楕円体高）の変換は実装していません。
#       本プロジェクトでは石川県の林野庁公開数値標高0.5mデータ（航空レーザ測量）
#       のみを使用しており、すべて標高（東京湾平均海面基準）で統一されているため、
#       垂直基準変換は不要です。
#       異なるデータソース（ALOS DSM、ドローン測量等）を混在させる場合は、
#       pyproj等によるジオイド変換の実装が必要になります。
# *******************************************************************
def ensure_dem_mosaic(dem_root: Path, mosaic_name: str = "mergedem.vrt", subdir: str = "merge") -> Optional[Path]:
    """Build or reuse a DEM VRT mosaic from rasters under dem_root."""
    dem_root = Path(dem_root)
    if not dem_root.exists():
        warnings.warn(f"DEM directory not found: {dem_root}")
        return None

    mosaic_dir = dem_root / subdir
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    mosaic_path = mosaic_dir / mosaic_name

    raster_patterns = ("*.tif", "*.tiff")
    dem_files = []
    for pattern in raster_patterns:
        dem_files.extend(
            path for path in dem_root.glob(f"**/{pattern}")
            if path.is_file() and mosaic_dir not in path.parents
        )

    if not dem_files:
        warnings.warn(f"No DEM rasters found under {dem_root}")
        return mosaic_path if mosaic_path.exists() else None

    latest_tile_mtime = max(path.stat().st_mtime for path in dem_files)
    needs_rebuild = (not mosaic_path.exists()) or (mosaic_path.stat().st_mtime < latest_tile_mtime)

    if not needs_rebuild:
        return mosaic_path

    if gdal is None:
        warnings.warn("GDAL is unavailable; cannot rebuild DEM VRT mosaic.")
        return mosaic_path if mosaic_path.exists() else None

    print(f"ℹ️  Building DEM VRT mosaic ({len(dem_files)} tiles, average resolution)...")
    vrt_options = gdal.BuildVRTOptions(resolution="average")
    vrt = gdal.BuildVRT(str(mosaic_path), [str(path) for path in dem_files], options=vrt_options)
    if vrt is None:
        warnings.warn(f"Failed to build DEM VRT mosaic at {mosaic_path}")
        return mosaic_path if mosaic_path.exists() else None
    vrt.FlushCache()
    del vrt
    return mosaic_path


def ensure_dem_reprojected(src_path: Path, target_epsg: str = "EPSG:6675") -> Optional[Path]:
    """Ensure a DEM is reprojected to the requested EPSG; returns projected path."""
    src_path = Path(src_path)
    if not src_path.exists():
        warnings.warn(f"DEM source not found for reprojection: {src_path}")
        return None

    if gdal is None:
        warnings.warn("GDAL is unavailable; cannot reproject DEM. Using source raster as-is.")
        return src_path

    target_suffix = target_epsg.lower().replace(":", "")
    projected_path = src_path.with_name(f"{src_path.stem}_{target_suffix}.tif")

    try:
        dataset = gdal.Open(str(src_path))
    except Exception as exc:
        warnings.warn(f"Failed to open DEM for reprojection {src_path}: {exc}")
        return None

    if dataset is None:
        warnings.warn(f"GDAL could not open DEM for reprojection: {src_path}")
        return None

    try:
        if osr is not None:
            source_srs = osr.SpatialReference()
            target_srs = osr.SpatialReference()
            try:
                target_srs.ImportFromEPSG(int(target_epsg.split(":")[1]))
            except Exception:
                target_srs = None

            source_wkt = dataset.GetProjection()
            same_crs = False
            if source_wkt and target_srs is not None:
                try:
                    source_srs.ImportFromWkt(source_wkt)
                    same_crs = bool(source_srs.IsSame(target_srs))
                except Exception:
                    same_crs = False
            if same_crs:
                dataset = None
                return src_path
    finally:
        if dataset is not None:
            dataset = None

    needs_rebuild = (not projected_path.exists()) or (projected_path.stat().st_mtime < src_path.stat().st_mtime)
    if not needs_rebuild:
        return projected_path

    print(f"ℹ️  Reprojecting DEM to {target_epsg}: {projected_path.name}")
    warp_options = gdal.WarpOptions(dstSRS=target_epsg, resampleAlg="bilinear")
    result = gdal.Warp(str(projected_path), str(src_path), options=warp_options)
    if result is None:
        warnings.warn(f"Failed to reproject DEM to {target_epsg}: {src_path}")
        return None
    result.FlushCache()
    del result
    # Build overviews for the reprojected raster to accelerate later preview reads
    try:
        ensure_raster_overviews(projected_path)
    except Exception:
        pass
    return projected_path


# *******************************************************************
# Data Processing Class for Landslide Image Chips (tip-based)
# *******************************************************************
class LandslideTipProcessor:
    def __init__(
        self,
        rgb_paths,
        landslide_path,
        legacy_path,
        dem_paths=None,
        vdvi_paths=None,
        glcm_paths=None,
        exg_paths=None,
        epsg_code=6675,
        resolution=0.2,
        dem_resampling=Resampling.bilinear,
        vdvi_resampling_order=1,
        alignment_preview_dir=None,
        patch_overlay_dir=None,
    ):
        self.rgb_paths = rgb_paths if isinstance(rgb_paths, list) else [rgb_paths]
        self.rgb_paths = [Path(p) for p in self.rgb_paths]
        self.epsg_code = epsg_code
        self.resolution = resolution  # m

        self.dem_paths = self._prepare_path_list(dem_paths, len(self.rgb_paths))
        self.vdvi_paths = self._prepare_path_list(vdvi_paths, len(self.rgb_paths))
        self.glcm_paths = self._prepare_path_list(glcm_paths, len(self.rgb_paths))
        self.exg_paths = self._prepare_path_list(exg_paths, len(self.rgb_paths))
        self.dem_resampling = dem_resampling
        self.vdvi_resampling_order = vdvi_resampling_order

        # Load polygons
        self.landslide_gdf = gpd.read_file(landslide_path)
        if self.landslide_gdf.crs != f"EPSG:{epsg_code}":
            self.landslide_gdf.to_crs(epsg=epsg_code, inplace=True)
        self.landslide_gdf['landslide_id'] = range(len(self.landslide_gdf))

        self.legacy_gdf = gpd.read_file(legacy_path)
        if self.legacy_gdf.crs != f"EPSG:{epsg_code}":
            self.legacy_gdf.to_crs(epsg=epsg_code, inplace=True)

        assert self.legacy_gdf.crs == self.landslide_gdf.crs

        # Simple in-memory cache for auxiliary rasters after reprojection/resampling.
        # Keyed by (path_string, target_width, target_height, target_epsg).
        # This avoids recreating WarpedVRT and reprojecting the same DEM/VDVI
        # multiple times when processing many patches from the same RGB image.
        self._aux_cache = {}
        self.enable_aux_cache = True
        self._dem_preview_cache = {}
        self.alignment_preview_dir = Path(alignment_preview_dir) if alignment_preview_dir else None
        if self.alignment_preview_dir is not None:
            self.alignment_preview_dir.mkdir(parents=True, exist_ok=True)
        self._previewed_paths = set()
        self.patch_overlay_dir = Path(patch_overlay_dir) if patch_overlay_dir else None
        if self.patch_overlay_dir is not None:
            self.patch_overlay_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_path_list(self, paths, reference_length):
        if reference_length <= 0:
            return []
        if paths is None:
            return [None] * reference_length
        if isinstance(paths, (str, Path)):
            return [Path(paths)] * reference_length
        if not isinstance(paths, (list, tuple)):
            raise TypeError("paths must be None, a string/Path, or a list/tuple of paths")
        if len(paths) == 1 and reference_length > 1:
            paths = list(paths) * reference_length
        if len(paths) != reference_length:
            raise ValueError("Number of auxiliary paths must match number of RGB paths")
        normalized = []
        for item in paths:
            if item is None or item == "":
                normalized.append(None)
            else:
                normalized.append(Path(item))
        return normalized

    def _load_dem_preview(self, dem_path, max_dimension=2500, bounds=None):
        if dem_path is None:
            return None, None, None

        path = Path(dem_path)
        if not path.exists():
            warnings.warn(f"DEM preview source not found: {path}")
            return None, None, None

        # Ensure we have overviews to make preview fast and robust
        try:
            ensure_raster_overviews(path)
        except Exception:
            # Non-fatal; continue with regular read
            pass

        rounded_bounds = None
        if bounds is not None:
            rounded_bounds = tuple(round(float(val), 6) for val in bounds)
        cache_key = (str(path), max_dimension, rounded_bounds)
        cached = self._dem_preview_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            with rasterio.open(path) as src:
                if src.count < 1:
                    warnings.warn(f"DEM preview source has no bands: {path}")
                    return None, None, None

                if bounds is None:
                    window = None
                    window_height = float(src.height)
                    window_width = float(src.width)
                    window_transform = src.transform
                else:
                    window = src.window(*bounds, boundless=True)
                    if window is None:
                        warnings.warn(f"Failed to compute DEM window for bounds {bounds}")
                        return None, None, None
                    window = window.round_offsets().round_lengths()
                    window_height = float(window.height)
                    window_width = float(window.width)
                    window_transform = src.window_transform(window)

                max_native_dim = max(window_height, window_width)
                scale_factor = 1.0
                if max_dimension and max_dimension > 0 and max_native_dim > max_dimension:
                    scale_factor = max_native_dim / max_dimension

                out_height = max(1, int(round(window_height / scale_factor)))
                out_width = max(1, int(round(window_width / scale_factor)))

                read_kwargs = {
                    "out_shape": (out_height, out_width),
                    "resampling": Resampling.bilinear,
                    "boundless": True,
                    "masked": True,
                }
                if bounds is not None:
                    read_kwargs["window"] = window

                dem_preview = src.read(1, **read_kwargs)

                if np.ma.isMaskedArray(dem_preview):
                    dem_preview = dem_preview.astype(np.float32, copy=False).filled(np.nan)
                else:
                    dem_preview = dem_preview.astype(np.float32, copy=False)

                if src.nodata is not None and not np.isnan(src.nodata):
                    dem_preview = np.where(dem_preview == src.nodata, np.nan, dem_preview)

                scale_x = window_width / out_width if out_width else 1.0
                scale_y = window_height / out_height if out_height else 1.0
                overview_transform = window_transform * Affine.scale(scale_x, scale_y)
                result = (dem_preview, overview_transform, src.crs)
                self._dem_preview_cache[cache_key] = result
                return result
        except Exception as exc:
            # Try once more after forcing overview construction
            try:
                if ensure_raster_overviews(path):
                    with rasterio.open(path) as src:
                        if bounds is None:
                            window = None
                            window_height = float(src.height)
                            window_width = float(src.width)
                            window_transform = src.transform
                        else:
                            window = src.window(*bounds, boundless=True)
                            window = window.round_offsets().round_lengths()
                            window_height = float(window.height)
                            window_width = float(window.width)
                            window_transform = src.window_transform(window)

                        max_native_dim = max(window_height, window_width)
                        scale_factor = 1.0
                        if max_dimension and max_dimension > 0 and max_native_dim > max_dimension:
                            scale_factor = max_native_dim / max_dimension

                        out_height = max(1, int(round(window_height / scale_factor)))
                        out_width = max(1, int(round(window_width / scale_factor)))

                        read_kwargs = {
                            "out_shape": (out_height, out_width),
                            "resampling": Resampling.bilinear,
                            "boundless": True,
                            "masked": True,
                        }
                        if bounds is not None:
                            read_kwargs["window"] = window

                        dem_preview = src.read(1, **read_kwargs)
                        if np.ma.isMaskedArray(dem_preview):
                            dem_preview = dem_preview.astype(np.float32, copy=False).filled(np.nan)
                        else:
                            dem_preview = dem_preview.astype(np.float32, copy=False)

                        if src.nodata is not None and not np.isnan(src.nodata):
                            dem_preview = np.where(dem_preview == src.nodata, np.nan, dem_preview)

                        scale_x = window_width / out_width if out_width else 1.0
                        scale_y = window_height / out_height if out_height else 1.0
                        overview_transform = window_transform * Affine.scale(scale_x, scale_y)
                        result = (dem_preview, overview_transform, src.crs)
                        self._dem_preview_cache[cache_key] = result
                        return result
            except Exception:
                pass

            warnings.warn(f"Failed to build DEM preview for {path}: {exc}")
            return None, None, None

    def _normalize_overlay_array(self, array):
        data = np.asarray(array, dtype=np.float32)
        if data.ndim == 2:
            mask = np.isfinite(data)
            if np.any(mask):
                max_val = np.nanmax(data[mask])
                if max_val > 0:
                    if max_val > 1.0:
                        data = data / max_val
            return np.clip(data, 0.0, 1.0)

        data = data[..., :3].astype(np.float32, copy=False)
        mask = np.isfinite(data)
        if np.any(mask):
            max_val = np.nanmax(data[mask])
            if max_val > 0:
                if max_val > 1.0:
                    data = data / max_val
        return np.clip(data, 0.0, 1.0)

    def _save_dem_alignment_preview(
        self,
        rgb_array,
        rgb_transform,
        dem_array,
        dem_transform,
        preview_name,
        rgb_crs=None,
        dem_crs=None,
    ):
        if self.alignment_preview_dir is None or dem_array is None:
            return
        if preview_name in self._previewed_paths:
            return

        if rgb_crs is not None and dem_crs is not None and rgb_crs != dem_crs:
            warnings.warn(
                f"Skipping alignment preview for {preview_name}; CRS mismatch {rgb_crs} vs {dem_crs}."
            )
            return

        dem_bounds = self._compute_bounds(dem_transform, dem_array.shape)
        if dem_bounds is None:
            return
        dem_extent = [dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]]

        rgb_bounds = self._compute_bounds(rgb_transform, rgb_array.shape[:2])
        if rgb_bounds is None:
            return
        rgb_extent = [rgb_bounds[0], rgb_bounds[2], rgb_bounds[1], rgb_bounds[3]]

        fig, ax = plt.subplots(figsize=(10, 7))

        dem_display = np.ma.masked_invalid(dem_array.astype(np.float32, copy=False))
        if dem_display.mask.all():
            plt.close(fig)
            print(f"    ⚠️  DEM for {preview_name} contains only NaN values; skipping alignment preview.")
            return

        im = ax.imshow(dem_display, cmap='gray', extent=dem_extent, origin='upper')
        cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
        cbar.set_label('DEM (m)')

        if rgb_array.ndim == 2:
            overlay = rgb_array.astype(np.float32, copy=False)
            finite_mask = np.isfinite(overlay)
            if np.any(finite_mask):
                max_val = np.nanmax(overlay[finite_mask])
                if max_val > 0:
                    if max_val > 1.0:
                        overlay = overlay / max_val
            overlay = np.clip(overlay, 0.0, 1.0)
            ax.imshow(overlay, cmap='viridis', extent=rgb_extent, origin='upper', alpha=0.9)
        else:
            overlay = rgb_array[..., :3].astype(np.float32, copy=False)
            finite_mask = np.isfinite(overlay)
            if np.any(finite_mask):
                max_val = np.nanmax(overlay[finite_mask])
                if max_val > 0:
                    if max_val > 1.0:
                        overlay = overlay / max_val
            overlay = np.clip(overlay, 0.0, 1.0)
            ax.imshow(overlay, extent=rgb_extent, origin='upper')

        rect = Rectangle(
            (rgb_extent[0], rgb_extent[2]),
            rgb_extent[1] - rgb_extent[0],
            rgb_extent[3] - rgb_extent[2],
            linewidth=1.2,
            edgecolor='cyan',
            facecolor='none',
        )
        ax.add_patch(rect)

        ax.set_title(f"{preview_name} DEM alignment")
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_aspect('equal')

        preview_path = self.alignment_preview_dir / f"{preview_name}_dem_alignment.png"
        fig.savefig(preview_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        self._previewed_paths.add(preview_name)
        print(f"    💾 Saved DEM alignment preview to {preview_path.name}")


    def _save_patch_overlay(self, image_name, ratio_map, dem_ras, extent, polygons_gdf):
        if self.patch_overlay_dir is None:
            return
        has_ratio = np.isfinite(ratio_map).any()
        has_dem = dem_ras is not None and np.isfinite(dem_ras).any()
        if not has_ratio and not has_dem:
            print(f"    ⚠️  No valid data to render patch overlay for {image_name}")
            return

        fig, ax = plt.subplots(figsize=(10, 7), facecolor='black')
        ax.set_facecolor('black')

        if has_dem:
            dem_display = np.ma.masked_invalid(dem_ras.astype(np.float32, copy=False))
            if not dem_display.mask.all():
                ax.imshow(dem_display, cmap='gray', extent=extent, origin='upper')

        if has_ratio:
            ratio_display = np.ma.masked_invalid(ratio_map.astype(np.float32, copy=False))
            if not ratio_display.mask.all():
                im = ax.imshow(
                    ratio_display,
                    cmap='plasma',
                    extent=extent,
                    origin='upper',
                    alpha=0.6,
                    vmin=0.0,
                    vmax=1.0,
                )
                cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
                cbar.set_label('Annotation ratio (0-1)', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        if polygons_gdf is not None and not polygons_gdf.empty:
            polygons_gdf.boundary.plot(ax=ax, linewidth=1.0, edgecolor='cyan')

        ax.set_xlabel('Easting', color='white')
        ax.set_ylabel('Northing', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_aspect('equal')
        ax.set_title(f"{image_name} | Legacy ratio overlay", color='white')

        output_path = self.patch_overlay_dir / f"{image_name}_patch_overlay.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"    💾 Saved patch overlay to {output_path.name}")


    def _save_vdvi_overlay(self, image_name, rgb_ras, vdvi_ras, extent):
        if self.patch_overlay_dir is None or vdvi_ras is None:
            return
        
        # Check for valid data
        if not np.isfinite(vdvi_ras).any():
             print(f"    ⚠️  No valid VDVI data to render overlay for {image_name}")
             return

        fig, ax = plt.subplots(figsize=(10, 7))

        # Display RGB if available
        if rgb_ras is not None:
             # Normalize RGB
             rgb_display = self._normalize_overlay_array(rgb_ras)
             if rgb_display.ndim == 2:
                 ax.imshow(rgb_display, cmap='gray', extent=extent, origin='upper')
             else:
                 ax.imshow(rgb_display, extent=extent, origin='upper')

        # Overlay VDVI
        vdvi_display = np.ma.masked_invalid(vdvi_ras.astype(np.float32, copy=False))
        if not vdvi_display.mask.all():
            im = ax.imshow(
                vdvi_display,
                cmap='RdYlGn', 
                extent=extent,
                origin='upper',
                alpha=0.5,
                vmin=-1.0, 
                vmax=1.0,
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
            cbar.set_label('VDVI')

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_aspect('equal')
        ax.set_title(f"{image_name} | VDVI overlay")

        output_path = self.patch_overlay_dir / f"{image_name}_vdvi_overlay.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"    💾 Saved VDVI overlay to {output_path.name}")

    def generate_alignment_preview(self):
        if self.alignment_preview_dir is None:
            return

        preview_entries = []
        dem_bounds_map: Dict[Path, List[float]] = {}

        for idx, rgb_path in enumerate(self.rgb_paths):
            dem_path = None
            if self.dem_paths and idx < len(self.dem_paths):
                dem_path = self.dem_paths[idx]
            if dem_path is None:
                continue

            dem_path = Path(dem_path)

            try:
                rgb_ras, rgb_transform, rgb_crs, _ = self._load_raster_data(rgb_path)
            except Exception as exc:
                print(f"    ⚠️  Failed to load RGB image {rgb_path} for alignment preview: {exc}")
                continue

            rgb_bounds = self._compute_bounds(rgb_transform, rgb_ras.shape[:2])
            if rgb_bounds is None:
                continue
            rgb_extent = [rgb_bounds[0], rgb_bounds[2], rgb_bounds[1], rgb_bounds[3]]

            dem_union = dem_bounds_map.get(dem_path)
            if dem_union is None:
                dem_bounds_map[dem_path] = list(rgb_bounds)
            else:
                dem_union[0] = min(dem_union[0], rgb_bounds[0])
                dem_union[1] = min(dem_union[1], rgb_bounds[1])
                dem_union[2] = max(dem_union[2], rgb_bounds[2])
                dem_union[3] = max(dem_union[3], rgb_bounds[3])

            overlay = self._normalize_overlay_array(rgb_ras)
            del rgb_ras

            preview_entries.append(
                {
                    "rgb_path": rgb_path,
                    "dem_path": dem_path,
                    "rgb_crs": rgb_crs,
                    "rgb_extent": rgb_extent,
                    "overlay": overlay,
                }
            )

        if not preview_entries:
            return

        dem_contexts: Dict[Path, Dict[str, object]] = {}

        for dem_path, bounds in dem_bounds_map.items():
            dem_preview, dem_transform, dem_crs = self._load_dem_preview(dem_path, bounds=tuple(bounds))
            if dem_preview is None:
                print(f"    ⚠️  DEM overview not available for alignment preview: {dem_path}")
                continue

            dem_bounds = self._compute_bounds(dem_transform, dem_preview.shape)
            if dem_bounds is None:
                continue
            dem_extent = [dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]]

            fig, ax = plt.subplots(figsize=(10, 7))
            dem_display = np.ma.masked_invalid(dem_preview.astype(np.float32, copy=False))
            if dem_display.mask.all():
                plt.close(fig)
                print(f"    ⚠️  DEM overview for {dem_path} contains only NaN values; skipping.")
                continue

            im = ax.imshow(dem_display, cmap='gray', extent=dem_extent, origin='upper')
            cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
            cbar.set_label('DEM (m)')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            ax.set_aspect('equal')

            dem_contexts[dem_path] = {
                "fig": fig,
                "ax": ax,
                "preview_name": dem_path.stem,
                "dem_crs": dem_crs,
                "overlays": 0,
                "dem_extent": dem_extent,
                "cmap": colormaps["tab10"],
            }

        for entry in preview_entries:
            dem_path = entry["dem_path"]
            context = dem_contexts.get(dem_path)
            if context is None:
                continue

            rgb_crs = entry["rgb_crs"]
            if context["dem_crs"] is not None and rgb_crs is not None and context["dem_crs"] != rgb_crs:
                warnings.warn(
                    f"Skipping overlay for {entry['rgb_path']}; CRS mismatch {rgb_crs} vs {context['dem_crs']}"
                )
                continue

            overlay = entry["overlay"]
            rgb_extent = entry["rgb_extent"]

            if overlay.ndim == 2:
                context["ax"].imshow(
                    overlay,
                    cmap='viridis',
                    extent=rgb_extent,
                    origin='upper',
                    alpha=0.9,
                )
            else:
                context["ax"].imshow(overlay, extent=rgb_extent, origin='upper')

            overlay_index = context["overlays"]
            cmap = context["cmap"]
            colors = getattr(cmap, "colors", None)
            if colors:
                edge_color = colors[overlay_index % len(colors)]
            else:
                edge_color = cmap(
                    (overlay_index % max(1, getattr(cmap, "N", 10)))
                    / max(1, getattr(cmap, "N", 10) - 1)
                )
            rect = Rectangle(
                (rgb_extent[0], rgb_extent[2]),
                rgb_extent[1] - rgb_extent[0],
                rgb_extent[3] - rgb_extent[2],
                linewidth=1.4,
                edgecolor=edge_color,
                facecolor='none',
            )
            context["ax"].add_patch(rect)

            label_x = rgb_extent[0] + 0.02 * (rgb_extent[1] - rgb_extent[0])
            label_y = rgb_extent[3] - 0.02 * (rgb_extent[3] - rgb_extent[2])
            context["ax"].text(
                label_x,
                label_y,
                Path(entry["rgb_path"]).stem,
                fontsize=7,
                color=edge_color,
                weight='bold',
            )

            context["overlays"] += 1
            entry["overlay"] = None

        for context in dem_contexts.values():
            fig = context["fig"]
            ax = context["ax"]
            overlays = context["overlays"]
            preview_name = context["preview_name"]
            dem_extent = context["dem_extent"]
            ax.set_xlim(dem_extent[0], dem_extent[1])
            ax.set_ylim(dem_extent[3], dem_extent[2])
            ax.set_title(f"DEM alignment overview ({overlays} image(s))")
            suffix = "_dem_alignment_overview" if overlays > 1 else "_dem_alignment"
            output_name = f"{preview_name}{suffix}.png"
            preview_path = self.alignment_preview_dir / output_name
            fig.savefig(preview_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            self._previewed_paths.add(output_name)
            print(f"    💾 Saved DEM alignment preview to {output_name}")


    def _load_raster_data(self, filepath):
        with rasterio.open(filepath) as src:
            data = src.read()
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
            # JPG/PNG with world file (jgw/pgw) may have transform but no CRS
            # Assume EPSG:6675 (JGD2011 / Japan Plane Rectangular CS VII) for Noto data
            if crs is None and transform is not None and not transform.is_identity:
                from rasterio.crs import CRS
                crs = CRS.from_epsg(6675)
                print(f"  [INFO] No CRS in {Path(filepath).name}, assuming EPSG:6675")
            
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0))
            else:
                data = data.squeeze()
        return data, transform, crs, (bounds.left, bounds.bottom, bounds.right, bounds.top)

    def _load_auxiliary_raster(self, filepath, target_shape, target_transform, target_crs, is_vdvi=False):
        if filepath is None:
            return None

        path = Path(filepath)
        if not path.exists():
            warnings.warn(f"Auxiliary raster not found: {path}")
            return None

        # Attempt cache hit
        if self.enable_aux_cache:
            cache_key = (str(path), target_shape[1], target_shape[0], str(target_crs))
            cached = self._aux_cache.get(cache_key)
            if cached is not None:
                return cached  # already float32

        if path.suffix.lower() == '.npy':
            data = np.asarray(np.load(path), dtype=np.float32)
            data = np.squeeze(data)
            if data.ndim > 2:
                data = data[..., 0]
            if data.shape != target_shape:
                zoom_factors = (
                    target_shape[0] / data.shape[0],
                    target_shape[1] / data.shape[1],
                )
                data = zoom(data, zoom_factors, order=self.vdvi_resampling_order)
            return data.astype(np.float32)

        try:
            with rasterio.open(path) as src:
                resampling_method = self.dem_resampling if not is_vdvi else Resampling.bilinear

                target_bounds = self._compute_bounds(target_transform, target_shape)
                
                # If target has no CRS but has valid bounds/transform, assume same CRS as source
                # This handles cases where RGB images lack CRS metadata but use same coordinate system
                effective_target_crs = target_crs if target_crs is not None else src.crs
                
                if src.crs is None or src.transform is None or target_transform is None or target_bounds is None:
                    # Fallback: simple resize without georeferencing
                    src_data = src.read(1).astype(np.float32)
                    if src_data.shape != target_shape:
                        zoom_factors = (
                            target_shape[0] / src_data.shape[0],
                            target_shape[1] / src_data.shape[1],
                        )
                        src_data = zoom(src_data, zoom_factors, order=1)
                    return src_data

                # Use bounds-based extraction: assume target bounds are in source CRS if target_crs is None
                src_bounds = target_bounds  # Assume same CRS when target_crs is None
                
                # Read data within the window corresponding to the target area
                window = src.window(*src_bounds)
                
                print(f"  [DEBUG] Extracting window: {window}")
                print(f"  [DEBUG] Target bounds in src CRS: {src_bounds}")
                
                # Read windowed data
                data = src.read(
                    1,
                    window=window,
                    out_shape=(target_shape[0], target_shape[1]),
                    resampling=resampling_method,
                    boundless=True,
                    masked=True,
                )
                if np.ma.isMaskedArray(data):
                    data = data.filled(np.nan)
                data = data.astype(np.float32, copy=False)
                if src.nodata is not None and not np.isnan(src.nodata):
                    data = np.where(data == src.nodata, np.nan, data)
                if self.enable_aux_cache:
                    self._aux_cache[cache_key] = data
                return data
        except Exception as exc:
            warnings.warn(f"Failed to load auxiliary raster {path}: {exc}")
            return None

    def _rasterize_polygons(self, gdf, id_col, rgb_ras, rgb_transform):
        shapes = [(geom, i) for i, geom in zip(gdf[id_col], gdf.geometry)]
        return rasioftr.rasterize(shapes, out_shape=rgb_ras.shape[:2], transform=rgb_transform, fill=np.nan, dtype=np.float32, all_touched=True)

    def _pixel_to_coord(self, col, row, rgb_transform):
        x, y = rgb_transform * (col + 0.5, row + 0.5)
        return x, y

    def _compute_bounds(self, transform, shape):
        if transform is None or shape is None:
            return None
        height, width = shape
        left = transform.c
        top = transform.f
        right = left + transform.a * width
        bottom = top + transform.e * height
        return (
            min(left, right),
            min(bottom, top),
            max(left, right),
            max(bottom, top),
        )

    def extract_tips(self, tip_size_m=5, target_size_m=None, slide_ratio=0.5, collect_data=True, show_progress=True, use_exg=True):
        """
        Extract image patches for landslide classification.
        
        Parameters:
        -----------
        tip_size_m : float
            Size of input patch in meters (Field of View). Default: 5m.
            With Center-Labeling strategy, this represents the context area (e.g., 20m).
        target_size_m : float, optional
            Size of center region in meters for label calculation. Default: None (same as tip_size_m).
            When specified, uses Center-Labeling strategy where labels are computed from
            the central target_size_m region while input includes full tip_size_m context.
        slide_ratio : float
            Sliding window stride as ratio of target_size_m (or tip_size_m if target_size_m is None).
        collect_data : bool
            Whether to collect and return patch data.
        show_progress : bool
            Whether to show progress bar.
            
        Notes:
        ------
        Center-Labeling Strategy (when target_size_m is specified):
        - Input size (tip_size_m): Full context area fed to model (e.g., 20m = 100px)
        - Target size (target_size_m): Central region for label calculation (e.g., 5m = 25px)
        - Padding: Reflect padding for features, NaN padding for labels
        - This allows edge regions to be evaluated with mirrored context
        """
        tip_size_grids = int(tip_size_m / self.resolution)
        
        # Center-Labeling戦略: target_size_mが指定された場合
        if target_size_m is not None:
            target_size_grids = int(target_size_m / self.resolution)
            # パディング量の計算: (入力サイズ - ターゲットサイズ) // 2
            pad_amount = (tip_size_grids - target_size_grids) // 2
            # ストライドはターゲットサイズ基準
            n_slide = max(1, int(target_size_grids * slide_ratio))
            use_center_labeling = True
            print(f"  [Center-Labeling] Input: {tip_size_m}m ({tip_size_grids}px), "
                  f"Target: {target_size_m}m ({target_size_grids}px), "
                  f"Padding: {pad_amount}px, Stride: {n_slide}px")
        else:
            target_size_grids = tip_size_grids
            pad_amount = 0
            n_slide = max(1, int(tip_size_grids * slide_ratio))
            use_center_labeling = False
        
        # 従来の方式（Center-Labeling不使用時）のバッファ計算
        l_buf = tip_size_grids // 2
        u_buf = tip_size_grids - l_buf - 1

        tips = []
        labels = []
        pixel_locations = []
        geo_locations = []
        landslide_ids = []

        for idx, rgb_path in enumerate(self.rgb_paths):
            print(f"Processing {rgb_path}...")
            # Load raster
            rgb_ras, rgb_transform, rgb_crs, rgb_bounds = self._load_raster_data(rgb_path)
            if rgb_ras.ndim == 2:
                rgb_ras = np.repeat(rgb_ras[:, :, np.newaxis], 3, axis=2)
            aoi_size = rgb_ras.shape[:2]
            i_range = range(l_buf, aoi_size[1] - u_buf, n_slide)
            j_range = range(l_buf, aoi_size[0] - u_buf, n_slide)
            rgb_extent = [rgb_bounds[0], rgb_bounds[2], rgb_bounds[1], rgb_bounds[3]]
            ratio_sum = np.zeros(aoi_size, dtype=np.float32)
            ratio_count = np.zeros(aoi_size, dtype=np.float32)

            # When using a shared DEM mosaic, each dem_paths[idx] may be identical; still processed once per RGB for windowing simplicity.
            dem_ras = self._load_auxiliary_raster(
                self.dem_paths[idx],
                aoi_size,
                rgb_transform,
                rgb_crs,
                is_vdvi=False,
            ) if self.dem_paths and self.dem_paths[idx] is not None else None

            # DEBUG: Check DEM loading
            if dem_ras is not None:
                print(f"  [DEBUG] RGB shape: {rgb_ras.shape}, RGB CRS: {rgb_crs}")
                print(f"  [DEBUG] RGB bounds: {rgb_bounds}")
                print(f"  [DEBUG] RGB transform: {rgb_transform}")
                print(f"  [DEBUG] DEM shape: {dem_ras.shape}")
                print(f"  [DEBUG] DEM valid pixels: {np.sum(~np.isnan(dem_ras))} / {dem_ras.size}")
                print(f"  [DEBUG] DEM value range: {np.nanmin(dem_ras):.2f} - {np.nanmax(dem_ras):.2f}")
                # Check source DEM info
                with rasterio.open(self.dem_paths[idx]) as src:
                    print(f"  [DEBUG] Source DEM CRS: {src.crs}")
                    print(f"  [DEBUG] Source DEM bounds: {src.bounds}")
                    print(f"  [DEBUG] Source DEM shape: {src.height} x {src.width}")

            vdvi_ras = self._load_auxiliary_raster(
                self.vdvi_paths[idx],
                aoi_size,
                rgb_transform,
                rgb_crs,
                is_vdvi=True,
            ) if self.vdvi_paths and self.vdvi_paths[idx] is not None else None

            # Load GLCM raster (contrast feature from .npy file)
            glcm_ras = None
            if self.glcm_paths and self.glcm_paths[idx] is not None:
                glcm_path = self.glcm_paths[idx]
                if Path(glcm_path).exists():
                    try:
                        glcm_data = np.load(glcm_path)
                        # Resize GLCM to match RGB size if needed
                        if glcm_data.shape != aoi_size:
                            from scipy.ndimage import zoom
                            zoom_factors = (aoi_size[0] / glcm_data.shape[0], 
                                          aoi_size[1] / glcm_data.shape[1])
                            glcm_ras = zoom(glcm_data, zoom_factors, order=1).astype(np.float32)
                        else:
                            glcm_ras = glcm_data.astype(np.float32)
                        print(f"  [DEBUG] GLCM shape: {glcm_ras.shape}")
                        print(f"  [DEBUG] GLCM value range: {np.nanmin(glcm_ras):.2f} - {np.nanmax(glcm_ras):.2f}")
                    except Exception as e:
                        warnings.warn(f"Failed to load GLCM from {glcm_path}: {e}")
                        glcm_ras = None
                else:
                    warnings.warn(f"GLCM file not found: {glcm_path}")

            # Load EXG raster (Excess Green Index from .npy file)
            exg_ras = None
            if use_exg and self.exg_paths and self.exg_paths[idx] is not None:
                exg_path = self.exg_paths[idx]
                if Path(exg_path).exists():
                    try:
                        exg_data = np.load(exg_path)
                        # Resize EXG to match RGB size if needed
                        if exg_data.shape != aoi_size:
                            from scipy.ndimage import zoom
                            zoom_factors = (aoi_size[0] / exg_data.shape[0], 
                                          aoi_size[1] / exg_data.shape[1])
                            exg_ras = zoom(exg_data, zoom_factors, order=1).astype(np.float32)
                        else:
                            exg_ras = exg_data.astype(np.float32)
                        print(f"  [DEBUG] EXG shape: {exg_ras.shape}")
                        print(f"  [DEBUG] EXG value range: {np.nanmin(exg_ras):.2f} - {np.nanmax(exg_ras):.2f}")
                    except Exception as e:
                        warnings.warn(f"Failed to load EXG from {exg_path}: {e}")
                        exg_ras = None
                else:
                    warnings.warn(f"EXG file not found: {exg_path}")

            # Clip polygons
            extent_gdf = gpd.GeoDataFrame(geometry=[box(*rgb_bounds)], crs=rgb_crs)
            
            # Fix invalid geometries before clipping
            def safe_clip(gdf, mask_gdf):
                """Safely clip GeoDataFrame with geometry validation and error handling."""
                if gdf.empty:
                    return gdf
                
                # Make a copy to avoid modifying original
                gdf_fixed = gdf.copy()
                
                # Fix invalid geometries using buffer(0) trick
                invalid_mask = ~gdf_fixed.geometry.is_valid
                if invalid_mask.any():
                    gdf_fixed.loc[invalid_mask, 'geometry'] = gdf_fixed.loc[invalid_mask, 'geometry'].buffer(0)
                
                # Remove empty geometries
                gdf_fixed = gdf_fixed[~gdf_fixed.geometry.is_empty]
                
                if gdf_fixed.empty:
                    return gdf_fixed
                
                try:
                    return gpd.clip(gdf_fixed, mask_gdf)
                except Exception as e:
                    # If clip fails, try with more aggressive geometry repair
                    warnings.warn(f"Clip failed, attempting repair: {e}")
                    try:
                        # Use make_valid if available (shapely >= 1.8)
                        from shapely.validation import make_valid
                        gdf_fixed['geometry'] = gdf_fixed['geometry'].apply(
                            lambda g: make_valid(g) if g is not None and not g.is_valid else g
                        )
                        return gpd.clip(gdf_fixed, mask_gdf)
                    except ImportError:
                        pass
                    except Exception as e2:
                        warnings.warn(f"Repair attempt also failed: {e2}")
                    
                    # Last resort: filter by intersection instead of clip
                    try:
                        mask_geom = mask_gdf.unary_union
                        intersects_mask = gdf_fixed.geometry.intersects(mask_geom)
                        result = gdf_fixed[intersects_mask].copy()
                        # Attempt intersection one by one, skipping failures
                        valid_rows = []
                        for idx, row in result.iterrows():
                            try:
                                clipped_geom = row.geometry.intersection(mask_geom)
                                if not clipped_geom.is_empty:
                                    row_copy = row.copy()
                                    row_copy['geometry'] = clipped_geom
                                    valid_rows.append(row_copy)
                            except Exception:
                                continue
                        if valid_rows:
                            return gpd.GeoDataFrame(valid_rows, crs=gdf.crs)
                        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
                    except Exception as e3:
                        warnings.warn(f"Fallback intersection also failed: {e3}")
                        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
            
            landslide_aoi_gdf = safe_clip(self.landslide_gdf, extent_gdf)
            legacy_aoi_gdf = safe_clip(self.legacy_gdf, extent_gdf)
            # Rasterize
            landslide_ras = self._rasterize_polygons(landslide_aoi_gdf, 'landslide_id', rgb_ras, rgb_transform)
            legacy_ras = self._rasterize_polygons(legacy_aoi_gdf, 'id', rgb_ras, rgb_transform)
            legacy_ras = np.where(np.isnan(landslide_ras), np.nan, legacy_ras)
            
            # ============================================================
            # Center-Labeling戦略: パディング処理
            # ============================================================
            if use_center_labeling and pad_amount > 0:
                print(f"  [Padding] Applying reflect padding ({pad_amount}px) for features...")
                
                # 特徴量（RGB）のReflect Padding
                rgb_ras_padded = np.pad(
                    rgb_ras,
                    ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)),
                    mode='reflect'
                )
                
                # DEM/VDVI/GLCMのReflect Padding
                if dem_ras is not None:
                    dem_ras_padded = np.pad(dem_ras, pad_amount, mode='reflect')
                else:
                    dem_ras_padded = None
                    
                if vdvi_ras is not None:
                    vdvi_ras_padded = np.pad(vdvi_ras, pad_amount, mode='reflect')
                else:
                    vdvi_ras_padded = None
                    
                if glcm_ras is not None:
                    glcm_ras_padded = np.pad(glcm_ras, pad_amount, mode='reflect')
                else:
                    glcm_ras_padded = None
                
                if use_exg and exg_ras is not None:
                    exg_ras_padded = np.pad(exg_ras, pad_amount, mode='reflect')
                else:
                    exg_ras_padded = None
                
                # ラベル・ID画像のConstant Padding (NaN埋め)
                landslide_ras_padded = np.pad(
                    landslide_ras, pad_amount, mode='constant', constant_values=np.nan
                )
                legacy_ras_padded = np.pad(
                    legacy_ras, pad_amount, mode='constant', constant_values=np.nan
                )
                
                # パディング後のサイズ
                padded_size = (aoi_size[0] + 2 * pad_amount, aoi_size[1] + 2 * pad_amount)
                
                # 元画像の中心を走査（パディングを考慮したオフセット）
                # 元画像の(0,0)は、パディング後は(pad_amount, pad_amount)
                # ストライドはtarget_size基準
                i_range = range(pad_amount, padded_size[1] - pad_amount, n_slide)
                j_range = range(pad_amount, padded_size[0] - pad_amount, n_slide)
                
                print(f"  [Padding] Padded size: {padded_size}, Scan range: i={len(list(i_range))}, j={len(list(j_range))}")
                # rangeを再生成（イテレータは消費されるため）
                i_range = range(pad_amount, padded_size[1] - pad_amount, n_slide)
                j_range = range(pad_amount, padded_size[0] - pad_amount, n_slide)
            else:
                # 従来のロジック（パディングなし）
                rgb_ras_padded = rgb_ras
                dem_ras_padded = dem_ras
                vdvi_ras_padded = vdvi_ras
                glcm_ras_padded = glcm_ras
                exg_ras_padded = exg_ras
                landslide_ras_padded = landslide_ras
                legacy_ras_padded = legacy_ras
                padded_size = aoi_size
            
            # Create tips
            total_iterations = len(list(i_range)) * len(list(j_range))
            # rangeを再生成
            if use_center_labeling and pad_amount > 0:
                i_range = range(pad_amount, padded_size[1] - pad_amount, n_slide)
                j_range = range(pad_amount, padded_size[0] - pad_amount, n_slide)
            else:
                i_range = range(l_buf, aoi_size[1] - u_buf, n_slide)
                j_range = range(l_buf, aoi_size[0] - u_buf, n_slide)
            
            progress = tqdm(total=total_iterations, desc="Extracting tips") if show_progress else None
            try:
                for i in i_range:
                    for j in j_range:
                        # ============================================================
                        # Center-Labeling: 中心領域のラベルをチェック
                        # ============================================================
                        if use_center_labeling and pad_amount > 0:
                            # 中心target_size領域のラベルを取得
                            half_target = target_size_grids // 2
                            center_j_start = j - half_target
                            center_j_end = j + (target_size_grids - half_target)
                            center_i_start = i - half_target
                            center_i_end = i + (target_size_grids - half_target)
                            
                            center_landslide = landslide_ras_padded[center_j_start:center_j_end, center_i_start:center_i_end]
                            center_legacy = legacy_ras_padded[center_j_start:center_j_end, center_i_start:center_i_end]
                            
                            # 中心領域にNaNが含まれていれば（=地すべり領域外）スキップ
                            if np.any(np.isnan(center_landslide)):
                                if progress is not None:
                                    progress.update(1)
                                continue
                            
                            # 代表landslide_idは中心点から取得
                            landslide_id = int(landslide_ras_padded[j, i])
                            
                            # 入力パッチ（tip_size_grids）の切り出し
                            half_tip = tip_size_grids // 2
                            tip_j_start = j - half_tip
                            tip_j_end = j + (tip_size_grids - half_tip)
                            tip_i_start = i - half_tip
                            tip_i_end = i + (tip_size_grids - half_tip)
                            
                            rgb_tip = rgb_ras_padded[tip_j_start:tip_j_end, tip_i_start:tip_i_end, :]
                            
                            # 中心領域でlegacy_ratioを計算
                            is_legacy = (center_legacy > 0) & ~(
                                (rgb_ras_padded[center_j_start:center_j_end, center_i_start:center_i_end, 0] == 0) & 
                                (rgb_ras_padded[center_j_start:center_j_end, center_i_start:center_i_end, 1] == 0) & 
                                (rgb_ras_padded[center_j_start:center_j_end, center_i_start:center_i_end, 2] == 0)
                            )
                            legacy_ratio = round(np.sum(is_legacy) / (target_size_grids ** 2), 5)
                            
                            # サイズ検証
                            if rgb_tip.shape[0] != tip_size_grids or rgb_tip.shape[1] != tip_size_grids:
                                if progress is not None:
                                    progress.update(1)
                                continue
                            
                            # 補助チャンネルの切り出し
                            patch_channels = [rgb_tip.astype(np.float32)]
                            valid_patch = True
                            
                            if dem_ras_padded is not None:
                                dem_tip = dem_ras_padded[tip_j_start:tip_j_end, tip_i_start:tip_i_end]
                                if dem_tip.shape != (tip_size_grids, tip_size_grids):
                                    valid_patch = False
                                else:
                                    patch_channels.append(dem_tip.astype(np.float32)[..., np.newaxis])
                            
                            if vdvi_ras_padded is not None and valid_patch:
                                vdvi_tip = vdvi_ras_padded[tip_j_start:tip_j_end, tip_i_start:tip_i_end]
                                if vdvi_tip.shape != (tip_size_grids, tip_size_grids):
                                    valid_patch = False
                                else:
                                    patch_channels.append(vdvi_tip.astype(np.float32)[..., np.newaxis])
                            
                            if glcm_ras_padded is not None and valid_patch:
                                glcm_tip = glcm_ras_padded[tip_j_start:tip_j_end, tip_i_start:tip_i_end]
                                if glcm_tip.shape != (tip_size_grids, tip_size_grids):
                                    valid_patch = False
                                else:
                                    patch_channels.append(glcm_tip.astype(np.float32)[..., np.newaxis])
                            
                            if use_exg and exg_ras_padded is not None and valid_patch:
                                exg_tip = exg_ras_padded[tip_j_start:tip_j_end, tip_i_start:tip_i_end]
                                if exg_tip.shape != (tip_size_grids, tip_size_grids):
                                    valid_patch = False
                                else:
                                    patch_channels.append(exg_tip.astype(np.float32)[..., np.newaxis])
                            
                            if not valid_patch:
                                if progress is not None:
                                    progress.update(1)
                                continue
                            
                            combined_tip = np.dstack(patch_channels).astype(np.float32)
                            
                            # ratio_mapは元画像サイズで計算（オフセット補正）
                            orig_j = j - pad_amount
                            orig_i = i - pad_amount
                            orig_half_target = target_size_grids // 2
                            row_slice = slice(max(0, orig_j - orig_half_target), min(aoi_size[0], orig_j + (target_size_grids - orig_half_target)))
                            col_slice = slice(max(0, orig_i - orig_half_target), min(aoi_size[1], orig_i + (target_size_grids - orig_half_target)))
                            ratio_sum[row_slice, col_slice] += legacy_ratio
                            ratio_count[row_slice, col_slice] += 1.0
                            
                            if collect_data:
                                tips.append(combined_tip)
                                labels.append(legacy_ratio)
                                pixel_locations.append((orig_i, orig_j))
                                geo_locations.append(self._pixel_to_coord(orig_i, orig_j, rgb_transform))
                                landslide_ids.append(landslide_id)
                            
                            if progress is not None:
                                progress.update(1)
                        else:
                            # ============================================================
                            # 従来のロジック（Center-Labeling不使用）
                            # ============================================================
                            if np.isnan(landslide_ras[j, i]):
                                if progress is not None:
                                    progress.update(1)
                                continue

                            landslide_id = int(landslide_ras[j, i])
                            legacy_type = 0 if np.isnan(legacy_ras[j, i]) else int(legacy_ras[j, i])

                            rgb_tip = rgb_ras[j - l_buf:j + u_buf + 1, i - l_buf:i + u_buf + 1, :]
                            legacy_tip = legacy_ras[j - l_buf:j + u_buf + 1, i - l_buf:i + u_buf + 1]
                            is_legacy = (legacy_tip > 0) & ~((rgb_tip[:, :, 0] == 0) & (rgb_tip[:, :, 1] == 0) & (rgb_tip[:, :, 2] == 0))
                            legacy_ratio = round(np.sum(is_legacy) / (tip_size_grids ** 2), 5)

                            if rgb_tip.shape[0] != tip_size_grids or rgb_tip.shape[1] != tip_size_grids:
                                if progress is not None:
                                    progress.update(1)
                                continue

                            patch_channels = [rgb_tip.astype(np.float32)]
                            valid_patch = True

                            if dem_ras is not None:
                                dem_tip = dem_ras[j - l_buf:j + u_buf + 1, i - l_buf:i + u_buf + 1]
                                if dem_tip.shape != (tip_size_grids, tip_size_grids):
                                    valid_patch = False
                                else:
                                    patch_channels.append(dem_tip.astype(np.float32)[..., np.newaxis])

                            if vdvi_ras is not None and valid_patch:
                                vdvi_tip = vdvi_ras[j - l_buf:j + u_buf + 1, i - l_buf:i + u_buf + 1]
                                if vdvi_tip.shape != (tip_size_grids, tip_size_grids):
                                    valid_patch = False
                                else:
                                    patch_channels.append(vdvi_tip.astype(np.float32)[..., np.newaxis])

                            # Add GLCM channel (after VDVI, as the last auxiliary channel)
                            if glcm_ras is not None and valid_patch:
                                glcm_tip = glcm_ras[j - l_buf:j + u_buf + 1, i - l_buf:i + u_buf + 1]
                                if glcm_tip.shape != (tip_size_grids, tip_size_grids):
                                    valid_patch = False
                                else:
                                    patch_channels.append(glcm_tip.astype(np.float32)[..., np.newaxis])

                            # Add EXG channel (after GLCM)
                            if use_exg and exg_ras is not None and valid_patch:
                                exg_tip = exg_ras[j - l_buf:j + u_buf + 1, i - l_buf:i + u_buf + 1]
                                if exg_tip.shape != (tip_size_grids, tip_size_grids):
                                    valid_patch = False
                                else:
                                    patch_channels.append(exg_tip.astype(np.float32)[..., np.newaxis])

                            if not valid_patch:
                                if progress is not None:
                                    progress.update(1)
                                continue

                            combined_tip = np.dstack(patch_channels).astype(np.float32)

                            row_slice = slice(j - l_buf, j + u_buf + 1)
                            col_slice = slice(i - l_buf, i + u_buf + 1)
                            ratio_sum[row_slice, col_slice] += legacy_ratio
                            ratio_count[row_slice, col_slice] += 1.0

                            if collect_data:
                                tips.append(combined_tip)
                                labels.append(legacy_ratio)
                                pixel_locations.append((i, j))
                                geo_locations.append(self._pixel_to_coord(i, j, rgb_transform))
                                landslide_ids.append(landslide_id)

                            if progress is not None:
                                progress.update(1)
            finally:
                if progress is not None:
                    progress.close()

            if ratio_count.any():
                ratio_map = np.full_like(ratio_sum, np.nan, dtype=np.float32)
                valid_mask = ratio_count > 0
                ratio_map[valid_mask] = ratio_sum[valid_mask] / ratio_count[valid_mask]
                ratio_map = np.where(np.isnan(landslide_ras), np.nan, ratio_map)
            else:
                ratio_map = np.full_like(ratio_sum, np.nan, dtype=np.float32)

            if self.patch_overlay_dir is not None:
                self._save_patch_overlay(
                    Path(rgb_path).stem,
                    ratio_map,
                    dem_ras,
                    rgb_extent,
                    landslide_aoi_gdf,
                )
                if vdvi_ras is not None:
                    self._save_vdvi_overlay(
                        Path(rgb_path).stem,
                        rgb_ras,
                        vdvi_ras,
                        rgb_extent
                    )

        tips_array = np.array(tips, dtype=np.float32) if tips else np.empty((0, tip_size_grids, tip_size_grids, 0), dtype=np.float32)
        return tips_array, np.array(labels), np.array(pixel_locations), np.array(geo_locations), np.array(landslide_ids)


# *******************************************************************
# 閾値最適化クラス
# *******************************************************************
def assign_class_from_thresholds(value: float, thresholds: List[float], eps: float = 1e-6) -> int:
    """
    連続値ラベル(0-1)をクラスインデックスに変換する
    
    thresholds = [0.0, 0.25, 0.5, 0.75] の場合:
    - Class 0: 0%（value ≈ 0.0）
    - Class 1: 0%超〜25%以下（0.0 < value <= 0.25）
    - Class 2: 25%超〜50%以下（0.25 < value <= 0.50）
    - Class 3: 50%超〜75%以下（0.50 < value <= 0.75）
    - Class 4: 75%超〜100%（0.75 < value <= 1.0）
    
    Parameters:
    -----------
    value : float
        連続値ラベル（0.0〜1.0）
    thresholds : List[float]
        クラス境界閾値 [0.0, 0.25, 0.5, 0.75]
    eps : float
        浮動小数点比較の許容誤差
    
    Returns:
    --------
    int
        クラスラベル（0〜n_classes-1）
    """
    # 0% の特別処理（完全崩壊）
    if abs(value) < eps:
        return 0
    
    # 100% の特別処理（ほぼ完全回復）
    if abs(value - 1.0) < eps:
        return len(thresholds)
    
    # 閾値に基づいて判定
    # thresholds[0] = 0.0 は Class 0 用なのでスキップし、thresholds[1:] から判定
    for i in range(1, len(thresholds)):
        if value <= thresholds[i] + eps:
            return i
    
    # 最後の閾値を超えた場合は最大クラス
    return len(thresholds)


def generate_class_names_from_thresholds(thresholds: Optional[List[float]], n_classes: int) -> List[str]:
    """Create human-readable class labels from threshold definitions."""
    if thresholds is None or len(thresholds) != n_classes - 1:
        return [f'Class {i}' for i in range(n_classes)]

    percentages = [int(round(t * 100)) for t in thresholds]
    bounds = [0] + percentages + [100]
    class_names: List[str] = []

    for idx in range(n_classes):
        lower = bounds[idx]
        upper = bounds[idx + 1]
        if idx == 0:
            class_names.append("0%")
            continue

        lower_str = f"<{lower}" if lower != 0 else "<0"
        upper_str = f"<{upper}" if upper < 100 else "<100"
        class_names.append(f"{lower_str}~{upper_str}%")

    return class_names


def to_serializable(value):
    """Convert numpy/scalar containers into JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


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
            classes[i] = assign_class_from_thresholds(label, thresholds)
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
# データ拡張（Geometric Augmentation for Training）
# *******************************************************************
class GeometricAugmentation:
    """
    学習データ用の幾何学的データ拡張
    
    航空写真チップ分類における回転不変性を利用し、過学習を抑制する。
    全チャンネル（RGB, DEM, Slope, VDVI）に同じ変換を適用。
    
    Parameters:
    -----------
    h_flip_prob : float
        水平反転の確率（デフォルト: 0.5）
    v_flip_prob : float
        垂直反転の確率（デフォルト: 0.5）
    rotate90_prob : float
        90度回転の確率（デフォルト: 0.5）
    """
    
    def __init__(self, h_flip_prob: float = 0.5, v_flip_prob: float = 0.5, 
                 rotate90_prob: float = 0.5):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.rotate90_prob = rotate90_prob
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        入力テンソルに幾何学的変換を適用
        
        Parameters:
        -----------
        image : torch.Tensor
            形状 (C, H, W) のテンソル
        
        Returns:
        --------
        torch.Tensor
            変換後のテンソル（同じ形状）
        """
        # Random Horizontal Flip
        if random.random() < self.h_flip_prob:
            image = torch.flip(image, dims=[2])  # W軸で反転
        
        # Random Vertical Flip
        if random.random() < self.v_flip_prob:
            image = torch.flip(image, dims=[1])  # H軸で反転
        
        # Random Rotate 90 degrees (0, 90, 180, 270 degrees)
        if random.random() < self.rotate90_prob:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            image = torch.rot90(image, k, dims=[1, 2])  # (H, W) 平面で回転
        
        return image


# *******************************************************************
# Advanced Augmentation with ColorJitter (Phase 3-1)
# *******************************************************************
class AdvancedAugmentation:
    """
    幾何学的変換 + RGB色調変換
    
    重要: DEM/Slope/VDVI等の補助チャンネルには色調変換を適用しない
    
    Parameters:
    -----------
    h_flip_prob : float
        水平反転の確率
    v_flip_prob : float
        垂直反転の確率
    rotate90_prob : float
        90度回転の確率
    color_jitter_prob : float
        ColorJitter適用の確率
    brightness : float
        明度の変動範囲
    contrast : float
        コントラストの変動範囲
    saturation : float
        彩度の変動範囲
    hue : float
        色相の変動範囲（植生の緑を維持するため控えめに）
    """
    
    def __init__(
        self,
        h_flip_prob: float = 0.5,
        v_flip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        color_jitter_prob: float = 0.3,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05  # 色相は控えめに
    ):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.rotate90_prob = rotate90_prob
        self.color_jitter_prob = color_jitter_prob
        
        # ColorJitter変換（RGB用）
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        入力テンソルに拡張を適用
        
        Parameters:
        -----------
        image : torch.Tensor
            形状 (C, H, W) のテンソル
            チャンネル順序: [RGB(0-2), DEM(3), Slope(4), VDVI(5), ...]
        
        Returns:
        --------
        torch.Tensor
            拡張後のテンソル
        """
        # === 幾何学的変換（全チャンネルに適用） ===
        
        # Random Horizontal Flip
        if random.random() < self.h_flip_prob:
            image = torch.flip(image, dims=[2])
        
        # Random Vertical Flip
        if random.random() < self.v_flip_prob:
            image = torch.flip(image, dims=[1])
        
        # Random Rotate 90 degrees
        if random.random() < self.rotate90_prob:
            k = random.randint(1, 3)
            image = torch.rot90(image, k, dims=[1, 2])
        
        # === ColorJitter（RGBチャンネルのみに適用） ===
        if random.random() < self.color_jitter_prob and image.shape[0] >= 3:
            # RGBチャンネルを分離
            rgb = image[:3, :, :]
            aux = image[3:, :, :] if image.shape[0] > 3 else None
            
            # ColorJitterを適用（0-1の範囲のテンソルに対して）
            rgb = self.color_jitter(rgb)
            
            # 再結合
            if aux is not None:
                image = torch.cat([rgb, aux], dim=0)
            else:
                image = rgb
        
        return image


# *******************************************************************
# GridMask Augmentation (Phase 3-2)
# *******************************************************************
class GridMaskAugmentation:
    """
    GridMask: 格子状にマスクを適用
    
    パッチの一部を遮蔽し、局所的特徴への過度な依存を防ぐ。
    
    Parameters:
    -----------
    prob : float
        GridMask適用の確率
    d_range : Tuple[int, int]
        格子のサイズ範囲 (min, max)
    ratio : float
        マスク領域の比率
    """
    
    def __init__(
        self,
        prob: float = 0.3,
        d_range: Tuple[int, int] = (10, 20),
        ratio: float = 0.5
    ):
        self.prob = prob
        self.d_min, self.d_max = d_range
        self.ratio = ratio
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        image : torch.Tensor
            形状 (C, H, W) のテンソル
        
        Returns:
        --------
        torch.Tensor
            マスク適用後のテンソル
        """
        if random.random() > self.prob:
            return image
        
        _, h, w = image.shape
        
        # 格子サイズをランダムに選択
        d = random.randint(self.d_min, self.d_max)
        
        # マスクの長さ
        l = int(d * self.ratio)
        
        # 開始位置をランダムに
        st_h = random.randint(0, d - 1)
        st_w = random.randint(0, d - 1)
        
        # マスクを作成
        mask = torch.ones_like(image[0])  # (H, W)
        
        for i in range(st_h, h, d):
            for j in range(st_w, w, d):
                mask[i:min(i + l, h), j:min(j + l, w)] = 0
        
        return image * mask.unsqueeze(0)


class CombinedAugmentation:
    """
    GeometricAugmentation + AdvancedAugmentation + GridMaskを組み合わせた拡張
    
    config dictで各拡張の有効/無効を制御
    """
    
    def __init__(
        self,
        use_geometric: bool = True,
        use_color_jitter: bool = True,
        use_gridmask: bool = False,
        h_flip_prob: float = 0.5,
        v_flip_prob: float = 0.5,
        rotate90_prob: float = 0.5,
        color_jitter_prob: float = 0.3,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        gridmask_prob: float = 0.3,
        gridmask_d_range: Tuple[int, int] = (10, 20),
        gridmask_ratio: float = 0.5
    ):
        self.use_geometric = use_geometric
        self.use_color_jitter = use_color_jitter
        self.use_gridmask = use_gridmask
        
        # 幾何学的変換パラメータ
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.rotate90_prob = rotate90_prob
        
        # ColorJitter（RGB専用）
        if use_color_jitter:
            self.color_jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )
            self.color_jitter_prob = color_jitter_prob
        
        # GridMask
        if use_gridmask:
            self.gridmask = GridMaskAugmentation(
                prob=gridmask_prob,
                d_range=gridmask_d_range,
                ratio=gridmask_ratio
            )
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        image : torch.Tensor
            形状 (C, H, W) のテンソル
        
        Returns:
        --------
        torch.Tensor
            拡張後のテンソル
        """
        # === 幾何学的変換 ===
        if self.use_geometric:
            if random.random() < self.h_flip_prob:
                image = torch.flip(image, dims=[2])
            
            if random.random() < self.v_flip_prob:
                image = torch.flip(image, dims=[1])
            
            if random.random() < self.rotate90_prob:
                k = random.randint(1, 3)
                image = torch.rot90(image, k, dims=[1, 2])
        
        # === ColorJitter（RGB専用） ===
        if self.use_color_jitter and image.shape[0] >= 3:
            if random.random() < self.color_jitter_prob:
                rgb = image[:3, :, :]
                aux = image[3:, :, :] if image.shape[0] > 3 else None
                
                rgb = self.color_jitter(rgb)
                
                if aux is not None:
                    image = torch.cat([rgb, aux], dim=0)
                else:
                    image = rgb
        
        # === GridMask ===
        if self.use_gridmask:
            image = self.gridmask(image)
        
        return image


# *******************************************************************
# PyTorch Dataset（閾値可変対応版）
# *******************************************************************
class LandslideDataset(Dataset):
    """PyTorch用のデータセットクラス（閾値可変対応）
    
    Parameters:
    -----------
    force_labels : np.ndarray, optional
        外部から強制的に設定するラベル配列。指定された場合、元のlabelsを上書きする。
        Pseudo-Labeling等で使用。配列長はimagesと一致する必要がある。
    """
    
    def __init__(self, images, labels, pixel_locations, geo_locations, 
                 region_idx=None, transform=None, patch_size=25, 
                 thresholds=None, n_classes=5, dem_range=None, vdvi_range=None,
                 glcm_range=None, force_labels=None):
        self.images = images
        self.pixel_locations = pixel_locations
        self.geo_locations = geo_locations
        self.region_idx = region_idx
        self.transform = transform
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.dem_range = dem_range
        self.vdvi_range = vdvi_range
        self.glcm_range = glcm_range  # GLCM正規化用の範囲 (min, max)
        
        # force_labelsが指定された場合は上書き（Pseudo-Labeling用）
        if force_labels is not None:
            if len(force_labels) != len(images):
                raise ValueError(
                    f"force_labels length ({len(force_labels)}) must match "
                    f"images length ({len(images)})"
                )
            self.labels = np.array(force_labels)
            self._using_force_labels = True
        else:
            self.labels = labels  # 連続値ラベル（0.0～1.0）
            self._using_force_labels = False
        
        # デフォルトの閾値設定
        if thresholds is None:
            if n_classes == 5:
                self.thresholds = FIVE_CLASS_FIXED_THRESHOLDS.copy()
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
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        if image_tensor.shape[1] != self.patch_size or image_tensor.shape[2] != self.patch_size:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.patch_size, self.patch_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # ========================================
        # チャンネルごとの正規化（チャンネル順序: [R, G, B, DEM, Slope, VDVI]）
        # ========================================
        
        # Index 0-2 (RGB): 255.0で除算し [0, 1] にクリップ
        if image_tensor.shape[0] >= 3:
            image_tensor[:3] = torch.clamp(image_tensor[:3] / 255.0, 0.0, 1.0)

        # Index 3 (DEM): dem_range を使用して正規化
        if image_tensor.shape[0] > 3 and self.dem_range is not None:
            dem_min, dem_max = self.dem_range
            if dem_max > dem_min:
                image_tensor[3] = (image_tensor[3] - dem_min) / (dem_max - dem_min + 1e-6)
                image_tensor[3] = torch.clamp(image_tensor[3], 0.0, 1.0)

        # Index 4 (Slope): 0.0〜90.0度の範囲を 90.0 で除算して正規化
        # 重要: SlopeにはVDVI rangeを適用してはいけない
        if image_tensor.shape[0] > 4:
            image_tensor[4] = torch.clamp(image_tensor[4] / 90.0, 0.0, 1.0)

        # Index 5 (VDVI): vdvi_range (-1.0〜1.0) を使用して正規化
        if image_tensor.shape[0] > 5 and self.vdvi_range is not None:
            vdvi_min, vdvi_max = self.vdvi_range
            if vdvi_max > vdvi_min:
                image_tensor[5] = (image_tensor[5] - vdvi_min) / (vdvi_max - vdvi_min + 1e-6)
                image_tensor[5] = torch.clamp(image_tensor[5], 0.0, 1.0)

        # Index 6 (GLCM): glcm_range を使用して正規化（または標準化）
        # GLCM Contrast値は広い範囲を持つため、Min-Maxスケーリングを適用
        if image_tensor.shape[0] > 6:
            if self.glcm_range is not None:
                glcm_min, glcm_max = self.glcm_range
                if glcm_max > glcm_min:
                    image_tensor[6] = (image_tensor[6] - glcm_min) / (glcm_max - glcm_min + 1e-6)
                    image_tensor[6] = torch.clamp(image_tensor[6], 0.0, 1.0)
            else:
                # glcm_rangeが指定されていない場合、パッチ内で標準化
                glcm_channel = image_tensor[6]
                glcm_min = glcm_channel.min()
                glcm_max = glcm_channel.max()
                if glcm_max > glcm_min:
                    image_tensor[6] = (glcm_channel - glcm_min) / (glcm_max - glcm_min + 1e-6)
                else:
                    image_tensor[6] = torch.zeros_like(glcm_channel)

        image = image_tensor

        if self.transform:
            image = self.transform(image)
        
        # 閾値に基づいてクラスラベルを生成（修正版：assign_class_from_thresholdsを使用）
        raw_label = self.labels[idx]
        class_label = assign_class_from_thresholds(raw_label, self.thresholds)
        
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


def _count_classes_from_labels(labels, thresholds, n_classes):
    """Convert continuous ratios to class counts using the provided thresholds."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for label in labels:
        class_idx = assign_class_from_thresholds(label, thresholds)
        counts[class_idx] += 1
    return counts


def compute_inverse_frequency_weights(labels, thresholds, n_classes):
    """Return weights proportional to inverse class frequency (sum normalized)."""
    counts = _count_classes_from_labels(labels, thresholds, n_classes)
    weights = np.zeros(n_classes, dtype=np.float32)
    mask = counts > 0
    if np.any(mask):
        weights[mask] = 1.0 / counts[mask].astype(np.float32)
        weights[mask] *= mask.sum() / np.sum(weights[mask])
    else:
        weights[:] = 1.0
    return weights, counts


# *******************************************************************
# Pseudo-Label Generator（Self-Training用）
# *******************************************************************
class PseudoLabelGenerator:
    """
    Teacherモデルを用いてPseudo-Labelを生成するクラス
    
    Self-Trainingパイプラインで使用し、学習データのラベルをリファインする。
    高確信度で現在のラベルと異なる予測のみを新しいラベルとして採用。
    
    Parameters:
    -----------
    model : nn.Module
        Teacherモデル（学習済み）
    device : str
        使用デバイス（'cuda' or 'cpu'）
    n_classes : int
        クラス数
    thresholds : List[float], optional
        クラス分類の閾値
    """
    
    def __init__(self, model, device, n_classes=5, thresholds=None):
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.thresholds = thresholds if thresholds is not None else FIVE_CLASS_FIXED_THRESHOLDS.copy()
        self.model.eval()
    
    def _label_to_class(self, label):
        """連続値ラベルをクラスインデックスに変換"""
        for i, threshold in enumerate(self.thresholds):
            if label <= threshold:
                return i
        return len(self.thresholds)
    
    def _class_to_label_center(self, class_idx):
        """クラスインデックスを代表的な連続値ラベルに変換"""
        bounds = [0.0] + list(self.thresholds) + [1.0]
        return (bounds[class_idx] + bounds[class_idx + 1]) / 2
    
    def generate(self, dataset, batch_size=64, threshold=0.9, 
                 require_class_change=True):
        """
        Pseudo-Labelを生成する
        
        Parameters:
        -----------
        dataset : LandslideDataset
            対象データセット
        batch_size : int
            推論時のバッチサイズ
        threshold : float
            Pseudo-Label採用の確信度閾値（softmax確率）
        require_class_change : bool
            Trueの場合、現在のラベルと異なる予測のみを採用
        
        Returns:
        --------
        new_labels : np.ndarray
            更新後のラベル配列（変更がない箇所は元のまま）
        change_indices : List[int]
            ラベルが変更されたサンプルのインデックス
        change_stats : Dict
            変更統計情報
        """
        from torch.utils.data import DataLoader
        
        # 元のラベルをコピー
        original_labels = np.array(dataset.labels).copy()
        new_labels = original_labels.copy()
        
        # 推論用DataLoader（transformなし）
        inference_dataset = LandslideDataset(
            images=dataset.images,
            labels=dataset.labels,
            pixel_locations=dataset.pixel_locations,
            geo_locations=dataset.geo_locations,
            region_idx=dataset.region_idx,
            transform=None,  # 推論時はaugmentationなし
            patch_size=dataset.patch_size,
            thresholds=dataset.thresholds,
            n_classes=dataset.n_classes,
            dem_range=dataset.dem_range,
            vdvi_range=dataset.vdvi_range,
        )
        loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
        
        # 推論結果を収集
        all_probs = []
        all_preds = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Generating pseudo-labels"):
                inputs = batch['image'].to(self.device)
                outputs = self.model(inputs)
                
                # 出力がtupleの場合（中間出力を含む場合）
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        all_preds = np.concatenate(all_preds)
        
        # 変更候補の抽出
        change_indices = []
        change_details = []  # (index, old_label, new_label, confidence, old_class, new_class)
        
        for i in range(len(original_labels)):
            max_prob = all_probs[i].max()
            pred_class = all_preds[i]
            old_class = self._label_to_class(original_labels[i])
            
            # 確信度が閾値以上
            if max_prob >= threshold:
                # 現在のクラスと異なる予測（オプション）
                if require_class_change and pred_class == old_class:
                    continue
                
                # 新しいラベル（クラスの中央値）を設定
                new_label = self._class_to_label_center(pred_class)
                new_labels[i] = new_label
                
                change_indices.append(i)
                change_details.append({
                    'index': i,
                    'old_label': original_labels[i],
                    'new_label': new_label,
                    'old_class': old_class,
                    'new_class': pred_class,
                    'confidence': max_prob,
                    'probs': all_probs[i].tolist(),
                })
        
        # 統計情報
        change_stats = {
            'total_samples': len(original_labels),
            'changed_samples': len(change_indices),
            'change_ratio': len(change_indices) / len(original_labels) if len(original_labels) > 0 else 0,
            'threshold': threshold,
            'require_class_change': require_class_change,
            'details': change_details,
            'class_change_matrix': self._compute_change_matrix(change_details),
            'avg_confidence': np.mean([d['confidence'] for d in change_details]) if change_details else 0,
        }
        
        return new_labels, change_indices, change_stats
    
    def _compute_change_matrix(self, change_details):
        """クラス間の変更を集計（confusion matrix形式）"""
        matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int32)
        for detail in change_details:
            matrix[detail['old_class'], detail['new_class']] += 1
        return matrix.tolist()


# *******************************************************************
# ResNet18カスタム実装
# *******************************************************************
class ResNet18Custom(nn.Module):
    """ResNet18 - 小さい画像パッチ対応版"""
    
    def __init__(self, num_classes=5, pretrained=True, dropout=0.3, input_size=25, in_channels=3):
        super(ResNet18Custom, self).__init__()
        
        self.backbone = models.resnet18(pretrained=pretrained)
        old_conv = self.backbone.conv1
        
        # 小さい画像への適応
        if input_size <= 32:
            # 最初の畳み込み層を調整
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            # MaxPoolを削除（小さい画像では不要）
            self.backbone.maxpool = nn.Identity()
        else:
            if in_channels != old_conv.in_channels:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )

        self._initialize_first_conv_weights(old_conv, in_channels, pretrained)
        
        # 分類器の変更（多クラス分類用に修正）
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)  # Sigmoidを削除（CrossEntropyLossで使用するため）
        )
    
    def _initialize_first_conv_weights(self, old_conv, in_channels, pretrained):
        if not pretrained:
            return

        new_conv = self.backbone.conv1
        with torch.no_grad():
            old_weight = old_conv.weight
            target_kernel = new_conv.weight.shape[2:]

            if old_weight.shape[2:] != target_kernel:
                old_weight = self._resize_conv_kernel(old_weight, target_kernel)

            original_channels = old_weight.shape[1]

            if in_channels == original_channels:
                new_conv.weight.copy_(old_weight)
            elif in_channels < original_channels:
                new_conv.weight.copy_(old_weight[:, :in_channels, :, :])
            else:
                new_conv.weight[:, :original_channels, :, :].copy_(old_weight)
                extra_channels = in_channels - original_channels
                mean_weights = old_weight.mean(dim=1, keepdim=True)
                repeated = mean_weights.repeat(1, extra_channels, 1, 1)
                new_conv.weight[:, original_channels:, :, :].copy_(repeated)

    def _resize_conv_kernel(self, weight, target_size):
        """Resize pretrained kernel weights to match a new convolution kernel size."""
        if weight.shape[2:] == target_size:
            return weight

        out_channels, in_channels, _, _ = weight.shape
        weight_reshaped = weight.view(out_channels * in_channels, 1, weight.shape[2], weight.shape[3])
        resized = F.interpolate(weight_reshaped, size=target_size, mode='bilinear', align_corners=False)
        return resized.view(out_channels, in_channels, target_size[0], target_size[1])
    
    def forward(self, x):
        return self.backbone(x)

# *******************************************************************
# MLP-Mixer Model (保持しておく)
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
# CBAM (Convolutional Block Attention Module)
# *******************************************************************
class ChannelAttention(nn.Module):
    """SE (Squeeze-and-Excitation) ベースのチャンネル注意機構"""
    
    def __init__(self, num_features: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共有MLP
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(num_features // reduction, num_features, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        
        # Max pooling path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out)
        
        # 結合
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    """空間的注意機構"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel方向のavg/maxをスタック
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    
    Channel Attention + Spatial Attentionを組み合わせた注意機構
    """
    
    def __init__(self, num_features: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(num_features, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# *******************************************************************
# Phase 5-2: Multi-Scale Feature Extraction (ASPP inspired)
# *******************************************************************
class MultiScaleFeatureExtractor(nn.Module):
    """
    複数スケールでの特徴抽出（ASPP inspired）
    
    異なるdilation rateのatrous convolutionで複数スケールの特徴を抽出し、
    細かいテクスチャと大きなパターンの両方を捉える。
    """
    
    def __init__(self, in_channels: int, out_channels: int, dilations: List[int] = [1, 2, 4, 8]):
        """
        Parameters:
        -----------
        in_channels : int
            入力チャンネル数
        out_channels : int
            出力チャンネル数
        dilations : List[int]
            各スケールのdilation rate
        """
        super().__init__()
        
        self.branches = nn.ModuleList()
        
        # 1x1 convolution (global context)
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels // len(dilations), 1, bias=False),
            nn.BatchNorm2d(out_channels // len(dilations)),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions with different dilation rates
        for d in dilations[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(dilations), 3, 
                         padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels // len(dilations)),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // len(dilations), 1, bias=False),
            nn.BatchNorm2d(out_channels // len(dilations)),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        total_channels = (out_channels // len(dilations)) * (len(dilations) + 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x : torch.Tensor
            入力 (B, C, H, W)
        
        Returns:
        --------
        torch.Tensor
            マルチスケール特徴 (B, out_channels, H, W)
        """
        size = x.shape[2:]
        
        # 各スケールの特徴を計算
        features = [branch(x) for branch in self.branches]
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        features.append(global_feat)
        
        # 結合 & 融合
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


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


#Hybrid CNN-ViT Model
class HybridModel(nn.Module):
    """
    斜面崩壊残存植生検出に特化したハイブリッドモデル（ResNet18バックボーン版）
    
    設計思想:
    - RGB: ResNet18（ImageNet事前学習済み）でテクスチャ・空間パターンを抽出
    - VDVI/DEM/Slope: MLPで直接処理（最も効果的だった手法を維持）
    - 融合: Attentionベースではなく単純な連結＋ゲーティング
    
    in_channels対応（チャンネル順序: [RGB, DEM, Slope, VDVI]）:
    - 3: RGBのみ
    - 4: RGB + DEM
    - 5: RGB + DEM + Slope
    - 6: RGB + DEM + Slope + VDVI
    """
    
    def __init__(self, num_classes=5, image_size=25, dropout=0.3, in_channels=5, 
                 use_pretrained=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.image_size = image_size
        # 新しいチャンネル順序: [RGB(0-2), DEM(3), Slope(4), VDVI(5)]
        self.use_dem = in_channels >= 4    # DEM is at index 3
        self.use_slope = in_channels >= 5  # Slope is at index 4
        self.use_vdvi = in_channels >= 6   # VDVI is at index 5
        
        # ========================================
        # Stream 1: RGB空間特徴抽出（ResNet18事前学習済みバックボーン）
        # ========================================
        # ImageNetで事前学習済みの重みを利用してテクスチャ抽出能力を向上
        resnet = models.resnet18(pretrained=use_pretrained)
        
        # 小さい画像パッチ（25x25）への適応
        if image_size <= 32:
            # 最初の畳み込み層を小さいカーネルに変更
            self.rgb_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # 事前学習済み重みの中央部分をコピー（可能な範囲で）
            if use_pretrained:
                with torch.no_grad():
                    # 7x7カーネルの中央3x3を抽出
                    old_weight = resnet.conv1.weight.data
                    center = old_weight[:, :, 2:5, 2:5]  # 中央の3x3を取得
                    self.rgb_conv1.weight.data = center
            self.rgb_bn1 = resnet.bn1
            self.rgb_relu = resnet.relu
            # MaxPoolは小さい画像では省略
            self.rgb_maxpool = nn.Identity()
        else:
            self.rgb_conv1 = resnet.conv1
            self.rgb_bn1 = resnet.bn1
            self.rgb_relu = resnet.relu
            self.rgb_maxpool = resnet.maxpool
        
        # ResNet18のLayer1〜Layer4を使用
        self.rgb_layer1 = resnet.layer1  # 64ch
        self.rgb_layer2 = resnet.layer2  # 128ch
        self.rgb_layer3 = resnet.layer3  # 256ch
        self.rgb_layer4 = resnet.layer4  # 512ch
        self.rgb_avgpool = nn.AdaptiveAvgPool2d(1)
        
        # RGB特徴量の次元（ResNet18のlayer4出力）
        self.rgb_dim = 512
        
        # ファインチューニング用: 初期層は凍結、後半層は学習可能
        if use_pretrained:
            # layer1, layer2を凍結（低レベル特徴は転用）
            for param in self.rgb_layer1.parameters():
                param.requires_grad = False
            for param in self.rgb_layer2.parameters():
                param.requires_grad = False
            # layer3, layer4は学習可能（高レベル特徴は適応）
        
        # ========================================
        # Stream 2: DEM/Slope/VDVI数値特徴処理（MLP）
        # ========================================
        # MLPが最も効果的だった観察を活かす
        # DEM, Slope, VDVIを別々に処理してから統合
        
        # パッチ全体の統計量を使用
        # 4チャンネル（mean, std, min, max）× 利用可能なAuxチャンネル数
        n_aux_channels = int(self.use_dem) + int(self.use_slope) + int(self.use_vdvi)
        self.aux_stats_dim = 4 * n_aux_channels if n_aux_channels > 0 else 0
        
        if self.aux_stats_dim > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(self.aux_stats_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),  # 軽めのdropout
                
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
            self.aux_dim = 128
        else:
            self.aux_mlp = None
            self.aux_dim = 0
        
        # ========================================
        # Stream 3: VDVIの空間パターン（オプション）
        # ========================================
        # CNNでVDVIを処理すると悪化したため、
        # 空間情報は非常に軽量な形でのみ抽出
        if self.use_vdvi:
            self.vdvi_spatial = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.vdvi_spatial_dim = 16
        else:
            self.vdvi_spatial = None
            self.vdvi_spatial_dim = 0
        
        # ========================================
        # 融合レイヤー（Gated Fusion）
        # ========================================
        # Attention機構は過剰なので、単純なゲーティングを使用
        total_dim = self.rgb_dim + self.aux_dim + self.vdvi_spatial_dim
        
        # ゲーティング：各ストリームの重要度を動的に調整
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid()
        )
        
        # ========================================
        # 分類ヘッド
        # ========================================
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(128, num_classes)
        )
        
        # 重み初期化（分類ヘッドのみ、ResNet部分は事前学習済み）
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """分類ヘッドとAux MLPの重みを初期化（ResNet部分は除く）"""
        for module in [self.aux_mlp, self.vdvi_spatial, self.gate, self.classifier]:
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def extract_rgb_features(self, rgb):
        """ResNet18バックボーンでRGB特徴を抽出"""
        x = self.rgb_conv1(rgb)
        x = self.rgb_bn1(x)
        x = self.rgb_relu(x)
        x = self.rgb_maxpool(x)
        
        x = self.rgb_layer1(x)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        
        x = self.rgb_avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def extract_aux_stats(self, dem, slope, vdvi):
        """DEM, Slope, VDVIから統計量を抽出（MLPで処理するため）
        
        チャンネル順序: [RGB(0-2), DEM(3), Slope(4), VDVI(5)]
        """
        stats_list = []
        # batch_sizeを取得（どれか一つでもあれば）
        batch_size = None
        device = None
        for tensor in [dem, slope, vdvi]:
            if tensor is not None:
                batch_size = tensor.size(0)
                device = tensor.device
                break
        
        if batch_size is None:
            raise ValueError("At least one of dem, slope, vdvi must be provided")
        
        # DEM統計量
        if dem is not None and dem.numel() > 0:
            dem_flat = dem.view(batch_size, -1)
            if dem_flat.size(1) > 0:
                dem_mean = dem_flat.mean(dim=1, keepdim=True)
                dem_std = dem_flat.std(dim=1, keepdim=True)
                dem_min = dem_flat.min(dim=1, keepdim=True)[0]
                dem_max = dem_flat.max(dim=1, keepdim=True)[0]
                stats_list.extend([dem_mean, dem_std, dem_min, dem_max])
        
        # Slope統計量
        if slope is not None and slope.numel() > 0:
            slope_flat = slope.view(batch_size, -1)
            if slope_flat.size(1) > 0:
                slope_mean = slope_flat.mean(dim=1, keepdim=True)
                slope_std = slope_flat.std(dim=1, keepdim=True)
                slope_min = slope_flat.min(dim=1, keepdim=True)[0]
                slope_max = slope_flat.max(dim=1, keepdim=True)[0]
                stats_list.extend([slope_mean, slope_std, slope_min, slope_max])
        
        # VDVI統計量
        if vdvi is not None and vdvi.numel() > 0:
            vdvi_flat = vdvi.view(batch_size, -1)
            if vdvi_flat.size(1) > 0:
                vdvi_mean = vdvi_flat.mean(dim=1, keepdim=True)
                vdvi_std = vdvi_flat.std(dim=1, keepdim=True)
                vdvi_min = vdvi_flat.min(dim=1, keepdim=True)[0]
                vdvi_max = vdvi_flat.max(dim=1, keepdim=True)[0]
                stats_list.extend([vdvi_mean, vdvi_std, vdvi_min, vdvi_max])
        
        # 連結
        if stats_list:
            stats = torch.cat(stats_list, dim=1)
        else:
            # Auxデータがない場合はダミーのゼロテンソルを返す
            stats = torch.zeros(batch_size, 0, device=device)
        
        return stats
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        チャンネル順序: [RGB(0-2), DEM(3), Slope(4), VDVI(5)]
        C = 3 (RGB only), 4 (RGB+DEM), 5 (RGB+DEM+Slope), 6 (RGB+DEM+Slope+VDVI)
        """
        batch_size = x.size(0)
        num_channels = x.size(1)
        
        # チャンネル分離（新しい順序: [RGB, DEM, Slope, VDVI]）
        rgb = x[:, :3, :, :]  # (B, 3, H, W)
        
        dem = None
        slope = None
        vdvi = None
        if num_channels >= 4 and self.use_dem:
            dem = x[:, 3:4, :, :]    # (B, 1, H, W) - DEM at index 3
        if num_channels >= 5 and self.use_slope:
            slope = x[:, 4:5, :, :]  # (B, 1, H, W) - Slope at index 4
        if num_channels >= 6 and self.use_vdvi:
            vdvi = x[:, 5:6, :, :]   # (B, 1, H, W) - VDVI at index 5
        
        # Stream 1: RGB空間特徴（ResNet18バックボーン）
        rgb_feat = self.extract_rgb_features(rgb)  # (B, 512)
        
        features_to_combine = [rgb_feat]
        
        # Stream 2: DEM/Slope/VDVI統計量 → MLP（利用可能な場合のみ）
        if self.aux_mlp is not None and (dem is not None or slope is not None or vdvi is not None):
            aux_stats = self.extract_aux_stats(dem, slope, vdvi)
            if aux_stats.size(1) > 0:
                aux_feat = self.aux_mlp(aux_stats)  # (B, 128)
                features_to_combine.append(aux_feat)
        
        # Stream 3: VDVI空間パターン（軽量）（利用可能な場合のみ）
        if self.vdvi_spatial is not None and vdvi is not None:
            vdvi_spatial_feat = self.vdvi_spatial(vdvi)
            vdvi_spatial_feat = vdvi_spatial_feat.view(batch_size, -1)  # (B, 16)
            features_to_combine.append(vdvi_spatial_feat)
        
        # 融合
        combined = torch.cat(features_to_combine, dim=1)
        
        # ゲーティング（各特徴の重要度を調整）
        gate_weights = self.gate(combined)
        gated_features = combined * gate_weights
        
        # 分類
        output = self.classifier(gated_features)
        
        return output


# *******************************************************************
# HybridModelV2: 柔軟なチャンネル設定に対応した改良版
# *******************************************************************
class HybridModelV2(nn.Module):
    """
    柔軟なチャンネル設定に対応したハイブリッドモデル（GLCM/EXG対応版）
    
    channel_configで任意のチャンネル構成を指定可能:
    - RGB: 必須（ResNet18バックボーンで空間特徴を抽出）
    - DEM/Slope/VDVI/GLCM/EXG: オプション（MLPで統計量を処理）
    
    デフォルトのチャンネル構成（最大8チャンネル）:
    - Index 0-2: RGB（必須）
    - Index 3: DEM（オプション）
    - Index 4: Slope（オプション）
    - Index 5: VDVI（オプション）
    - Index 6: GLCM（オプション）
    - Index 7: EXG（オプション）
    """
    
    DEFAULT_CHANNEL_CONFIG = {
        'RGB': {'indices': [0, 1, 2], 'active': True},
        'DEM': {'indices': [3], 'active': True},
        'Slope': {'indices': [4], 'active': True},
        'VDVI': {'indices': [5], 'active': True},
        'GLCM': {'indices': [6], 'active': True},
        'EXG': {'indices': [7], 'active': True},
    }
    
    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 100,
        dropout: float = 0.3,
        in_channels: int = 8,
        channel_config: Optional[Dict] = None,
        use_pretrained: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        
        # チャンネル設定の初期化
        if channel_config is None:
            self.channel_config = self._infer_channel_config(in_channels)
        else:
            self.channel_config = channel_config
        
        # 各モダリティの有効/無効を判定
        self.use_dem = self._is_channel_active('DEM')
        self.use_slope = self._is_channel_active('Slope')
        self.use_vdvi = self._is_channel_active('VDVI')
        self.use_glcm = self._is_channel_active('GLCM')
        self.use_exg = self._is_channel_active('EXG')
        
        print(f"  HybridModelV2 initialized with channels: in_channels={in_channels}")
        print(f"    DEM={self.use_dem}, Slope={self.use_slope}, VDVI={self.use_vdvi}, "
              f"GLCM={self.use_glcm}, EXG={self.use_exg}")
        
        # ========================================
        # Stream 1: RGB空間特徴抽出（ResNet18事前学習済みバックボーン）
        # ========================================
        resnet = models.resnet18(pretrained=use_pretrained)
        
        if image_size <= 32:
            self.rgb_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if use_pretrained:
                with torch.no_grad():
                    old_weight = resnet.conv1.weight.data
                    center = old_weight[:, :, 2:5, 2:5]
                    self.rgb_conv1.weight.data = center
            self.rgb_bn1 = resnet.bn1
            self.rgb_relu = resnet.relu
            self.rgb_maxpool = nn.Identity()
        else:
            self.rgb_conv1 = resnet.conv1
            self.rgb_bn1 = resnet.bn1
            self.rgb_relu = resnet.relu
            self.rgb_maxpool = resnet.maxpool
        
        self.rgb_layer1 = resnet.layer1
        self.rgb_layer2 = resnet.layer2
        self.rgb_layer3 = resnet.layer3
        self.rgb_layer4 = resnet.layer4
        self.rgb_avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.rgb_dim = 512
        
        # ファインチューニング設定
        if use_pretrained:
            for param in self.rgb_layer1.parameters():
                param.requires_grad = False
            for param in self.rgb_layer2.parameters():
                param.requires_grad = False
        
        # ========================================
        # Stream 2: 補助チャンネル統計量処理（MLP）
        # ========================================
        # DEM, Slope, VDVI, GLCM, EXGの統計量（各4値: mean, std, min, max）
        n_aux_channels = (int(self.use_dem) + int(self.use_slope) + 
                         int(self.use_vdvi) + int(self.use_glcm) + int(self.use_exg))
        self.aux_stats_dim = 4 * n_aux_channels if n_aux_channels > 0 else 0
        
        if self.aux_stats_dim > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(self.aux_stats_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
            self.aux_dim = 128
        else:
            self.aux_mlp = None
            self.aux_dim = 0
        
        # ========================================
        # Stream 3: GLCM空間パターン（テクスチャ専用）
        # ========================================
        if self.use_glcm:
            self.texture_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.texture_dim = 32
        else:
            self.texture_cnn = None
            self.texture_dim = 0
        
        # ========================================
        # Stream 4: 植生指標空間パターン（VDVI + EXG）
        # ========================================
        vegetation_active = self.use_vdvi or self.use_exg
        if vegetation_active:
            self.vegetation_spatial = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.vegetation_spatial_dim = 16
        else:
            self.vegetation_spatial = None
            self.vegetation_spatial_dim = 0
        
        # ========================================
        # 融合レイヤー（Gated Fusion）
        # ========================================
        total_dim = self.rgb_dim + self.aux_dim + self.texture_dim + self.vegetation_spatial_dim
        
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid()
        )
        
        # ========================================
        # 分類ヘッド
        # ========================================
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        self._init_classifier_weights()
    
    def _infer_channel_config(self, in_channels: int) -> Dict:
        """in_channelsからchannel_configを推論"""
        config = {
            'RGB': {'indices': [0, 1, 2], 'active': True},
            'DEM': {'indices': [3], 'active': in_channels >= 4},
            'Slope': {'indices': [4], 'active': in_channels >= 5},
            'VDVI': {'indices': [5], 'active': in_channels >= 6},
            'GLCM': {'indices': [6], 'active': in_channels >= 7},
            'EXG': {'indices': [7], 'active': in_channels >= 8},
        }
        return config
    
    def _is_channel_active(self, channel_name: str) -> bool:
        """指定チャンネルがアクティブかどうかを判定"""
        if channel_name in self.channel_config:
            return self.channel_config[channel_name].get('active', False)
        return False
    
    def _get_channel_indices(self, channel_name: str) -> List[int]:
        """指定チャンネルのインデックスを取得"""
        if channel_name in self.channel_config:
            return self.channel_config[channel_name].get('indices', [])
        return []
    
    def _init_classifier_weights(self):
        """分類ヘッドとAux MLPの重みを初期化"""
        for module in [self.aux_mlp, self.texture_cnn, self.vegetation_spatial, 
                       self.gate, self.classifier]:
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def extract_rgb_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """ResNet18バックボーンでRGB特徴を抽出"""
        x = self.rgb_conv1(rgb)
        x = self.rgb_bn1(x)
        x = self.rgb_relu(x)
        x = self.rgb_maxpool(x)
        x = self.rgb_layer1(x)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        x = self.rgb_avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def extract_channel_stats(self, x: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """指定インデックスのチャンネルから統計量を抽出"""
        if not indices or x.size(1) <= max(indices):
            return None
        
        channel_data = x[:, indices, :, :]
        batch_size = x.size(0)
        flat = channel_data.view(batch_size, -1)
        
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)
        min_val = flat.min(dim=1, keepdim=True)[0]
        max_val = flat.max(dim=1, keepdim=True)[0]
        
        return torch.cat([mean, std, min_val, max_val], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        batch_size = x.size(0)
        num_channels = x.size(1)
        
        # Stream 1: RGB空間特徴
        rgb_indices = self._get_channel_indices('RGB')
        rgb = x[:, rgb_indices, :, :]
        rgb_feat = self.extract_rgb_features(rgb)
        
        features_to_combine = [rgb_feat]
        
        # Stream 2: 補助チャンネル統計量
        if self.aux_mlp is not None:
            aux_stats_list = []
            
            for channel_name in ['DEM', 'Slope', 'VDVI', 'GLCM', 'EXG']:
                if self._is_channel_active(channel_name):
                    indices = self._get_channel_indices(channel_name)
                    if indices and num_channels > max(indices):
                        stats = self.extract_channel_stats(x, indices)
                        if stats is not None:
                            aux_stats_list.append(stats)
            
            if aux_stats_list:
                aux_stats = torch.cat(aux_stats_list, dim=1)
                aux_feat = self.aux_mlp(aux_stats)
                features_to_combine.append(aux_feat)
        
        # Stream 3: GLCMテクスチャ空間パターン
        if self.texture_cnn is not None and self.use_glcm:
            glcm_indices = self._get_channel_indices('GLCM')
            if glcm_indices and num_channels > max(glcm_indices):
                glcm = x[:, glcm_indices, :, :]
                texture_feat = self.texture_cnn(glcm)
                texture_feat = texture_feat.view(batch_size, -1)
                features_to_combine.append(texture_feat)
        
        # Stream 4: 植生指標空間パターン（VDVIまたはEXG）
        if self.vegetation_spatial is not None:
            veg_channel = None
            if self.use_vdvi:
                vdvi_indices = self._get_channel_indices('VDVI')
                if vdvi_indices and num_channels > max(vdvi_indices):
                    veg_channel = x[:, vdvi_indices, :, :]
            elif self.use_exg:
                exg_indices = self._get_channel_indices('EXG')
                if exg_indices and num_channels > max(exg_indices):
                    veg_channel = x[:, exg_indices, :, :]
            
            if veg_channel is not None:
                veg_feat = self.vegetation_spatial(veg_channel)
                veg_feat = veg_feat.view(batch_size, -1)
                features_to_combine.append(veg_feat)
        
        # 融合
        combined = torch.cat(features_to_combine, dim=1)
        gate_weights = self.gate(combined)
        gated_features = combined * gate_weights
        
        # 分類
        output = self.classifier(gated_features)
        return output


class HybridModelLegacy(nn.Module):
    """
    斜面崩壊残存植生検出に特化したハイブリッドモデル（旧版：浅いCNN）
    
    設計思想:
    - RGB: 浅いCNNで空間パターンを抽出（過学習防止）
    - VDVI/DEM: MLPで直接処理（最も効果的だった手法を維持）
    - 融合: Attentionベースではなく単純な連結＋ゲーティング
    
    in_channels対応（チャンネル順序: [RGB, DEM, Slope, VDVI]）:
    - 3: RGBのみ
    - 4: RGB + DEM
    - 5: RGB + DEM + Slope
    - 6: RGB + DEM + Slope + VDVI
    """
    
    def __init__(self, num_classes=5, image_size=25, dropout=0.3, in_channels=5):
        super().__init__()
        
        self.in_channels = in_channels
        # 新しいチャンネル順序: [RGB(0-2), DEM(3), Slope(4), VDVI(5)]
        self.use_dem = in_channels >= 4    # DEM is at index 3
        self.use_slope = in_channels >= 5  # Slope is at index 4
        self.use_vdvi = in_channels >= 6   # VDVI is at index 5
        
        # ========================================
        # Stream 1: RGB空間特徴抽出（浅いCNN）
        # ========================================
        # ResNetの過学習問題を回避するため、2-3層に制限
        self.rgb_encoder = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 -> 12
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 -> 6
            
            # Block 3: 64 -> 128（深くしすぎない）
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global pooling
        )
        
        self.rgb_dim = 128
        
        # ========================================
        # Stream 2: DEM/Slope/VDVI数値特徴処理（MLP）
        # ========================================
        # MLPが最も効果的だった観察を活かす
        # DEM, Slope, VDVIを別々に処理してから統合
        
        # パッチ全体の統計量を使用
        # 4チャンネル（mean, std, min, max）× 利用可能なAuxチャンネル数
        n_aux_channels = int(self.use_dem) + int(self.use_slope) + int(self.use_vdvi)
        self.aux_stats_dim = 4 * n_aux_channels if n_aux_channels > 0 else 0
        
        if self.aux_stats_dim > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(self.aux_stats_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),  # 軽めのdropout
                
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.GELU(),
            )
            self.aux_dim = 64
        else:
            self.aux_mlp = None
            self.aux_dim = 0
        
        # ========================================
        # Stream 3: VDVIの空間パターン（オプション）
        # ========================================
        # CNNでVDVIを処理すると悪化したため、
        # 空間情報は非常に軽量な形でのみ抽出
        if self.use_vdvi:
            self.vdvi_spatial = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.vdvi_spatial_dim = 8
        else:
            self.vdvi_spatial = None
            self.vdvi_spatial_dim = 0
        
        # ========================================
        # 融合レイヤー（Gated Fusion）
        # ========================================
        # Attention機構は過剰なので、単純なゲーティングを使用
        total_dim = self.rgb_dim + self.aux_dim + self.vdvi_spatial_dim
        
        # ゲーティング：各ストリームの重要度を動的に調整
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.Sigmoid()
        )
        
        # ========================================
        # 分類ヘッド
        # ========================================
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(64, num_classes)
        )
        
        # 重み初期化（過学習防止）
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def extract_aux_stats(self, dem, slope, vdvi):
        """DEM, Slope, VDVIから統計量を抽出（MLPで処理するため）
        
        チャンネル順序: [RGB(0-2), DEM(3), Slope(4), VDVI(5)]
        """
        stats_list = []
        # batch_sizeを取得（どれか一つでもあれば）
        batch_size = None
        device = None
        for tensor in [dem, slope, vdvi]:
            if tensor is not None:
                batch_size = tensor.size(0)
                device = tensor.device
                break
        
        if batch_size is None:
            raise ValueError("At least one of dem, slope, vdvi must be provided")
        
        # DEM統計量
        if dem is not None and dem.numel() > 0:
            dem_flat = dem.view(batch_size, -1)
            if dem_flat.size(1) > 0:
                dem_mean = dem_flat.mean(dim=1, keepdim=True)
                dem_std = dem_flat.std(dim=1, keepdim=True)
                dem_min = dem_flat.min(dim=1, keepdim=True)[0]
                dem_max = dem_flat.max(dim=1, keepdim=True)[0]
                stats_list.extend([dem_mean, dem_std, dem_min, dem_max])
        
        # Slope統計量
        if slope is not None and slope.numel() > 0:
            slope_flat = slope.view(batch_size, -1)
            if slope_flat.size(1) > 0:
                slope_mean = slope_flat.mean(dim=1, keepdim=True)
                slope_std = slope_flat.std(dim=1, keepdim=True)
                slope_min = slope_flat.min(dim=1, keepdim=True)[0]
                slope_max = slope_flat.max(dim=1, keepdim=True)[0]
                stats_list.extend([slope_mean, slope_std, slope_min, slope_max])
        
        # VDVI統計量
        if vdvi is not None and vdvi.numel() > 0:
            vdvi_flat = vdvi.view(batch_size, -1)
            if vdvi_flat.size(1) > 0:
                vdvi_mean = vdvi_flat.mean(dim=1, keepdim=True)
                vdvi_std = vdvi_flat.std(dim=1, keepdim=True)
                vdvi_min = vdvi_flat.min(dim=1, keepdim=True)[0]
                vdvi_max = vdvi_flat.max(dim=1, keepdim=True)[0]
                stats_list.extend([vdvi_mean, vdvi_std, vdvi_min, vdvi_max])
        
        # 連結
        if stats_list:
            stats = torch.cat(stats_list, dim=1)
        else:
            # Auxデータがない場合はダミーのゼロテンソルを返す
            stats = torch.zeros(batch_size, 0, device=device)
        
        return stats
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        チャンネル順序: [RGB(0-2), DEM(3), Slope(4), VDVI(5)]
        C = 3 (RGB only), 4 (RGB+DEM), 5 (RGB+DEM+Slope), 6 (RGB+DEM+Slope+VDVI)
        """
        batch_size = x.size(0)
        num_channels = x.size(1)
        
        # チャンネル分離（新しい順序: [RGB, DEM, Slope, VDVI]）
        rgb = x[:, :3, :, :]  # (B, 3, H, W)
        
        dem = None
        slope = None
        vdvi = None
        if num_channels >= 4 and self.use_dem:
            dem = x[:, 3:4, :, :]    # (B, 1, H, W) - DEM at index 3
        if num_channels >= 5 and self.use_slope:
            slope = x[:, 4:5, :, :]  # (B, 1, H, W) - Slope at index 4
        if num_channels >= 6 and self.use_vdvi:
            vdvi = x[:, 5:6, :, :]   # (B, 1, H, W) - VDVI at index 5
        
        # Stream 1: RGB空間特徴
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = rgb_feat.view(batch_size, -1)  # (B, 128)
        
        features_to_combine = [rgb_feat]
        
        # Stream 2: DEM/Slope/VDVI統計量 → MLP（利用可能な場合のみ）
        if self.aux_mlp is not None and (dem is not None or slope is not None or vdvi is not None):
            aux_stats = self.extract_aux_stats(dem, slope, vdvi)
            if aux_stats.size(1) > 0:
                aux_feat = self.aux_mlp(aux_stats)  # (B, 64)
                features_to_combine.append(aux_feat)
        
        # Stream 3: VDVI空間パターン（軽量）（利用可能な場合のみ）
        if self.vdvi_spatial is not None and vdvi is not None:
            vdvi_spatial_feat = self.vdvi_spatial(vdvi)
            vdvi_spatial_feat = vdvi_spatial_feat.view(batch_size, -1)  # (B, 8)
            features_to_combine.append(vdvi_spatial_feat)
        
        # 融合
        combined = torch.cat(features_to_combine, dim=1)
        
        # ゲーティング（各特徴の重要度を調整）
        gate_weights = self.gate(combined)
        gated_features = combined * gate_weights
        
        # 分類
        output = self.classifier(gated_features)
        
        return output


class LandslideModelWithRegularization(nn.Module):
    """
    正則化を強化したバージョン（ResNetの過学習問題対策）
    """
    
    def __init__(self, num_classes=5, image_size=25, dropout=0.4):
        super().__init__()
        
        # 基本モデル
        self.base = HybridModel(
            num_classes=num_classes,
            image_size=image_size,
            dropout=dropout
        )
        
        # Label Smoothing用のクラス数を保持
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.base(x)
    
    @staticmethod
    def get_loss_function(label_smoothing=0.1):
        """Label Smoothing付きCross Entropy"""
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    @staticmethod
    def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
        """AdamW with decoupled weight decay"""
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    
    @staticmethod
    def get_scheduler(optimizer, num_epochs, warmup_epochs=5):
        """Cosine Annealing with Warmup"""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
class MixupAugmentation:
    """
    Mixup/CutMix for better generalization
    ResNetの過学習対策として有効
    """
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def train_step_with_mixup(model, x, y, criterion, optimizer, mixup):
    """Mixup付きの学習ステップ"""
    mixed_x, y_a, y_b, lam = mixup(x, y)
    
    outputs = model(mixed_x)
    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient Clipping（過学習防止）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()



class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) for regularization"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class MLP(nn.Module):
    """MLP block with GELU activation"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# =============================================================================
# Transformer用Attention機構
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with optional attention dropout
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CrossAttention(nn.Module):
    """
    Cross-Attention: Query from one modality, Key/Value from another
    RGB特徴をQueryとし、補助データ特徴をKey/Valueとして情報を統合
    """
    
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, query, context):
        """
        query: (B, N_q, C) - RGB tokens
        context: (B, N_c, C) - Auxiliary tokens
        """
        B, N_q, C = query.shape
        N_c = context.shape[1]
        
        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, N_c, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# =============================================================================
# Transformer Encoder Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Standard Transformer Encoder Block
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 dropout=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=dropout
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, dropout=dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class CrossModalBlock(nn.Module):
    """
    Cross-Modal Transformer Block
    Self-Attention + Cross-Attention + FFN
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 dropout=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=dropout
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # Cross-Attention
        self.norm2 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads // 2, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=dropout
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # FFN
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features=mlp_hidden_dim, dropout=dropout)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x, context):
        # Self-Attention
        x = x + self.drop_path1(self.self_attn(self.norm1(x)))
        # Cross-Attention
        x = x + self.drop_path2(self.cross_attn(self.norm2(x), self.norm_context(context)))
        # FFN
        x = x + self.drop_path3(self.mlp(self.norm3(x)))
        return x


# =============================================================================
# 補助データ（VDVI/DEM）処理モジュール
# =============================================================================

class AuxiliaryStatisticsEncoder(nn.Module):
    """
    VDVI/DEMの統計量を抽出してエンコードするモジュール
    
    MLPがVDVI/DEMで最も効果的だった観察に基づき、
    空間的な処理ではなく統計量ベースの処理を行う
    
    可変チャンネル対応:
    - VDVIのみ、DEMのみ、両方の3パターンに対応
    """
    
    def __init__(self, output_dim=64, dropout=0.1, n_aux_channels=2):
        super().__init__()
        
        self.n_aux_channels = n_aux_channels
        # 統計量: mean, std, min, max, median, q25, q75, range × aux_channels
        self.stats_per_channel = 8
        self.stats_dim = self.stats_per_channel * max(1, n_aux_channels)
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.stats_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
    
    def extract_statistics(self, x):
        """
        x: (B, 1, H, W) - single channel (VDVI or DEM)
        Returns: (B, 8) - statistics
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        
        # サイズが0の場合はゼロテンソルを返す
        if x_flat.size(1) == 0:
            return torch.zeros(B, self.stats_per_channel, device=x.device, dtype=x.dtype)
        
        mean = x_flat.mean(dim=1, keepdim=True)
        std = x_flat.std(dim=1, keepdim=True)
        min_val = x_flat.min(dim=1, keepdim=True)[0]
        max_val = x_flat.max(dim=1, keepdim=True)[0]
        median = x_flat.median(dim=1, keepdim=True)[0]
        q25 = x_flat.quantile(0.25, dim=1, keepdim=True)
        q75 = x_flat.quantile(0.75, dim=1, keepdim=True)
        range_val = max_val - min_val
        
        stats = torch.cat([mean, std, min_val, max_val, median, q25, q75, range_val], dim=1)
        return stats
    
    def forward(self, vdvi, dem, batch_size=None, device=None):
        """
        vdvi: (B, 1, H, W) or None
        dem: (B, 1, H, W) or None
        batch_size: バッチサイズ（vdvi/demがNoneの場合に使用）
        device: デバイス（vdvi/demがNoneの場合に使用）
        Returns: (B, output_dim)
        """
        stats_list = []
        
        if vdvi is not None and vdvi.numel() > 0:
            stats_list.append(self.extract_statistics(vdvi))
        
        if dem is not None and dem.numel() > 0:
            stats_list.append(self.extract_statistics(dem))
        
        if not stats_list:
            # 補助データがない場合はゼロテンソルを返す
            if batch_size is None:
                batch_size = vdvi.size(0) if vdvi is not None else (dem.size(0) if dem is not None else 1)
            if device is None:
                device = vdvi.device if vdvi is not None else (dem.device if dem is not None else 'cpu')
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        combined_stats = torch.cat(stats_list, dim=1)
        
        # stats_dimと合わない場合はパディングまたはトランケート
        if combined_stats.size(1) < self.stats_dim:
            padding = torch.zeros(combined_stats.size(0), self.stats_dim - combined_stats.size(1), 
                                  device=combined_stats.device, dtype=combined_stats.dtype)
            combined_stats = torch.cat([combined_stats, padding], dim=1)
        elif combined_stats.size(1) > self.stats_dim:
            combined_stats = combined_stats[:, :self.stats_dim]
        
        encoded = self.encoder(combined_stats)
        
        return encoded


class LightweightSpatialEncoder(nn.Module):
    """
    補助データの空間情報を軽量に抽出
    
    CNN/ViTでVDVI導入時に精度が悪化した観察から、
    複雑な空間処理は避け、非常に軽量な構造にする
    
    可変チャンネル対応
    """
    
    def __init__(self, in_channels=2, output_dim=32):
        super().__init__()
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        # 非常に浅い構造（1層のみ）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.proj = nn.Linear(16, output_dim)
    
    def forward(self, vdvi, dem, batch_size=None, device=None):
        """
        vdvi: (B, 1, H, W) or None
        dem: (B, 1, H, W) or None
        batch_size: バッチサイズ（vdvi/demがNoneの場合に使用）
        device: デバイス（vdvi/demがNoneの場合に使用）
        Returns: (B, output_dim)
        """
        channels = []
        if vdvi is not None and vdvi.numel() > 0:
            channels.append(vdvi)
        if dem is not None and dem.numel() > 0:
            channels.append(dem)
        
        if not channels:
            # 補助データがない場合はゼロテンソルを返す
            if batch_size is None:
                batch_size = vdvi.size(0) if vdvi is not None else (dem.size(0) if dem is not None else 1)
            if device is None:
                device = vdvi.device if vdvi is not None else (dem.device if dem is not None else 'cpu')
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        x = torch.cat(channels, dim=1)  # (B, n_channels, H, W)
        
        # チャンネル数が期待と異なる場合は調整
        if x.size(1) < self.in_channels:
            # ゼロパディング
            padding = torch.zeros(x.size(0), self.in_channels - x.size(1), 
                                  x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > self.in_channels:
            x = x[:, :self.in_channels, :, :]
        
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x


# =============================================================================
# メインモデル: Transformer版
# =============================================================================

class LandslideTransformerV1(nn.Module):
    """
    斜面崩壊残存植生検出に特化したTransformerモデル（Version 1）
    
    特徴:
    - RGBのみをTransformerで処理
    - VDVI/DEMは統計量としてMLPで処理（最も効果的だった手法）
    - 後段で特徴を融合
    
    設計根拠:
    - ViTでVDVI導入時に精度悪化 → VDVIはTransformerに入れない
    - MLPがVDVI/DEMで最良 → 統計量ベースで処理
    """
    
    def __init__(
        self,
        image_size: int = 25,
        patch_size: int = 5,
        in_channels: int = 5,  # RGB(3) + VDVI(1) + DEM(1)
        num_classes: int = 5,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # ========================================
        # RGB Patch Embedding
        # ========================================
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # ========================================
        # Transformer Encoder (RGB only)
        # ========================================
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # ========================================
        # Auxiliary Data Encoder (VDVI/DEM)
        # ========================================
        self.aux_stats_encoder = AuxiliaryStatisticsEncoder(output_dim=64, dropout=dropout)
        
        # ========================================
        # Feature Fusion & Classification
        # ========================================
        fusion_dim = embed_dim + 64  # RGB features + Aux stats
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, C, H, W) - C can be 3 (RGB), 4 (RGB+VDVI), or 5 (RGB+VDVI+DEM)
        """
        B = x.size(0)
        num_channels = x.size(1)
        
        # Split channels
        rgb = x[:, :3, :, :]
        vdvi = x[:, 3:4, :, :] if num_channels >= 4 else None
        dem = x[:, 4:5, :, :] if num_channels >= 5 else None
        
        # ========================================
        # RGB Processing via Transformer
        # ========================================
        rgb_tokens = self.patch_embed(rgb)  # (B, num_patches, embed_dim)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        rgb_tokens = torch.cat([cls_tokens, rgb_tokens], dim=1)
        rgb_tokens = rgb_tokens + self.pos_embed
        rgb_tokens = self.pos_drop(rgb_tokens)
        
        for block in self.blocks:
            rgb_tokens = block(rgb_tokens)
        
        rgb_tokens = self.norm(rgb_tokens)
        rgb_feat = rgb_tokens[:, 0]  # CLS token
        
        # ========================================
        # Auxiliary Data Processing via MLP
        # ========================================
        aux_feat = self.aux_stats_encoder(vdvi, dem, batch_size=B, device=x.device)
        
        # ========================================
        # Fusion & Classification
        # ========================================
        combined = torch.cat([rgb_feat, aux_feat], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
        return output


class LandslideTransformerV2(nn.Module):
    """
    斜面崩壊残存植生検出に特化したTransformerモデル（Version 2）
    
    特徴:
    - Dual-Stream Architecture
    - RGB Stream: Full Transformer
    - Aux Stream: 軽量な処理 + Cross-Attention for selective integration
    
    設計根拠:
    - Cross-Attentionにより、RGBがAuxから「必要な情報だけ」を選択的に取得
    - ViTでVDVI精度悪化の原因は「全情報を同等に扱った」ことと推測
    """
    
    def __init__(
        self,
        image_size: int = 25,
        patch_size: int = 5,
        in_channels: int = 5,
        num_classes: int = 5,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # ========================================
        # RGB Stream
        # ========================================
        self.rgb_patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.rgb_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # ========================================
        # Auxiliary Stream (lightweight)
        # ========================================
        # 統計量エンコーダ
        self.aux_stats_encoder = AuxiliaryStatisticsEncoder(output_dim=embed_dim, dropout=dropout)
        
        # 軽量な空間エンコーダ（Cross-Attention用のコンテキスト）
        self.aux_spatial_encoder = LightweightSpatialEncoder(in_channels=2, output_dim=embed_dim)
        
        # Aux tokens: [stats_token, spatial_token]
        self.aux_pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        
        # ========================================
        # Transformer Blocks with Cross-Attention
        # ========================================
        self.pos_drop = nn.Dropout(dropout)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        # 前半: RGB only (Self-Attention)
        self.rgb_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=dpr[i]
            )
            for i in range(depth // 2)
        ])
        
        # 後半: Cross-Modal (Self-Attention + Cross-Attention)
        self.cross_modal_blocks = nn.ModuleList([
            CrossModalBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=dpr[i + depth // 2]
            )
            for i in range(depth - depth // 2)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # ========================================
        # Classification Head
        # ========================================
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.rgb_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.rgb_cls_token, std=0.02)
        nn.init.trunc_normal_(self.aux_pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, C, H, W) - C can be 3 (RGB), 4 (RGB+VDVI), or 5 (RGB+VDVI+DEM)
        """
        B = x.size(0)
        num_channels = x.size(1)
        
        # Split channels
        rgb = x[:, :3, :, :]
        vdvi = x[:, 3:4, :, :] if num_channels >= 4 else None
        dem = x[:, 4:5, :, :] if num_channels >= 5 else None
        
        # ========================================
        # RGB Embedding
        # ========================================
        rgb_tokens = self.rgb_patch_embed(rgb)
        cls_tokens = self.rgb_cls_token.expand(B, -1, -1)
        rgb_tokens = torch.cat([cls_tokens, rgb_tokens], dim=1)
        rgb_tokens = rgb_tokens + self.rgb_pos_embed
        rgb_tokens = self.pos_drop(rgb_tokens)
        
        # ========================================
        # Auxiliary Embedding
        # ========================================
        aux_stats = self.aux_stats_encoder(vdvi, dem, batch_size=B, device=x.device).unsqueeze(1)  # (B, 1, embed_dim)
        aux_spatial = self.aux_spatial_encoder(vdvi, dem, batch_size=B, device=x.device).unsqueeze(1)  # (B, 1, embed_dim)
        aux_tokens = torch.cat([aux_stats, aux_spatial], dim=1)  # (B, 2, embed_dim)
        aux_tokens = aux_tokens + self.aux_pos_embed
        
        # ========================================
        # RGB Self-Attention (前半)
        # ========================================
        for block in self.rgb_blocks:
            rgb_tokens = block(rgb_tokens)
        
        # ========================================
        # Cross-Modal Attention (後半)
        # ========================================
        for block in self.cross_modal_blocks:
            rgb_tokens = block(rgb_tokens, aux_tokens)
        
        rgb_tokens = self.norm(rgb_tokens)
        cls_feat = rgb_tokens[:, 0]
        
        # ========================================
        # Classification
        # ========================================
        output = self.classifier(cls_feat)
        
        return output


class LandslideTransformerV3(nn.Module):
    """
    斜面崩壊残存植生検出に特化したTransformerモデル（Version 3）
    
    特徴:
    - Gated Multimodal Fusion
    - 各モダリティの重要度を動的に学習
    - 最も軽量なTransformer構造
    
    設計根拠:
    - ResNetの過学習問題 → パラメータ数を最小化
    - MLPのVDVI/DEM処理能力 → ゲーティングで重要度を制御
    """
    
    def __init__(
        self,
        image_size: int = 25,
        patch_size: int = 5,
        in_channels: int = 5,
        num_classes: int = 5,
        embed_dim: int = 96,  # 小さめ
        depth: int = 3,  # 浅め
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,  # 強めの正則化
        attn_drop: float = 0.1,
        drop_path: float = 0.15,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # ========================================
        # RGB Transformer (軽量)
        # ========================================
        self.rgb_patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.rgb_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.rgb_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        self.rgb_norm = nn.LayerNorm(embed_dim)
        
        # ========================================
        # Auxiliary MLP (統計量ベース)
        # ========================================
        self.aux_encoder = AuxiliaryStatisticsEncoder(output_dim=embed_dim, dropout=dropout)
        
        # ========================================
        # Gated Fusion
        # ========================================
        # 各モダリティの重要度を学習するゲート
        self.rgb_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.aux_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # ========================================
        # Classification Head
        # ========================================
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.rgb_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.rgb_cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, C, H, W) - C can be 3 (RGB), 4 (RGB+VDVI), or 5 (RGB+VDVI+DEM)
        """
        B = x.size(0)
        num_channels = x.size(1)
        
        # Split channels
        rgb = x[:, :3, :, :]
        vdvi = x[:, 3:4, :, :] if num_channels >= 4 else None
        dem = x[:, 4:5, :, :] if num_channels >= 5 else None
        
        # ========================================
        # RGB Processing
        # ========================================
        rgb_tokens = self.rgb_patch_embed(rgb)
        cls_tokens = self.rgb_cls_token.expand(B, -1, -1)
        rgb_tokens = torch.cat([cls_tokens, rgb_tokens], dim=1)
        rgb_tokens = rgb_tokens + self.rgb_pos_embed
        rgb_tokens = self.pos_drop(rgb_tokens)
        
        for block in self.rgb_blocks:
            rgb_tokens = block(rgb_tokens)
        
        rgb_tokens = self.rgb_norm(rgb_tokens)
        rgb_feat = rgb_tokens[:, 0]  # (B, embed_dim)
        
        # ========================================
        # Auxiliary Processing
        # ========================================
        aux_feat = self.aux_encoder(vdvi, dem, batch_size=B, device=x.device)  # (B, embed_dim)
        
        # ========================================
        # Gated Fusion
        # ========================================
        combined = torch.cat([rgb_feat, aux_feat], dim=1)  # (B, 2*embed_dim)
        
        rgb_weight = self.rgb_gate(combined)  # (B, embed_dim)
        aux_weight = self.aux_gate(combined)  # (B, embed_dim)
        
        # Weighted combination
        fused = rgb_feat * rgb_weight + aux_feat * aux_weight
        
        # ========================================
        # Classification
        # ========================================
        output = self.classifier(fused)
        
        return output


# =============================================================================
# Hybrid Model: CNN + Transformer
# =============================================================================

class LandslideHybridTransformer(nn.Module):
    """
    CNN + Transformer ハイブリッドモデル
    
    特徴:
    - RGB: 浅いCNNで局所特徴抽出 → Transformerで大域的関係性
    - VDVI/DEM: 統計量ベースMLP
    
    設計根拠:
    - ResNetの過学習 → CNNは2層に制限
    - Transformerの表現力 → 局所特徴抽出後に適用
    """
    
    def __init__(
        self,
        image_size: int = 25,
        in_channels: int = 5,
        num_classes: int = 5,
        embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.15,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        
        # ========================================
        # RGB CNN Feature Extractor (浅い)
        # ========================================
        self.cnn_stem = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 25 -> 12
            
            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),  # 12 -> 6
        )
        
        self.cnn_proj = nn.Conv2d(128, embed_dim, kernel_size=1)
        
        # Feature map size after CNN: 6x6 = 36 patches
        self.num_patches = 36
        
        # ========================================
        # Transformer
        # ========================================
        self.flatten = Rearrange('b c h w -> b (h w) c')
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # ========================================
        # Auxiliary Encoder
        # ========================================
        self.aux_encoder = AuxiliaryStatisticsEncoder(output_dim=64, dropout=dropout)
        
        # ========================================
        # Fusion & Classification
        # ========================================
        fusion_dim = embed_dim + 64
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, C, H, W) - C can be 3 (RGB), 4 (RGB+VDVI), or 5 (RGB+VDVI+DEM)
        """
        B = x.size(0)
        num_channels = x.size(1)
        
        # Split channels
        rgb = x[:, :3, :, :]
        vdvi = x[:, 3:4, :, :] if num_channels >= 4 else None
        dem = x[:, 4:5, :, :] if num_channels >= 5 else None
        
        # ========================================
        # CNN Feature Extraction
        # ========================================
        cnn_feat = self.cnn_stem(rgb)
        cnn_feat = self.cnn_proj(cnn_feat)
        
        # ========================================
        # Transformer
        # ========================================
        tokens = self.flatten(cnn_feat)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        
        for block in self.blocks:
            tokens = block(tokens)
        
        tokens = self.norm(tokens)
        rgb_feat = tokens[:, 0]
        
        # ========================================
        # Auxiliary Processing
        # ========================================
        aux_feat = self.aux_encoder(vdvi, dem, batch_size=B, device=x.device)
        
        # ========================================
        # Fusion & Classification
        # ========================================
        combined = torch.cat([rgb_feat, aux_feat], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        
        return output


# =============================================================================
# 中間値予測に特化したモデル群
# Intermediate Value Prediction Models
# =============================================================================

class OrdinalRegressionHead(nn.Module):
    """
    順序回帰ヘッド: クラス間の順序関係を明示的にモデル化
    
    累積確率をモデル化し、隣接クラスとの関係を学習
    P(Y > k) を各閾値kについて予測
    """
    
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.n_thresholds = n_classes - 1
        
        # 各閾値に対する累積確率を予測
        self.cumulative_logits = nn.Linear(in_features, self.n_thresholds)
        
        # 閾値は学習可能だが順序を保持
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, self.n_thresholds))
    
    def forward(self, x):
        """
        Returns:
            class_probs: (B, n_classes) - 各クラスの確率
            cumulative_probs: (B, n_thresholds) - 累積確率
        """
        # 累積ロジット
        logits = self.cumulative_logits(x)
        
        # 順序制約付き閾値（単調増加を保証）
        sorted_thresholds = torch.cumsum(F.softplus(self.thresholds), dim=0)
        
        # 累積確率 P(Y > k)
        cumulative_probs = torch.sigmoid(logits - sorted_thresholds)
        
        # クラス確率に変換
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        ones = torch.ones(x.size(0), 1, device=x.device)
        zeros = torch.zeros(x.size(0), 1, device=x.device)
        
        # P(Y > -1) = 1, P(Y > n_classes-1) = 0
        extended_cumulative = torch.cat([ones, cumulative_probs, zeros], dim=1)
        class_probs = extended_cumulative[:, :-1] - extended_cumulative[:, 1:]
        
        # 数値安定性のためクリップ
        class_probs = torch.clamp(class_probs, min=1e-7, max=1.0)
        class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
        
        return class_probs, cumulative_probs


class ContinuousRegressionHead(nn.Module):
    """
    連続値回帰ヘッド: 0-1の連続値を直接予測
    
    Beta分布のパラメータを予測し、より柔軟な分布をモデル化
    """
    
    def __init__(self, in_features):
        super().__init__()
        
        # Beta分布のα, βパラメータを予測
        self.alpha_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # α > 0
        )
        self.beta_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # β > 0
        )
        
        # 直接の平均予測
        self.mean_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Returns:
            mean: (B, 1) - 予測平均
            alpha: (B, 1) - Beta分布のα
            beta: (B, 1) - Beta分布のβ
        """
        alpha = self.alpha_head(x) + 1.0  # α >= 1 for numerical stability
        beta = self.beta_head(x) + 1.0
        mean = self.mean_head(x)
        
        return mean, alpha, beta


class SoftLabelHead(nn.Module):
    """
    ソフトラベル予測ヘッド: 隣接クラスへの確率分布を予測
    
    中間値の場合、複数のクラスに確率を分散させる
    """
    
    def __init__(self, in_features, n_classes, temperature=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.temperature = temperature
        
        # クラスロジットと不確実性を同時に予測
        self.logits_head = nn.Linear(in_features, n_classes)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Returns:
            probs: (B, n_classes) - クラス確率
            uncertainty: (B, 1) - 予測の不確実性（0-1）
        """
        logits = self.logits_head(x)
        uncertainty = self.uncertainty_head(x)
        
        # 不確実性が高いほど分布を平滑化
        effective_temp = self.temperature + uncertainty * 2.0
        probs = F.softmax(logits / effective_temp, dim=1)
        
        return probs, uncertainty


class BoundaryAwareEncoder(nn.Module):
    """
    境界認識エンコーダ: クラス境界付近の特徴を強調
    
    中間値は境界付近に位置するため、境界特徴を明示的に抽出
    """
    
    def __init__(self, in_dim, out_dim, n_boundaries=4):
        super().__init__()
        self.n_boundaries = n_boundaries
        
        # 各境界に対する特徴抽出
        self.boundary_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim // n_boundaries),
                nn.LayerNorm(out_dim // n_boundaries),
                nn.GELU()
            )
            for _ in range(n_boundaries)
        ])
        
        # 境界位置（0.25, 0.5, 0.75など）への近さを学習
        self.boundary_positions = nn.Parameter(
            torch.linspace(0.25, 0.75, n_boundaries)
        )
        
        # 境界特徴の統合
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        """
        x: (B, in_dim) - 入力特徴
        Returns: (B, out_dim) - 境界認識特徴
        """
        boundary_features = []
        for proj in self.boundary_projections:
            boundary_features.append(proj(x))
        
        combined = torch.cat(boundary_features, dim=1)
        return self.fusion(combined)


class IntermediateValueTransformer(nn.Module):
    """
    中間値予測に特化したTransformerモデル
    
    マルチタスク学習:
    1. 順序回帰（クラス間の順序関係）
    2. 連続値回帰（直接的な値予測）
    3. ソフトラベル分類（確率分布予測）
    4. 境界認識（クラス境界の検出）
    """
    
    def __init__(
        self,
        image_size: int = 25,
        patch_size: int = 5,
        in_channels: int = 5,
        n_classes: int = 5,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        
        # RGB Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        # Position & CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Auxiliary Encoder (VDVI/DEM) - チャンネル数に応じて調整
        aux_input_dim = 4 * (in_channels - 3) if in_channels > 3 else 4
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
        )
        
        # Feature Fusion
        fusion_dim = embed_dim + embed_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Boundary-Aware Encoder
        self.boundary_encoder = BoundaryAwareEncoder(
            embed_dim, embed_dim, n_boundaries=n_classes - 1
        )
        
        # Multi-Task Heads
        self.ordinal_head = OrdinalRegressionHead(embed_dim * 2, n_classes)
        self.regression_head = ContinuousRegressionHead(embed_dim * 2)
        self.soft_label_head = SoftLabelHead(embed_dim * 2, n_classes)
        
        # Final Classification (ensemble of heads)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def extract_aux_stats(self, vdvi, dem):
        """補助データから統計量を抽出"""
        stats = []
        B = vdvi.size(0) if vdvi is not None else (dem.size(0) if dem is not None else 1)
        device = vdvi.device if vdvi is not None else (dem.device if dem is not None else 'cpu')
        
        for data in [vdvi, dem]:
            if data is not None and data.numel() > 0:
                flat = data.view(B, -1)
                stats.extend([
                    flat.mean(dim=1, keepdim=True),
                    flat.std(dim=1, keepdim=True),
                    flat.min(dim=1, keepdim=True)[0],
                    flat.max(dim=1, keepdim=True)[0],
                ])
        
        if not stats:
            return torch.zeros(B, 4, device=device)
        
        combined = torch.cat(stats, dim=1)
        return combined
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns:
            class_probs: (B, n_classes) - 最終クラス確率
            outputs: dict - 各ヘッドの出力
        """
        B = x.size(0)
        C = x.size(1)
        
        # Channel split
        rgb = x[:, :3, :, :]
        vdvi = x[:, 3:4, :, :] if C >= 4 else None
        dem = x[:, 4:5, :, :] if C >= 5 else None
        
        # RGB → Transformer
        tokens = self.patch_embed(rgb)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        rgb_feat = tokens[:, 0]
        
        # Auxiliary features
        aux_stats = self.extract_aux_stats(vdvi, dem)
        # パディング
        expected_aux_dim = 4 * (self.in_channels - 3) if self.in_channels > 3 else 4
        if aux_stats.size(1) < expected_aux_dim:
            padding = torch.zeros(B, expected_aux_dim - aux_stats.size(1), device=x.device)
            aux_stats = torch.cat([aux_stats, padding], dim=1)
        aux_feat = self.aux_encoder(aux_stats[:, :expected_aux_dim])
        
        # Fusion
        combined = torch.cat([rgb_feat, aux_feat], dim=1)
        base_feat = self.fusion(combined)
        
        # Boundary-aware features
        boundary_feat = self.boundary_encoder(base_feat)
        
        # Combined features for multi-task heads
        final_feat = torch.cat([base_feat, boundary_feat], dim=1)
        
        # Multi-task outputs
        ordinal_probs, cumulative_probs = self.ordinal_head(final_feat)
        reg_mean, reg_alpha, reg_beta = self.regression_head(final_feat)
        soft_probs, uncertainty = self.soft_label_head(final_feat)
        
        # Ensemble: weighted combination of probabilities
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # 回帰出力をクラス確率に変換
        reg_probs = self._regression_to_probs(reg_mean, self.n_classes, x.device)
        
        # 加重平均
        class_probs = (
            weights[0] * ordinal_probs +
            weights[1] * reg_probs +
            weights[2] * soft_probs
        )
        
        outputs = {
            'ordinal_probs': ordinal_probs,
            'cumulative_probs': cumulative_probs,
            'regression_mean': reg_mean,
            'regression_alpha': reg_alpha,
            'regression_beta': reg_beta,
            'soft_probs': soft_probs,
            'uncertainty': uncertainty,
            'ensemble_weights': weights,
        }
        
        return class_probs, outputs
    
    def _regression_to_probs(self, mean, n_classes, device):
        """連続値予測をクラス確率に変換"""
        # クラス中心
        class_centers = torch.linspace(0, 1, n_classes, device=device)
        
        # 各クラス中心との距離に基づく確率
        distances = (mean - class_centers.view(1, -1)).abs()
        probs = F.softmax(-distances * 5.0, dim=1)  # 温度パラメータ
        
        return probs


class IntermediateFocusedCNN(nn.Module):
    """
    中間値に焦点を当てたCNNモデル
    
    特徴:
    - マルチスケール特徴抽出
    - 境界検出モジュール
    - 順序回帰ヘッド
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        n_classes: int = 5,
        base_channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.in_channels = in_channels
        
        # RGB処理
        self.rgb_encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        rgb_dim = base_channels * 4
        
        # マルチスケール特徴
        self.multiscale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            )
            for k in [3, 5, 7]
        ])
        multiscale_dim = 16 * 3
        
        # 補助データ処理 - チャンネル数に応じて調整
        aux_input_dim = 4 * (in_channels - 3) if in_channels > 3 else 4
        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        aux_dim = 64
        
        # 特徴統合
        total_dim = rgb_dim + multiscale_dim + aux_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        # マルチタスクヘッド
        self.ordinal_head = OrdinalRegressionHead(128, n_classes)
        self.regression_head = ContinuousRegressionHead(128)
        self.classifier = nn.Linear(128, n_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def extract_aux_stats(self, vdvi, dem):
        stats = []
        B = vdvi.size(0) if vdvi is not None else (dem.size(0) if dem is not None else 1)
        device = vdvi.device if vdvi is not None else (dem.device if dem is not None else 'cpu')
        
        for data in [vdvi, dem]:
            if data is not None and data.numel() > 0:
                flat = data.view(B, -1)
                stats.extend([
                    flat.mean(dim=1, keepdim=True),
                    flat.std(dim=1, keepdim=True),
                    flat.min(dim=1, keepdim=True)[0],
                    flat.max(dim=1, keepdim=True)[0],
                ])
        
        if not stats:
            return torch.zeros(B, 4, device=device)
        
        combined = torch.cat(stats, dim=1)
        return combined
    
    def forward(self, x):
        B = x.size(0)
        C = x.size(1)
        
        rgb = x[:, :3, :, :]
        vdvi = x[:, 3:4, :, :] if C >= 4 else None
        dem = x[:, 4:5, :, :] if C >= 5 else None
        
        # RGB特徴
        rgb_feat = self.rgb_encoder(rgb).view(B, -1)
        
        # マルチスケール特徴
        ms_feats = [branch(rgb).view(B, -1) for branch in self.multiscale]
        ms_feat = torch.cat(ms_feats, dim=1)
        
        # 補助特徴
        aux_stats = self.extract_aux_stats(vdvi, dem)
        # パディング
        expected_aux_dim = 4 * (self.in_channels - 3) if self.in_channels > 3 else 4
        if aux_stats.size(1) < expected_aux_dim:
            padding = torch.zeros(B, expected_aux_dim - aux_stats.size(1), device=x.device)
            aux_stats = torch.cat([aux_stats, padding], dim=1)
        aux_feat = self.aux_encoder(aux_stats[:, :expected_aux_dim])
        
        # 統合
        combined = torch.cat([rgb_feat, ms_feat, aux_feat], dim=1)
        features = self.fusion(combined)
        
        # マルチタスク出力
        ordinal_probs, cumulative_probs = self.ordinal_head(features)
        reg_mean, reg_alpha, reg_beta = self.regression_head(features)
        class_logits = self.classifier(features)
        class_probs = F.softmax(class_logits, dim=1)
        
        # アンサンブル
        reg_probs = self._reg_to_probs(reg_mean, x.device)
        final_probs = 0.4 * ordinal_probs + 0.3 * class_probs + 0.3 * reg_probs
        
        outputs = {
            'ordinal_probs': ordinal_probs,
            'cumulative_probs': cumulative_probs,
            'regression_mean': reg_mean,
            'regression_alpha': reg_alpha,
            'regression_beta': reg_beta,
            'class_logits': class_logits,
        }
        
        return final_probs, outputs
    
    def _reg_to_probs(self, mean, device):
        class_centers = torch.linspace(0, 1, self.n_classes, device=device)
        distances = (mean - class_centers.view(1, -1)).abs()
        probs = F.softmax(-distances * 5.0, dim=1)
        return probs


class MultiTaskLoss(nn.Module):
    """
    マルチタスク損失関数
    
    1. 順序回帰損失（累積確率のBCE）
    2. 連続値回帰損失（L1 + Beta NLL）
    3. ソフトラベル損失（KL divergence with label smoothing）
    4. 境界認識損失（境界付近サンプルの重み付け）
    """
    
    def __init__(self, n_classes=5, thresholds=None):
        super().__init__()
        self.n_classes = n_classes
        
        if thresholds is None:
            self.thresholds = torch.linspace(0, 1, n_classes + 1)[1:-1]
        else:
            self.thresholds = torch.tensor(thresholds)
        
        # 損失の重み（学習可能）
        self.loss_weights = nn.Parameter(torch.ones(4))
    
    def forward(self, outputs, class_probs, true_labels, raw_labels):
        """
        outputs: モデルの全出力
        class_probs: 最終クラス確率
        true_labels: クラスラベル (B,)
        raw_labels: 連続値ラベル (B,)
        """
        device = class_probs.device
        B = class_probs.size(0)
        
        # 1. 分類損失（交差エントロピー）
        ce_loss = F.cross_entropy(
            torch.log(class_probs + 1e-7),
            true_labels
        )
        
        # 2. 順序回帰損失
        cumulative_probs = outputs['cumulative_probs']
        thresholds = self.thresholds.to(device)
        
        # 真の累積ラベル: P(Y > k) = 1 if y > threshold[k]
        true_cumulative = (raw_labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
        ordinal_loss = F.binary_cross_entropy(cumulative_probs, true_cumulative)
        
        # 3. 回帰損失
        reg_mean = outputs['regression_mean'].squeeze()
        reg_loss = F.l1_loss(reg_mean, raw_labels)
        
        # 4. ソフトラベル損失（中間値に対して隣接クラスへの確率を許容）
        soft_probs = outputs.get('soft_probs')
        if soft_probs is not None:
            soft_targets = self._generate_soft_targets(raw_labels, self.n_classes, device)
            soft_loss = F.kl_div(
                torch.log(soft_probs + 1e-7),
                soft_targets,
                reduction='batchmean'
            )
        else:
            soft_loss = torch.tensor(0.0, device=device)
        
        # 5. 境界認識損失（境界付近のサンプルを重視）
        boundary_weights = self._compute_boundary_weights(raw_labels, device)
        weighted_ce_loss = (F.cross_entropy(
            torch.log(class_probs + 1e-7),
            true_labels,
            reduction='none'
        ) * boundary_weights).mean()
        
        # 重み付き合計
        weights = F.softmax(self.loss_weights, dim=0)
        total_loss = (
            weights[0] * ce_loss +
            weights[1] * ordinal_loss +
            weights[2] * reg_loss +
            weights[3] * soft_loss +
            0.5 * weighted_ce_loss  # 追加のボーナス項
        )
        
        loss_dict = {
            'total': total_loss,
            'ce': ce_loss,
            'ordinal': ordinal_loss,
            'regression': reg_loss,
            'soft_label': soft_loss,
            'boundary_weighted': weighted_ce_loss,
        }
        
        return total_loss, loss_dict
    
    def _generate_soft_targets(self, raw_labels, n_classes, device):
        """連続値ラベルからソフトターゲットを生成"""
        B = raw_labels.size(0)
        
        # クラス境界
        boundaries = torch.linspace(0, 1, n_classes + 1, device=device)
        class_centers = (boundaries[:-1] + boundaries[1:]) / 2
        
        # 各クラス中心との距離
        distances = (raw_labels.unsqueeze(1) - class_centers.unsqueeze(0)).abs()
        
        # 距離に基づくソフトターゲット（近いほど高確率）
        sigma = 0.15
        soft_targets = torch.exp(-distances**2 / (2 * sigma**2))
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        
        return soft_targets
    
    def _compute_boundary_weights(self, raw_labels, device):
        """境界付近のサンプルに高い重みを付与"""
        thresholds = self.thresholds.to(device)
        
        # 各閾値との最小距離
        distances = (raw_labels.unsqueeze(1) - thresholds.unsqueeze(0)).abs()
        min_distance = distances.min(dim=1)[0]
        
        # 距離が小さいほど重み大（境界付近）
        weights = 1.0 + torch.exp(-min_distance * 10)
        
        return weights


class CORALLoss(nn.Module):
    """
    CORAL損失関数
    
    各閾値での二値分類損失の合計 + 回帰損失
    """
    
    def __init__(self, n_classes=5, thresholds=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_thresholds = n_classes - 1
        
        if thresholds is None:
            self.thresholds = torch.linspace(0, 1, n_classes + 1)[1:-1]
        else:
            self.thresholds = torch.tensor(thresholds)
    
    def forward(self, outputs, class_probs, true_labels, raw_labels):
        device = class_probs.device
        thresholds = self.thresholds.to(device)
        
        cumulative_logits = outputs['cumulative_logits']
        
        # 真の累積ラベル
        true_cumulative = (raw_labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
        
        # 二値交差エントロピー（各閾値）
        coral_loss = F.binary_cross_entropy_with_logits(
            cumulative_logits, true_cumulative
        )
        
        # 分類損失
        ce_loss = F.cross_entropy(torch.log(class_probs + 1e-7), true_labels)
        
        # 回帰損失
        reg_mean = outputs['regression_mean'].squeeze()
        reg_loss = F.l1_loss(reg_mean, raw_labels)
        
        total_loss = coral_loss + 0.5 * ce_loss + 0.3 * reg_loss
        
        loss_dict = {
            'total': total_loss,
            'coral': coral_loss,
            'ce': ce_loss,
            'regression': reg_loss,
        }
        
        return total_loss, loss_dict


# 中間値予測モデル用の定数
INTERMEDIATE_MODEL_NAMES = [
    'IntermediateValueTransformer',
    'IntermediateFocusedCNN'
]


def is_intermediate_model(model_name: str) -> bool:
    """モデルが中間値予測モデルかどうかを判定"""
    return model_name in INTERMEDIATE_MODEL_NAMES


def get_loss_fn_for_model(model_name: str, n_classes: int = 5, thresholds=None):
    """モデルに対応する損失関数を取得"""
    if is_intermediate_model(model_name):
        return MultiTaskLoss(n_classes=n_classes, thresholds=thresholds)
    else:
        return None  # 通常のCrossEntropyを使用


# =============================================================================
# 学習用ユーティリティ
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # Smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-true_dist * log_preds).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = torch.tensor(self.alpha, device=pred.device)[target]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MixupCutmix:
    """Mixup and Cutmix augmentation for regularization"""
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, x, y):
        if np.random.rand() > self.prob:
            return x, y, y, 1.0
        
        if np.random.rand() > 0.5 and self.cutmix_alpha > 0:
            return self._cutmix(x, y)
        else:
            return self._mixup(x, y)
    
    def _mixup(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def _cutmix(self, x, y):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """
    Cosine learning rate schedule with warmup
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# モデル選択ヘルパー
# =============================================================================

def create_landslide_model(
    model_name: str,
    image_size: int = 25,
    patch_size: int = 5,
    in_channels: int = 5,
    num_classes: int = 5,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    モデル名からモデルインスタンスを作成
    
    Available models:
    - 'TransformerV1': Basic Transformer (RGB only) + MLP (Aux)
    - 'TransformerV2': Cross-Modal Transformer
    - 'TransformerV3': Gated Multimodal Transformer (lightweight)
    - 'HybridTransformer': CNN + Transformer Hybrid
    """
    
    model_dict = {
        'TransformerV1': LandslideTransformerV1,
        'TransformerV2': LandslideTransformerV2,
        'TransformerV3': LandslideTransformerV3,
        'HybridTransformer': LandslideHybridTransformer,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    
    model_cls = model_dict[model_name]
    
    # Build kwargs based on model requirements
    model_kwargs = {
        'image_size': image_size,
        'num_classes': num_classes,
        'dropout': dropout,
    }
    
    if model_name != 'HybridTransformer':
        model_kwargs['patch_size'] = patch_size
        model_kwargs['in_channels'] = in_channels
    else:
        model_kwargs['in_channels'] = in_channels
    
    # Add any additional kwargs
    model_kwargs.update(kwargs)
    
    return model_cls(**model_kwargs)


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 5, 25, 25)) -> dict:
    """
    モデルの概要情報を取得
    """
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Forward pass to check output shape
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'output_shape': tuple(output.shape),
        'input_shape': input_size,
    }

def plot_training_curves_enhanced(train_losses, val_losses, train_f1s, val_f1s, 
                                 train_accs, val_accs, save_path, fold_num=None):
    """
    改良版の学習曲線を可視化して保存する関数
    
    Parameters:
    - train_losses, val_losses: 訓練/検証損失のリスト
    - train_f1s, val_f1s: 訓練/検証F1スコアのリスト
    - train_accs, val_accs: 訓練/検証精度のリスト
    - save_path: 保存先パス
    - fold_num: Fold番号
    """
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('black')
    
    epochs = range(1, len(train_losses) + 1)
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors=palette['cream'])
        for spine in ax.spines.values():
            spine.set_color(palette['cream'])
        ax.grid(True, alpha=0.3, color=palette['mist'])
    
    # Loss plot
    ax = axes[0]
    ax.plot(epochs, train_losses, color=palette['teal'], label='Training Loss', linewidth=2, alpha=0.9)
    ax.plot(epochs, val_losses, color=palette['magenta'], label='Validation Loss', linewidth=2, alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=11, color=palette['cream'])
    ax.set_ylabel('Loss', fontsize=11, color=palette['cream'])
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold', color=palette['cream'])
    ax.legend(loc='upper right')
    ax.set_xlim(1, len(epochs))
    
    # 最小値を強調
    min_val_loss_idx = np.argmin(val_losses)
    ax.plot(min_val_loss_idx + 1, val_losses[min_val_loss_idx], marker='o', color=palette['ice'], markersize=8)
    ax.annotate(f'Min: {val_losses[min_val_loss_idx]:.4f}', 
                xy=(min_val_loss_idx + 1, val_losses[min_val_loss_idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=9, color=palette['cream'])
    
    # F1 Score plot
    ax = axes[1]
    ax.plot(epochs, train_f1s, color=palette['teal'], label='Training F1', linewidth=2, alpha=0.9)
    ax.plot(epochs, val_f1s, color=palette['magenta'], label='Validation F1', linewidth=2, alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=11, color=palette['cream'])
    ax.set_ylabel('F1 Score (Macro)', fontsize=11, color=palette['cream'])
    ax.set_title('Training and Validation F1 Score', fontsize=12, fontweight='bold', color=palette['cream'])
    ax.legend(loc='lower right')
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1)
    
    # 最大値を強調
    max_val_f1_idx = np.argmax(val_f1s)
    ax.plot(max_val_f1_idx + 1, val_f1s[max_val_f1_idx], marker='o', color=palette['ice'], markersize=8)
    ax.annotate(f'Max: {val_f1s[max_val_f1_idx]:.4f}', 
                xy=(max_val_f1_idx + 1, val_f1s[max_val_f1_idx]),
                xytext=(5, -15), textcoords='offset points', fontsize=9, color=palette['cream'])
    
    # Accuracy plot
    ax = axes[2]
    ax.plot(epochs, train_accs, color=palette['teal'], label='Training Accuracy', linewidth=2, alpha=0.9)
    ax.plot(epochs, val_accs, color=palette['magenta'], label='Validation Accuracy', linewidth=2, alpha=0.9)
    ax.set_xlabel('Epoch', fontsize=11, color=palette['cream'])
    ax.set_ylabel('Accuracy', fontsize=11, color=palette['cream'])
    ax.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold', color=palette['cream'])
    ax.legend(loc='lower right')
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1)
    
    # 最大値を強調
    max_val_acc_idx = np.argmax(val_accs)
    ax.plot(max_val_acc_idx + 1, val_accs[max_val_acc_idx], marker='o', color=palette['ice'], markersize=8)
    ax.annotate(f'Max: {val_accs[max_val_acc_idx]:.4f}', 
                xy=(max_val_acc_idx + 1, val_accs[max_val_acc_idx]),
                xytext=(5, -15), textcoords='offset points', fontsize=9, color=palette['cream'])
    
    # Combined metrics plot (normalized)
    ax = axes[3]
    
    # 損失を正規化（0-1範囲）
    norm_train_loss = (np.array(train_losses) - np.min(train_losses)) / (np.max(train_losses) - np.min(train_losses))
    norm_val_loss = (np.array(val_losses) - np.min(val_losses)) / (np.max(val_losses) - np.min(val_losses))
    
    ax.plot(epochs, 1 - norm_train_loss, linestyle='--', color=palette['teal'], label='Train (1-Loss_norm)', linewidth=1.5, alpha=0.8)
    ax.plot(epochs, 1 - norm_val_loss, linestyle='--', color=palette['magenta'], label='Val (1-Loss_norm)', linewidth=1.5, alpha=0.8)
    ax.plot(epochs, train_f1s, color=palette['ice'], label='Train F1', linewidth=2, alpha=0.9)
    ax.plot(epochs, val_f1s, color=palette['mist'], label='Val F1', linewidth=2, alpha=0.9)
    ax.plot(epochs, train_accs, linestyle=':', color=palette['teal'], label='Train Acc', linewidth=1.5, alpha=0.8)
    ax.plot(epochs, val_accs, linestyle=':', color=palette['magenta'], label='Val Acc', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=11, color=palette['cream'])
    ax.set_ylabel('Score', fontsize=11, color=palette['cream'])
    ax.set_title('All Metrics (Normalized)', fontsize=12, fontweight='bold', color=palette['cream'])
    ax.legend(loc='center right', fontsize=9, ncol=2)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1)
    
    if fold_num is not None:
        fig.suptitle(f'Fold {fold_num} Training Curves', fontsize=14, fontweight='bold', y=1.02, color=palette['cream'])
    else:
        fig.suptitle('Training Curves', fontsize=14, fontweight='bold', y=1.02, color=palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 最終エポックの値も返す
    return {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_f1': train_f1s[-1],
        'final_val_f1': val_f1s[-1],
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_val_acc': val_accs[-1] if val_accs else 0,
        'best_val_f1': max(val_f1s),
        'best_val_acc': max(val_accs) if val_accs else 0,
        'best_epoch': val_f1s.index(max(val_f1s)) + 1
    }


def plot_cv_scores_enhanced(fold_results, save_dir, model_name, class_names=None):
    """
    改良版のクロスバリデーション結果の可視化
    """
    n_folds = len(fold_results)
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    # フォールドごとの色を生成（フォールド数に応じた色）
    fold_cmap = plt.cm.get_cmap('tab10' if n_folds <= 10 else 'tab20')
    fold_colors = [fold_cmap(i / n_folds) for i in range(n_folds)]
    
    # データの準備
    metrics_data = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1_macro': [],
        'precision_macro': [],
        'recall_macro': []
    }
    
    for result in fold_results:
        for metric in metrics_data.keys():
            metrics_data[metric].append(result['metrics'][metric])
    
    # 1. 各メトリクスの箱ひげ図
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('black')
    axes = axes.flatten()
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors=palette['cream'])
        for spine in ax.spines.values():
            spine.set_color(palette['cream'])
        ax.grid(True, alpha=0.3, axis='y', color=palette['mist'])
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx]
        
        # 箱ひげ図
        bp = ax.boxplot(
            [values],
            labels=[metric_name.replace('_', ' ').title()],
            patch_artist=True,
            widths=0.6,
            boxprops=dict(color=palette['cream']),
            medianprops=dict(color=palette['magenta']),
            whiskerprops=dict(color=palette['cream']),
            capprops=dict(color=palette['cream'])
        )
        bp['boxes'][0].set_facecolor(palette['deep_teal'])
        bp['boxes'][0].set_alpha(0.6)
        
        # 個々のポイントをフォールドごとの色で表示
        x = np.ones(len(values)) + np.random.normal(0, 0.04, len(values))
        for i, (xi, vi) in enumerate(zip(x, values)):
            ax.scatter(xi, vi, alpha=0.9, s=80, color=fold_colors[i], 
                      zorder=3, edgecolors='white', linewidths=0.5,
                      label=f'Fold {i+1}' if idx == 0 else None)
        
        # 平均値のライン
        ax.axhline(
            y=np.mean(values),
            color=palette['ice'],
            linestyle='--',
            label=f'Mean: {np.mean(values):.4f}' if idx == 0 else f'{np.mean(values):.4f}',
            alpha=0.7
        )
        
        ax.set_ylabel('Score', fontsize=10, color=palette['cream'])
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=11, fontweight='bold', color=palette['cream'])
        ax.set_ylim([max(0, min(values) - 0.05), min(1, max(values) + 0.05)])
    
    # 最後のサブプロットに凡例とテーブルを表示
    ax = axes[5]
    ax.axis('off')
    ax.set_facecolor('black')
    
    # フォールドの凡例を追加
    legend_elements = [plt.scatter([], [], c=[fold_colors[i]], s=80, 
                                   edgecolors='white', linewidths=0.5,
                                   label=f'Fold {i+1}') for i in range(n_folds)]
    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
                      title='Fold Colors', title_fontsize=11)
    legend.get_frame().set_facecolor(palette['deep_teal'])
    legend.get_frame().set_edgecolor(palette['mist'])
    legend.get_title().set_color(palette['cream'])
    for text in legend.get_texts():
        text.set_color(palette['cream'])
    
    # 統計テーブル（位置調整）
    table_data = []
    for metric_name, values in metrics_data.items():
        table_data.append([
            metric_name.replace('_', ' ').title(),
            f'{np.mean(values):.4f}',
            f'{np.std(values):.4f}',
            f'{np.min(values):.4f}',
            f'{np.max(values):.4f}'
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
        cellLoc='center',
        loc='lower center',
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)
    
    # ヘッダーの色設定
    for i in range(5):
        table[(0, i)].set_facecolor(palette['deep_teal'])
        table[(0, i)].set_text_props(weight='bold', color=palette['cream'])
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            continue
        cell.set_facecolor('#000000')
        cell.set_text_props(color=palette['cream'])
        cell.set_edgecolor(palette['mist'])
    
    plt.suptitle(f'{model_name} - Cross-Validation Metrics Summary', 
                fontsize=14, fontweight='bold', y=1.02, color=palette['cream'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_metrics_summary.png'), dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 2. 各foldのメトリクス推移（フォールドごとの色付き）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('black')
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors=palette['cream'])
        for spine in ax.spines.values():
            spine.set_color(palette['cream'])
        ax.grid(True, alpha=0.3, color=palette['mist'])
    
    folds = list(range(1, n_folds + 1))
    
    # F1とAccuracyの推移（各foldを色分け）
    ax = axes[0]
    # 折れ線グラフ（全体の傾向）
    ax.plot(folds, metrics_data['f1_macro'], '-', alpha=0.5, linewidth=1.5, color=palette['magenta'])
    ax.plot(folds, metrics_data['accuracy'], '-', alpha=0.5, linewidth=1.5, color=palette['teal'])
    ax.plot(folds, metrics_data['balanced_accuracy'], '-', alpha=0.5, linewidth=1.5, color=palette['ice'])
    
    # 各foldのポイント（フォールドごとの色）
    for i, fold_num in enumerate(folds):
        ax.scatter(fold_num, metrics_data['f1_macro'][i], s=120, color=fold_colors[i], 
                  marker='o', edgecolors='white', linewidths=1, zorder=3,
                  label=f'Fold {fold_num}' if i == 0 else None)
        ax.scatter(fold_num, metrics_data['accuracy'][i], s=100, color=fold_colors[i], 
                  marker='s', edgecolors='white', linewidths=1, zorder=3)
        ax.scatter(fold_num, metrics_data['balanced_accuracy'][i], s=100, color=fold_colors[i], 
                  marker='^', edgecolors='white', linewidths=1, zorder=3)
    
    # メトリクスの凡例（マーカー形状）
    from matplotlib.lines import Line2D
    metric_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=palette['magenta'], 
               markersize=10, label='F1 Macro', linestyle='-', linewidth=1.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=palette['teal'], 
               markersize=10, label='Accuracy', linestyle='-', linewidth=1.5),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=palette['ice'], 
               markersize=10, label='Balanced Acc', linestyle='-', linewidth=1.5),
    ]
    legend1 = ax.legend(handles=metric_handles, loc='lower left', fontsize=9, title='Metrics')
    legend1.get_frame().set_facecolor(palette['deep_teal'])
    legend1.get_frame().set_edgecolor(palette['mist'])
    legend1.get_title().set_color(palette['cream'])
    for text in legend1.get_texts():
        text.set_color(palette['cream'])
    
    ax.set_xlabel('Fold', fontsize=12, color=palette['cream'])
    ax.set_ylabel('Score', fontsize=12, color=palette['cream'])
    ax.set_title('Performance Metrics by Fold', fontsize=13, fontweight='bold', color=palette['cream'])
    ax.set_xticks(folds)
    ax.set_ylim([min(min(metrics_data['f1_macro']), 
                    min(metrics_data['accuracy']), 
                    min(metrics_data['balanced_accuracy'])) - 0.02, 
                max(max(metrics_data['f1_macro']), 
                    max(metrics_data['accuracy']), 
                    max(metrics_data['balanced_accuracy'])) + 0.02])
    
    # Precision/Recallの推移（各foldを色分け）
    ax = axes[1]
    # 折れ線グラフ
    ax.plot(folds, metrics_data['precision_macro'], '-', alpha=0.5, linewidth=1.5, color=palette['magenta'])
    ax.plot(folds, metrics_data['recall_macro'], '-', alpha=0.5, linewidth=1.5, color=palette['teal'])
    
    # 各foldのポイント
    for i, fold_num in enumerate(folds):
        ax.scatter(fold_num, metrics_data['precision_macro'][i], s=120, color=fold_colors[i], 
                  marker='D', edgecolors='white', linewidths=1, zorder=3)
        ax.scatter(fold_num, metrics_data['recall_macro'][i], s=100, color=fold_colors[i], 
                  marker='p', edgecolors='white', linewidths=1, zorder=3)
    
    metric_handles2 = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor=palette['magenta'], 
               markersize=10, label='Precision Macro', linestyle='-', linewidth=1.5),
        Line2D([0], [0], marker='p', color='w', markerfacecolor=palette['teal'], 
               markersize=10, label='Recall Macro', linestyle='-', linewidth=1.5),
    ]
    # フォールドの色凡例
    fold_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=fold_colors[i], 
                          markersize=8, label=f'Fold {i+1}', linestyle='') for i in range(n_folds)]
    
    legend2 = ax.legend(handles=metric_handles2, loc='lower left', fontsize=9, title='Metrics')
    legend2.get_frame().set_facecolor(palette['deep_teal'])
    legend2.get_frame().set_edgecolor(palette['mist'])
    legend2.get_title().set_color(palette['cream'])
    for text in legend2.get_texts():
        text.set_color(palette['cream'])
    
    # フォールド色の凡例を右側に追加
    legend3 = ax.legend(handles=fold_handles, loc='lower right', fontsize=8, title='Folds', ncol=2)
    legend3.get_frame().set_facecolor(palette['deep_teal'])
    legend3.get_frame().set_edgecolor(palette['mist'])
    legend3.get_title().set_color(palette['cream'])
    for text in legend3.get_texts():
        text.set_color(palette['cream'])
    ax.add_artist(legend2)  # 最初の凡例を再追加
    
    ax.set_xlabel('Fold', fontsize=12, color=palette['cream'])
    ax.set_ylabel('Score', fontsize=12, color=palette['cream'])
    ax.set_title('Precision and Recall by Fold', fontsize=13, fontweight='bold', color=palette['cream'])
    ax.set_xticks(folds)
    ax.set_ylim([min(min(metrics_data['precision_macro']), 
                    min(metrics_data['recall_macro'])) - 0.02,
                max(max(metrics_data['precision_macro']), 
                    max(metrics_data['recall_macro'])) + 0.02])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_metrics_by_fold.png'), dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 3. クラス別F1スコアのヒートマップ（各foldのクラス別性能）
    if 'f1' in fold_results[0]['metrics']:
        n_classes = len(fold_results[0]['metrics']['f1'])
        f1_matrix = np.zeros((n_folds, n_classes))
        
        for i, result in enumerate(fold_results):
            f1_matrix[i, :] = result['metrics']['f1']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        custom_cmap = ListedColormap([palette['deep_teal'], palette['teal'], palette['ice'], palette['cream'], palette['magenta']])
        im = ax.imshow(f1_matrix, cmap=custom_cmap, aspect='auto', vmin=0, vmax=1)
        
        # 各セルに値を表示
        for i in range(n_folds):
            for j in range(n_classes):
                ax.text(j, i, f'{f1_matrix[i, j]:.3f}',
                        ha="center", va="center", color='black', fontsize=10, fontweight='bold')
        
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_folds))
        if class_names and len(class_names) == n_classes:
            ax.set_xticklabels(class_names, color=palette['cream'])
        else:
            ax.set_xticklabels([f'Class {i}' for i in range(n_classes)], color=palette['cream'])
        ax.set_yticklabels([f'Fold {i+1}' for i in range(n_folds)], color=palette['cream'])
        
        ax.set_xlabel('Class', fontsize=12, color=palette['cream'])
        ax.set_ylabel('Fold', fontsize=12, color=palette['cream'])
        ax.set_title(f'{model_name} - F1 Score by Class and Fold', fontsize=13, fontweight='bold', color=palette['cream'])
        
        # カラーバーを追加
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_facecolor('black')
        cbar.ax.yaxis.set_tick_params(color=palette['cream'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=palette['cream'])
        cbar.set_label('F1 Score', rotation=270, labelpad=15, color=palette['cream'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cv_f1_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    
    print(f"  📊 Cross-validation visualizations saved to {save_dir}")


# *******************************************************************
# 学習曲線の可視化関数
# *******************************************************************
def plot_training_curves(train_losses, val_losses, train_f1s, val_f1s, 
                        save_path, fold_num=None):
    """
    学習曲線を可視化して保存する関数
    
    Parameters:
    - train_losses: 訓練損失のリスト
    - val_losses: 検証損失のリスト
    - train_f1s: 訓練F1スコアのリスト
    - val_f1s: 検証F1スコアのリスト
    - save_path: 保存先パス
    - fold_num: Fold番号（クロスバリデーションの場合）
    """
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('black')
    axes = (ax1, ax2)

    epochs = range(1, len(train_losses) + 1)

    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors=palette['cream'])
        for spine in ax.spines.values():
            spine.set_color(palette['cream'])
        ax.grid(True, alpha=0.3, color=palette['mist'])

    # Loss plot
    ax1.plot(epochs, train_losses, color=palette['teal'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, color=palette['magenta'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, color=palette['cream'])
    ax1.set_ylabel('Loss', fontsize=12, color=palette['cream'])
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold', color=palette['cream'])
    legend = ax1.legend()
    if legend:
        legend.get_frame().set_facecolor(palette['deep_teal'])
        legend.get_frame().set_edgecolor(palette['mist'])
        for text in legend.get_texts():
            text.set_color(palette['cream'])
    ax1.set_xlim(1, len(epochs))

    # F1 Score plot
    ax2.plot(epochs, train_f1s, color=palette['teal'], label='Training F1', linewidth=2)
    ax2.plot(epochs, val_f1s, color=palette['magenta'], label='Validation F1', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, color=palette['cream'])
    ax2.set_ylabel('F1 Score (Macro)', fontsize=12, color=palette['cream'])
    ax2.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold', color=palette['cream'])
    legend = ax2.legend(loc='lower right')
    if legend:
        legend.get_frame().set_facecolor(palette['deep_teal'])
        legend.get_frame().set_edgecolor(palette['mist'])
        for text in legend.get_texts():
            text.set_color(palette['cream'])
    ax2.set_xlim(1, len(epochs))
    ax2.set_ylim(0, 1)

    if fold_num is not None:
        fig.suptitle(f'Fold {fold_num} Training Curves', fontsize=16, y=1.02, color=palette['cream'])
    else:
        fig.suptitle('Training Curves', fontsize=16, y=1.02, color=palette['cream'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # 最終エポックの値も返す
    return {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_f1': train_f1s[-1],
        'final_val_f1': val_f1s[-1],
        'best_val_f1': max(val_f1s),
        'best_epoch': val_f1s.index(max(val_f1s)) + 1
    }


# *******************************************************************
# 最適閾値探索関数（改良版）
# *******************************************************************
def find_optimal_thresholds(y_true_continuous, y_pred_probs, n_classes=5, 
                           metric='f1_macro', n_iter=100):
    """
    最適な閾値を探索する関数
    
    Parameters:
    - y_true_continuous: 連続値の真のラベル（0.0～1.0）
    - y_pred_probs: モデルの予測確率 (n_samples, n_classes)
    - n_classes: クラス数
    - metric: 最適化する指標
    - n_iter: 最適化の反復回数
    
    Returns:
    - best_thresholds: 最適な閾値のリスト
    - best_score: 最適スコア
    - threshold_history: 探索履歴
    """
    from scipy.optimize import differential_evolution
    import warnings
    warnings.filterwarnings('ignore')
    
    # 履歴を記録
    threshold_history = []
    
    def label_to_class(labels, thresholds):
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
    
    def objective(thresholds):
        # 閾値を昇順にソート
        thresholds = sorted(thresholds)
        
        # 連続値ラベルをクラスラベルに変換
        y_true_classes = label_to_class(y_true_continuous, thresholds)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        # 評価指標を計算
        if metric == 'f1_macro':
            score = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true_classes, y_pred_classes)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true_classes, y_pred_classes)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # 履歴に追加
        threshold_history.append({
            'thresholds': thresholds.copy(),
            'score': score
        })
        
        return -score  # 最小化問題に変換
    
    # 初期閾値の範囲を設定
    bounds = [(0.0, 1.0) for _ in range(n_classes - 1)]
    
    # 差分進化法で最適化
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=n_iter, 
        popsize=15, 
        seed=42,
        disp=False
    )
    
    best_thresholds = sorted(result.x)
    best_score = -result.fun
    
    # クラス分布の分析
    y_true_classes = label_to_class(y_true_continuous, best_thresholds)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    class_distribution = {
        'true': np.bincount(y_true_classes, minlength=n_classes),
        'pred': np.bincount(y_pred_classes, minlength=n_classes)
    }
    
    return best_thresholds, best_score, threshold_history, class_distribution


# *******************************************************************
# 混同行列の可視化関数
# *******************************************************************
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', 
                         save_path=None, normalize=False, thresholds=None):
    """
    混同行列を可視化する関数
    
    Parameters:
    - cm: 混同行列
    - classes: クラス名のリスト
    - title: グラフのタイトル
    - save_path: 保存先パス（Noneの場合は保存しない）
    - normalize: 正規化するかどうか
    - thresholds: クラス分類に使用した閾値リスト（表示用）
    """
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    cmap = LinearSegmentedColormap.from_list(
        'teal_gradient',
        [palette['cream'], palette['mist'], palette['teal'], palette['deep_teal']]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor=palette['mist'],
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    # タイトルに閾値情報を追加
    if thresholds is not None:
        threshold_str = ', '.join([f'{t:.2f}' for t in thresholds])
        full_title = f"{title}\nThresholds: [{threshold_str}]"
    else:
        full_title = title
    
    ax.set_title(full_title, color=palette['cream'], fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', color=palette['cream'], fontsize=12)
    ax.set_xlabel('Predicted Label', color=palette['cream'], fontsize=12)

    ax.tick_params(axis='both', colors=palette['cream'])
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(palette['cream'])

    # Annotated text visibility on the dark gradient
    for text in heatmap.texts:
        text.set_color('black')
        text.set_fontweight('bold')

    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_facecolor('black')
    cbar.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(cbar.ax.get_yticklabels(), color=palette['cream'])
    cbar.set_label('Count' if not normalize else 'Proportion', color=palette['cream'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"  [plot_confusion_matrix] Figure saved to: {save_path}")
        plt.close(fig)
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
# モデル初期化関数（ResNet18追加版）
# *******************************************************************
def init_model(model_name, hyperparams, n_classes=5):
    """
    モデルを初期化する関数
    
    Parameters:
    - model_name: 'ResNet18', 'MLP-Mixer', 'CNN', 'ViT'
    - hyperparams: ハイパーパラメータの辞書
    - n_classes: クラス数
    """
    in_channels = hyperparams.get('in_channels', 3)
    image_size = hyperparams.get('image_size', 100)  # デフォルト100px (20m / 0.2m/px)
    
    if model_name == 'ResNet18':
        model = ResNet18Custom(
            num_classes=n_classes,
            pretrained=hyperparams.get('pretrained', True),
            dropout=hyperparams.get('dropout', 0.3),  # デフォルト0.3
            input_size=hyperparams.get('input_size', image_size),
            in_channels=in_channels,
        )
    elif model_name == 'MLP-Mixer':
        model = MLPMixer(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=hyperparams.get('patch_size', 10),
            num_classes=n_classes,
            dim=hyperparams.get('dim', 512),
            depth=hyperparams.get('depth', 12),
            token_dim=hyperparams.get('token_dim', 256),
            channel_dim=hyperparams.get('channel_dim', 2048)
        )
    elif model_name == 'CNN':
        model = CNN(
            in_channels=in_channels,
            num_classes=n_classes,
            hidden_dim=hyperparams.get('hidden_dim', 128),
            depth=hyperparams.get('depth', 4),
            kernel_size=hyperparams.get('kernel_size', 3)
        )
    elif model_name == 'ViT':
        model = ViT(
            image_size=image_size,
            patch_size=hyperparams.get('patch_size', 10),
            in_channels=in_channels,
            num_classes=n_classes,
            dim=hyperparams.get('dim', 256),
            depth=hyperparams.get('depth', 6),
            heads=hyperparams.get('heads', 8),
            mlp_dim=hyperparams.get('mlp_dim', 512),
            dropout=hyperparams.get('dropout', 0.1)
        )
    elif model_name == 'HybridModel':
        # ハイブリッドCNN-ViTモデル（RGB: 浅いCNN、VDVI/DEM: MLP）
        model = HybridModel(
            num_classes=n_classes,
            image_size=image_size,
            dropout=hyperparams.get('dropout', 0.3),
            in_channels=in_channels
        )
    elif model_name == 'HybridModelV2':
        # 柔軟なチャンネル設定対応ハイブリッドモデル（GLCM/EXG対応版）
        channel_config = hyperparams.get('channel_config', None)
        model = HybridModelV2(
            num_classes=n_classes,
            image_size=image_size,
            dropout=hyperparams.get('dropout', 0.3),
            in_channels=in_channels,
            channel_config=channel_config,
            use_pretrained=hyperparams.get('use_pretrained', True)
        )
    elif model_name == 'TransformerV1':
        # 基本Transformer（RGB only）+ MLP（Aux）
        model = LandslideTransformerV1(
            image_size=image_size,
            patch_size=hyperparams.get('patch_size', 10),
            in_channels=in_channels,
            num_classes=n_classes,
            embed_dim=hyperparams.get('embed_dim', 128),
            depth=hyperparams.get('depth', 4),
            num_heads=hyperparams.get('num_heads', 4),
            mlp_ratio=hyperparams.get('mlp_ratio', 2.0),
            dropout=hyperparams.get('dropout', 0.1),
            attn_drop=hyperparams.get('attn_drop', 0.0),
            drop_path=hyperparams.get('drop_path', 0.1)
        )
    elif model_name == 'TransformerV2':
        # Cross-Modal Transformer
        model = LandslideTransformerV2(
            image_size=image_size,
            patch_size=hyperparams.get('patch_size', 10),
            in_channels=in_channels,
            num_classes=n_classes,
            embed_dim=hyperparams.get('embed_dim', 128),
            depth=hyperparams.get('depth', 4),
            num_heads=hyperparams.get('num_heads', 4),
            mlp_ratio=hyperparams.get('mlp_ratio', 2.0),
            dropout=hyperparams.get('dropout', 0.1),
            attn_drop=hyperparams.get('attn_drop', 0.0),
            drop_path=hyperparams.get('drop_path', 0.1)
        )
    elif model_name == 'TransformerV3':
        # Gated Multimodal Transformer（軽量版）
        model = LandslideTransformerV3(
            image_size=image_size,
            patch_size=hyperparams.get('patch_size', 10),
            in_channels=in_channels,
            num_classes=n_classes,
            embed_dim=hyperparams.get('embed_dim', 96),
            depth=hyperparams.get('depth', 3),
            num_heads=hyperparams.get('num_heads', 3),
            mlp_ratio=hyperparams.get('mlp_ratio', 2.0),
            dropout=hyperparams.get('dropout', 0.2),
            attn_drop=hyperparams.get('attn_drop', 0.1),
            drop_path=hyperparams.get('drop_path', 0.15)
        )
    elif model_name == 'HybridTransformer':
        # CNN + Transformerハイブリッド
        model = LandslideHybridTransformer(
            image_size=image_size,
            in_channels=in_channels,
            num_classes=n_classes,
            embed_dim=hyperparams.get('embed_dim', 128),
            depth=hyperparams.get('depth', 3),
            num_heads=hyperparams.get('num_heads', 4),
            mlp_ratio=hyperparams.get('mlp_ratio', 2.0),
            dropout=hyperparams.get('dropout', 0.15),
            attn_drop=hyperparams.get('attn_drop', 0.0),
            drop_path=hyperparams.get('drop_path', 0.1)
        )
    # ========================================
    # 中間値予測に特化したモデル群
    # ========================================
    elif model_name == 'IntermediateValueTransformer':
        model = IntermediateValueTransformer(
            image_size=image_size,
            patch_size=hyperparams.get('patch_size', 10),
            in_channels=in_channels,
            n_classes=n_classes,
            embed_dim=hyperparams.get('embed_dim', 128),
            depth=hyperparams.get('depth', 4),
            num_heads=hyperparams.get('num_heads', 4),
            mlp_ratio=hyperparams.get('mlp_ratio', 2.0),
            dropout=hyperparams.get('dropout', 0.1),
        )
    elif model_name == 'IntermediateFocusedCNN':
        model = IntermediateFocusedCNN(
            in_channels=in_channels,
            n_classes=n_classes,
            base_channels=hyperparams.get('base_channels', 32),
            dropout=hyperparams.get('dropout', 0.2),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


# *******************************************************************
# ハイパーパラメータ探索のためのObjective関数（ResNet18対応版）
# *******************************************************************
def create_objective(model_name, train_dataset, val_dataset, n_classes, device, n_epochs=20):
    """Optunaのobjective関数を作成"""

    sample = train_dataset[0]['image']
    inferred_in_channels = int(sample.shape[0])
    
    trial_count = [0]  # トライアル番号を追跡するためのカウンター
    
    def objective(trial):
        trial_count[0] += 1
        print(f"\n{'='*70}")
        print(f"Trial #{trial_count[0]} (Optuna Trial #{trial.number}) Starting...")
        print(f"{'='*70}")
        
        # モデル別のハイパーパラメータ探索空間
        if model_name == 'ResNet18':
            hyperparams = {
                'pretrained': trial.suggest_categorical('pretrained', [True, False]),
                'dropout': 0.3,  # 固定値として0.3を使用
                'input_size': 25  # 固定
            }
        elif model_name == 'MLP-Mixer':
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
        elif model_name == 'HybridModel':
            # ハイブリッドCNN-ViTモデル
            hyperparams = {
                'image_size': 25,  # 固定
                'dropout': trial.suggest_float('dropout', 0.2, 0.5),
            }
        elif model_name == 'TransformerV1':
            # 基本Transformer（RGB only）+ MLP（Aux）
            hyperparams = {
                'image_size': 25,
                'patch_size': trial.suggest_categorical('patch_size', [5]),
                'embed_dim': trial.suggest_categorical('embed_dim', [64, 128, 192]),
                'depth': trial.suggest_int('depth', 2, 6),
                'num_heads': trial.suggest_categorical('num_heads', [2, 4]),
                'mlp_ratio': trial.suggest_categorical('mlp_ratio', [1.5, 2.0, 3.0]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3),
                'attn_drop': trial.suggest_float('attn_drop', 0.0, 0.2),
                'drop_path': trial.suggest_float('drop_path', 0.0, 0.2),
            }
        elif model_name == 'TransformerV2':
            # Cross-Modal Transformer
            hyperparams = {
                'image_size': 25,
                'patch_size': trial.suggest_categorical('patch_size', [5]),
                'embed_dim': trial.suggest_categorical('embed_dim', [64, 128, 192]),
                'depth': trial.suggest_int('depth', 2, 6),
                'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
                'mlp_ratio': trial.suggest_categorical('mlp_ratio', [1.5, 2.0, 3.0]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3),
                'attn_drop': trial.suggest_float('attn_drop', 0.0, 0.2),
                'drop_path': trial.suggest_float('drop_path', 0.0, 0.2),
            }
        elif model_name == 'TransformerV3':
            # Gated Multimodal Transformer（軽量版）
            hyperparams = {
                'image_size': 25,
                'patch_size': trial.suggest_categorical('patch_size', [5]),
                'embed_dim': trial.suggest_categorical('embed_dim', [64, 96, 128]),
                'depth': trial.suggest_int('depth', 2, 4),
                'num_heads': trial.suggest_categorical('num_heads', [2, 3, 4]),
                'mlp_ratio': trial.suggest_categorical('mlp_ratio', [1.5, 2.0]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                'attn_drop': trial.suggest_float('attn_drop', 0.0, 0.2),
                'drop_path': trial.suggest_float('drop_path', 0.1, 0.25),
            }
        elif model_name == 'HybridTransformer':
            # CNN + Transformerハイブリッド
            hyperparams = {
                'image_size': 25,
                'embed_dim': trial.suggest_categorical('embed_dim', [64, 128, 192]),
                'depth': trial.suggest_int('depth', 2, 4),
                'num_heads': trial.suggest_categorical('num_heads', [2, 4]),
                'mlp_ratio': trial.suggest_categorical('mlp_ratio', [1.5, 2.0, 3.0]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                'attn_drop': trial.suggest_float('attn_drop', 0.0, 0.15),
                'drop_path': trial.suggest_float('drop_path', 0.0, 0.2),
            }
        # ========================================
        # 中間値予測に特化したモデル群
        # ========================================
        elif model_name == 'IntermediateValueTransformer':
            hyperparams = {
                'image_size': 25,
                'patch_size': trial.suggest_categorical('patch_size', [5]),
                'embed_dim': trial.suggest_categorical('embed_dim', [64, 128, 192]),
                'depth': trial.suggest_int('depth', 2, 6),
                'num_heads': trial.suggest_categorical('num_heads', [2, 4]),
                'mlp_ratio': trial.suggest_categorical('mlp_ratio', [1.5, 2.0, 3.0]),
                'dropout': trial.suggest_float('dropout', 0.05, 0.25),
            }
        elif model_name == 'IntermediateFocusedCNN':
            hyperparams = {
                'base_channels': trial.suggest_categorical('base_channels', [16, 32, 64]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        hyperparams['in_channels'] = inferred_in_channels
        
        # 共通のハイパーパラメータ
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # 選択されたハイパーパラメータを表示
        print("\n📊 Selected Hyperparameters:")
        print(f"  Model-specific params: {hyperparams}")
        print(f"  Learning rate: {learning_rate:.6f}")
        print(f"  Batch size: {batch_size}")
        print(f"  Weight decay: {weight_decay:.6f}")
        print(f"\n{'─'*70}")
        trial.set_user_attr('in_channels', inferred_in_channels)
        
        # モデルの初期化
        model = init_model(model_name, hyperparams, n_classes)
        model = model.to(device)
        
        # モデルのパラメータ数を表示
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model initialized: {total_params:,} total params ({trainable_params:,} trainable)")
        
        # データローダーの作成
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 損失関数とオプティマイザ
        class_weights, class_counts = compute_inverse_frequency_weights(
            train_dataset.labels, train_dataset.thresholds, n_classes
        )
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"  Class counts: {class_counts.tolist()} -> weights: {[round(w, 4) for w in class_weights]}")
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        print(f"\n📈 Training for {n_epochs} epochs...")
        print(f"{'─'*70}")
        
        # 訓練
        best_val_f1 = 0
        best_epoch = 0
        epoch_results = []
        
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
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['image'].to(device)
                    labels = batch['label']
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.to(device))
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu()
                    
                    val_preds.extend(preds.numpy())
                    val_labels.extend(labels.numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            
            # F1スコアと精度の計算
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            val_acc = accuracy_score(val_labels, val_preds)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                improvement_mark = " ⭐ NEW BEST!"
            else:
                improvement_mark = ""
            
            # エポックごとの結果を表示（5エポックごとまたは最良スコア更新時）
            if (epoch + 1) % 5 == 0 or improvement_mark:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                      f"Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}{improvement_mark}")
            
            scheduler.step()
            
            # Pruning（早期終了）
            trial.report(val_f1, epoch)
            if trial.should_prune():
                print(f"\n⚠️  Trial pruned at epoch {epoch+1} (F1: {val_f1:.4f})")
                print(f"{'='*70}")
                raise optuna.TrialPruned()
        
        # トライアル終了時のサマリー
        print(f"\n{'─'*70}")
        print(f"✅ Trial #{trial_count[0]} Completed!")
        print(f"  Best F1 Score: {best_val_f1:.4f} (at epoch {best_epoch})")
        print(f"{'='*70}")
        
         # メモリクリア（強化版）
        if best_model_state is not None:
            del best_model_state
            best_model_state = None
        
        del train_loader
        del val_loader
        del optimizer
        del scheduler
        del model
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        gc.collect()
        gc.collect()
        return best_val_f1
    
    return objective


# *******************************************************************
# Cross-Validation with Confusion Matrix per Fold (Spatial K-Fold版)
# *******************************************************************
def cross_validate_with_confusion_matrix(model_name, hyperparams, images, labels, 
                                        pixel_locations, geo_locations, landslide_ids,
                                        n_classes=5, n_folds=5, n_epochs=60, 
                                        batch_size=48, device='cuda', save_dir='results',
                                        optimize_threshold=True, use_spatial_kfold=True,
                                        config=None):
    """
    クロスバリデーションを実行し、各foldの混同行列と学習曲線を生成
    Balanced Spatial K-Foldをサポート
    """
    
    # Spatial K-Foldまたは通常のStratified K-Foldを選択
    fixed_thresholds = config.get('fixed_thresholds') if config else None
    thresholds_for_split = fixed_thresholds if fixed_thresholds is not None else [i / n_classes for i in range(1, n_classes)]
    class_names_template = generate_class_names_from_thresholds(fixed_thresholds, n_classes)

    if use_spatial_kfold and geo_locations is not None:
        if config is not None and config.get('spatial_folds_cache') is not None:
            spatial_folds = config['spatial_folds_cache']
            n_folds = len(np.unique(spatial_folds))
            print("\n🗺️  Using cached Balanced Spatial K-Fold assignments...")
        else:
            print("\n🗺️  Using Balanced Spatial K-Fold...")
            spatial_folds, gdf_prj = apply_balanced_spatial_kfold(
                images, labels, geo_locations, landslide_ids, config
            )
            n_folds = len(np.unique(spatial_folds))
            if config is not None:
                config['spatial_folds_cache'] = spatial_folds
                config['spatial_folds_gdf'] = gdf_prj
    else:
        print("\n📊 Using Stratified K-Fold...")
        # クラスラベルの生成（層化分割用）
        class_labels = []
        for label in labels:
            for i, threshold in enumerate(thresholds_for_split):
                if label <= threshold:
                    class_labels.append(i)
                    break
            else:
                class_labels.append(n_classes - 1)
        
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        spatial_folds = None
    
    fold_results = []
    all_cms = []
    best_thresholds_per_fold = []
    all_predictions_df = []  # 予測結果を保存するためのリスト
    
    # 最良モデル追跡用
    best_overall_f1 = 0
    best_overall_fold = 0
    best_overall_model_state = None
    best_overall_metrics = None
    best_overall_epoch = 0
    
    # 結果保存用ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    curves_dir = os.path.join(model_save_dir, 'training_curves')
    os.makedirs(curves_dir, exist_ok=True)
    models_dir = os.path.join(model_save_dir, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 全体の進捗管理
    total_folds = n_folds
    dem_range = config.get('dem_range') if config else None
    vdvi_range = config.get('vdvi_range') if config else None
    
    # Foldごとの処理
    for fold in range(n_folds):
        fold_start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"📁 Fold {fold + 1}/{n_folds} - {(fold/total_folds)*100:.1f}% Complete")
        print(f"{'='*70}")
        
        # インデックスの取得
        if use_spatial_kfold:
            val_idx = np.where(spatial_folds == fold)[0]
            train_idx = np.where(spatial_folds != fold)[0]
        else:
            # Stratified K-Foldの場合
            train_val_splits = list(kf.split(images, class_labels))
            train_idx, val_idx = train_val_splits[fold]
        
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # データセットの作成
        train_dataset = LandslideDataset(
            images[train_idx], labels[train_idx],
            pixel_locations[train_idx], geo_locations[train_idx],
            n_classes=n_classes,
            thresholds=fixed_thresholds,
            dem_range=dem_range,
            vdvi_range=vdvi_range
        )
        val_dataset = LandslideDataset(
            images[val_idx], labels[val_idx],
            pixel_locations[val_idx], geo_locations[val_idx],
            n_classes=n_classes,
            thresholds=fixed_thresholds,
            dem_range=dem_range,
            vdvi_range=vdvi_range
        )
        
        # バッチサイズは事前定義されたものを使用
        if 'batch_size' in hyperparams:
            actual_batch_size = hyperparams['batch_size']
        else:
            actual_batch_size = batch_size
        
        # データローダーの作成
        train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False)
        
        # モデルの初期化と訓練
        model = init_model(model_name, hyperparams, n_classes)
        model = model.to(device)
        
        # 中間値予測モデルかどうかを判定
        use_intermediate_loss = is_intermediate_model(model_name)
        
        # 学習率とweight_decayを取得
        learning_rate = hyperparams.get('learning_rate', 1e-4)
        weight_decay = hyperparams.get('weight_decay', 1e-4)
        
        class_weights, class_counts = compute_inverse_frequency_weights(
            train_dataset.labels, train_dataset.thresholds, n_classes
        )
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"  Class counts (fold {fold + 1}): {class_counts.tolist()} -> weights: {[round(w, 4) for w in class_weights]}")
        
        # 損失関数の設定（中間値予測モデル用の特別な損失関数を使用）
        if use_intermediate_loss:
            intermediate_loss_fn = get_loss_fn_for_model(model_name, n_classes, fixed_thresholds)
            if intermediate_loss_fn is not None:
                intermediate_loss_fn = intermediate_loss_fn.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)  # フォールバック用
            print(f"  Using specialized loss function for {model_name}")
        else:
            intermediate_loss_fn = None
            criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
        # 学習曲線用のリスト
        train_losses = []
        val_losses = []
        train_f1s = []
        val_f1s = []
        train_accs = []
        val_accs = []
        
        # 訓練
        print(f"\n📈 Training for {n_epochs} epochs...")
        print(f"{'─'*70}")
        best_model_state = None
        best_val_f1 = 0
        best_epoch = 0
        best_val_preds = None
        best_val_labels = None
        best_val_probs = None
        
        for epoch in range(n_epochs):
            epoch_start = datetime.now()
            
            # Training
            model.train()
            train_loss = 0
            train_preds = []
            train_labels_epoch = []
            
            for batch in train_loader:
                inputs = batch['image'].to(device)
                labels_batch = batch['label'].to(device)
                raw_labels_batch = batch['raw_label'].to(device)
                
                optimizer.zero_grad()
                
                # 中間値予測モデルの場合
                if use_intermediate_loss:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        class_probs, model_outputs = outputs
                        # 専用損失関数を使用
                        if intermediate_loss_fn is not None:
                            loss, _ = intermediate_loss_fn(model_outputs, class_probs, labels_batch, raw_labels_batch)
                        else:
                            loss = criterion(torch.log(class_probs + 1e-7), labels_batch)
                        preds = class_probs.argmax(dim=1).cpu()
                    else:
                        loss = criterion(outputs, labels_batch)
                        preds = torch.argmax(outputs, dim=1).cpu()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                    preds = torch.argmax(outputs, dim=1).cpu()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(preds.numpy())
                train_labels_epoch.extend(labels_batch.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            train_f1 = f1_score(train_labels_epoch, train_preds, average='macro', zero_division=0)
            train_acc = accuracy_score(train_labels_epoch, train_preds)
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels_epoch = []
            val_probs = []
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['image'].to(device)
                    labels_batch = batch['label']
                    raw_labels_batch = batch['raw_label'].to(device)
                    
                    # 中間値予測モデルの場合
                    if use_intermediate_loss:
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            class_probs, model_outputs = outputs
                            if intermediate_loss_fn is not None:
                                loss, _ = intermediate_loss_fn(model_outputs, class_probs, labels_batch.to(device), raw_labels_batch)
                            else:
                                loss = criterion(torch.log(class_probs + 1e-7), labels_batch.to(device))
                            probs = class_probs.cpu()
                            preds = class_probs.argmax(dim=1).cpu()
                        else:
                            loss = criterion(outputs, labels_batch.to(device))
                            probs = torch.softmax(outputs, dim=1).cpu()
                            preds = torch.argmax(outputs, dim=1).cpu()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels_batch.to(device))
                        probs = torch.softmax(outputs, dim=1).cpu()
                        preds = torch.argmax(outputs, dim=1).cpu()
                    
                    val_loss += loss.item()
                    
                    val_probs.append(probs.numpy())
                    val_preds.extend(preds.numpy())
                    val_labels_epoch.extend(labels_batch.numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(val_labels_epoch, val_preds, average='macro', zero_division=0)
            val_acc = accuracy_score(val_labels_epoch, val_preds)
            
            # 学習曲線データの保存
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # 特定エポック（1, 5, 10）で混同行列を保存
            checkpoint_epochs = [1, 5, 10]
            if (epoch + 1) in checkpoint_epochs:
                # 混同行列を計算
                epoch_cm = confusion_matrix(val_labels_epoch, val_preds)
                epoch_cm_dir = os.path.join(model_save_dir, 'epoch_confusion_matrices')
                os.makedirs(epoch_cm_dir, exist_ok=True)
                
                # 混同行列を保存
                plot_confusion_matrix(
                    epoch_cm, class_names_template,
                    title=f'{model_name} - Fold {fold + 1} Epoch {epoch + 1} Confusion Matrix',
                    save_path=os.path.join(epoch_cm_dir, f'cm_fold_{fold+1}_epoch_{epoch+1}.png'),
                    normalize=False,
                    thresholds=fixed_thresholds
                )
                # 正規化版も保存
                plot_confusion_matrix(
                    epoch_cm, class_names_template,
                    title=f'{model_name} - Fold {fold + 1} Epoch {epoch + 1} Normalized CM',
                    save_path=os.path.join(epoch_cm_dir, f'cm_fold_{fold+1}_epoch_{epoch+1}_normalized.png'),
                    normalize=True,
                    thresholds=fixed_thresholds
                )
                print(f"  📊 Saved confusion matrix for epoch {epoch + 1}")
            
            # ベストモデルの保存
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
                # ベスト時点の予測結果と混同行列を保存
                best_val_preds = val_preds.copy() if isinstance(val_preds, list) else val_preds.tolist()
                best_val_labels = val_labels_epoch.copy() if isinstance(val_labels_epoch, list) else val_labels_epoch.tolist()
                best_val_probs = np.vstack(val_probs).copy()
                improvement = " ⭐ NEW BEST!"
            else:
                improvement = ""
            
            # 進捗表示（5エポックごとまたは最初/最後、ベスト更新時）
            epoch_progress = (epoch + 1) / n_epochs * 100
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == n_epochs - 1 or improvement:
                epoch_time = (datetime.now() - epoch_start).total_seconds()
                print(f"  Epoch {epoch+1:3d}/{n_epochs} ({epoch_progress:5.1f}%): "
                      f"Loss[T:{avg_train_loss:.4f}/V:{avg_val_loss:.4f}] "
                      f"F1[T:{train_f1:.4f}/V:{val_f1:.4f}] "
                      f"Acc[T:{train_acc:.4f}/V:{val_acc:.4f}] "
                      f"Time:{epoch_time:.1f}s{improvement}")
            
            scheduler.step()
        
        # 学習曲線の保存（改良版）
        curve_path = os.path.join(curves_dir, f'fold_{fold+1}_curves.png')
        curve_stats = plot_training_curves_enhanced(
            train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs,
            save_path=curve_path, fold_num=fold+1
        )
        print(f"\n📊 Training Summary:")
        print(f"  Best F1: {curve_stats['best_val_f1']:.4f} at epoch {curve_stats['best_epoch']}")
        print(f"  Best Acc: {curve_stats['best_val_acc']:.4f}")
        print(f"  Final F1: {curve_stats['final_val_f1']:.4f}")
        print(f"  Learning curves saved to: {curve_path}")
        
        # 最良モデルをロード
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最終評価
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []
        val_raw_labels = []
        val_indices = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                labels_batch = batch['label']
                raw_labels_batch = batch['raw_label']
                
                # 中間値予測モデルの場合
                if use_intermediate_loss:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        class_probs, model_outputs = outputs
                        probs = class_probs.cpu()
                        preds = class_probs.argmax(dim=1).cpu()
                    else:
                        probs = torch.softmax(outputs, dim=1).cpu()
                        preds = torch.argmax(outputs, dim=1).cpu()
                else:
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1).cpu()
                    preds = torch.argmax(outputs, dim=1).cpu()
                
                val_probs.append(probs.numpy())
                val_preds.extend(preds.numpy())
                val_labels.extend(labels_batch.numpy())
                val_raw_labels.extend(raw_labels_batch.numpy())
        
        val_probs = np.vstack(val_probs)
        
        # Validation用の座標情報を取得
        val_geo_locs = geo_locations[val_idx[:len(val_labels)]]
        val_pixel_locs = pixel_locations[val_idx[:len(val_labels)]]
        
        # 予測結果をDataFrameに保存
        fold_predictions_df = pd.DataFrame({
            'fold': fold,
            'true_label': val_labels,
            'predicted_label': val_preds,
            'raw_label': val_raw_labels,
            'val_index': val_idx[:len(val_labels)],
            'geo_x': val_geo_locs[:, 0],
            'geo_y': val_geo_locs[:, 1],
            'pixel_x': val_pixel_locs[:, 0],
            'pixel_y': val_pixel_locs[:, 1],
        })
        
        # 各クラスの予測確率も追加
        for i in range(n_classes):
            fold_predictions_df[f'prob_class_{i}'] = val_probs[:, i]
        
        all_predictions_df.append(fold_predictions_df)
        
        # 閾値の最適化（最初のfoldでのみ実施）
        if optimize_threshold and fold == 0:
            print(f"\n🔍 Optimizing thresholds...")
            best_thresholds, best_score, threshold_history, class_dist = find_optimal_thresholds(
                np.array(val_raw_labels), val_probs, n_classes=n_classes, metric='f1_macro'
            )
            best_thresholds_per_fold.append(best_thresholds)
            print(f"  Optimized thresholds: {[f'{t:.3f}' for t in best_thresholds]}")
            print(f"  Optimized F1 score: {best_score:.4f}")
            print(f"  Class distribution (true): {class_dist['true']}")
            print(f"  Class distribution (pred): {class_dist['pred']}")
        
        # メトリクスと混同行列の計算
        metrics, cm = calculate_metrics(val_labels, val_preds, val_probs)
        
        # 結果の保存
        fold_result = {
            'fold': fold + 1,
            'metrics': metrics,
            'confusion_matrix': cm,
            'training_stats': curve_stats,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        }
        fold_results.append(fold_result)
        all_cms.append(cm)
        
        # 混同行列の可視化と保存（最終エポック）
        class_names = class_names_template
        plot_confusion_matrix(
            cm, class_names,
            title=f'{model_name} - Fold {fold + 1} Final Confusion Matrix',
            save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}_final.png'),
            normalize=False,
            thresholds=fixed_thresholds
        )
        
        # 正規化版も保存
        plot_confusion_matrix(
            cm, class_names,
            title=f'{model_name} - Fold {fold + 1} Final Normalized Confusion Matrix',
            save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}_final_normalized.png'),
            normalize=True,
            thresholds=fixed_thresholds
        )
        
        # ベストエポック時点の混同行列を保存
        if best_val_preds is not None:
            best_cm = confusion_matrix(best_val_labels, best_val_preds)
            plot_confusion_matrix(
                best_cm, class_names,
                title=f'{model_name} - Fold {fold + 1} Best (Epoch {best_epoch}) Confusion Matrix',
                save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}_best.png'),
                normalize=False,
                thresholds=fixed_thresholds
            )
            plot_confusion_matrix(
                best_cm, class_names,
                title=f'{model_name} - Fold {fold + 1} Best (Epoch {best_epoch}) Normalized CM',
                save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}_best_normalized.png'),
                normalize=True,
                thresholds=fixed_thresholds
            )
            print(f"  📊 Best epoch ({best_epoch}) confusion matrix saved")
        
        # foldごとの結果を表示
        fold_time = (datetime.now() - fold_start_time).total_seconds() / 60
        print(f"\n✅ Fold {fold + 1} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  F1 per class: {[f'{f:.3f}' for f in metrics['f1']]}")
        print(f"  Time taken: {fold_time:.1f} minutes")
        
        # 全体の最良モデルを追跡（保存は最後に1つのみ）
        if best_model_state is not None:
            if metrics['f1_macro'] > best_overall_f1:
                best_overall_f1 = metrics['f1_macro']
                best_overall_fold = fold + 1
                best_overall_model_state = best_model_state.copy()
                best_overall_metrics = metrics.copy()
                best_overall_epoch = best_epoch
                print(f"  🏆 New best model! (F1 Macro: {best_overall_f1:.4f})")
        
         # メモリクリア（強化版）
        if best_model_state is not None:
            del best_model_state
            best_model_state = None
        
        del model
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        gc.collect()
        gc.collect()

    # 全予測結果を保存
    all_predictions_df = pd.concat(all_predictions_df, ignore_index=True)
    predictions_path = os.path.join(model_save_dir, 'all_predictions.csv')
    all_predictions_df.to_csv(predictions_path, index=False)
    print(f"\n💾 All predictions saved to: {predictions_path}")
    
    # 全foldで最も良いモデルのみを保存
    if best_overall_model_state is not None:
        best_model_path = os.path.join(models_dir, 'best_model.pth')
        # 保存前にCPUテンソルであることを確認
        clean_state_dict = {
            k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v 
            for k, v in best_overall_model_state.items()
        }
        torch.save({
            'model_state_dict': best_overall_model_state,
            'model_name': model_name,
            'hyperparams': hyperparams,
            'n_classes': n_classes,
            'best_fold': best_overall_fold,
            'best_epoch': best_overall_epoch,
            'best_f1_macro': best_overall_f1,
            'metrics': best_overall_metrics,
            'thresholds': fixed_thresholds
        }, best_model_path)
        print(f"🏆 Best model saved (Fold {best_overall_fold}, F1={best_overall_f1:.4f}): {best_model_path}")
    
    # クロスバリデーション結果の可視化
    plot_cv_scores_enhanced(fold_results, model_save_dir, model_name, class_names=class_names_template)
    
    # 全体の混同行列（平均）
    avg_cm = np.mean(all_cms, axis=0).astype(int)
    plot_confusion_matrix(
        avg_cm, class_names_template,
        title=f'{model_name} - Average Confusion Matrix across all folds',
        save_path=os.path.join(model_save_dir, 'confusion_matrix_average.png'),
        normalize=False,
        thresholds=fixed_thresholds
    )
    
    # 正規化版も保存
    plot_confusion_matrix(
        avg_cm, class_names_template,
        title=f'{model_name} - Average Normalized Confusion Matrix across all folds',
        save_path=os.path.join(model_save_dir, 'confusion_matrix_average_normalized.png'),
        normalize=True,
        thresholds=fixed_thresholds
    )
    
    # 結果のサマリーを作成
    summary = {
        'model': model_name,
        'n_folds': n_folds,
        'n_classes': n_classes,
        'n_epochs': n_epochs,
        'use_spatial_kfold': use_spatial_kfold,
        'hyperparams': hyperparams,
        'fold_results': fold_results,
        'average_confusion_matrix': avg_cm.tolist(),
        'best_thresholds': best_thresholds_per_fold if optimize_threshold else None,
        'predictions_path': predictions_path,
        'best_overall_fold': best_overall_fold,
        'best_overall_f1': best_overall_f1,
        'best_model_path': os.path.join(models_dir, 'best_model_overall.pth') if best_overall_model_state is not None else None
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
    print(f"\n{'='*70}")
    print(f"📊 {model_name} - Cross-Validation Summary")
    print(f"{'='*70}")
    
    avg_metrics = {}
    for key in fold_results[0]['metrics'].keys():
        if key in ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            values = [fold['metrics'][key] for fold in fold_results]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            print(f"{key}: {avg_metrics[key]['mean']:.4f} ± {avg_metrics[key]['std']:.4f}")
    
    print(f"\n💾 All results saved to: {model_save_dir}")
    print(f"{'='*70}")
    
    return summary


def evaluate_and_save_ensemble(model_names, results, config):
    """平均確率アンサンブルを作成し、単独モデルと同様の成果物を保存する。"""
    if len(model_names) < 2:
        return None

    prob_cols = [f'prob_class_{i}' for i in range(config['n_classes'])]
    merged_df = None
    processed_models = []

    for model_name in model_names:
        model_info = results.get(model_name)
        if not model_info:
            print(f"⚠️  Ensemble skipped for {model_name}: no result info.")
            continue
        cv_results = model_info.get('cv_results', {})
        predictions_path = cv_results.get('predictions_path')
        if not predictions_path or not os.path.exists(predictions_path):
            print(f"⚠️  Ensemble skipped for {model_name}: predictions file not found.")
            continue

        df = pd.read_csv(predictions_path)
        required_cols = {'fold', 'val_index', 'true_label', 'raw_label', *prob_cols}
        if not required_cols.issubset(df.columns):
            print(f"⚠️  Ensemble skipped for {model_name}: missing probability columns.")
            continue

        rename_map = {col: f"{col}_{model_name}" for col in prob_cols}
        df_model = df[['fold', 'val_index', 'true_label', 'raw_label'] + prob_cols].copy()
        df_model = df_model.rename(columns=rename_map)

        if merged_df is None:
            merged_df = df_model
            processed_models.append(model_name)
        else:
            before = len(merged_df)
            merged_df = merged_df.merge(
                df_model.drop(columns=['true_label', 'raw_label']),
                on=['fold', 'val_index'],
                how='inner'
            )
            if len(merged_df) < before:
                print(f"⚠️  Ensemble merge dropped {before - len(merged_df)} rows when aligning with {model_name}.")
            processed_models.append(model_name)

    if merged_df is None or len(processed_models) < 2:
        print("⚠️  Ensemble evaluation skipped: insufficient aligned predictions.")
        return None

    ensemble_prob_cols = []
    for i in range(config['n_classes']):
        model_prob_cols = [f"prob_class_{i}_{model}" for model in processed_models]
        missing_cols = [col for col in model_prob_cols if col not in merged_df.columns]
        if missing_cols:
            print(f"⚠️  Ensemble missing columns {missing_cols}; skipping class {i}.")
            continue
        target_col = f'ensemble_prob_class_{i}'
        merged_df[target_col] = merged_df[model_prob_cols].mean(axis=1)
        ensemble_prob_cols.append(target_col)

    if not ensemble_prob_cols:
        print("⚠️  Ensemble evaluation failed: no probability columns after averaging.")
        return None

    probs = merged_df[ensemble_prob_cols].values
    merged_df['ensemble_pred'] = np.argmax(probs, axis=1)

    ensemble_dir = Path(config['output_dir']) / 'Ensemble'
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = ensemble_dir / 'ensemble_predictions.csv'
    save_cols = ['fold', 'val_index', 'true_label', 'raw_label', 'ensemble_pred'] + ensemble_prob_cols
    merged_df[save_cols].to_csv(predictions_path, index=False)
    print(f"\n🤝 Ensemble predictions saved to: {predictions_path}")

    fold_results = []
    for fold_id in sorted(merged_df['fold'].unique()):
        mask = merged_df['fold'] == fold_id
        y_true = merged_df.loc[mask, 'true_label'].to_numpy()
        y_prob = merged_df.loc[mask, ensemble_prob_cols].to_numpy()
        y_pred = merged_df.loc[mask, 'ensemble_pred'].to_numpy()
        metrics, cm = calculate_metrics(y_true, y_pred, y_prob)
        fold_results.append({
            'fold': int(fold_id) + 1,
            'metrics': metrics,
            'confusion_matrix': cm,
            'train_size': None,
            'val_size': len(y_true)
        })

    class_names = generate_class_names_from_thresholds(config.get('fixed_thresholds'), config['n_classes'])
    ensemble_thresholds = config.get('fixed_thresholds')
    plot_cv_scores_enhanced(fold_results, ensemble_dir, 'Ensemble', class_names=class_names)
    for fold_result in fold_results:
        fold_idx = fold_result['fold']
        cm = fold_result['confusion_matrix']
        plot_confusion_matrix(
            cm, class_names,
            title=f'Ensemble - Fold {fold_idx} Confusion Matrix',
            save_path=ensemble_dir / f'confusion_matrix_fold_{fold_idx}.png',
            normalize=False,
            thresholds=ensemble_thresholds
        )
        plot_confusion_matrix(
            cm, class_names,
            title=f'Ensemble - Fold {fold_idx} Normalized Confusion Matrix',
            save_path=ensemble_dir / f'confusion_matrix_fold_{fold_idx}_normalized.png',
            normalize=True,
            thresholds=ensemble_thresholds
        )

    avg_cm = np.mean([fr['confusion_matrix'] for fr in fold_results], axis=0)
    plot_confusion_matrix(
        avg_cm.astype(int), class_names,
        title='Ensemble - Average Confusion Matrix',
        save_path=ensemble_dir / 'confusion_matrix_average.png',
        normalize=False,
        thresholds=ensemble_thresholds
    )

    summary = {
        'model': 'Ensemble',
        'base_models': processed_models,
        'n_folds': len(fold_results),
        'n_classes': config['n_classes'],
        'fold_results': fold_results,
        'average_confusion_matrix': avg_cm.tolist(),
        'predictions_path': str(predictions_path),
        'base_model_paths': {}
    }
    
    # 各ベースモデルの最良モデルパスを記録
    for model_name in processed_models:
        model_info = results.get(model_name, {})
        cv_results = model_info.get('cv_results', {})
        best_model_path = cv_results.get('best_model_path')
        if best_model_path:
            summary['base_model_paths'][model_name] = best_model_path

    with open(ensemble_dir / 'cv_results.json', 'w') as f:
        json_summary = summary.copy()
        json_summary['fold_results'] = []
        for fr in fold_results:
            serialized = {
                'fold': fr['fold'],
                'metrics': to_serializable(fr['metrics']),
                'confusion_matrix': fr['confusion_matrix'].tolist(),
                'train_size': fr['train_size'],
                'val_size': fr['val_size']
            }
            json_summary['fold_results'].append(serialized)
        json.dump(to_serializable(json_summary), f, indent=2)

    print("\n✅ Ensemble summary saved to:", ensemble_dir / 'cv_results.json')
    
    # ============================================================
    # アンサンブルモデル構成の保存（推論時に使用可能）
    # ============================================================
    ensemble_model_info = {
        'ensemble_type': 'average_probability',
        'base_models': processed_models,
        'n_classes': config['n_classes'],
        'thresholds': config.get('fixed_thresholds'),
        'base_model_paths': {},
        'base_model_hyperparams': {}
    }
    
    # 各ベースモデルの情報を収集
    for model_name in processed_models:
        model_info = results.get(model_name, {})
        cv_results = model_info.get('cv_results', {})
        best_params = model_info.get('best_params', {})
        
        # 最良モデルのパス
        best_model_path = cv_results.get('best_model_path')
        if best_model_path and Path(best_model_path).exists():
            ensemble_model_info['base_model_paths'][model_name] = str(best_model_path)
        else:
            # 代替: saved_models/best_model_overall.pth を探す
            alt_path = Path(config['output_dir']) / model_name / 'saved_models' / 'best_model_overall.pth'
            if alt_path.exists():
                ensemble_model_info['base_model_paths'][model_name] = str(alt_path)
        
        # ハイパーパラメータ
        if best_params:
            ensemble_model_info['base_model_hyperparams'][model_name] = to_serializable(best_params)
    
    # アンサンブルモデル構成を保存
    ensemble_config_path = ensemble_dir / 'ensemble_model_config.json'
    with open(ensemble_config_path, 'w') as f:
        json.dump(ensemble_model_info, f, indent=2)
    print(f"💾 Ensemble model config saved to: {ensemble_config_path}")
    
    # PyTorchフォーマットでも保存（推論用）
    ensemble_pth_path = ensemble_dir / 'ensemble_model.pth'
    torch.save({
        'ensemble_type': 'average_probability',
        'base_models': processed_models,
        'n_classes': config['n_classes'],
        'thresholds': config.get('fixed_thresholds'),
        'base_model_paths': ensemble_model_info['base_model_paths'],
        'base_model_hyperparams': ensemble_model_info['base_model_hyperparams'],
        'average_metrics': {
            'accuracy': np.mean([fr['metrics']['accuracy'] for fr in fold_results]),
            'f1_macro': np.mean([fr['metrics']['f1_macro'] for fr in fold_results]),
            'balanced_accuracy': np.mean([fr['metrics']['balanced_accuracy'] for fr in fold_results]),
        }
    }, ensemble_pth_path)
    print(f"💾 Ensemble model info saved to: {ensemble_pth_path}")
    
    # ベースモデルのパスを表示
    if ensemble_model_info['base_model_paths']:
        print("\n🏆 Ensemble base models (best overall):")
        for model_name, model_path in ensemble_model_info['base_model_paths'].items():
            print(f"  - {model_name}: {model_path}")

    return summary


# *******************************************************************
# Adjacent-Tolerant Metrics (隣接クラス許容指標)
# *******************************************************************

def calculate_adjacent_tolerant_metrics(y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        y_prob: Optional[np.ndarray] = None,
                                        n_classes: int = 5,
                                        tolerance: int = 1) -> Dict:
    """
    隣接クラス間の誤分類を許容した精度指標を計算
    
    植生被覆率のような順序性のあるクラス分類では、隣接クラス間の誤分類は
    完全な誤りとは言えない場合がある。この関数は±toleranceクラス以内の
    予測を「許容」として扱う指標を計算する。
    
    Parameters:
    - y_true: 真のラベル (N,)
    - y_pred: 予測ラベル (N,)
    - y_prob: 予測確率 (N, n_classes), オプション
    - n_classes: クラス数
    - tolerance: 許容するクラス差（デフォルト1=隣接クラスまで許容）
    
    Returns:
    - Dict: 各種指標を含む辞書
    """
    n_samples = len(y_true)
    
    # 基本指標（許容なし）
    exact_correct = y_true == y_pred
    exact_accuracy = np.mean(exact_correct)
    
    # 隣接許容正解
    class_diff = np.abs(y_true - y_pred)
    tolerant_correct = class_diff <= tolerance
    tolerant_accuracy = np.mean(tolerant_correct)
    
    # 誤分類の内訳
    adjacent_errors = (class_diff == 1)  # 隣接クラスへの誤分類
    distant_errors = (class_diff > 1)    # 2クラス以上離れた誤分類
    
    n_exact_correct = np.sum(exact_correct)
    n_adjacent_errors = np.sum(adjacent_errors)
    n_distant_errors = np.sum(distant_errors)
    
    # 重み付き精度（距離に応じたペナルティ）
    # 完全正解=1.0, 隣接誤分類=0.5, 2クラス差=0.25, ...
    weights = 1.0 / (1.0 + class_diff)
    weighted_accuracy = np.mean(weights)
    
    # 平均絶対誤差（MAE）- クラス間距離
    mae = np.mean(class_diff)
    
    # 二乗平均平方根誤差（RMSE）
    rmse = np.sqrt(np.mean(class_diff ** 2))
    
    # クラス別の隣接許容精度
    class_tolerant_accuracy = {}
    class_exact_accuracy = {}
    for c in range(n_classes):
        mask = y_true == c
        if np.sum(mask) > 0:
            class_exact_accuracy[c] = float(np.mean(exact_correct[mask]))
            class_tolerant_accuracy[c] = float(np.mean(tolerant_correct[mask]))
    
    # Quadratic Weighted Kappa（順序性を考慮した一致度）
    qwk = _calculate_quadratic_weighted_kappa(y_true, y_pred, n_classes)
    
    # Ordinal Classification Index (OCI) - 順序分類専用指標
    oci = _calculate_ordinal_classification_index(y_true, y_pred, n_classes)
    
    # 境界精度（各閾値境界での分類精度）
    boundary_accuracy = _calculate_boundary_accuracy(y_true, y_pred, n_classes)
    
    metrics = {
        # 基本指標
        'exact_accuracy': float(exact_accuracy),
        'tolerant_accuracy': float(tolerant_accuracy),
        'tolerance_level': tolerance,
        
        # 誤分類内訳
        'n_samples': n_samples,
        'n_exact_correct': int(n_exact_correct),
        'n_adjacent_errors': int(n_adjacent_errors),
        'n_distant_errors': int(n_distant_errors),
        'adjacent_error_rate': float(n_adjacent_errors / n_samples),
        'distant_error_rate': float(n_distant_errors / n_samples),
        
        # 距離ベース指標
        'weighted_accuracy': float(weighted_accuracy),
        'mae': float(mae),
        'rmse': float(rmse),
        
        # 順序分類指標
        'quadratic_weighted_kappa': float(qwk),
        'ordinal_classification_index': float(oci),
        
        # クラス別精度
        'class_exact_accuracy': class_exact_accuracy,
        'class_tolerant_accuracy': class_tolerant_accuracy,
        
        # 境界精度
        'boundary_accuracy': boundary_accuracy,
    }
    
    # 確率がある場合の追加指標
    if y_prob is not None:
        prob_metrics = _calculate_probability_based_metrics(y_true, y_pred, y_prob, n_classes)
        metrics.update(prob_metrics)
    
    return metrics


def _calculate_quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, 
                                        n_classes: int) -> float:
    """
    Quadratic Weighted Kappa (QWK) を計算
    順序性のあるクラス分類の一致度を評価する標準的な指標
    
    QWK = 1 - (Σ w_ij * O_ij) / (Σ w_ij * E_ij)
    w_ij = (i - j)^2 / (N - 1)^2 (二次の重み)
    """
    # 混同行列
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    # 重み行列（二次）
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    
    # 期待度数
    hist_true = np.sum(cm, axis=1)
    hist_pred = np.sum(cm, axis=0)
    n_samples = np.sum(cm)
    
    if n_samples == 0:
        return 0.0
    
    expected = np.outer(hist_true, hist_pred) / n_samples
    
    # QWK計算
    observed_weighted = np.sum(weights * cm)
    expected_weighted = np.sum(weights * expected)
    
    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else 0.0
    
    qwk = 1.0 - (observed_weighted / expected_weighted)
    
    return qwk


def _calculate_ordinal_classification_index(y_true: np.ndarray, y_pred: np.ndarray,
                                            n_classes: int) -> float:
    """
    Ordinal Classification Index (OCI) を計算
    順序分類の性能を0-1で表現する指標
    
    OCI = 1 - MAE / MAE_max
    MAE_max = (n_classes - 1) * 0.5 （ランダム予測の期待MAE近似）
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mae_max = (n_classes - 1) * 0.5  # ランダム予測時の近似期待値
    
    oci = 1.0 - (mae / mae_max) if mae_max > 0 else 1.0
    oci = max(0.0, min(1.0, oci))  # 0-1にクリップ
    
    return oci


def _calculate_boundary_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                 n_classes: int) -> Dict:
    """
    各クラス境界での分類精度を計算
    境界付近（隣接クラス間）でどれだけ正確に分類できているか
    """
    boundary_results = {}
    
    for boundary in range(n_classes - 1):
        # 境界の下側クラス（boundary）と上側クラス（boundary + 1）
        lower_class = boundary
        upper_class = boundary + 1
        
        # 境界に関係するサンプル
        boundary_mask = (y_true == lower_class) | (y_true == upper_class)
        
        if np.sum(boundary_mask) == 0:
            continue
        
        # 境界サンプルでの正確な分類
        boundary_correct = y_true[boundary_mask] == y_pred[boundary_mask]
        boundary_acc = np.mean(boundary_correct)
        
        # 下側→上側への誤分類
        lower_to_upper = np.sum((y_true == lower_class) & (y_pred == upper_class))
        # 上側→下側への誤分類  
        upper_to_lower = np.sum((y_true == upper_class) & (y_pred == lower_class))
        
        n_lower = np.sum(y_true == lower_class)
        n_upper = np.sum(y_true == upper_class)
        
        boundary_results[f'boundary_{lower_class}_{upper_class}'] = {
            'accuracy': float(boundary_acc),
            'n_samples': int(np.sum(boundary_mask)),
            'lower_to_upper_rate': float(lower_to_upper / n_lower) if n_lower > 0 else 0.0,
            'upper_to_lower_rate': float(upper_to_lower / n_upper) if n_upper > 0 else 0.0,
        }
    
    return boundary_results


def _calculate_probability_based_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                         y_prob: np.ndarray, n_classes: int) -> Dict:
    """
    予測確率に基づく追加指標を計算
    """
    # 期待クラス（確率加重平均）
    class_indices = np.arange(n_classes)
    expected_class = np.sum(y_prob * class_indices, axis=1)
    
    # 期待値と真値の差
    expected_mae = np.mean(np.abs(y_true - expected_class))
    
    # 隣接クラスへの確率漏れ
    adjacent_probability_leak = []
    for i, (true_label, probs) in enumerate(zip(y_true, y_prob)):
        # 真のクラスの隣接クラスへの確率
        leak = 0.0
        if true_label > 0:
            leak += probs[true_label - 1]
        if true_label < n_classes - 1:
            leak += probs[true_label + 1]
        adjacent_probability_leak.append(leak)
    
    mean_adjacent_leak = np.mean(adjacent_probability_leak)
    
    # 確信度（最大確率）と距離の関係
    max_probs = np.max(y_prob, axis=1)
    class_diff = np.abs(y_true - y_pred)
    
    # 高確信度での遠距離誤分類
    high_conf_distant_error = np.sum((max_probs > 0.8) & (class_diff > 1))
    
    return {
        'expected_mae': float(expected_mae),
        'mean_adjacent_probability_leak': float(mean_adjacent_leak),
        'high_confidence_distant_errors': int(high_conf_distant_error),
    }


def evaluate_with_adjacent_tolerance(predictions_df: pd.DataFrame,
                                     class_names: List[str],
                                     save_dir: Path,
                                     model_name: str = "Model",
                                     tolerance: int = 1) -> Dict:
    """
    隣接クラス許容指標による評価を実行し、結果を保存
    
    Parameters:
    - predictions_df: 予測結果のDataFrame
    - class_names: クラス名リスト
    - save_dir: 保存先
    - model_name: モデル名
    - tolerance: 許容クラス差
    
    Returns:
    - Dict: 評価結果
    """
    save_dir = Path(save_dir)
    tolerant_dir = save_dir / 'adjacent_tolerant_metrics'
    tolerant_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📏 Adjacent-Tolerant Metrics Evaluation - {model_name}")
    print(f"   Tolerance Level: ±{tolerance} class(es)")
    print(f"{'='*70}")
    
    # データ取得
    y_true = predictions_df['true_label'].values
    y_pred = predictions_df['pred_label'].values if 'pred_label' in predictions_df.columns \
             else predictions_df['predicted_class'].values
    
    prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
    y_prob = predictions_df[prob_cols].values if prob_cols else None
    
    n_classes = len(class_names)
    
    # 指標計算
    metrics = calculate_adjacent_tolerant_metrics(
        y_true, y_pred, y_prob, n_classes, tolerance
    )
    metrics['model_name'] = model_name
    metrics['class_names'] = class_names
    
    # 結果表示
    print(f"\n📊 Results:")
    print(f"  Exact Accuracy:     {metrics['exact_accuracy']*100:.2f}%")
    print(f"  Tolerant Accuracy:  {metrics['tolerant_accuracy']*100:.2f}% (±{tolerance} class)")
    print(f"  Improvement:        +{(metrics['tolerant_accuracy'] - metrics['exact_accuracy'])*100:.2f}%")
    print(f"\n  Adjacent Errors:    {metrics['n_adjacent_errors']:,} ({metrics['adjacent_error_rate']*100:.2f}%)")
    print(f"  Distant Errors:     {metrics['n_distant_errors']:,} ({metrics['distant_error_rate']*100:.2f}%)")
    print(f"\n  Weighted Accuracy:  {metrics['weighted_accuracy']*100:.2f}%")
    print(f"  MAE:                {metrics['mae']:.3f}")
    print(f"  RMSE:               {metrics['rmse']:.3f}")
    print(f"\n  Quadratic Weighted Kappa: {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  Ordinal Classification Index: {metrics['ordinal_classification_index']:.4f}")
    
    # 可視化
    _plot_adjacent_tolerant_metrics(metrics, class_names, tolerant_dir, model_name)
    _plot_error_distance_distribution(y_true, y_pred, tolerant_dir, model_name)
    _plot_boundary_accuracy_chart(metrics['boundary_accuracy'], class_names, tolerant_dir, model_name)
    
    # 結果保存
    # JSON保存
    with open(tolerant_dir / 'adjacent_tolerant_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(to_serializable(metrics), f, indent=2, ensure_ascii=False)
    
    # CSV保存（サマリー）
    summary_df = pd.DataFrame([{
        'Model': model_name,
        'Exact_Accuracy': metrics['exact_accuracy'],
        'Tolerant_Accuracy': metrics['tolerant_accuracy'],
        'Tolerance': tolerance,
        'Adjacent_Errors': metrics['n_adjacent_errors'],
        'Distant_Errors': metrics['n_distant_errors'],
        'Weighted_Accuracy': metrics['weighted_accuracy'],
        'MAE': metrics['mae'],
        'RMSE': metrics['rmse'],
        'QWK': metrics['quadratic_weighted_kappa'],
        'OCI': metrics['ordinal_classification_index'],
    }])
    summary_df.to_csv(tolerant_dir / 'metrics_summary.csv', index=False)
    
    # クラス別精度のCSV
    class_df = pd.DataFrame({
        'Class': class_names,
        'Exact_Accuracy': [metrics['class_exact_accuracy'].get(i, np.nan) for i in range(n_classes)],
        'Tolerant_Accuracy': [metrics['class_tolerant_accuracy'].get(i, np.nan) for i in range(n_classes)],
    })
    class_df.to_csv(tolerant_dir / 'class_accuracy.csv', index=False)
    
    # テキストレポート
    _generate_tolerant_metrics_report(metrics, class_names, tolerant_dir / 'report.txt')
    
    print(f"\n💾 Results saved to: {tolerant_dir}")
    
    return metrics


def _plot_adjacent_tolerant_metrics(metrics: Dict, class_names: List[str],
                                    save_dir: Path, model_name: str):
    """隣接許容指標の可視化"""
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('black')
    
    # 1. 精度比較（Exact vs Tolerant）
    ax1 = axes[0, 0]
    ax1.set_facecolor('black')
    
    accuracy_types = ['Exact\nAccuracy', 'Tolerant\nAccuracy', 'Weighted\nAccuracy']
    accuracy_values = [
        metrics['exact_accuracy'] * 100,
        metrics['tolerant_accuracy'] * 100,
        metrics['weighted_accuracy'] * 100
    ]
    colors = [palette['magenta'], palette['teal'], palette['ice']]
    
    bars = ax1.bar(accuracy_types, accuracy_values, color=colors, alpha=0.8, edgecolor=palette['cream'])
    ax1.set_ylabel('Accuracy (%)', color=palette['cream'])
    ax1.set_title(f'{model_name} - Accuracy Comparison', color=palette['cream'])
    ax1.set_ylim(0, 100)
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # 値をバーの上に表示
    for bar, val in zip(bars, accuracy_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', color=palette['cream'], fontsize=11)
    
    # 2. 誤分類の内訳（円グラフ）
    ax2 = axes[0, 1]
    ax2.set_facecolor('black')
    
    sizes = [metrics['n_exact_correct'], metrics['n_adjacent_errors'], metrics['n_distant_errors']]
    labels = ['Correct', 'Adjacent\nError', 'Distant\nError']
    colors_pie = [palette['teal'], palette['ice'], palette['magenta']]
    explode = (0, 0.05, 0.1)
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                       explode=explode, autopct='%1.1f%%',
                                       startangle=90, textprops={'color': palette['cream']})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    ax2.set_title(f'{model_name} - Error Classification', color=palette['cream'])
    
    # 3. クラス別精度比較
    ax3 = axes[1, 0]
    ax3.set_facecolor('black')
    
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.35
    
    exact_acc = [metrics['class_exact_accuracy'].get(i, 0) * 100 for i in range(n_classes)]
    tolerant_acc = [metrics['class_tolerant_accuracy'].get(i, 0) * 100 for i in range(n_classes)]
    
    bars1 = ax3.bar(x - width/2, exact_acc, width, label='Exact', color=palette['magenta'], alpha=0.8)
    bars2 = ax3.bar(x + width/2, tolerant_acc, width, label='Tolerant', color=palette['teal'], alpha=0.8)
    
    ax3.set_xlabel('Class', color=palette['cream'])
    ax3.set_ylabel('Accuracy (%)', color=palette['cream'])
    ax3.set_title(f'{model_name} - Class-wise Accuracy', color=palette['cream'])
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax3.set_ylim(0, 100)
    ax3.tick_params(colors=palette['cream'])
    for spine in ax3.spines.values():
        spine.set_color(palette['cream'])
    
    # 4. 順序分類指標
    ax4 = axes[1, 1]
    ax4.set_facecolor('black')
    
    ordinal_metrics = ['QWK', 'OCI', '1-MAE/4', '1-RMSE/4']
    ordinal_values = [
        metrics['quadratic_weighted_kappa'],
        metrics['ordinal_classification_index'],
        1 - metrics['mae'] / (n_classes - 1),
        1 - metrics['rmse'] / (n_classes - 1)
    ]
    
    bars = ax4.barh(ordinal_metrics, ordinal_values, color=palette['deep_teal'], alpha=0.8, edgecolor=palette['cream'])
    ax4.set_xlim(0, 1)
    ax4.set_xlabel('Score (0-1)', color=palette['cream'])
    ax4.set_title(f'{model_name} - Ordinal Classification Metrics', color=palette['cream'])
    ax4.tick_params(colors=palette['cream'])
    for spine in ax4.spines.values():
        spine.set_color(palette['cream'])
    
    # 値を表示
    for bar, val in zip(bars, ordinal_values):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', color=palette['cream'], fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'adjacent_tolerant_metrics.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def _plot_error_distance_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                                      save_dir: Path, model_name: str):
    """誤分類距離の分布を可視化"""
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    class_diff = y_true - y_pred  # 符号付き差分
    abs_diff = np.abs(class_diff)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('black')
    
    # 1. 絶対距離の分布
    ax1 = axes[0]
    ax1.set_facecolor('black')
    
    max_diff = int(np.max(abs_diff)) if len(abs_diff) > 0 else 4
    bins = np.arange(-0.5, max_diff + 1.5, 1)
    counts, _, patches = ax1.hist(abs_diff, bins=bins, color=palette['teal'], alpha=0.8, edgecolor=palette['cream'])
    
    # 色分け（距離に応じて）
    for i, patch in enumerate(patches):
        if i == 0:
            patch.set_facecolor(palette['teal'])  # 正解
        elif i == 1:
            patch.set_facecolor(palette['ice'])   # 隣接誤分類
        else:
            patch.set_facecolor(palette['magenta'])  # 遠距離誤分類
    
    ax1.set_xlabel('Absolute Class Distance', color=palette['cream'])
    ax1.set_ylabel('Count', color=palette['cream'])
    ax1.set_title(f'{model_name} - Error Distance Distribution', color=palette['cream'])
    ax1.set_xticks(range(max_diff + 1))
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # 2. 符号付き距離の分布（過大評価 vs 過小評価）
    ax2 = axes[1]
    ax2.set_facecolor('black')
    
    # 誤分類のみ
    error_diff = class_diff[class_diff != 0]
    
    if len(error_diff) > 0:
        bins = np.arange(-4.5, 5.5, 1)
        n, bins_out, patches = ax2.hist(error_diff, bins=bins, color=palette['ice'], alpha=0.8, edgecolor=palette['cream'])
        
        # 色分け
        for i, patch in enumerate(patches):
            bin_center = (bins_out[i] + bins_out[i+1]) / 2
            if bin_center < 0:
                patch.set_facecolor(palette['magenta'])  # 過小評価
            else:
                patch.set_facecolor(palette['ice'])  # 過大評価
    
    ax2.axvline(x=0, color=palette['cream'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Signed Class Difference (Pred - True)', color=palette['cream'])
    ax2.set_ylabel('Count', color=palette['cream'])
    ax2.set_title(f'{model_name} - Error Direction\n(Left=Underestimate, Right=Overestimate)', color=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_distance_distribution.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def _plot_boundary_accuracy_chart(boundary_results: Dict, class_names: List[str],
                                  save_dir: Path, model_name: str):
    """境界精度の可視化"""
    if not boundary_results:
        return
    
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('black')
    
    boundaries = list(boundary_results.keys())
    accuracies = [boundary_results[b]['accuracy'] * 100 for b in boundaries]
    n_samples = [boundary_results[b]['n_samples'] for b in boundaries]
    
    # 境界ラベルを読みやすく
    boundary_labels = []
    for b in boundaries:
        parts = b.split('_')
        if len(parts) >= 3:
            lower_idx = int(parts[1])
            upper_idx = int(parts[2])
            if lower_idx < len(class_names) and upper_idx < len(class_names):
                boundary_labels.append(f'{class_names[lower_idx]}\n↔\n{class_names[upper_idx]}')
            else:
                boundary_labels.append(b)
        else:
            boundary_labels.append(b)
    
    # 1. 境界精度
    ax1 = axes[0]
    ax1.set_facecolor('black')
    
    x = np.arange(len(boundaries))
    bars = ax1.bar(x, accuracies, color=palette['teal'], alpha=0.8, edgecolor=palette['cream'])
    
    ax1.set_xlabel('Class Boundary', color=palette['cream'])
    ax1.set_ylabel('Accuracy (%)', color=palette['cream'])
    ax1.set_title(f'{model_name} - Boundary Accuracy', color=palette['cream'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(boundary_labels, fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # 値を表示
    for bar, val, n in zip(bars, accuracies, n_samples):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%\n(n={n})', ha='center', va='bottom', 
                color=palette['cream'], fontsize=8)
    
    # 2. 境界での誤分類方向
    ax2 = axes[1]
    ax2.set_facecolor('black')
    
    lower_to_upper = [boundary_results[b]['lower_to_upper_rate'] * 100 for b in boundaries]
    upper_to_lower = [boundary_results[b]['upper_to_lower_rate'] * 100 for b in boundaries]
    
    width = 0.35
    bars1 = ax2.bar(x - width/2, lower_to_upper, width, label='Lower→Upper', 
                   color=palette['ice'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, upper_to_lower, width, label='Upper→Lower',
                   color=palette['magenta'], alpha=0.8)
    
    ax2.set_xlabel('Class Boundary', color=palette['cream'])
    ax2.set_ylabel('Misclassification Rate (%)', color=palette['cream'])
    ax2.set_title(f'{model_name} - Cross-Boundary Errors', color=palette['cream'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(boundary_labels, fontsize=8)
    ax2.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'boundary_accuracy.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def _generate_tolerant_metrics_report(metrics: Dict, class_names: List[str], save_path: Path):
    """隣接許容指標のテキストレポートを生成"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"隣接クラス許容精度評価レポート - {metrics['model_name']}\n")
        f.write("Adjacent-Tolerant Metrics Evaluation Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"許容レベル / Tolerance Level: ±{metrics['tolerance_level']} class(es)\n\n")
        
        f.write("【精度指標 / Accuracy Metrics】\n")
        f.write(f"  厳密精度 (Exact Accuracy):       {metrics['exact_accuracy']*100:.2f}%\n")
        f.write(f"  許容精度 (Tolerant Accuracy):    {metrics['tolerant_accuracy']*100:.2f}%\n")
        f.write(f"  精度向上幅:                      +{(metrics['tolerant_accuracy']-metrics['exact_accuracy'])*100:.2f}%\n")
        f.write(f"  重み付き精度 (Weighted):         {metrics['weighted_accuracy']*100:.2f}%\n\n")
        
        f.write("【誤分類内訳 / Error Breakdown】\n")
        f.write(f"  総サンプル数:     {metrics['n_samples']:,}\n")
        f.write(f"  正解数:           {metrics['n_exact_correct']:,}\n")
        f.write(f"  隣接誤分類:       {metrics['n_adjacent_errors']:,} ({metrics['adjacent_error_rate']*100:.2f}%)\n")
        f.write(f"  遠距離誤分類:     {metrics['n_distant_errors']:,} ({metrics['distant_error_rate']*100:.2f}%)\n\n")
        
        f.write("【距離ベース指標 / Distance-Based Metrics】\n")
        f.write(f"  平均絶対誤差 (MAE):              {metrics['mae']:.4f}\n")
        f.write(f"  二乗平均平方根誤差 (RMSE):       {metrics['rmse']:.4f}\n\n")
        
        f.write("【順序分類指標 / Ordinal Classification Metrics】\n")
        f.write(f"  Quadratic Weighted Kappa (QWK):  {metrics['quadratic_weighted_kappa']:.4f}\n")
        f.write(f"  Ordinal Classification Index:   {metrics['ordinal_classification_index']:.4f}\n\n")
        
        f.write("【クラス別精度 / Class-wise Accuracy】\n")
        f.write(f"  {'Class':<20} {'Exact':>10} {'Tolerant':>10} {'Improve':>10}\n")
        f.write("  " + "-" * 52 + "\n")
        for i, name in enumerate(class_names):
            exact = metrics['class_exact_accuracy'].get(i, 0) * 100
            tolerant = metrics['class_tolerant_accuracy'].get(i, 0) * 100
            improve = tolerant - exact
            f.write(f"  {name:<20} {exact:>9.1f}% {tolerant:>9.1f}% {improve:>+9.1f}%\n")
        f.write("\n")
        
        f.write("【境界精度 / Boundary Accuracy】\n")
        for boundary, data in metrics['boundary_accuracy'].items():
            parts = boundary.split('_')
            if len(parts) >= 3:
                lower_idx = int(parts[1])
                upper_idx = int(parts[2])
                if lower_idx < len(class_names) and upper_idx < len(class_names):
                    boundary_name = f"{class_names[lower_idx]} ↔ {class_names[upper_idx]}"
                else:
                    boundary_name = boundary
            else:
                boundary_name = boundary
            f.write(f"  {boundary_name}: {data['accuracy']*100:.1f}% (n={data['n_samples']})\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by class5-v9.py - Adjacent-Tolerant Metrics Module\n")


def run_adjacent_tolerant_evaluation_for_all_models(results: Dict, config: Dict,
                                                    tolerance: int = 1) -> Dict:
    """
    全モデルに対して隣接許容指標評価を実行
    
    Parameters:
    - results: 各モデルの結果辞書
    - config: 設定辞書
    - tolerance: 許容クラス差
    
    Returns:
    - Dict: 全モデルの評価結果
    """
    print("\n" + "=" * 80)
    print("📏 Running Adjacent-Tolerant Evaluation for All Models")
    print(f"   Tolerance: ±{tolerance} class(es)")
    print("=" * 80)
    
    class_names = generate_class_names_from_thresholds(
        config.get('fixed_thresholds'), config['n_classes']
    )
    
    all_evaluations = {}
    comparison_data = []
    
    for model_name, model_result in results.items():
        cv_results = model_result.get('cv_results', {})
        predictions_path = cv_results.get('predictions_path')
        
        if not predictions_path or not os.path.exists(predictions_path):
            print(f"⚠️  Skipping {model_name}: predictions file not found")
            continue
        
        try:
            predictions_df = pd.read_csv(predictions_path)
            
            # pred_labelがない場合
            if 'pred_label' not in predictions_df.columns:
                prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
                if prob_cols:
                    predictions_df['pred_label'] = predictions_df[prob_cols].values.argmax(axis=1)
                elif 'predicted_class' in predictions_df.columns:
                    predictions_df['pred_label'] = predictions_df['predicted_class']
                else:
                    print(f"⚠️  Skipping {model_name}: cannot determine predicted labels")
                    continue
            
            model_save_dir = Path(config['output_dir']) / model_name
            
            evaluation = evaluate_with_adjacent_tolerance(
                predictions_df, class_names, model_save_dir, model_name, tolerance
            )
            
            all_evaluations[model_name] = evaluation
            
            comparison_data.append({
                'Model': model_name,
                'Exact_Accuracy': evaluation['exact_accuracy'],
                'Tolerant_Accuracy': evaluation['tolerant_accuracy'],
                'Improvement': evaluation['tolerant_accuracy'] - evaluation['exact_accuracy'],
                'Adjacent_Error_Rate': evaluation['adjacent_error_rate'],
                'Distant_Error_Rate': evaluation['distant_error_rate'],
                'MAE': evaluation['mae'],
                'RMSE': evaluation['rmse'],
                'QWK': evaluation['quadratic_weighted_kappa'],
                'OCI': evaluation['ordinal_classification_index'],
            })
            
        except Exception as e:
            print(f"⚠️  Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 全モデル比較を保存
    if comparison_data:
        comparison_dir = Path(config['output_dir']) / 'adjacent_tolerant_comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Tolerant_Accuracy', ascending=False)
        comparison_df.to_csv(comparison_dir / 'model_comparison.csv', index=False)
        
        # 比較可視化
        _plot_tolerant_comparison(comparison_df, comparison_dir)
        
        print(f"\n📊 Comparison saved to: {comparison_dir}")
    
    return all_evaluations


def _plot_tolerant_comparison(df: pd.DataFrame, save_dir: Path):
    """全モデルの隣接許容精度比較を可視化"""
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('black')
    
    # 1. Exact vs Tolerant Accuracy
    ax1 = axes[0, 0]
    ax1.set_facecolor('black')
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.barh(x - width/2, df['Exact_Accuracy'] * 100, width, 
                    label='Exact', color=palette['magenta'], alpha=0.8)
    bars2 = ax1.barh(x + width/2, df['Tolerant_Accuracy'] * 100, width,
                    label='Tolerant', color=palette['teal'], alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(df['Model'])
    ax1.set_xlabel('Accuracy (%)', color=palette['cream'])
    ax1.set_title('Exact vs Tolerant Accuracy', color=palette['cream'])
    ax1.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # 2. Improvement by tolerance
    ax2 = axes[0, 1]
    ax2.set_facecolor('black')
    
    bars = ax2.barh(df['Model'], df['Improvement'] * 100, color=palette['ice'], alpha=0.8)
    ax2.set_xlabel('Accuracy Improvement (%)', color=palette['cream'])
    ax2.set_title('Improvement by Adjacent Tolerance', color=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    
    # 3. QWK comparison
    ax3 = axes[1, 0]
    ax3.set_facecolor('black')
    
    bars = ax3.barh(df['Model'], df['QWK'], color=palette['deep_teal'], alpha=0.8)
    ax3.set_xlabel('Quadratic Weighted Kappa', color=palette['cream'])
    ax3.set_title('Quadratic Weighted Kappa (Ordinal Agreement)', color=palette['cream'])
    ax3.set_xlim(0, 1)
    ax3.tick_params(colors=palette['cream'])
    for spine in ax3.spines.values():
        spine.set_color(palette['cream'])
    
    # 4. MAE comparison
    ax4 = axes[1, 1]
    ax4.set_facecolor('black')
    
    bars = ax4.barh(df['Model'], df['MAE'], color=palette['mist'], alpha=0.8)
    ax4.set_xlabel('Mean Absolute Error (classes)', color=palette['cream'])
    ax4.set_title('MAE - Lower is Better', color=palette['cream'])
    ax4.tick_params(colors=palette['cream'])
    for spine in ax4.spines.values():
        spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'tolerant_comparison.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


# *******************************************************************
# 誤分類の生態学的解釈モジュール
# Ecological Interpretation of Misclassification Module
# *******************************************************************

def analyze_misclassification_patterns(predictions_df: pd.DataFrame, 
                                       class_names: List[str],
                                       save_dir: Path,
                                       model_name: str = "Model") -> Dict:
    """
    誤分類パターンを詳細に解析する関数
    
    Parameters:
    - predictions_df: 予測結果のDataFrame（true_label, pred_label, prob_class_*を含む）
    - class_names: クラス名のリスト
    - save_dir: 保存先ディレクトリ
    - model_name: モデル名
    
    Returns:
    - Dict: 誤分類解析結果
    """
    save_dir = Path(save_dir)
    misclass_dir = save_dir / 'misclassification_analysis'
    misclass_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🔬 Misclassification Ecological Analysis - {model_name}")
    print(f"{'='*70}")
    
    # 基本情報
    y_true = predictions_df['true_label'].values
    y_pred = predictions_df['pred_label'].values if 'pred_label' in predictions_df.columns else predictions_df['predicted_class'].values
    
    # 確率列を取得
    prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
    y_prob = predictions_df[prob_cols].values if prob_cols else None
    
    # 誤分類されたサンプルのインデックス
    misclassified_mask = y_true != y_pred
    misclassified_idx = np.where(misclassified_mask)[0]
    
    n_total = len(y_true)
    n_misclassified = len(misclassified_idx)
    misclass_rate = n_misclassified / n_total * 100
    
    print(f"\n📊 Overview:")
    print(f"  Total samples: {n_total:,}")
    print(f"  Misclassified: {n_misclassified:,} ({misclass_rate:.2f}%)")
    
    # 隣接許容指標も計算
    n_classes = len(class_names)
    tolerant_metrics = calculate_adjacent_tolerant_metrics(y_true, y_pred, y_prob, n_classes)
    print(f"\n📏 Adjacent-Tolerant Metrics:")
    print(f"  Exact Accuracy:     {tolerant_metrics['exact_accuracy']*100:.2f}%")
    print(f"  Tolerant Accuracy:  {tolerant_metrics['tolerant_accuracy']*100:.2f}% (±1 class)")
    print(f"  Adjacent Errors:    {tolerant_metrics['n_adjacent_errors']:,} ({tolerant_metrics['adjacent_error_rate']*100:.2f}%)")
    print(f"  Distant Errors:     {tolerant_metrics['n_distant_errors']:,} ({tolerant_metrics['distant_error_rate']*100:.2f}%)")
    print(f"  QWK:                {tolerant_metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  MAE:                {tolerant_metrics['mae']:.3f}")
    
    # 1. クラス間誤分類マトリクス（遷移パターン解析）
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 誤分類ペアの抽出
    misclass_pairs = []
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j:
                rate = cm_normalized[i, j]
                count = cm[i, j]
                if count > 0:
                    misclass_pairs.append({
                        'true_class': true_class,
                        'true_class_idx': i,
                        'predicted_class': pred_class,
                        'predicted_class_idx': j,
                        'count': int(count),
                        'rate': float(rate),
                        'transition_direction': 'overestimate' if j > i else 'underestimate'
                    })
    
    # 誤分類率でソート
    misclass_pairs = sorted(misclass_pairs, key=lambda x: x['rate'], reverse=True)
    
    print(f"\n📈 Top Misclassification Patterns (rate > 5%):")
    for pair in misclass_pairs[:10]:
        if pair['rate'] > 0.05:
            direction = "↑" if pair['transition_direction'] == 'overestimate' else "↓"
            print(f"  {pair['true_class']} → {pair['predicted_class']}: {pair['count']:,} samples ({pair['rate']*100:.1f}%) {direction}")
    
    # 2. 確信度別誤分類解析
    confidence_analysis = None
    if y_prob is not None:
        max_probs = np.max(y_prob, axis=1)
        
        # 確信度分布（正分類 vs 誤分類）
        correct_confidences = max_probs[~misclassified_mask]
        wrong_confidences = max_probs[misclassified_mask]
        
        confidence_analysis = {
            'correct_mean': float(np.mean(correct_confidences)),
            'correct_std': float(np.std(correct_confidences)),
            'wrong_mean': float(np.mean(wrong_confidences)),
            'wrong_std': float(np.std(wrong_confidences)),
            'high_confidence_errors': int(np.sum((max_probs > 0.8) & misclassified_mask)),
            'low_confidence_errors': int(np.sum((max_probs < 0.5) & misclassified_mask))
        }
        
        print(f"\n📊 Confidence Analysis:")
        print(f"  Correct predictions - Mean confidence: {confidence_analysis['correct_mean']:.3f} ± {confidence_analysis['correct_std']:.3f}")
        print(f"  Wrong predictions   - Mean confidence: {confidence_analysis['wrong_mean']:.3f} ± {confidence_analysis['wrong_std']:.3f}")
        print(f"  High-confidence errors (>0.8): {confidence_analysis['high_confidence_errors']:,}")
        print(f"  Low-confidence errors (<0.5): {confidence_analysis['low_confidence_errors']:,}")
        
        # 確信度分布の可視化
        _plot_confidence_distribution(correct_confidences, wrong_confidences, 
                                     misclass_dir / 'confidence_distribution.png', model_name)
    
    # 3. 閾値境界解析（生態学的解釈の核心）
    boundary_analysis = _analyze_threshold_boundaries(
        predictions_df, class_names, misclass_dir, model_name
    )
    
    # 4. 誤分類遷移図の作成
    _plot_misclassification_flow(cm, class_names, misclass_dir / 'misclassification_flow.png', model_name)
    
    # 5. クラス別誤分類率の詳細
    class_analysis = []
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_total = np.sum(class_mask)
        if class_total > 0:
            class_correct = np.sum((y_true == i) & (y_pred == i))
            class_accuracy = class_correct / class_total
            
            # どのクラスに誤分類されやすいか
            error_distribution = {}
            for j, target_class in enumerate(class_names):
                if i != j:
                    error_count = np.sum((y_true == i) & (y_pred == j))
                    if error_count > 0:
                        error_distribution[target_class] = int(error_count)
            
            class_analysis.append({
                'class': class_name,
                'class_idx': i,
                'total': int(class_total),
                'correct': int(class_correct),
                'accuracy': float(class_accuracy),
                'error_distribution': error_distribution
            })
    
    # 6. 生態学的解釈の生成
    ecological_interpretation = _generate_ecological_interpretation(
        misclass_pairs, class_names, class_analysis
    )
    
    # 結果のまとめ
    analysis_result = {
        'model_name': model_name,
        'total_samples': n_total,
        'misclassified_samples': n_misclassified,
        'misclassification_rate': misclass_rate,
        'misclassification_pairs': misclass_pairs,
        'confidence_analysis': confidence_analysis,
        'boundary_analysis': boundary_analysis,
        'class_analysis': class_analysis,
        'ecological_interpretation': ecological_interpretation,
        # 隣接許容指標を追加
        'adjacent_tolerant_metrics': tolerant_metrics
    }
    
    # JSONとして保存
    with open(misclass_dir / 'misclassification_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(to_serializable(analysis_result), f, indent=2, ensure_ascii=False)
    
    # 隣接許容指標の詳細評価と保存
    evaluate_with_adjacent_tolerance(predictions_df, class_names, save_dir, model_name)
    
    # サマリーレポートの生成
    _generate_summary_report(analysis_result, misclass_dir / 'summary_report.txt', class_names)
    
    print(f"\n💾 Analysis results saved to: {misclass_dir}")
    
    return analysis_result


def _plot_confidence_distribution(correct_conf: np.ndarray, wrong_conf: np.ndarray,
                                  save_path: Path, model_name: str):
    """確信度分布の可視化"""
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('black')
    
    # ヒストグラム
    ax1 = axes[0]
    ax1.set_facecolor('black')
    ax1.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color=palette['teal'], density=True)
    ax1.hist(wrong_conf, bins=30, alpha=0.7, label='Misclassified', color=palette['magenta'], density=True)
    ax1.set_xlabel('Confidence', color=palette['cream'])
    ax1.set_ylabel('Density', color=palette['cream'])
    ax1.set_title(f'{model_name} - Confidence Distribution', color=palette['cream'])
    ax1.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # 箱ひげ図
    ax2 = axes[1]
    ax2.set_facecolor('black')
    bp = ax2.boxplot([correct_conf, wrong_conf], labels=['Correct', 'Misclassified'],
                     patch_artist=True)
    colors = [palette['teal'], palette['magenta']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color(palette['cream'])
    ax2.set_ylabel('Confidence', color=palette['cream'])
    ax2.set_title(f'{model_name} - Confidence by Prediction Result', color=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def _analyze_threshold_boundaries(predictions_df: pd.DataFrame, class_names: List[str],
                                  save_dir: Path, model_name: str) -> Dict:
    """
    閾値境界での誤分類を解析
    連続値（raw_label）がある場合、閾値付近での誤分類パターンを分析
    """
    if 'raw_label' not in predictions_df.columns:
        return {'note': 'raw_label not available for boundary analysis'}
    
    raw_labels = predictions_df['raw_label'].values
    y_true = predictions_df['true_label'].values
    y_pred = predictions_df['pred_label'].values if 'pred_label' in predictions_df.columns else predictions_df['predicted_class'].values
    
    # 閾値（0%, 25%, 50%, 75%）付近のサンプルを解析
    thresholds = [0.0, 0.25, 0.50, 0.75, 1.0]
    boundary_width = 0.05  # 閾値から±5%の範囲
    
    boundary_results = []
    
    for i, threshold in enumerate(thresholds[:-1]):
        # 閾値付近のサンプル
        lower_bound = threshold - boundary_width
        upper_bound = threshold + boundary_width
        
        # 下側のサンプル（閾値未満）
        below_mask = (raw_labels >= lower_bound) & (raw_labels < threshold)
        # 上側のサンプル（閾値以上）
        above_mask = (raw_labels >= threshold) & (raw_labels <= upper_bound)
        
        below_samples = np.sum(below_mask)
        above_samples = np.sum(above_mask)
        
        if below_samples > 0:
            below_misclass = np.sum((y_true != y_pred) & below_mask)
            below_error_rate = below_misclass / below_samples
        else:
            below_error_rate = 0.0
            below_misclass = 0
            
        if above_samples > 0:
            above_misclass = np.sum((y_true != y_pred) & above_mask)
            above_error_rate = above_misclass / above_samples
        else:
            above_error_rate = 0.0
            above_misclass = 0
        
        boundary_results.append({
            'threshold': float(threshold),
            'threshold_pct': int(threshold * 100),
            'below_samples': int(below_samples),
            'below_misclassified': int(below_misclass),
            'below_error_rate': float(below_error_rate),
            'above_samples': int(above_samples),
            'above_misclassified': int(above_misclass),
            'above_error_rate': float(above_error_rate),
            'total_boundary_samples': int(below_samples + above_samples),
            'total_boundary_errors': int(below_misclass + above_misclass)
        })
    
    # 境界解析の可視化
    _plot_boundary_analysis(boundary_results, raw_labels, y_true, y_pred,
                           save_dir / 'boundary_analysis.png', model_name)
    
    # 連続値に対する誤分類分布
    _plot_raw_label_error_distribution(raw_labels, y_true, y_pred,
                                       save_dir / 'raw_label_error_distribution.png', model_name)
    
    return {
        'boundary_results': boundary_results,
        'boundary_width': boundary_width,
        'total_boundary_samples': sum(br['total_boundary_samples'] for br in boundary_results),
        'total_boundary_errors': sum(br['total_boundary_errors'] for br in boundary_results)
    }


def _plot_boundary_analysis(boundary_results: List[Dict], raw_labels: np.ndarray,
                           y_true: np.ndarray, y_pred: np.ndarray,
                           save_path: Path, model_name: str):
    """閾値境界での誤分類率を可視化"""
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('black')
    
    # 左図：閾値別誤分類率
    ax1 = axes[0]
    ax1.set_facecolor('black')
    
    thresholds = [br['threshold_pct'] for br in boundary_results]
    below_rates = [br['below_error_rate'] * 100 for br in boundary_results]
    above_rates = [br['above_error_rate'] * 100 for br in boundary_results]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, below_rates, width, label='Below threshold', color=palette['ice'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, above_rates, width, label='Above threshold', color=palette['magenta'], alpha=0.8)
    
    ax1.set_xlabel('Threshold (%)', color=palette['cream'])
    ax1.set_ylabel('Error Rate (%)', color=palette['cream'])
    ax1.set_title(f'{model_name} - Error Rate near Thresholds', color=palette['cream'])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{t}%' for t in thresholds])
    ax1.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # 右図：連続値と誤分類の関係
    ax2 = axes[1]
    ax2.set_facecolor('black')
    
    misclassified = y_true != y_pred
    
    # ビン別誤分類率
    bins = np.linspace(0, 1, 21)  # 5%刻み
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_error_rates = []
    
    for i in range(len(bins) - 1):
        mask = (raw_labels >= bins[i]) & (raw_labels < bins[i+1])
        if np.sum(mask) > 0:
            error_rate = np.sum(misclassified & mask) / np.sum(mask) * 100
        else:
            error_rate = 0
        bin_error_rates.append(error_rate)
    
    ax2.bar(bin_centers * 100, bin_error_rates, width=4.5, color=palette['teal'], alpha=0.8, edgecolor=palette['cream'])
    
    # 閾値ラインを追加
    for threshold in [0, 25, 50, 75]:
        ax2.axvline(x=threshold, color=palette['magenta'], linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Raw Label Value (%)', color=palette['cream'])
    ax2.set_ylabel('Error Rate (%)', color=palette['cream'])
    ax2.set_title(f'{model_name} - Error Rate by Raw Label', color=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def _plot_raw_label_error_distribution(raw_labels: np.ndarray, y_true: np.ndarray,
                                       y_pred: np.ndarray, save_path: Path, model_name: str):
    """連続値における誤分類サンプルの分布を可視化"""
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    misclassified = y_true != y_pred
    correct = ~misclassified
    
    # KDE（密度推定）プロット
    correct_labels = raw_labels[correct]
    wrong_labels = raw_labels[misclassified]
    
    if len(correct_labels) > 10:
        kde_correct = gaussian_kde(correct_labels)
        x_range = np.linspace(0, 1, 200)
        ax.fill_between(x_range * 100, kde_correct(x_range), alpha=0.5, 
                       label='Correct', color=palette['teal'])
    
    if len(wrong_labels) > 10:
        kde_wrong = gaussian_kde(wrong_labels)
        x_range = np.linspace(0, 1, 200)
        ax.fill_between(x_range * 100, kde_wrong(x_range), alpha=0.5,
                       label='Misclassified', color=palette['magenta'])
    
    # 閾値ラインを追加
    for threshold in [0, 25, 50, 75]:
        ax.axvline(x=threshold, color=palette['cream'], linestyle='--', alpha=0.5, linewidth=1)
        ax.text(threshold + 1, ax.get_ylim()[1] * 0.95, f'{threshold}%', 
               color=palette['cream'], fontsize=9)
    
    ax.set_xlabel('Raw Label Value (% Vegetation)', color=palette['cream'])
    ax.set_ylabel('Density', color=palette['cream'])
    ax.set_title(f'{model_name} - Distribution of Samples by Raw Label', color=palette['cream'])
    ax.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax.tick_params(colors=palette['cream'])
    for spine in ax.spines.values():
        spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def _plot_misclassification_flow(cm: np.ndarray, class_names: List[str],
                                 save_path: Path, model_name: str):
    """誤分類フロー（Sankey風）の可視化"""
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    n_classes = len(class_names)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # 正規化した混同行列
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # ノードの位置
    left_x = 0.2
    right_x = 0.8
    
    # y位置の計算
    y_positions = np.linspace(0.9, 0.1, n_classes)
    
    # ノードの描画
    node_height = 0.08
    for i, (class_name, y) in enumerate(zip(class_names, y_positions)):
        # 左側（真のラベル）
        rect_left = plt.Rectangle((left_x - 0.05, y - node_height/2), 0.1, node_height,
                                  facecolor=palette['teal'], alpha=0.8, edgecolor=palette['cream'])
        ax.add_patch(rect_left)
        ax.text(left_x - 0.15, y, class_name, ha='right', va='center', 
               color=palette['cream'], fontsize=10, fontweight='bold')
        
        # 右側（予測ラベル）
        rect_right = plt.Rectangle((right_x - 0.05, y - node_height/2), 0.1, node_height,
                                   facecolor=palette['ice'], alpha=0.8, edgecolor=palette['cream'])
        ax.add_patch(rect_right)
        ax.text(right_x + 0.15, y, class_name, ha='left', va='center',
               color=palette['cream'], fontsize=10, fontweight='bold')
    
    # フロー（矢印）の描画
    for i in range(n_classes):
        for j in range(n_classes):
            if cm_normalized[i, j] > 0.02:  # 2%以上のフローのみ描画
                start_y = y_positions[i]
                end_y = y_positions[j]
                
                # 誤分類か正分類かで色を変える
                if i == j:
                    color = palette['teal']
                    alpha = 0.3
                else:
                    color = palette['magenta']
                    alpha = min(0.8, cm_normalized[i, j] * 2)  # 誤分類率に応じて透明度を調整
                
                # 線の太さを流量に比例させる
                linewidth = max(0.5, cm_normalized[i, j] * 10)
                
                ax.annotate('', xy=(right_x - 0.05, end_y), xytext=(left_x + 0.05, start_y),
                           arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, 
                                          lw=linewidth, connectionstyle='arc3,rad=0.1'))
                
                # ラベル（割合）を追加
                if i != j and cm_normalized[i, j] > 0.05:
                    mid_x = (left_x + right_x) / 2
                    mid_y = (start_y + end_y) / 2
                    ax.text(mid_x, mid_y, f'{cm_normalized[i, j]*100:.1f}%',
                           ha='center', va='center', color=palette['cream'],
                           fontsize=8, alpha=0.9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'{model_name} - Misclassification Flow', color=palette['cream'], 
                fontsize=14, fontweight='bold')
    ax.text(left_x, 0.02, 'True Label', ha='center', color=palette['cream'], fontsize=11)
    ax.text(right_x, 0.02, 'Predicted Label', ha='center', color=palette['cream'], fontsize=11)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


# *******************************************************************
# Advanced Visualization for Difficult-to-Classify Samples
# *******************************************************************

def visualize_model_confusion_regions(
    predictions_df: pd.DataFrame,
    original_image: Optional[np.ndarray],
    save_dir: Path,
    model_name: str,
    patch_size: int = 25,
    grid_resolution: int = 50,
    crs: str = 'EPSG:6675',
    class_names: Optional[List[str]] = None,
    output_geotiff: bool = True
) -> Dict:
    """
    入力画像領域の中でどの部分の正答率が低く、モデルが迷っているかを可視化する関数
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        予測結果のDataFrame。以下の列が必要:
        - 'x' or 'lon': x座標
        - 'y' or 'lat': y座標
        - 'true_label': 真のラベル
        - 'pred_label' or 'predicted_class': 予測ラベル
        - 'prob_class_0', 'prob_class_1', ... : 各クラスの予測確率（オプション）
    original_image : np.ndarray, optional
        元画像（背景表示用）。(H, W, C) または (C, H, W)
    save_dir : Path
        出力保存先ディレクトリ
    model_name : str
        モデル名（ファイル名に使用）
    patch_size : int
        各パッチのサイズ
    grid_resolution : int
        ヒートマップのグリッド解像度
    crs : str
        座標参照系
    class_names : List[str], optional
        クラス名のリスト
    output_geotiff : bool
        GeoTIFF出力を行うか
    
    Returns:
    --------
    Dict : 解析結果のサマリー
        - accuracy_map: 正答率マップの統計
        - entropy_map: エントロピーマップの統計
        - confusion_hotspots: 判別困難なホットスポットの座標
        - class_wise_analysis: クラス別の誤分類分析
    """
    save_dir = Path(save_dir)
    confusion_dir = save_dir / 'model_confusion_analysis'
    confusion_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🔍 Model Confusion Region Analysis for {model_name}")
    print(f"{'='*70}")
    
    # カラーパレット
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
        'dark': '#1a1a2e',
        'warning': '#ff6b6b',
    }
    
    # 座標列の取得
    if 'x' in predictions_df.columns:
        x_coords = predictions_df['x'].values
        y_coords = predictions_df['y'].values
    elif 'lon' in predictions_df.columns:
        x_coords = predictions_df['lon'].values
        y_coords = predictions_df['lat'].values
    else:
        raise ValueError("座標列（'x'/'y' または 'lon'/'lat'）が見つかりません")
    
    # ラベル列の取得
    y_true = predictions_df['true_label'].values
    if 'pred_label' in predictions_df.columns:
        y_pred = predictions_df['pred_label'].values
    else:
        y_pred = predictions_df['predicted_class'].values
    
    # 確率列の取得
    prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
    has_probs = len(prob_cols) > 0
    
    if has_probs:
        y_prob = predictions_df[prob_cols].values
        n_classes = len(prob_cols)
    else:
        n_classes = len(np.unique(y_true))
        y_prob = None
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # 正誤フラグ
    is_correct = (y_true == y_pred)
    
    print(f"  📊 Total samples: {len(predictions_df):,}")
    print(f"  ✅ Correct: {np.sum(is_correct):,} ({100*np.mean(is_correct):.1f}%)")
    print(f"  ❌ Misclassified: {np.sum(~is_correct):,} ({100*np.mean(~is_correct):.1f}%)")
    
    # ============================================================
    # 1. 空間的な正答率/エントロピーグリッドの作成
    # ============================================================
    print(f"\n  📐 Creating spatial accuracy grid ({grid_resolution}x{grid_resolution})...")
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # グリッドの作成
    x_edges = np.linspace(x_min, x_max, grid_resolution + 1)
    y_edges = np.linspace(y_min, y_max, grid_resolution + 1)
    
    # グリッドセル内の正答率と統計を計算
    accuracy_grid = np.full((grid_resolution, grid_resolution), np.nan)
    count_grid = np.zeros((grid_resolution, grid_resolution), dtype=int)
    entropy_grid = np.full((grid_resolution, grid_resolution), np.nan)
    max_prob_grid = np.full((grid_resolution, grid_resolution), np.nan)
    
    # 各グリッドセルの詳細分析
    cell_details = {}
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            # このセル内のサンプルを抽出
            mask = (
                (x_coords >= x_edges[j]) & (x_coords < x_edges[j+1]) &
                (y_coords >= y_edges[i]) & (y_coords < y_edges[i+1])
            )
            count = np.sum(mask)
            count_grid[i, j] = count
            
            if count > 0:
                cell_correct = is_correct[mask]
                accuracy_grid[i, j] = np.mean(cell_correct)
                
                if has_probs:
                    cell_probs = y_prob[mask]
                    # エントロピー計算
                    epsilon = 1e-10
                    cell_entropy = -np.sum(cell_probs * np.log(cell_probs + epsilon), axis=1)
                    max_entropy = np.log(n_classes)
                    entropy_grid[i, j] = np.mean(cell_entropy) / max_entropy  # 正規化
                    max_prob_grid[i, j] = np.mean(np.max(cell_probs, axis=1))
                
                # 詳細情報を保存
                cell_details[(i, j)] = {
                    'count': int(count),
                    'accuracy': float(accuracy_grid[i, j]),
                    'center_x': float((x_edges[j] + x_edges[j+1]) / 2),
                    'center_y': float((y_edges[i] + y_edges[i+1]) / 2),
                    'misclassified_count': int(np.sum(~cell_correct)),
                }
    
    # ============================================================
    # 2. 正答率ヒートマップの可視化
    # ============================================================
    print("  🎨 Creating accuracy heatmap...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor(palette['dark'])
    
    # (0,0) 正答率マップ
    ax1 = axes[0, 0]
    ax1.set_facecolor(palette['dark'])
    im1 = ax1.imshow(accuracy_grid, extent=[x_min, x_max, y_min, y_max],
                     origin='lower', cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax1.set_title(f'{model_name}\nSpatial Accuracy Map (Green=High, Red=Low)', 
                 color=palette['cream'], fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Coordinate', color=palette['cream'])
    ax1.set_ylabel('Y Coordinate', color=palette['cream'])
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Accuracy', color=palette['cream'])
    cbar1.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color=palette['cream'])
    
    # (0,1) サンプル数マップ（対数スケール）
    ax2 = axes[0, 1]
    ax2.set_facecolor(palette['dark'])
    count_display = np.log1p(count_grid)  # log(1+x) for better visualization
    im2 = ax2.imshow(count_display, extent=[x_min, x_max, y_min, y_max],
                     origin='lower', cmap='Blues', aspect='auto')
    ax2.set_title(f'{model_name}\nSample Density (log scale)', 
                 color=palette['cream'], fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Coordinate', color=palette['cream'])
    ax2.set_ylabel('Y Coordinate', color=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('log(1 + count)', color=palette['cream'])
    cbar2.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color=palette['cream'])
    
    # (1,0) 不確実性マップ（エントロピーまたは誤分類率）
    ax3 = axes[1, 0]
    ax3.set_facecolor(palette['dark'])
    if has_probs:
        uncertainty_map = entropy_grid
        uncertainty_label = 'Normalized Entropy'
    else:
        uncertainty_map = 1 - accuracy_grid  # 誤分類率
        uncertainty_label = 'Error Rate'
    
    im3 = ax3.imshow(uncertainty_map, extent=[x_min, x_max, y_min, y_max],
                     origin='lower', cmap='hot', vmin=0, vmax=1, aspect='auto')
    ax3.set_title(f'{model_name}\nModel Uncertainty Map (Bright=High Uncertainty)', 
                 color=palette['cream'], fontsize=12, fontweight='bold')
    ax3.set_xlabel('X Coordinate', color=palette['cream'])
    ax3.set_ylabel('Y Coordinate', color=palette['cream'])
    ax3.tick_params(colors=palette['cream'])
    for spine in ax3.spines.values():
        spine.set_color(palette['cream'])
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label(uncertainty_label, color=palette['cream'])
    cbar3.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color=palette['cream'])
    
    # (1,1) 判別困難領域のハイライト
    ax4 = axes[1, 1]
    ax4.set_facecolor(palette['dark'])
    
    # 閾値ベースでホットスポットを特定
    difficulty_threshold = 0.7  # 正答率70%未満を「困難」と定義
    min_samples_threshold = 5  # 最低サンプル数
    
    hotspot_mask = (accuracy_grid < difficulty_threshold) & (count_grid >= min_samples_threshold)
    hotspot_highlight = np.where(hotspot_mask, 1 - accuracy_grid, np.nan)
    
    # 背景として正答率マップを薄く表示
    ax4.imshow(accuracy_grid, extent=[x_min, x_max, y_min, y_max],
               origin='lower', cmap='Greys_r', alpha=0.3, aspect='auto')
    
    # ホットスポットを強調表示
    im4 = ax4.imshow(hotspot_highlight, extent=[x_min, x_max, y_min, y_max],
                     origin='lower', cmap='Reds', vmin=0, vmax=1, aspect='auto', alpha=0.8)
    ax4.set_title(f'{model_name}\nDifficult Regions (Accuracy < {difficulty_threshold*100:.0f}%)', 
                 color=palette['cream'], fontsize=12, fontweight='bold')
    ax4.set_xlabel('X Coordinate', color=palette['cream'])
    ax4.set_ylabel('Y Coordinate', color=palette['cream'])
    ax4.tick_params(colors=palette['cream'])
    for spine in ax4.spines.values():
        spine.set_color(palette['cream'])
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
    cbar4.set_label('Error Rate', color=palette['cream'])
    cbar4.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(plt.getp(cbar4.ax.axes, 'yticklabels'), color=palette['cream'])
    
    plt.tight_layout()
    plt.savefig(confusion_dir / 'spatial_accuracy_analysis.png', dpi=300, 
               bbox_inches='tight', facecolor=palette['dark'])
    plt.close(fig)
    
    # ============================================================
    # 3. クラス別の空間的誤分類パターン
    # ============================================================
    print("  📊 Analyzing class-wise confusion patterns...")
    
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(5 * ((n_classes + 1) // 2), 10))
    fig.patch.set_facecolor(palette['dark'])
    axes = axes.flatten()
    
    class_analysis = []
    
    for c in range(n_classes):
        ax = axes[c]
        ax.set_facecolor(palette['dark'])
        
        # このクラスのサンプルのみ抽出
        class_mask = (y_true == c)
        if np.sum(class_mask) == 0:
            ax.set_title(f'{class_names[c]}\n(No samples)', color=palette['cream'])
            ax.axis('off')
            continue
        
        class_x = x_coords[class_mask]
        class_y = y_coords[class_mask]
        class_correct = is_correct[class_mask]
        
        # 正解と誤分類を異なる色でプロット
        ax.scatter(class_x[class_correct], class_y[class_correct], 
                  c=palette['teal'], s=5, alpha=0.4, label='Correct')
        ax.scatter(class_x[~class_correct], class_y[~class_correct], 
                  c=palette['magenta'], s=15, alpha=0.7, label='Misclassified', marker='x')
        
        class_acc = np.mean(class_correct)
        ax.set_title(f'{class_names[c]}\nAccuracy: {class_acc*100:.1f}% (n={np.sum(class_mask):,})', 
                    color=palette['cream'], fontsize=10)
        ax.set_xlabel('X', color=palette['cream'], fontsize=8)
        ax.set_ylabel('Y', color=palette['cream'], fontsize=8)
        ax.tick_params(colors=palette['cream'], labelsize=7)
        ax.legend(fontsize=7, facecolor=palette['dark'], edgecolor=palette['cream'], 
                 labelcolor=palette['cream'], loc='upper right')
        for spine in ax.spines.values():
            spine.set_color(palette['cream'])
        ax.set_aspect('equal')
        
        # クラス分析情報を収集
        class_analysis.append({
            'class_idx': c,
            'class_name': class_names[c],
            'total_samples': int(np.sum(class_mask)),
            'correct_samples': int(np.sum(class_correct)),
            'misclassified_samples': int(np.sum(~class_correct)),
            'accuracy': float(class_acc),
        })
    
    # 余分なaxesを非表示
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')
        axes[idx].set_facecolor(palette['dark'])
    
    plt.suptitle(f'{model_name} - Class-wise Spatial Distribution of Errors',
                color=palette['cream'], fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(confusion_dir / 'class_wise_error_distribution.png', dpi=300, 
               bbox_inches='tight', facecolor=palette['dark'])
    plt.close(fig)
    
    # ============================================================
    # 4. 誤分類のヒートマップ（混同パターン別）
    # ============================================================
    print("  🔥 Creating confusion pattern heatmaps...")
    
    # 主要な誤分類パターンを特定
    confusion_patterns = {}
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            pattern_key = (int(y_true[i]), int(y_pred[i]))
            if pattern_key not in confusion_patterns:
                confusion_patterns[pattern_key] = []
            confusion_patterns[pattern_key].append(i)
    
    # 上位6パターンをプロット
    top_patterns = sorted(confusion_patterns.items(), 
                         key=lambda x: len(x[1]), reverse=True)[:6]
    
    if top_patterns:
        n_patterns = len(top_patterns)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.patch.set_facecolor(palette['dark'])
        axes = axes.flatten()
        
        for idx, ((true_c, pred_c), indices) in enumerate(top_patterns):
            ax = axes[idx]
            ax.set_facecolor(palette['dark'])
            
            pattern_x = x_coords[indices]
            pattern_y = y_coords[indices]
            
            # ヒートマップとして表示
            if len(indices) > 10:
                # 2D histogram
                h, xedges, yedges = np.histogram2d(pattern_x, pattern_y, bins=30)
                im = ax.imshow(h.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                              origin='lower', cmap='YlOrRd', aspect='auto')
                plt.colorbar(im, ax=ax, shrink=0.6)
            else:
                ax.scatter(pattern_x, pattern_y, c=palette['magenta'], s=30, alpha=0.7)
            
            ax.set_title(f'{class_names[true_c]} → {class_names[pred_c]}\n({len(indices)} errors)',
                        color=palette['cream'], fontsize=10)
            ax.tick_params(colors=palette['cream'], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(palette['cream'])
        
        # 未使用のaxes
        for idx in range(n_patterns, 6):
            axes[idx].axis('off')
            axes[idx].set_facecolor(palette['dark'])
        
        plt.suptitle(f'{model_name} - Top Confusion Patterns (Spatial Distribution)',
                    color=palette['cream'], fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(confusion_dir / 'confusion_pattern_heatmaps.png', dpi=300,
                   bbox_inches='tight', facecolor=palette['dark'])
        plt.close(fig)
    
    # ============================================================
    # 5. 確信度別の空間分布（確率がある場合）
    # ============================================================
    if has_probs:
        print("  📈 Creating confidence-based analysis...")
        
        max_probs = np.max(y_prob, axis=1)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor(palette['dark'])
        
        confidence_bins = [(0, 0.4, 'Very Low (<40%)'),
                          (0.4, 0.7, 'Medium (40-70%)'),
                          (0.7, 1.0, 'High (>70%)')]
        
        for idx, (low, high, label) in enumerate(confidence_bins):
            ax = axes[idx]
            ax.set_facecolor(palette['dark'])
            
            conf_mask = (max_probs >= low) & (max_probs < high)
            if np.sum(conf_mask) > 0:
                conf_correct = is_correct[conf_mask]
                ax.scatter(x_coords[conf_mask][conf_correct], 
                          y_coords[conf_mask][conf_correct],
                          c=palette['teal'], s=5, alpha=0.4, label='Correct')
                ax.scatter(x_coords[conf_mask][~conf_correct],
                          y_coords[conf_mask][~conf_correct],
                          c=palette['magenta'], s=15, alpha=0.7, label='Wrong', marker='x')
                
                acc = np.mean(conf_correct) * 100
                ax.set_title(f'Confidence {label}\nn={np.sum(conf_mask):,}, Acc={acc:.1f}%',
                            color=palette['cream'], fontsize=11)
            else:
                ax.set_title(f'Confidence {label}\n(No samples)', color=palette['cream'])
            
            ax.set_xlabel('X', color=palette['cream'])
            ax.set_ylabel('Y', color=palette['cream'])
            ax.tick_params(colors=palette['cream'])
            ax.legend(fontsize=8, facecolor=palette['dark'], 
                     edgecolor=palette['cream'], labelcolor=palette['cream'])
            for spine in ax.spines.values():
                spine.set_color(palette['cream'])
            ax.set_aspect('equal')
        
        plt.suptitle(f'{model_name} - Spatial Distribution by Confidence Level',
                    color=palette['cream'], fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(confusion_dir / 'confidence_spatial_distribution.png', dpi=300,
                   bbox_inches='tight', facecolor=palette['dark'])
        plt.close(fig)
    
    # ============================================================
    # 6. ホットスポットの特定と統計
    # ============================================================
    print("  🔴 Identifying confusion hotspots...")
    
    # 低正答率セルを抽出
    hotspot_cells = []
    for (i, j), details in cell_details.items():
        if details['accuracy'] < difficulty_threshold and details['count'] >= min_samples_threshold:
            hotspot_cells.append({
                'grid_i': i,
                'grid_j': j,
                'center_x': details['center_x'],
                'center_y': details['center_y'],
                'accuracy': details['accuracy'],
                'error_rate': 1 - details['accuracy'],
                'sample_count': details['count'],
                'misclassified_count': details['misclassified_count'],
            })
    
    # エラー率でソート
    hotspot_cells = sorted(hotspot_cells, key=lambda x: x['error_rate'], reverse=True)
    
    # ホットスポットリストを保存
    hotspot_df = pd.DataFrame(hotspot_cells)
    if len(hotspot_df) > 0:
        hotspot_df.to_csv(confusion_dir / 'confusion_hotspots.csv', index=False)
    
    # ============================================================
    # 7. GeoTIFF出力（オプション）
    # ============================================================
    if output_geotiff and len(x_coords) > 0:
        print("  📁 Exporting GeoTIFF files...")
        try:
            # 正答率ラスタ
            _export_grid_as_geotiff(
                accuracy_grid, x_edges, y_edges,
                confusion_dir / f'{model_name}_accuracy.tif',
                crs=crs, nodata=np.nan
            )
            
            # エントロピーラスタ
            if has_probs:
                _export_grid_as_geotiff(
                    entropy_grid, x_edges, y_edges,
                    confusion_dir / f'{model_name}_entropy.tif',
                    crs=crs, nodata=np.nan
                )
            
            print(f"    ✅ GeoTIFF files exported to {confusion_dir}")
        except Exception as e:
            print(f"    ⚠️ GeoTIFF export failed: {e}")
    
    # ============================================================
    # 8. サマリー統計
    # ============================================================
    valid_accuracy_cells = accuracy_grid[~np.isnan(accuracy_grid)]
    
    summary = {
        'model_name': model_name,
        'total_samples': len(predictions_df),
        'overall_accuracy': float(np.mean(is_correct)),
        'grid_resolution': grid_resolution,
        'accuracy_grid_stats': {
            'mean': float(np.nanmean(accuracy_grid)),
            'std': float(np.nanstd(accuracy_grid)),
            'min': float(np.nanmin(accuracy_grid)) if len(valid_accuracy_cells) > 0 else None,
            'max': float(np.nanmax(accuracy_grid)) if len(valid_accuracy_cells) > 0 else None,
            'cells_below_70pct': int(np.sum(accuracy_grid < 0.7)),
            'cells_below_50pct': int(np.sum(accuracy_grid < 0.5)),
        },
        'hotspot_count': len(hotspot_cells),
        'top_10_hotspots': hotspot_cells[:10],
        'class_analysis': class_analysis,
        'output_files': {
            'main_plot': str(confusion_dir / 'spatial_accuracy_analysis.png'),
            'class_distribution': str(confusion_dir / 'class_wise_error_distribution.png'),
            'hotspots_csv': str(confusion_dir / 'confusion_hotspots.csv'),
        }
    }
    
    # サマリーJSONを保存
    with open(confusion_dir / 'confusion_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    # ============================================================
    # 9. コンソール出力
    # ============================================================
    print(f"\n  📊 Analysis Summary:")
    print(f"     Overall accuracy: {summary['overall_accuracy']*100:.2f}%")
    print(f"     Grid cells analyzed: {np.sum(count_grid > 0)}")
    print(f"     Hotspots (acc < 70%): {summary['hotspot_count']}")
    print(f"     Cells below 50% accuracy: {summary['accuracy_grid_stats']['cells_below_50pct']}")
    
    if hotspot_cells:
        print(f"\n  🔴 Top 5 Confusion Hotspots:")
        for i, hs in enumerate(hotspot_cells[:5]):
            print(f"     {i+1}. ({hs['center_x']:.1f}, {hs['center_y']:.1f}): "
                  f"acc={hs['accuracy']*100:.1f}%, n={hs['sample_count']}")
    
    print(f"\n  ✅ Analysis complete. Results saved to: {confusion_dir}")
    
    return summary


def _export_grid_as_geotiff(grid: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray,
                            output_path: Path, crs: str = 'EPSG:6675', nodata: float = np.nan):
    """グリッドデータをGeoTIFFとして出力"""
    height, width = grid.shape
    
    # GeoTransform計算
    x_min = x_edges[0]
    y_max = y_edges[-1]  # 上端
    res_x = (x_edges[-1] - x_edges[0]) / width
    res_y = (y_edges[-1] - y_edges[0]) / height
    
    transform = rasterio.transform.from_bounds(
        x_edges[0], y_edges[0], x_edges[-1], y_edges[-1], width, height
    )
    
    # Y軸を反転（ラスタ座標系）
    grid_flipped = np.flipud(grid)
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata
    ) as dst:
        dst.write(grid_flipped.astype(np.float32), 1)


def visualize_prediction_uncertainty_map(predictions_df: pd.DataFrame,
                                         save_dir: Path,
                                         model_name: str,
                                         output_geotiff: bool = True) -> Dict:
    """
    予測確信度の空間マップを作成
    低確信度領域（判別困難領域）を可視化
    
    Parameters:
    - predictions_df: 予測結果のDataFrame（x, y座標とprob_class_*を含む）
    - save_dir: 保存先ディレクトリ
    - model_name: モデル名
    - output_geotiff: GeoTIFF出力を行うか
    
    Returns:
    - Dict: 不確実性解析結果
    """
    save_dir = Path(save_dir)
    uncertainty_dir = save_dir / 'uncertainty_maps'
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📍 Creating Uncertainty Map for {model_name}...")
    
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    # 確率列を取得
    prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
    if not prob_cols:
        print("  ⚠️ No probability columns found, skipping uncertainty map")
        return {'note': 'probability columns not available'}
    
    y_prob = predictions_df[prob_cols].values
    
    # 確信度（最大確率）
    max_confidence = np.max(y_prob, axis=1)
    
    # エントロピー（不確実性の指標）
    # H = -Σ p_i * log(p_i)
    epsilon = 1e-10
    entropy = -np.sum(y_prob * np.log(y_prob + epsilon), axis=1)
    max_entropy = np.log(len(prob_cols))  # 最大エントロピー（一様分布時）
    normalized_entropy = entropy / max_entropy  # 0-1に正規化
    
    # 座標情報の取得
    has_coords = 'x' in predictions_df.columns and 'y' in predictions_df.columns
    
    if has_coords:
        x_coords = predictions_df['x'].values
        y_coords = predictions_df['y'].values
    else:
        # pixel_x, pixel_y があるか確認
        if 'pixel_x' in predictions_df.columns and 'pixel_y' in predictions_df.columns:
            x_coords = predictions_df['pixel_x'].values
            y_coords = predictions_df['pixel_y'].values
            has_coords = True
        else:
            print("  ⚠️ No coordinate columns found, creating histogram only")
            has_coords = False
    
    # 不確実性統計
    uncertainty_stats = {
        'mean_confidence': float(np.mean(max_confidence)),
        'std_confidence': float(np.std(max_confidence)),
        'mean_entropy': float(np.mean(normalized_entropy)),
        'std_entropy': float(np.std(normalized_entropy)),
        'low_confidence_ratio': float(np.mean(max_confidence < 0.5)),
        'high_uncertainty_ratio': float(np.mean(normalized_entropy > 0.5)),
        'very_uncertain_count': int(np.sum(max_confidence < 0.4)),
    }
    
    # 1. 確信度・エントロピーヒストグラム
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('black')
    
    # 確信度分布
    ax1 = axes[0]
    ax1.set_facecolor('black')
    ax1.hist(max_confidence, bins=50, color=palette['teal'], alpha=0.8, edgecolor=palette['cream'])
    ax1.axvline(x=0.5, color=palette['magenta'], linestyle='--', linewidth=2, label='Threshold=0.5')
    ax1.set_xlabel('Max Confidence', color=palette['cream'])
    ax1.set_ylabel('Count', color=palette['cream'])
    ax1.set_title(f'{model_name} - Confidence Distribution', color=palette['cream'])
    ax1.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # エントロピー分布
    ax2 = axes[1]
    ax2.set_facecolor('black')
    ax2.hist(normalized_entropy, bins=50, color=palette['magenta'], alpha=0.8, edgecolor=palette['cream'])
    ax2.axvline(x=0.5, color=palette['teal'], linestyle='--', linewidth=2, label='High Uncertainty')
    ax2.set_xlabel('Normalized Entropy', color=palette['cream'])
    ax2.set_ylabel('Count', color=palette['cream'])
    ax2.set_title(f'{model_name} - Uncertainty (Entropy) Distribution', color=palette['cream'])
    ax2.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    
    # 確信度 vs エントロピー散布図
    ax3 = axes[2]
    ax3.set_facecolor('black')
    scatter = ax3.scatter(max_confidence, normalized_entropy, c=normalized_entropy,
                         cmap='magma', alpha=0.3, s=5)
    ax3.set_xlabel('Max Confidence', color=palette['cream'])
    ax3.set_ylabel('Normalized Entropy', color=palette['cream'])
    ax3.set_title(f'{model_name} - Confidence vs Entropy', color=palette['cream'])
    ax3.tick_params(colors=palette['cream'])
    for spine in ax3.spines.values():
        spine.set_color(palette['cream'])
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Uncertainty', color=palette['cream'])
    cbar.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=palette['cream'])
    
    plt.tight_layout()
    plt.savefig(uncertainty_dir / 'uncertainty_distributions.png', dpi=300, 
               bbox_inches='tight', facecolor='black')
    plt.close(fig)
    
    # 2. 空間的な不確実性マップ（座標がある場合）
    if has_coords:
        _create_spatial_uncertainty_map(
            x_coords, y_coords, max_confidence, normalized_entropy,
            uncertainty_dir, model_name, palette
        )
        
        # 3. GeoTIFF出力（オプション）
        if output_geotiff:
            _export_uncertainty_geotiff(
                predictions_df, max_confidence, normalized_entropy,
                uncertainty_dir, model_name
            )
    
    # 不確実サンプルのリストを保存
    uncertain_mask = max_confidence < 0.5
    uncertain_df = predictions_df[uncertain_mask].copy()
    uncertain_df['confidence'] = max_confidence[uncertain_mask]
    uncertain_df['entropy'] = normalized_entropy[uncertain_mask]
    uncertain_df.to_csv(uncertainty_dir / 'uncertain_samples.csv', index=False)
    
    print(f"  ✅ Uncertainty map saved to: {uncertainty_dir}")
    print(f"  📊 Low confidence samples (<0.5): {uncertainty_stats['very_uncertain_count']:,}")
    
    return uncertainty_stats


def _create_spatial_uncertainty_map(x_coords: np.ndarray, y_coords: np.ndarray,
                                    confidence: np.ndarray, entropy: np.ndarray,
                                    save_dir: Path, model_name: str, palette: Dict):
    """空間的な不確実性マップを作成"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('black')
    
    # 確信度マップ（低いほど赤）
    ax1 = axes[0]
    ax1.set_facecolor('black')
    scatter1 = ax1.scatter(x_coords, y_coords, c=confidence, cmap='RdYlGn',
                          s=2, alpha=0.7, vmin=0, vmax=1)
    ax1.set_xlabel('X Coordinate', color=palette['cream'])
    ax1.set_ylabel('Y Coordinate', color=palette['cream'])
    ax1.set_title(f'{model_name} - Spatial Confidence Map\n(Red=Low Confidence)', 
                 color=palette['cream'])
    ax1.tick_params(colors=palette['cream'])
    ax1.set_aspect('equal')
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.7)
    cbar1.set_label('Confidence', color=palette['cream'])
    cbar1.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color=palette['cream'])
    
    # エントロピーマップ（高いほど赤）
    ax2 = axes[1]
    ax2.set_facecolor('black')
    scatter2 = ax2.scatter(x_coords, y_coords, c=entropy, cmap='hot',
                          s=2, alpha=0.7, vmin=0, vmax=1)
    ax2.set_xlabel('X Coordinate', color=palette['cream'])
    ax2.set_ylabel('Y Coordinate', color=palette['cream'])
    ax2.set_title(f'{model_name} - Spatial Uncertainty Map\n(Bright=High Uncertainty)',
                 color=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    ax2.set_aspect('equal')
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.7)
    cbar2.set_label('Entropy', color=palette['cream'])
    cbar2.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color=palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'spatial_uncertainty_map.png', dpi=300,
               bbox_inches='tight', facecolor='black')
    plt.close(fig)


def _export_uncertainty_geotiff(predictions_df: pd.DataFrame, confidence: np.ndarray,
                                entropy: np.ndarray, save_dir: Path, model_name: str):
    """不確実性をGeoTIFFとして出力（QGISで確認可能）"""
    try:
        # 座標からグリッドを推定
        if 'x' not in predictions_df.columns or 'y' not in predictions_df.columns:
            print("  ⚠️ Cannot export GeoTIFF: coordinate columns not found")
            return
        
        x = predictions_df['x'].values
        y = predictions_df['y'].values
        
        # ユニークな座標を取得してグリッドサイズを推定
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        
        if len(unique_x) < 2 or len(unique_y) < 2:
            print("  ⚠️ Cannot export GeoTIFF: insufficient coordinate range")
            return
        
        # 解像度を推定
        res_x = np.min(np.diff(np.sort(unique_x)))
        res_y = np.min(np.diff(np.sort(unique_y)))
        
        # グリッドサイズ
        width = int((np.max(x) - np.min(x)) / res_x) + 1
        height = int((np.max(y) - np.min(y)) / res_y) + 1
        
        # GeoTransform
        transform = rasterio.transform.from_bounds(
            np.min(x) - res_x/2, np.min(y) - res_y/2,
            np.max(x) + res_x/2, np.max(y) + res_y/2,
            width, height
        )
        
        # 空のラスタを作成
        confidence_raster = np.full((height, width), np.nan, dtype=np.float32)
        entropy_raster = np.full((height, width), np.nan, dtype=np.float32)
        
        # データを配置
        for i, (xi, yi, conf, ent) in enumerate(zip(x, y, confidence, entropy)):
            col = int((xi - np.min(x)) / res_x)
            row = int((np.max(y) - yi) / res_y)  # Y軸は反転
            if 0 <= row < height and 0 <= col < width:
                confidence_raster[row, col] = conf
                entropy_raster[row, col] = ent
        
        # 確信度ラスタを保存
        confidence_path = save_dir / f'{model_name}_confidence.tif'
        with rasterio.open(
            confidence_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs='EPSG:6675',  # 仮定
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(confidence_raster, 1)
        
        # エントロピーラスタを保存
        entropy_path = save_dir / f'{model_name}_entropy.tif'
        with rasterio.open(
            entropy_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs='EPSG:6675',
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(entropy_raster, 1)
        
        print(f"  📁 GeoTIFF exported: {confidence_path.name}, {entropy_path.name}")
        
    except Exception as e:
        print(f"  ⚠️ GeoTIFF export failed: {e}")


def create_misclassification_gallery(predictions_df: pd.DataFrame,
                                     images: np.ndarray,
                                     class_names: List[str],
                                     save_dir: Path,
                                     model_name: str,
                                     n_samples_per_pair: int = 6) -> Dict:
    """
    誤分類サンプルのギャラリー表示を作成
    各クラスペアの代表的な誤分類サンプルをタイル表示
    
    Parameters:
    - predictions_df: 予測結果のDataFrame
    - images: 画像データ (N, C, H, W) or (N, H, W, C)
    - class_names: クラス名リスト
    - save_dir: 保存先
    - model_name: モデル名
    - n_samples_per_pair: 各クラスペアあたりの表示サンプル数
    
    Returns:
    - Dict: ギャラリー作成結果
    """
    save_dir = Path(save_dir)
    gallery_dir = save_dir / 'misclassification_gallery'
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🖼️ Creating Misclassification Gallery for {model_name}...")
    
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    y_true = predictions_df['true_label'].values
    y_pred = predictions_df['pred_label'].values if 'pred_label' in predictions_df.columns \
             else predictions_df['predicted_class'].values
    
    # 確率列を取得
    prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
    y_prob = predictions_df[prob_cols].values if prob_cols else None
    
    n_classes = len(class_names)
    gallery_info = {'pairs': []}
    
    # 各クラスペアについてギャラリーを作成
    for true_class in range(n_classes):
        for pred_class in range(n_classes):
            if true_class == pred_class:
                continue
            
            # このペアの誤分類サンプルを抽出
            pair_mask = (y_true == true_class) & (y_pred == pred_class)
            pair_indices = np.where(pair_mask)[0]
            
            if len(pair_indices) == 0:
                continue
            
            # 確信度でソート（高確信度での誤分類を優先表示）
            if y_prob is not None:
                confidences = y_prob[pair_indices, pred_class]
                sorted_idx = np.argsort(confidences)[::-1]  # 降順
                selected_indices = pair_indices[sorted_idx[:n_samples_per_pair]]
            else:
                selected_indices = pair_indices[:n_samples_per_pair]
            
            if len(selected_indices) == 0:
                continue
            
            # ギャラリー画像の作成
            n_selected = len(selected_indices)
            n_cols = min(n_selected, 6)
            n_rows = (n_selected + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 3))
            fig.patch.set_facecolor('black')
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, idx in enumerate(selected_indices):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                ax.set_facecolor('black')
                
                # 画像を取得
                img = images[idx]
                
                # (C, H, W) → (H, W, C) に変換
                if img.ndim == 3 and img.shape[0] in [3, 4, 5]:
                    img = np.transpose(img, (1, 2, 0))
                
                # RGB部分のみ表示
                if img.shape[-1] > 3:
                    img = img[..., :3]
                
                # 正規化（0-1）
                if img.max() > 1:
                    img = img / 255.0
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                
                # 確信度を表示
                if y_prob is not None:
                    conf = y_prob[idx, pred_class]
                    title = f"Conf: {conf:.2f}"
                else:
                    title = f"Sample {i+1}"
                
                ax.set_title(title, color=palette['cream'], fontsize=9)
                ax.axis('off')
            
            # 未使用のaxesを非表示
            for i in range(n_selected, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].axis('off')
                axes[row, col].set_facecolor('black')
            
            # タイトル
            fig.suptitle(f'{model_name}\nTrue: {class_names[true_class]} → Pred: {class_names[pred_class]}\n({len(pair_indices)} total samples)',
                        color=palette['cream'], fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            filename = f'gallery_{true_class}to{pred_class}_{class_names[true_class]}_to_{class_names[pred_class]}.png'
            # ファイル名に使えない文字を除去
            filename = filename.replace('<', '').replace('>', '').replace('%', 'pct')
            plt.savefig(gallery_dir / filename, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            
            gallery_info['pairs'].append({
                'true_class': int(true_class),
                'pred_class': int(pred_class),
                'true_name': class_names[true_class],
                'pred_name': class_names[pred_class],
                'total_samples': int(len(pair_indices)),
                'displayed_samples': int(len(selected_indices)),
                'filename': filename
            })
    
    # サマリー画像（全クラスペアの代表サンプルを1枚に）
    _create_gallery_summary(predictions_df, images, class_names, y_true, y_pred,
                           y_prob, gallery_dir, model_name, palette)
    
    print(f"  ✅ Gallery saved to: {gallery_dir}")
    print(f"  📸 Created {len(gallery_info['pairs'])} pair galleries")
    
    return gallery_info


def _create_gallery_summary(predictions_df: pd.DataFrame, images: np.ndarray,
                           class_names: List[str], y_true: np.ndarray,
                           y_pred: np.ndarray, y_prob: Optional[np.ndarray],
                           save_dir: Path, model_name: str, palette: Dict):
    """全クラスペアの代表サンプルを1枚のサマリー画像に"""
    n_classes = len(class_names)
    
    # 誤分類ペアを収集（最大9ペア）
    misclass_pairs = []
    for true_class in range(n_classes):
        for pred_class in range(n_classes):
            if true_class == pred_class:
                continue
            pair_mask = (y_true == true_class) & (y_pred == pred_class)
            count = np.sum(pair_mask)
            if count > 0:
                misclass_pairs.append((true_class, pred_class, count, pair_mask))
    
    # 誤分類数でソート、上位9ペアを表示
    misclass_pairs = sorted(misclass_pairs, key=lambda x: x[2], reverse=True)[:9]
    
    if not misclass_pairs:
        return
    
    n_pairs = len(misclass_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.patch.set_facecolor('black')
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (true_class, pred_class, count, mask) in enumerate(misclass_pairs):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        ax.set_facecolor('black')
        
        # 代表サンプルを1つ選択（最も確信度が高い誤分類）
        indices = np.where(mask)[0]
        if y_prob is not None:
            best_idx = indices[np.argmax(y_prob[indices, pred_class])]
        else:
            best_idx = indices[0]
        
        img = images[best_idx]
        if img.ndim == 3 and img.shape[0] in [3, 4, 5]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] > 3:
            img = img[..., :3]
        if img.max() > 1:
            img = img / 255.0
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        title = f"{class_names[true_class]} → {class_names[pred_class]}\n({count} samples)"
        ax.set_title(title, color=palette['cream'], fontsize=10)
        ax.axis('off')
    
    # 未使用のaxesを非表示
    for i in range(n_pairs, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
        axes[row, col].set_facecolor('black')
    
    fig.suptitle(f'{model_name} - Top Misclassification Examples',
                color=palette['cream'], fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'gallery_summary.png', dpi=200, bbox_inches='tight', facecolor='black')
    plt.close(fig)


def visualize_feature_space(features: np.ndarray,
                            labels: np.ndarray,
                            predictions: np.ndarray,
                            class_names: List[str],
                            save_dir: Path,
                            model_name: str,
                            method: str = 'tsne') -> Dict:
    """
    t-SNE/UMAPで特徴空間を可視化
    正解・誤分類を色分けして、なぜ判別困難かを分析
    
    Parameters:
    - features: 特徴ベクトル (N, D)
    - labels: 真のラベル (N,)
    - predictions: 予測ラベル (N,)
    - class_names: クラス名リスト
    - save_dir: 保存先
    - model_name: モデル名
    - method: 'tsne' or 'umap'
    
    Returns:
    - Dict: 可視化結果情報
    """
    save_dir = Path(save_dir)
    feature_dir = save_dir / 'feature_space'
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔮 Creating Feature Space Visualization ({method.upper()}) for {model_name}...")
    
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    # サンプル数が多すぎる場合はサブサンプリング
    max_samples = 5000
    if len(features) > max_samples:
        print(f"  ⚠️ Subsampling {len(features)} → {max_samples} samples for visualization")
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
        predictions = predictions[indices]
    
    # 次元削減
    n_samples = len(features)
    if method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(features)
        except ImportError:
            print("  ⚠️ UMAP not available, falling back to t-SNE")
            method = 'tsne'
    
    if method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        perplexity = min(30, n_samples - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        embedding = reducer.fit_transform(features)
    
    # 誤分類フラグ
    is_correct = labels == predictions
    
    n_classes = len(class_names)
    
    # 可視化1: クラス別（正解/誤分類で形状を変える）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('black')
    
    # クラス別カラーマップ
    class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # 左図: 真のラベルで色分け
    ax1 = axes[0]
    ax1.set_facecolor('black')
    
    for c in range(n_classes):
        mask = labels == c
        correct_mask = mask & is_correct
        wrong_mask = mask & ~is_correct
        
        # 正解: 丸
        if np.sum(correct_mask) > 0:
            ax1.scatter(embedding[correct_mask, 0], embedding[correct_mask, 1],
                       c=[class_colors[c]], marker='o', s=20, alpha=0.6,
                       label=f'{class_names[c]} (correct)')
        
        # 誤分類: ×
        if np.sum(wrong_mask) > 0:
            ax1.scatter(embedding[wrong_mask, 0], embedding[wrong_mask, 1],
                       c=[class_colors[c]], marker='x', s=30, alpha=0.8)
    
    ax1.set_title(f'{model_name} - Feature Space (by True Label)\n○=Correct, ×=Misclassified',
                 color=palette['cream'], fontsize=11)
    ax1.set_xlabel(f'{method.upper()} Dimension 1', color=palette['cream'])
    ax1.set_ylabel(f'{method.upper()} Dimension 2', color=palette['cream'])
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # 右図: 正解/誤分類で色分け
    ax2 = axes[1]
    ax2.set_facecolor('black')
    
    ax2.scatter(embedding[is_correct, 0], embedding[is_correct, 1],
               c=palette['teal'], marker='o', s=15, alpha=0.4, label='Correct')
    ax2.scatter(embedding[~is_correct, 0], embedding[~is_correct, 1],
               c=palette['magenta'], marker='x', s=25, alpha=0.7, label='Misclassified')
    
    ax2.set_title(f'{model_name} - Feature Space (Correct vs Misclassified)',
                 color=palette['cream'], fontsize=11)
    ax2.set_xlabel(f'{method.upper()} Dimension 1', color=palette['cream'])
    ax2.set_ylabel(f'{method.upper()} Dimension 2', color=palette['cream'])
    ax2.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(feature_dir / f'feature_space_{method}.png', dpi=200,
               bbox_inches='tight', facecolor='black')
    plt.close(fig)
    
    # 可視化2: クラス境界の重なりを解析
    _plot_class_overlap_analysis(embedding, labels, predictions, class_names,
                                feature_dir, model_name, method, palette)
    
    # 統計情報
    result = {
        'method': method,
        'n_samples': n_samples,
        'n_correct': int(np.sum(is_correct)),
        'n_misclassified': int(np.sum(~is_correct)),
        'embedding_saved': str(feature_dir / f'embedding_{method}.npy')
    }
    
    # embeddingを保存
    np.save(feature_dir / f'embedding_{method}.npy', embedding)
    
    print(f"  ✅ Feature space visualization saved to: {feature_dir}")
    
    return result


def _plot_class_overlap_analysis(embedding: np.ndarray, labels: np.ndarray,
                                 predictions: np.ndarray, class_names: List[str],
                                 save_dir: Path, model_name: str, method: str, palette: Dict):
    """クラス境界の重なり分析"""
    n_classes = len(class_names)
    
    # クラスごとの中心点を計算
    class_centers = []
    for c in range(n_classes):
        mask = labels == c
        if np.sum(mask) > 0:
            center = embedding[mask].mean(axis=0)
        else:
            center = np.array([np.nan, np.nan])
        class_centers.append(center)
    class_centers = np.array(class_centers)
    
    # クラス間距離行列
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(class_centers))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('black')
    
    # 左図: クラス中心間距離のヒートマップ
    ax1 = axes[0]
    ax1.set_facecolor('black')
    im = ax1.imshow(distances, cmap='viridis')
    ax1.set_xticks(np.arange(n_classes))
    ax1.set_yticks(np.arange(n_classes))
    ax1.set_xticklabels(class_names, rotation=45, ha='right', color=palette['cream'])
    ax1.set_yticklabels(class_names, color=palette['cream'])
    ax1.set_title(f'{model_name} - Class Center Distances in {method.upper()} Space',
                 color=palette['cream'])
    
    # 数値をセルに表示
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax1.text(j, i, f'{distances[i, j]:.1f}',
                          ha='center', va='center', color='white', fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Distance', color=palette['cream'])
    cbar.ax.yaxis.set_tick_params(color=palette['cream'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=palette['cream'])
    
    # 右図: クラス別誤分類率 vs 最近傍クラスとの距離
    ax2 = axes[1]
    ax2.set_facecolor('black')
    
    class_error_rates = []
    min_distances = []
    for c in range(n_classes):
        mask = labels == c
        if np.sum(mask) > 0:
            error_rate = np.sum((labels == c) & (predictions != c)) / np.sum(mask)
            # 最近傍クラスとの距離
            dist_to_others = distances[c].copy()
            dist_to_others[c] = np.inf  # 自分自身を除外
            min_dist = np.min(dist_to_others)
        else:
            error_rate = 0
            min_dist = 0
        class_error_rates.append(error_rate)
        min_distances.append(min_dist)
    
    ax2.scatter(min_distances, class_error_rates, c=palette['magenta'], s=100, alpha=0.8)
    for i, name in enumerate(class_names):
        ax2.annotate(name, (min_distances[i], class_error_rates[i]),
                    color=palette['cream'], fontsize=9,
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Distance to Nearest Class Center', color=palette['cream'])
    ax2.set_ylabel('Error Rate', color=palette['cream'])
    ax2.set_title(f'{model_name} - Error Rate vs Class Separation',
                 color=palette['cream'])
    ax2.tick_params(colors=palette['cream'])
    for spine in ax2.spines.values():
        spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'class_overlap_analysis_{method}.png', dpi=200,
               bbox_inches='tight', facecolor='black')
    plt.close(fig)


def create_sankey_diagram(confusion_matrix: np.ndarray,
                          class_names: List[str],
                          save_dir: Path,
                          model_name: str,
                          threshold: float = 0.02) -> Path:
    """
    Plotlyを使用した高品質なサンキーダイアグラムを作成
    誤分類フローを直感的に可視化
    
    Parameters:
    - confusion_matrix: 混同行列 (n_classes, n_classes)
    - class_names: クラス名リスト
    - save_dir: 保存先
    - model_name: モデル名
    - threshold: 表示する最小フロー割合（これ未満は省略）
    
    Returns:
    - Path: 保存したファイルのパス
    """
    save_dir = Path(save_dir)
    sankey_dir = save_dir / 'sankey_diagrams'
    sankey_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Creating Sankey Diagram for {model_name}...")
    
    try:
        import plotly.graph_objects as go
        use_plotly = True
    except ImportError:
        print("  ⚠️ Plotly not available, using matplotlib fallback")
        use_plotly = False
    
    cm = confusion_matrix
    n_classes = len(class_names)
    
    # 正規化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if use_plotly:
        # ノードラベル（左側: True, 右側: Predicted）
        labels = [f'True: {name}' for name in class_names] + \
                 [f'Pred: {name}' for name in class_names]
        
        # リンク（フロー）を構築
        sources = []
        targets = []
        values = []
        colors = []
        
        color_palette = [
            'rgba(134, 206, 203, 0.7)',  # teal - correct
            'rgba(225, 40, 133, 0.6)',   # magenta - error
            'rgba(195, 229, 231, 0.6)',  # ice
        ]
        
        for i in range(n_classes):
            for j in range(n_classes):
                flow = cm_normalized[i, j]
                if flow >= threshold:
                    sources.append(i)
                    targets.append(n_classes + j)
                    values.append(float(cm[i, j]))
                    
                    if i == j:
                        colors.append(color_palette[0])  # 正解
                    else:
                        colors.append(color_palette[1])  # 誤分類
        
        # Sankey図を作成
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=labels,
                color=['rgba(134, 206, 203, 0.8)'] * n_classes + \
                      ['rgba(195, 229, 231, 0.8)'] * n_classes
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=colors
            )
        )])
        
        fig.update_layout(
            title_text=f'{model_name} - Classification Flow Diagram',
            font_size=12,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font_color='#fffeec'
        )
        
        # HTML形式で保存（インタラクティブ）
        html_path = sankey_dir / f'{model_name}_sankey.html'
        fig.write_html(str(html_path))
        
        # PNG形式でも保存
        try:
            png_path = sankey_dir / f'{model_name}_sankey.png'
            fig.write_image(str(png_path), scale=2)
            print(f"  ✅ Sankey diagram saved: {html_path.name}, {png_path.name}")
            return png_path
        except Exception:
            print(f"  ✅ Sankey diagram saved (HTML only): {html_path.name}")
            return html_path
    
    else:
        # Matplotlib fallback（既存の_plot_misclassification_flowを利用）
        fallback_path = sankey_dir / f'{model_name}_flow.png'
        _plot_misclassification_flow(cm, class_names, fallback_path, model_name)
        print(f"  ✅ Flow diagram saved: {fallback_path.name}")
        return fallback_path


def run_comprehensive_visualization(predictions_df: pd.DataFrame,
                                    images: Optional[np.ndarray],
                                    features: Optional[np.ndarray],
                                    class_names: List[str],
                                    save_dir: Path,
                                    model_name: str) -> Dict:
    """
    判別困難領域の包括的な可視化を実行
    
    Parameters:
    - predictions_df: 予測結果
    - images: 画像データ（ギャラリー用、Noneの場合はスキップ）
    - features: 特徴ベクトル（t-SNE用、Noneの場合はスキップ）
    - class_names: クラス名
    - save_dir: 保存先
    - model_name: モデル名
    
    Returns:
    - Dict: 全可視化結果のサマリー
    """
    print(f"\n{'='*70}")
    print(f"🎨 Comprehensive Visualization of Difficult-to-Classify Samples")
    print(f"   Model: {model_name}")
    print(f"{'='*70}")
    
    results = {}
    
    # 1. 予測確信度の空間マップ
    try:
        results['uncertainty_map'] = visualize_prediction_uncertainty_map(
            predictions_df, save_dir, model_name
        )
    except Exception as e:
        print(f"  ⚠️ Uncertainty map failed: {e}")
        results['uncertainty_map'] = {'error': str(e)}
    
    # 2. サンキーダイアグラム
    y_true = predictions_df['true_label'].values
    y_pred = predictions_df['pred_label'].values if 'pred_label' in predictions_df.columns \
             else predictions_df['predicted_class'].values
    cm = confusion_matrix(y_true, y_pred)
    
    try:
        results['sankey'] = str(create_sankey_diagram(cm, class_names, save_dir, model_name))
    except Exception as e:
        print(f"  ⚠️ Sankey diagram failed: {e}")
        results['sankey'] = {'error': str(e)}
    
    # 3. 誤分類サンプルのギャラリー
    if images is not None:
        try:
            results['gallery'] = create_misclassification_gallery(
                predictions_df, images, class_names, save_dir, model_name
            )
        except Exception as e:
            print(f"  ⚠️ Gallery creation failed: {e}")
            results['gallery'] = {'error': str(e)}
    else:
        print("  ℹ️ Images not provided, skipping gallery")
        results['gallery'] = {'note': 'images not provided'}
    
    # 4. 特徴空間可視化（t-SNE/UMAP）
    if features is not None:
        try:
            # まずUMAPを試行、失敗したらt-SNE
            try:
                results['feature_space'] = visualize_feature_space(
                    features, y_true, y_pred, class_names, save_dir, model_name, method='umap'
                )
            except Exception:
                results['feature_space'] = visualize_feature_space(
                    features, y_true, y_pred, class_names, save_dir, model_name, method='tsne'
                )
        except Exception as e:
            print(f"  ⚠️ Feature space visualization failed: {e}")
            results['feature_space'] = {'error': str(e)}
    else:
        print("  ℹ️ Features not provided, skipping feature space visualization")
        results['feature_space'] = {'note': 'features not provided'}
    
    # 結果サマリーをJSON保存
    summary_path = save_dir / 'visualization_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(results), f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Comprehensive visualization complete: {save_dir}")
    
    return results


def _generate_ecological_interpretation(misclass_pairs: List[Dict], 
                                        class_names: List[str],
                                        class_analysis: List[Dict]) -> Dict:
    """
    誤分類パターンから生態学的解釈を生成
    
    5クラス分類（植生被覆率）に基づく解釈：
    - Class 0 (0%): 裸地・崩壊地
    - Class 1 (<25%): 回復初期段階
    - Class 2 (25-50%): 遷移初期
    - Class 3 (50-75%): 遷移中期
    - Class 4 (75-100%): 森林・完全被覆
    """
    
    ecological_meanings = {
        0: {'name': '裸地・崩壊地', 'stage': '撹乱直後', 'eng': 'Bare/Landslide'},
        1: {'name': '回復初期', 'stage': '植生回復開始', 'eng': 'Early Recovery'},
        2: {'name': '遷移初期', 'stage': '草本類優占', 'eng': 'Early Succession'},
        3: {'name': '遷移中期', 'stage': '低木類出現', 'eng': 'Mid Succession'},
        4: {'name': '森林', 'stage': '樹冠閉鎖', 'eng': 'Forest/Closed Canopy'}
    }
    
    interpretations = []
    
    # 上位の誤分類パターンを解釈
    for pair in misclass_pairs[:10]:
        if pair['rate'] < 0.03:  # 3%未満は無視
            continue
            
        true_idx = pair['true_class_idx']
        pred_idx = pair['predicted_class_idx']
        
        true_eco = ecological_meanings.get(true_idx, {'name': f'Class {true_idx}', 'stage': 'Unknown'})
        pred_eco = ecological_meanings.get(pred_idx, {'name': f'Class {pred_idx}', 'stage': 'Unknown'})
        
        # 遷移方向
        if pred_idx > true_idx:
            direction = 'overestimate'
            direction_ja = '過大評価'
            explanation = f"実際より植生被覆が高いと判定。{true_eco['stage']}の状態を{pred_eco['stage']}と誤認。"
        else:
            direction = 'underestimate'
            direction_ja = '過小評価'
            explanation = f"実際より植生被覆が低いと判定。{true_eco['stage']}の状態を{pred_eco['stage']}と誤認。"
        
        # 隣接クラス間の誤分類か
        is_adjacent = abs(pred_idx - true_idx) == 1
        
        # 生態学的な理由の推定
        if is_adjacent:
            ecological_reason = "閾値境界付近の遷移段階が連続的であるため、分類が困難。"
        elif abs(pred_idx - true_idx) >= 2:
            ecological_reason = "大きな分類誤差。画像品質、季節変動、または特殊な植生パターンの可能性。"
        else:
            ecological_reason = ""
        
        interpretations.append({
            'true_class': pair['true_class'],
            'true_ecological': true_eco['name'],
            'predicted_class': pair['predicted_class'],
            'predicted_ecological': pred_eco['name'],
            'rate': pair['rate'],
            'count': pair['count'],
            'direction': direction,
            'direction_ja': direction_ja,
            'is_adjacent_classes': is_adjacent,
            'explanation': explanation,
            'ecological_reason': ecological_reason
        })
    
    # 全体的な傾向の分析
    overestimate_count = sum(1 for i in interpretations if i['direction'] == 'overestimate')
    underestimate_count = sum(1 for i in interpretations if i['direction'] == 'underestimate')
    
    if overestimate_count > underestimate_count * 1.5:
        overall_trend = 'モデルは植生被覆を過大評価する傾向がある'
    elif underestimate_count > overestimate_count * 1.5:
        overall_trend = 'モデルは植生被覆を過小評価する傾向がある'
    else:
        overall_trend = 'モデルの予測バイアスは特に顕著ではない'
    
    # 隣接クラス誤分類の割合
    adjacent_errors = sum(1 for i in interpretations if i['is_adjacent_classes'])
    adjacent_ratio = adjacent_errors / len(interpretations) if interpretations else 0
    
    return {
        'interpretations': interpretations,
        'overall_trend': overall_trend,
        'overestimate_count': overestimate_count,
        'underestimate_count': underestimate_count,
        'adjacent_class_error_ratio': adjacent_ratio,
        'ecological_meanings': ecological_meanings,
        'recommendations': _generate_recommendations(interpretations, adjacent_ratio)
    }


def _generate_recommendations(interpretations: List[Dict], adjacent_ratio: float) -> List[str]:
    """誤分類パターンに基づく改善推奨事項を生成"""
    recommendations = []
    
    if adjacent_ratio > 0.7:
        recommendations.append(
            "隣接クラス間の誤分類が多い。閾値の調整または連続値予測への変更を検討。"
        )
    
    # 特定のクラスペアに関する推奨
    high_error_pairs = [i for i in interpretations if i['rate'] > 0.15]
    for pair in high_error_pairs:
        if pair['is_adjacent_classes']:
            recommendations.append(
                f"{pair['true_ecological']}と{pair['predicted_ecological']}の境界付近のサンプルを増やすことを推奨。"
            )
        else:
            recommendations.append(
                f"{pair['true_ecological']}クラスの学習データの品質を確認することを推奨。"
            )
    
    if not recommendations:
        recommendations.append("現在の分類精度は良好。継続的なモニタリングを推奨。")
    
    return recommendations


def _generate_summary_report(analysis: Dict, save_path: Path, class_names: List[str]):
    """サマリーレポートをテキストファイルとして生成"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"誤分類解析レポート - {analysis['model_name']}\n")
        f.write("Misclassification Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【概要 / Overview】\n")
        f.write(f"  総サンプル数: {analysis['total_samples']:,}\n")
        f.write(f"  誤分類数: {analysis['misclassified_samples']:,}\n")
        f.write(f"  誤分類率: {analysis['misclassification_rate']:.2f}%\n\n")
        
        f.write("【主要な誤分類パターン / Top Misclassification Patterns】\n")
        for pair in analysis['misclassification_pairs'][:5]:
            if pair['rate'] > 0.03:
                f.write(f"  {pair['true_class']} → {pair['predicted_class']}: ")
                f.write(f"{pair['count']:,}件 ({pair['rate']*100:.1f}%)\n")
        f.write("\n")
        
        if analysis['confidence_analysis']:
            f.write("【確信度分析 / Confidence Analysis】\n")
            ca = analysis['confidence_analysis']
            f.write(f"  正解予測の平均確信度: {ca['correct_mean']:.3f} ± {ca['correct_std']:.3f}\n")
            f.write(f"  誤分類の平均確信度: {ca['wrong_mean']:.3f} ± {ca['wrong_std']:.3f}\n")
            f.write(f"  高確信度での誤分類(>0.8): {ca['high_confidence_errors']:,}件\n\n")
        
        if 'boundary_analysis' in analysis and 'boundary_results' in analysis['boundary_analysis']:
            f.write("【閾値境界解析 / Threshold Boundary Analysis】\n")
            for br in analysis['boundary_analysis']['boundary_results']:
                f.write(f"  {br['threshold_pct']}%閾値付近: ")
                f.write(f"誤分類率 下側{br['below_error_rate']*100:.1f}% / 上側{br['above_error_rate']*100:.1f}%\n")
            f.write("\n")
        
        if 'ecological_interpretation' in analysis:
            eco = analysis['ecological_interpretation']
            f.write("【生態学的解釈 / Ecological Interpretation】\n")
            f.write(f"  全体傾向: {eco['overall_trend']}\n")
            f.write(f"  過大評価パターン数: {eco['overestimate_count']}\n")
            f.write(f"  過小評価パターン数: {eco['underestimate_count']}\n")
            f.write(f"  隣接クラス誤分類率: {eco['adjacent_class_error_ratio']*100:.1f}%\n\n")
            
            f.write("【推奨事項 / Recommendations】\n")
            for i, rec in enumerate(eco['recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by class5-v9.py - Ecological Misclassification Analysis Module\n")


def run_misclassification_analysis_for_all_models(results: Dict, config: Dict,
                                                   images: Optional[np.ndarray] = None,
                                                   features: Optional[np.ndarray] = None,
                                                   enable_comprehensive_viz: bool = True,
                                                   enable_confusion_region_viz: bool = True):
    """
    全モデルの誤分類解析を実行
    
    Parameters:
    - results: 各モデルの結果を含む辞書
    - config: 設定辞書
    - images: 画像データ（ギャラリー作成用、オプション）
    - features: 特徴ベクトル（t-SNE/UMAP用、オプション）
    - enable_comprehensive_viz: 包括的可視化を有効化するか
    - enable_confusion_region_viz: 判別困難領域の可視化を有効化するか
    """
    print("\n" + "=" * 80)
    print("🔬 Running Misclassification Analysis for All Models")
    print("=" * 80)
    
    class_names = generate_class_names_from_thresholds(
        config.get('fixed_thresholds'), config['n_classes']
    )
    
    all_analyses = {}
    
    for model_name, model_result in results.items():
        cv_results = model_result.get('cv_results', {})
        predictions_path = cv_results.get('predictions_path')
        
        if not predictions_path or not os.path.exists(predictions_path):
            print(f"⚠️  Skipping {model_name}: predictions file not found")
            continue
        
        try:
            predictions_df = pd.read_csv(predictions_path)
            
            # pred_labelがない場合は予測クラスを計算
            if 'pred_label' not in predictions_df.columns:
                prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
                if prob_cols:
                    predictions_df['pred_label'] = predictions_df[prob_cols].values.argmax(axis=1)
                elif 'predicted_class' in predictions_df.columns:
                    predictions_df['pred_label'] = predictions_df['predicted_class']
                else:
                    print(f"⚠️  Skipping {model_name}: cannot determine predicted labels")
                    continue
            
            model_save_dir = Path(config['output_dir']) / model_name
            
            # 基本的な誤分類パターン解析
            analysis = analyze_misclassification_patterns(
                predictions_df, class_names, model_save_dir, model_name
            )
            
            # =========================================================
            # 判別困難領域の可視化（新機能）
            # =========================================================
            if enable_confusion_region_viz:
                try:
                    print(f"\n  🔍 Creating confusion region visualization for {model_name}...")
                    confusion_results = visualize_model_confusion_regions(
                        predictions_df=predictions_df,
                        original_image=None,  # 背景画像（オプション）
                        save_dir=model_save_dir,
                        model_name=model_name,
                        grid_resolution=50,
                        class_names=class_names,
                        output_geotiff=True
                    )
                    analysis['confusion_region_analysis'] = confusion_results
                    print(f"    ✅ Confusion region analysis complete for {model_name}")
                except Exception as cr_e:
                    print(f"    ⚠️ Confusion region visualization failed for {model_name}: {cr_e}")
                    import traceback
                    traceback.print_exc()
                    analysis['confusion_region_analysis'] = {'error': str(cr_e)}
            
            # 包括的な可視化（オプション）
            if enable_comprehensive_viz:
                try:
                    viz_results = run_comprehensive_visualization(
                        predictions_df=predictions_df,
                        images=images,
                        features=features,
                        class_names=class_names,
                        save_dir=model_save_dir,
                        model_name=model_name
                    )
                    analysis['comprehensive_visualization'] = viz_results
                except Exception as viz_e:
                    print(f"  ⚠️ Comprehensive visualization failed for {model_name}: {viz_e}")
                    analysis['comprehensive_visualization'] = {'error': str(viz_e)}
            
            all_analyses[model_name] = analysis
            
        except Exception as e:
            print(f"⚠️  Error analyzing {model_name}: {e}")
            continue
    
    # 全モデル比較サマリーを作成
    if all_analyses:
        _create_model_comparison_summary(all_analyses, config['output_dir'], class_names)
    
    return all_analyses


def _create_model_comparison_summary(all_analyses: Dict, output_dir: Path, class_names: List[str]):
    """全モデルの誤分類パターンを比較するサマリーを作成"""
    output_dir = Path(output_dir)
    comparison_dir = output_dir / 'misclassification_comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # DataFrameで比較表を作成
    comparison_data = []
    for model_name, analysis in all_analyses.items():
        row = {
            'Model': model_name,
            'Total Samples': analysis['total_samples'],
            'Misclassified': analysis['misclassified_samples'],
            'Error Rate (%)': round(analysis['misclassification_rate'], 2)
        }
        
        # 確信度情報があれば追加
        if analysis['confidence_analysis']:
            row['Correct Conf Mean'] = round(analysis['confidence_analysis']['correct_mean'], 3)
            row['Wrong Conf Mean'] = round(analysis['confidence_analysis']['wrong_mean'], 3)
        
        # 生態学的解釈があれば追加
        if 'ecological_interpretation' in analysis:
            eco = analysis['ecological_interpretation']
            row['Overestimate Count'] = eco['overestimate_count']
            row['Underestimate Count'] = eco['underestimate_count']
            row['Adjacent Error Ratio'] = round(eco['adjacent_class_error_ratio'] * 100, 1)
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Error Rate (%)')
    
    # CSVとして保存
    comparison_df.to_csv(comparison_dir / 'model_comparison.csv', index=False)
    
    # 可視化
    _plot_model_comparison(comparison_df, comparison_dir / 'model_comparison.png')
    
    print(f"\n📊 Model comparison saved to: {comparison_dir}")


def _plot_model_comparison(df: pd.DataFrame, save_path: Path):
    """モデル間の誤分類率比較を可視化"""
    palette = {
        'mist': '#bec8d1',
        'cream': '#fffeec',
        'ice': '#c3e5e7',
        'teal': '#86cecb',
        'deep_teal': '#137a7f',
        'magenta': '#e12885',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('black')
    
    # 左図：誤分類率の比較
    ax1 = axes[0]
    ax1.set_facecolor('black')
    bars = ax1.barh(df['Model'], df['Error Rate (%)'], color=palette['teal'], alpha=0.8)
    ax1.set_xlabel('Error Rate (%)', color=palette['cream'])
    ax1.set_title('Misclassification Rate by Model', color=palette['cream'])
    ax1.tick_params(colors=palette['cream'])
    for spine in ax1.spines.values():
        spine.set_color(palette['cream'])
    
    # 右図：過大評価 vs 過小評価
    if 'Overestimate Count' in df.columns:
        ax2 = axes[1]
        ax2.set_facecolor('black')
        x = np.arange(len(df))
        width = 0.35
        ax2.barh(x - width/2, df['Overestimate Count'], width, 
                label='Overestimate', color=palette['magenta'], alpha=0.8)
        ax2.barh(x + width/2, df['Underestimate Count'], width,
                label='Underestimate', color=palette['ice'], alpha=0.8)
        ax2.set_yticks(x)
        ax2.set_yticklabels(df['Model'])
        ax2.set_xlabel('Count', color=palette['cream'])
        ax2.set_title('Over/Underestimate Patterns', color=palette['cream'])
        ax2.legend(facecolor='black', edgecolor=palette['cream'], labelcolor=palette['cream'])
        ax2.tick_params(colors=palette['cream'])
        for spine in ax2.spines.values():
            spine.set_color(palette['cream'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close(fig)


# *******************************************************************
# Main Training Pipeline with Hyperparameter Search
# *******************************************************************
def train_with_hyperparameter_search(model_name, images, labels, pixel_locations, 
                                    geo_locations, n_classes=5, n_trials=50, 
                                    device='cuda', save_dir='results',
                                    dem_range=None, vdvi_range=None,
                                    fixed_thresholds=None):
    """
    ハイパーパラメータ探索を含む訓練パイプライン
    """
    print(f"\n{'='*80}")
    print(f"🔍 Hyperparameter Search for {model_name}")
    print(f"{'='*80}")
    print(f"Total trials to run: {n_trials}")
    print(f"Device: {device}")
    print(f"Number of classes: {n_classes}")
    print(f"{'='*80}")
    
    # データの分割（訓練用と検証用）
    from sklearn.model_selection import train_test_split
    
    # クラスラベルの生成（層化分割用）
    class_labels = []
    thresholds_for_split = fixed_thresholds if fixed_thresholds is not None else [i / n_classes for i in range(1, n_classes)]
    for label in labels:
        for i, threshold in enumerate(thresholds_for_split):
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
    
    print(f"\n📊 Dataset Split:")
    print(f"  Training samples: {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")
    
    # データセットの作成
    train_dataset = LandslideDataset(
        images[train_idx], labels[train_idx],
        pixel_locations[train_idx], geo_locations[train_idx],
        n_classes=n_classes,
        thresholds=fixed_thresholds,
        dem_range=dem_range,
        vdvi_range=vdvi_range
    )
    val_dataset = LandslideDataset(
        images[val_idx], labels[val_idx],
        pixel_locations[val_idx], geo_locations[val_idx],
        n_classes=n_classes,
        thresholds=fixed_thresholds,
        dem_range=dem_range,
        vdvi_range=vdvi_range
    )
    
    # Optunaでハイパーパラメータ探索
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name=f'{model_name}_optimization'
    )
    
    # カスタムコールバック関数を追加（進捗状況の表示用）
    def print_trial_summary(study, trial):
        print(f"\n{'─'*70}")
        print(f"📊 Study Progress Summary:")
        print(f"  Completed trials: {len(study.trials)}")
        print(f"  Current best value: {study.best_value:.4f}")
        if len(study.trials) >= 5:
            recent_values = [t.value for t in study.trials[-5:] if t.value is not None]
            if recent_values:
                print(f"  Recent 5 trials average: {np.mean(recent_values):.4f}")
        print(f"{'─'*70}\n")
    
    objective = create_objective(
        model_name, train_dataset, val_dataset, 
        n_classes, device, n_epochs=30
    )
    
    print(f"\n🚀 Starting {n_trials} optimization trials...")
    print(f"{'='*80}")
    
    # Optunaの最適化を実行（進捗バーを無効化し、カスタム表示を使用）
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=False,
        callbacks=[print_trial_summary]
    )
    
    # 最良のハイパーパラメータを取得
    best_params = study.best_params
    best_value = study.best_value
    best_in_channels = study.best_trial.user_attrs.get('in_channels', int(train_dataset[0]['image'].shape[0]))
    best_params['in_channels'] = best_in_channels
    
    print(f"\n{'='*80}")
    print(f"🎉 Optimization Complete!")
    print(f"{'='*80}")
    print(f"\n📊 Best Trial Results:")
    print(f"  Best F1 macro score: {best_value:.4f}")
    print(f"  Best trial number: {study.best_trial.number}")
    print(f"\n📋 Best Hyperparameters:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")
    
    # トライアルの統計情報
    print(f"\n📈 Trial Statistics:")
    all_values = [t.value for t in study.trials if t.value is not None]
    if all_values:
        print(f"  Completed trials: {len(all_values)}/{n_trials}")
        print(f"  Average F1 score: {np.mean(all_values):.4f}")
        print(f"  Std deviation: {np.std(all_values):.4f}")
        print(f"  Min F1 score: {np.min(all_values):.4f}")
        print(f"  Max F1 score: {np.max(all_values):.4f}")
        
        # Top 5 trials
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else -1, reverse=True)[:5]
        print(f"\n🏆 Top 5 Trials:")
        for i, trial in enumerate(sorted_trials, 1):
            if trial.value:
                print(f"  {i}. Trial #{trial.number}: F1={trial.value:.4f}")
    
    # 結果を保存
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # ハイパーパラメータをJSONファイルに保存
    with open(os.path.join(model_save_dir, 'best_hyperparams.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # 全トライアルの結果も保存
    trial_results = []
    for trial in study.trials:
        trial_dict = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state),
            'in_channels': trial.user_attrs.get('in_channels')
        }
        trial_results.append(trial_dict)
    
    with open(os.path.join(model_save_dir, 'all_trials.json'), 'w') as f:
        json.dump(trial_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {model_save_dir}")
    
    # Optunaの可視化
    try:
        import optuna.visualization as vis
        
        # 最適化履歴
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(model_save_dir, 'optimization_history.html'))
        
        # パラメータの重要度
        if len(study.trials) >= 10:  # 十分なトライアルがある場合のみ
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(model_save_dir, 'param_importances.html'))
        
        # パラメータの関係
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(model_save_dir, 'parallel_coordinate.html'))
        
        print(f"📊 Visualization files created in {model_save_dir}")
        
    except Exception as e:
        print(f"⚠️  Could not create Optuna visualizations: {e}")
    
    print(f"{'='*80}\n")
    
    return best_params


# *******************************************************************
# Process Multiple Datasets
# *******************************************************************
def process_multiple_datasets(dataset_configs, landslide_path, epsg_code=6675, 
                             tip_size_m=5, target_size_m=None, slide_ratio=0.5, cache_dir=None,
                             required_channels: Optional[int] = None,
                             alignment_preview_dir=None,
                             patch_overlay_dir=None,
                             glcm_root=None,
                             exg_root=None,
                             use_exg=True):
    """
    複数のデータセットを処理して統合
    
    Parameters:
    -----------
    tip_size_m : float
        Size of input patch in meters (Field of View). Default: 5m.
        With Center-Labeling strategy, this represents the context area (e.g., 20m).
    target_size_m : float, optional
        Size of center region in meters for label calculation. Default: None (same as tip_size_m).
        When specified, uses Center-Labeling strategy where labels are computed from
        the central target_size_m region while input includes full tip_size_m context.
    glcm_root : str or Path, optional
        GLCMファイルのルートディレクトリ。
        ファイルは {glcm_root}/{dataset_id}/{dataset_id}_contrast.npy の形式で配置されている想定
    exg_root : str or Path, optional
        EXG（Excess Green Index）ファイルのルートディレクトリ。
        ファイルは {exg_root}/{dataset_id}.npy の形式で配置されている想定
    """
    all_tips = []
    all_labels = []
    all_pixel_locs = []
    all_geo_locs = []
    all_landslide_ids = []
    all_dataset_names = []
    processed_dataset_names = []
    channel_origin = []  # record original channel counts before padding
    
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    alignment_preview_dir = Path(alignment_preview_dir) if alignment_preview_dir else None
    if alignment_preview_dir is not None:
        alignment_preview_dir.mkdir(parents=True, exist_ok=True)
    patch_overlay_dir = Path(patch_overlay_dir) if patch_overlay_dir else None
    if patch_overlay_dir is not None:
        patch_overlay_dir.mkdir(parents=True, exist_ok=True)
    glcm_root = Path(glcm_root) if glcm_root else None
    exg_root = Path(exg_root) if exg_root else None

    for config in dataset_configs:
        print(f"\nProcessing dataset: {config['dataset_name']}...")

        cached = False
        cache_file = None
        if cache_dir is not None:
            cache_file = cache_dir / f"{config['dataset_name']}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_payload = pickle.load(f)
                    tips = np.asarray(cache_payload['tips'], dtype=np.float32)
                    labels = np.asarray(cache_payload['labels'], dtype=np.float32)
                    pixel_locs = np.asarray(cache_payload['pixel_locs'], dtype=np.int64)
                    geo_locs = np.asarray(cache_payload['geo_locs'], dtype=np.float64)
                    landslide_ids = np.asarray(cache_payload['landslide_ids'], dtype=np.int64)
                    # Validate channel count against current requirement (if provided)
                    if required_channels is not None and tips.ndim == 4 and tips.shape[-1] != required_channels:
                        print(f"  ⚠️  Cached channels ({tips.shape[-1]}) != required ({required_channels}); recomputing {cache_file.name}...")
                        cached = False
                    else:
                        cached = True
                        print(f"  ♻️  Loaded cached patches from {cache_file.name}")
                except Exception as exc:
                    print(f"  ⚠️  Failed to load cache {cache_file.name}: {exc}. Recomputing...")
                    cached = False
        
        dem_path = config.get('dem_path')
        vdvi_path = config.get('vdvi_path')
        
        # GLCMパスの解決
        glcm_path = config.get('glcm_path')
        if glcm_path is None and glcm_root is not None:
            # glcm_root/{dataset_id}/{dataset_id}_contrast.npy の形式で探索
            dataset_id = config['dataset_name']
            glcm_candidate = glcm_root / dataset_id / f"{dataset_id}_contrast.npy"
            if glcm_candidate.exists():
                glcm_path = glcm_candidate
                print(f"  ✅ Found GLCM file: {glcm_path}")
            else:
                # 代替パス: glcm_root/{dataset_id}_contrast.npy
                glcm_candidate_alt = glcm_root / f"{dataset_id}_contrast.npy"
                if glcm_candidate_alt.exists():
                    glcm_path = glcm_candidate_alt
                    print(f"  ✅ Found GLCM file (alt): {glcm_path}")
                else:
                    print(f"  ⚠️  GLCM file not found for {dataset_id}")
        
        # EXGパスの解決
        exg_path = config.get('exg_path')
        if exg_path is None and exg_root is not None:
            # exg_root/{dataset_id}.npy の形式で探索（VDVIと同じ形式）
            dataset_id = config['dataset_name']
            exg_candidate = exg_root / f"{dataset_id}.npy"
            if exg_candidate.exists():
                exg_path = exg_candidate
                print(f"  ✅ Found EXG file: {exg_path}")
            else:
                print(f"  ⚠️  EXG file not found for {dataset_id}")
        
        preview_needed = False
        if alignment_preview_dir is not None and dem_path:
            preview_file = alignment_preview_dir / f"{config['dataset_name']}_dem_alignment.png"
            preview_needed = not preview_file.exists()
        overlay_needed = False
        if patch_overlay_dir is not None:
            rgb_sources = config['rgb_path'] if isinstance(config['rgb_path'], (list, tuple)) else [config['rgb_path']]
            overlay_needed = any(
                not (patch_overlay_dir / f"{Path(path).stem}_patch_overlay.png").exists()
                for path in rgb_sources
            )

        processor = None
        if (not cached) or preview_needed or overlay_needed:
            processor = LandslideTipProcessor(
                rgb_paths=config['rgb_path'],
                landslide_path=str(landslide_path),
                legacy_path=str(config['legacy_path']),
                dem_paths=dem_path if dem_path else None,
                vdvi_paths=vdvi_path if vdvi_path else None,
                glcm_paths=glcm_path if glcm_path else None,
                exg_paths=exg_path if exg_path else None,
                epsg_code=epsg_code,
                resolution=0.2,
                alignment_preview_dir=alignment_preview_dir,
                patch_overlay_dir=patch_overlay_dir,
            )

        if not cached:
            assert processor is not None
            tips, labels, pixel_locs, geo_locs, landslide_ids = processor.extract_tips(
                tip_size_m=tip_size_m,
                target_size_m=target_size_m,
                slide_ratio=slide_ratio,
                use_exg=use_exg,
            )
            if preview_needed:
                processor.generate_alignment_preview()
        else:
            if overlay_needed and processor is not None:
                processor.extract_tips(
                    tip_size_m=tip_size_m,
                    target_size_m=target_size_m,
                    slide_ratio=slide_ratio,
                    collect_data=False,
                    show_progress=False,
                    use_exg=use_exg,
                )
            if preview_needed and processor is not None:
                processor.generate_alignment_preview()
        
        if len(tips) == 0:
            print(f"  ⚠️  No tips extracted from {config['dataset_name']}. Skipping this dataset.")
            continue

        print(f"  Extracted {len(tips)} tips from {config['dataset_name']}")
        print(f"  Label distribution: min={labels.min():.3f}, max={labels.max():.3f}, mean={labels.mean():.3f}")

        if not cached and cache_file is not None:
            try:
                payload = {
                    'tips': tips.astype(np.float32),
                    'labels': labels.astype(np.float32),
                    'pixel_locs': pixel_locs.astype(np.int64),
                    'geo_locs': geo_locs.astype(np.float64),
                    'landslide_ids': landslide_ids.astype(np.int64),
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"  💾 Cached patches to {cache_file.name}")
            except Exception as exc:
                print(f"  ⚠️  Failed to cache dataset {config['dataset_name']}: {exc}")
        
        channel_origin.append(tips.shape[-1] if tips.ndim == 4 else 0)
        all_tips.append(tips)
        all_labels.append(labels)
        all_pixel_locs.append(pixel_locs)
        all_geo_locs.append(geo_locs)
        all_landslide_ids.append(landslide_ids)
        all_dataset_names.extend([config['dataset_name']] * len(tips))
        processed_dataset_names.append(config['dataset_name'])
    
    # 全データを結合
    if not all_tips:
        raise ValueError("No tips were extracted from the provided datasets. Verify input paths and annotations.")

    observed_channels = [t.shape[-1] for t in all_tips]
    max_channels = max(observed_channels)
    target_channels = required_channels if required_channels is not None else max_channels
    if required_channels is not None and target_channels != max_channels:
        if target_channels > max_channels:
            print(f"  ℹ️  Forcing channel count to {target_channels} (observed max={max_channels}). Missing bands will be zero-padded.")
        else:
            print(f"  ℹ️  Forcing channel count to {target_channels} (observed max={max_channels}). Extra bands will be truncated.")

    # Align channel count across datasets so downstream models receive consistent tensors
    if any(t.shape[-1] != target_channels for t in all_tips):
        normalized_tips = []
        for idx, tips in enumerate(all_tips):
            current = tips
            current_channels = tips.shape[-1]
            if current_channels < target_channels:
                channel_diff = target_channels - current_channels
                padding = np.zeros(
                    (tips.shape[0], tips.shape[1], tips.shape[2], channel_diff),
                    dtype=tips.dtype
                )
                current = np.concatenate([tips, padding], axis=-1)
                print(
                    f"  ⚠️  Padded dataset {processed_dataset_names[idx]} with {channel_diff} zero channel(s) to match {target_channels} channels. (original={channel_origin[idx]})"
                )
            elif current_channels > target_channels:
                current = tips[..., :target_channels]
                print(
                    f"  ⚠️  Trimmed dataset {processed_dataset_names[idx]} from {current_channels} to {target_channels} channels."
                )
            normalized_tips.append(current)
        all_tips = normalized_tips

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
def compose_trial_name(
    output_root: Path,
    base_prefix: str,
    model_names: List[str],
    use_dem: bool,
    use_vdvi: bool,
) -> str:
    """Build versioned trial name reflecting model usage and auxiliary rasters."""
    tags = []
    if model_names:
        tags.append("+".join(model_names))
    if use_dem:
        tags.append("dem")
    if use_vdvi:
        tags.append("vdvi")

    name_stem = "-".join([base_prefix] + tags)
    pattern = re.compile(rf"{re.escape(name_stem)}-v(\d+)$")
    max_version = 0
    if output_root.exists():
        for entry in output_root.iterdir():
            if not entry.is_dir():
                continue
            match = pattern.match(entry.name)
            if match:
                max_version = max(max_version, int(match.group(1)))

    version = max_version + 1 if max_version > 0 else 1
    candidate = f"{name_stem}-v{version:03d}"
    while (output_root / candidate).exists():
        version += 1
        candidate = f"{name_stem}-v{version:03d}"
    return candidate


def main():
    """メイン実行関数"""
    
    # pyprojrootを使用してプロジェクトルートを取得
    try:
        from pyprojroot import here
        root_dir = here()
    except ImportError:
        print("pyprojroot not found, using current directory as root")
        root_dir = Path(".")
    
    base_trial_prefix = "Noto-class5"
    # 利用可能なモデル一覧:
    # 既存モデル: "CNN", "MLP-Mixer", "ViT", "HybridModel", "HybridModelV2", "TransformerV1", "TransformerV2", "TransformerV3", "HybridTransformer"
    # 中間値予測特化モデル: "IntermediateValueTransformer", "IntermediateFocusedCNN"
    model_names = ["CNN", "MLP-Mixer","HybridModel"]
    # 中間値予測モデルを追加する場合:
    # model_names = ["IntermediateValueTransformer", "IntermediateFocusedCNN"]
    use_dem = True
    use_vdvi = True

    output_root = root_dir / "output"
    trial_name = compose_trial_name(output_root, base_trial_prefix, model_names, use_dem, use_vdvi)

    # ディレクトリの設定
    data_dir = root_dir / 'data'
    data_annotation_dir = data_dir / "annotation"
    data_Noto_April_dir = data_dir / "Noto_April"
    data_sample_dir = data_Noto_April_dir / "annotation_sample"
    dem_dir = data_dir / 'dem'
    #dem_dir = Path("/home/ge/Documents/DEM_Noto")#6000マシンで回す場合こっち
    vdvi_root = data_dir / 'vdvi'
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / trial_name
    fig_dir = output_dir / "figs"

    expected_in_channels = 3 + int(use_dem) + int(use_vdvi)
    
    # ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    dem_alignment_dir = fig_dir / "dem_alignment_previews"
    if use_dem:
        dem_alignment_dir.mkdir(parents=True, exist_ok=True)
    else:
        dem_alignment_dir = None
    patch_overlay_dir = fig_dir / "patch_overlays"
    patch_overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # データセット設定
    dataset_entries = [
        #('07ed874', data_sample_dir / '07ed874.jpg', data_annotation_dir / '07ed874' / '07ed874.shp'),
        #('07ed882', data_sample_dir / '07ed882.jpg', data_annotation_dir / '07ed882' / '07ed882.shp'),
        #('07ed921', data_sample_dir / '07ed921.jpg', data_annotation_dir / '07ed921' / '07ed921.geojson'),
        #('07ed922', data_sample_dir / '07ed922.jpg', data_annotation_dir / '07ed922' / '07ed922.shp'),
        ('07ee703', data_sample_dir / '07ee703.jpg', data_annotation_dir / '07ee703' / '07ee703.geojson'),
        ('07fd041', data_sample_dir / '07fd041.jpg', data_annotation_dir / 'Porigon_april_fix.geojson'),
        ('07fd111', data_sample_dir / '07fd111.jpg', data_annotation_dir / '07fd111' / '07fd111.geojson'),
        ('07ed684', data_sample_dir / '07ed684.jpg', data_annotation_dir / '07ed684' / '07ed684.geojson'),
    ]

    shared_dem_path = None
    if use_dem:
        preferred_dem_path = Path(r"C:\Users\GE\pj24_legacy-kaken-chami\Noto\data\dem\mergedem.tif")
        if preferred_dem_path.exists():
            projected_path = ensure_dem_reprojected(preferred_dem_path, "EPSG:6675")
            if projected_path is not None:
                shared_dem_path = projected_path
                print(f"ℹ️  Using shared DEM raster for all datasets: {shared_dem_path}")
            else:
                print(f"⚠️  Failed to prepare projected DEM from {preferred_dem_path}.")
        else:
            print(f"⚠️  Preferred DEM raster not found at {preferred_dem_path}. Falling back to automatic mosaic discovery.")
            shared_dem_path = ensure_dem_mosaic(dem_dir)
            if shared_dem_path is not None:
                print(f"ℹ️  Using shared DEM mosaic: {shared_dem_path}")
            else:
                print("⚠️  Shared DEM mosaic could not be prepared. Per-dataset search will be used.")

    def resolve_dem_path(dataset_name: str):
        direct_candidates = [
            dem_dir / f"{dataset_name}.vrt",
            dem_dir / f"{dataset_name}.tif",
            dem_dir / f"dem_{dataset_name}.vrt",
            dem_dir / f"dem_{dataset_name}.tif",
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return candidate, None

        search_tokens = [dataset_name, dataset_name[:6], dataset_name[:5], dataset_name[:4], dataset_name[-3:]]
        for token in filter(None, search_tokens):
            matches = sorted(itertools.chain(dem_dir.glob(f"*{token}*.vrt"), dem_dir.glob(f"*{token}*.tif")))
            if len(matches) == 1:
                return matches[0], None
            if len(matches) > 1:
                message = (
                    f"Multiple DEM candidates found for {dataset_name}: "
                    f"{', '.join(m.name for m in matches)}. Using {matches[0].name}."
                )
                return matches[0], message

        return None, f"DEM not found for {dataset_name}. Update dataset configuration manually."

    dataset_configs = []
    for dataset_name, rgb_path, legacy_path in dataset_entries:
        vdvi_candidate = None
        vdvi_warning = None
        if use_vdvi:
            candidate = vdvi_root / f'{dataset_name}.npy'
            if candidate.exists():
                vdvi_candidate = candidate
            else:
                vdvi_warning = f"VDVI file not found for {dataset_name} at {candidate}."

        dem_candidate = None
        dem_warning = None
        if use_dem:
            if shared_dem_path is not None:
                dem_candidate = shared_dem_path
            else:
                dem_candidate, dem_warning = resolve_dem_path(dataset_name)

        if dem_warning:
            print(f"⚠️  {dem_warning}")
        if vdvi_warning:
            print(f"⚠️  {vdvi_warning}")

        dataset_configs.append({
            'rgb_path': rgb_path,
            'dataset_name': dataset_name,
            'legacy_path': legacy_path,
            'dem_path': dem_candidate,
            'vdvi_path': vdvi_candidate
        })
    
    # 設定パラメータ
    config = {
        'dataset_configs': dataset_configs,
        'landslide_path': data_dir / 'noto_landslide' / 'hokai-all.geojson',
        'output_dir': output_dir,
        'fig_dir': fig_dir,
        'epsg_code': 6675,
        'tip_size_m': 5,
        'slide_ratio': 0.5,
        'n_outer_cv': 8,  # クロスバリデーションのfold数
        'n_epochs': 30,  # エポック数を30に変更
        'batch_size': 64,  # バッチサイズ（CVでは大きめに調整）
        'model_names': model_names,  # 使用するモデル
        'n_trials': 32,  # ハイパーパラメータ探索のトライアル数
        'n_classes': 5,  # クラス数（5で固定）
        'optimize_threshold': False,  # 閾値最適化を行うか
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # ハイパーパラメータ探索のON/OFF
        'hyperparameter_search': True,  # Falseで探索をスキップ
        
        # Spatial K-Fold設定
        'use_spatial_kfold': True,  # Balanced Spatial K-Foldを使用
        'min_points_per_fold': 1000,  # 各foldの最小サンプル数
        'visualize_folds': True,  # foldの空間分布を可視化
        'base_seed': 42,  # ランダムシード
        'dem_dir': dem_dir,
        'vdvi_root': vdvi_root,
        'use_dem': use_dem,
        'use_vdvi': use_vdvi,
        'dem_range': (-100.0, 1500.0) if use_dem else None,
        'vdvi_range': (-1.0, 1.0) if use_vdvi else None,
        'expected_in_channels': expected_in_channels,
        'cache_dir': data_dir / 'pkl',
        'fixed_thresholds': FIVE_CLASS_FIXED_THRESHOLDS.copy(),
        'dem_alignment_dir': dem_alignment_dir,
        'patch_overlay_dir': patch_overlay_dir,
        
        # 事前定義されたハイパーパラメータ（探索をOFFにした場合に使用）
        'predefined_hyperparams': {
            'ResNet18': {
                'pretrained': True,
                'learning_rate': 0.00042470585622618684,
                'batch_size': 64,  # 32から64に増加
                'weight_decay': 0.0009133995846860973,
                'dropout': 0.3,  # 固定値
                'in_channels': expected_in_channels
            },
        # CNN用の定義を追加
            'CNN': {
                'hidden_dim': 128,  # ResNetよりシンプルにするため 128 から減らす
                'depth': 4,
                'kernel_size': 3,
                'learning_rate': 0.001,
                'batch_size': 128,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            'MLP-Mixer': {
                'patch_size': 5,
                'dim': 5,
                'depth': 4,
                'token_dim': 6,
                'channel_dim': 7,
                'learning_rate': 0.001,
                'batch_size': 128,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            'ViT': {
                'patch_size': 1,
                'dim': 4,
                'depth': 4,
                'heads': 2,
                'mlp_dim': 6,
                'dropout': 0.35,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            # 新規追加: ハイブリッドCNN-ViTモデル
            'HybridModel': {
                'image_size': 25,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            # 新規追加: 基本Transformer（RGB only）+ MLP（Aux）
            'TransformerV1': {
                'image_size': 25,
                'patch_size': 5,
                'embed_dim': 128,
                'depth': 4,
                'num_heads': 4,
                'mlp_ratio': 2.0,
                'dropout': 0.3,
                'attn_drop': 0.0,
                'drop_path': 0.1,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            # 新規追加: Cross-Modal Transformer
            'TransformerV2': {
                'image_size': 25,
                'patch_size': 5,
                'embed_dim': 128,
                'depth': 4,
                'num_heads': 4,
                'mlp_ratio': 2.0,
                'dropout': 0.3,
                'attn_drop': 0.0,
                'drop_path': 0.1,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            # 新規追加: Gated Multimodal Transformer（軽量版）
            'TransformerV3': {
                'image_size': 25,
                'patch_size': 5,
                'embed_dim': 96,
                'depth': 3,
                'num_heads': 3,
                'mlp_ratio': 2.0,
                'dropout': 0.3,
                'attn_drop': 0.1,
                'drop_path': 0.15,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            # 新規追加: CNN + Transformerハイブリッド
            'HybridTransformer': {
                'image_size': 25,
                'embed_dim': 128,
                'depth': 3,
                'num_heads': 4,
                'mlp_ratio': 2.0,
                'dropout': 0.3,
                'attn_drop': 0.0,
                'drop_path': 0.1,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            # ========================================
            # 中間値予測に特化したモデル群
            # ========================================
            'IntermediateValueTransformer': {
                'image_size': 25,
                'patch_size': 5,
                'embed_dim': 128,
                'depth': 4,
                'num_heads': 4,
                'mlp_ratio': 2.0,
                'dropout': 0.1,
                'learning_rate': 0.0008,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            },
            'IntermediateFocusedCNN': {
                'base_channels': 32,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 0.0001,
                'in_channels': expected_in_channels
            }
        }
    }
    
    print(f"{'='*80}")
    print(f"Landslide Classification Pipeline with ResNet18")
    print(f"{'='*80}")
    print(f"Trial name: {trial_name}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {config['device']}")
    print(f"Number of datasets: {len(dataset_configs)}")
    print(f"Models to train: {config['model_names']}")
    print(f"Number of classes: {config['n_classes']}")
    print(f"Use DEM: {config['use_dem']} | Use VDVI: {config['use_vdvi']}")
    print(f"{'='*80}")
    
    # 複数データセットの処理と統合
    print("\nProcessing multiple datasets...")
    all_tips, all_labels, all_pixel_locs, all_geo_locs, all_landslide_ids, all_dataset_names = \
        process_multiple_datasets(
            dataset_configs=config['dataset_configs'],
            landslide_path=config['landslide_path'],
            epsg_code=config['epsg_code'],
            tip_size_m=config['tip_size_m'],
            slide_ratio=config['slide_ratio'],
            cache_dir=config.get('cache_dir'),
            required_channels=config.get('expected_in_channels'),
            alignment_preview_dir=config.get('dem_alignment_dir'),
            patch_overlay_dir=config.get('patch_overlay_dir')
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
        print(f"🤖 Processing {model_name}")
        print(f"{'='*80}")
        
        # 結果保存ディレクトリの作成
        model_output_dir = config['output_dir'] / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ハイパーパラメータの取得または探索
        if config['hyperparameter_search']:
            # ハイパーパラメータ探索を実行
            print(f"\n🔍 Starting hyperparameter search with {config['n_trials']} trials...")
            best_params = train_with_hyperparameter_search(
                model_name=model_name,
                images=all_tips,
                labels=all_labels,
                pixel_locations=all_pixel_locs,
                geo_locations=all_geo_locs,
                n_classes=config['n_classes'],
                n_trials=config['n_trials'],
                device=config['device'],
                save_dir=str(model_output_dir),
                dem_range=config.get('dem_range'),
                vdvi_range=config.get('vdvi_range'),
                fixed_thresholds=config.get('fixed_thresholds')
            )
        else:
            # 事前定義されたハイパーパラメータを使用
            print(f"\n📋 Using predefined hyperparameters (skipping search)")
            if model_name in config['predefined_hyperparams']:
                best_params = config['predefined_hyperparams'][model_name].copy()
                print(f"  Loaded hyperparameters for {model_name}:")
                for key, value in best_params.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.6f}")
                    else:
                        print(f"    {key}: {value}")
                
                # ハイパーパラメータをファイルに保存（記録用）
                with open(model_output_dir / 'used_hyperparams.json', 'w') as f:
                    json.dump(best_params, f, indent=2)
            else:
                raise ValueError(f"No predefined hyperparameters found for {model_name}")
        
        # ResNet18の場合、dropoutを0.3に固定
        if model_name == 'ResNet18':
            best_params['dropout'] = 0.3
        
        if all_tips.ndim == 4 and all_tips.shape[-1]:
            best_params['in_channels'] = all_tips.shape[-1]
        else:
            best_params.setdefault('in_channels', 3)
        
        # 最適化されたハイパーパラメータでクロスバリデーション
        print(f"\n🔄 Performing {config['n_outer_cv']}-fold cross-validation...")
        print(f"  Using {config['n_epochs']} epochs per fold")
        print(f"  Batch size: {best_params.get('batch_size', config['batch_size'])}")
        if config['use_spatial_kfold']:
            print(f"  Using Balanced Spatial K-Fold")
        else:
            print(f"  Using Stratified K-Fold")
        
        cv_results = cross_validate_with_confusion_matrix(
            model_name=model_name,
            hyperparams=best_params,
            images=all_tips,
            labels=all_labels,
            pixel_locations=all_pixel_locs,
            geo_locations=all_geo_locs,
            landslide_ids=all_landslide_ids,  # 追加
            n_classes=config['n_classes'],
            n_folds=config['n_outer_cv'],
            n_epochs=config['n_epochs'],
            batch_size=config['batch_size'],
            device=config['device'],
            save_dir=str(model_output_dir),
            optimize_threshold=config['optimize_threshold'],
            use_spatial_kfold=config['use_spatial_kfold'],  # 追加
            config=config  # 追加
        )
        
        results[model_name] = {
            'cv_results': cv_results,
            'best_params': best_params,
            'dataset_names': all_dataset_names,
            'predictions_path': cv_results.get('predictions_path')
        }

    ensemble_summary = evaluate_and_save_ensemble(config['model_names'], results, config)
    if ensemble_summary is not None:
        results['Ensemble'] = {
            'cv_results': ensemble_summary,
            'best_params': None,
            'dataset_names': all_dataset_names,
            'predictions_path': ensemble_summary.get('predictions_path')
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
    
    # =========================================================
    # 誤分類の生態学的解析を実行
    # =========================================================
    print(f"\n{'='*80}")
    print("🔬 Running Ecological Misclassification Analysis")
    print(f"{'='*80}")
    
    try:
        misclass_analyses = run_misclassification_analysis_for_all_models(results, config)
        
        if misclass_analyses:
            print(f"\n✅ Misclassification analysis completed for {len(misclass_analyses)} models")
            
            # 最も誤分類率が低いモデルを特定
            best_model = min(misclass_analyses.items(), 
                           key=lambda x: x[1]['misclassification_rate'])
            print(f"🏆 Best model (lowest error rate): {best_model[0]} "
                  f"({best_model[1]['misclassification_rate']:.2f}%)")
            
            # 生態学的解釈のハイライト
            for model_name, analysis in misclass_analyses.items():
                if 'ecological_interpretation' in analysis:
                    eco = analysis['ecological_interpretation']
                    print(f"\n📊 {model_name} Ecological Summary:")
                    print(f"   Overall trend: {eco['overall_trend']}")
                    if eco['recommendations']:
                        print(f"   Top recommendation: {eco['recommendations'][0]}")
        else:
            print("⚠️  No misclassification analyses were generated")
            
    except Exception as e:
        print(f"⚠️  Error during misclassification analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"All results saved to {config['output_dir']}")
    print("Pipeline completed successfully!")
    print(f"{'='*80}")


# *******************************************************************
# Standalone Function: Visualize Model Confusion from CSV
# *******************************************************************
def analyze_confusion_from_predictions_csv(
    predictions_csv: str,
    output_dir: str,
    model_name: str = "Model",
    class_names: Optional[List[str]] = None,
    n_classes: int = 5,
    grid_resolution: int = 50
) -> Dict:
    """
    予測結果のCSVファイルから判別困難領域を可視化するスタンドアロン関数
    
    使い方:
    -------
    ```python
    from class5_v9 import analyze_confusion_from_predictions_csv
    
    result = analyze_confusion_from_predictions_csv(
        predictions_csv="output/CNN/all_predictions.csv",
        output_dir="output/CNN",
        model_name="CNN",
        n_classes=5,
        grid_resolution=50
    )
    ```
    
    Parameters:
    -----------
    predictions_csv : str
        予測結果のCSVファイルパス。以下の列が必要:
        - 'x' または 'lon': x座標
        - 'y' または 'lat': y座標  
        - 'true_label': 真のラベル
        - 'pred_label' または 'predicted_class': 予測ラベル
        - 'prob_class_0', 'prob_class_1', ... : 各クラスの予測確率（オプション）
    output_dir : str
        出力保存先ディレクトリ
    model_name : str
        モデル名（ファイル名に使用）
    class_names : List[str], optional
        クラス名のリスト。Noneの場合は自動生成
    n_classes : int
        クラス数
    grid_resolution : int
        空間グリッドの解像度
    
    Returns:
    --------
    Dict : 解析結果のサマリー
    
    出力ファイル:
    ------------
    - spatial_accuracy_analysis.png: 正答率の空間マップ
    - class_wise_error_distribution.png: クラス別誤分類分布
    - confusion_pattern_heatmaps.png: 主要な誤分類パターンのヒートマップ
    - confidence_spatial_distribution.png: 確信度別の空間分布（確率列がある場合）
    - confusion_hotspots.csv: 判別困難なホットスポットのリスト
    - {model_name}_accuracy.tif: 正答率のGeoTIFF
    - {model_name}_entropy.tif: エントロピーのGeoTIFF（確率列がある場合）
    - confusion_analysis_summary.json: 解析サマリー
    """
    print(f"\n{'='*70}")
    print(f"🔍 Standalone Confusion Region Analysis")
    print(f"   Input: {predictions_csv}")
    print(f"   Output: {output_dir}")
    print(f"{'='*70}")
    
    # CSVを読み込み
    predictions_df = pd.read_csv(predictions_csv)
    print(f"  📊 Loaded {len(predictions_df):,} samples from CSV")
    
    # pred_labelがない場合は計算
    if 'pred_label' not in predictions_df.columns:
        prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
        if prob_cols:
            predictions_df['pred_label'] = predictions_df[prob_cols].values.argmax(axis=1)
        elif 'predicted_class' in predictions_df.columns:
            predictions_df['pred_label'] = predictions_df['predicted_class']
        else:
            raise ValueError("CSVに'pred_label'または確率列が見つかりません")
    
    # クラス名の生成
    if class_names is None:
        thresholds = FIVE_CLASS_FIXED_THRESHOLDS.copy()
        class_names = generate_class_names_from_thresholds(thresholds, n_classes)
    
    # 可視化の実行
    result = visualize_model_confusion_regions(
        predictions_df=predictions_df,
        original_image=None,
        save_dir=Path(output_dir),
        model_name=model_name,
        grid_resolution=grid_resolution,
        class_names=class_names,
        output_geotiff=True
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"   Results saved to: {output_dir}/model_confusion_analysis/")
    
    return result


def run_confusion_analysis_cli():
    """
    コマンドラインから判別困難領域の可視化を実行するエントリポイント
    
    使い方:
    -------
    python class5-v9.py --analyze-confusion \\
        --predictions output/CNN/all_predictions.csv \\
        --output output/CNN \\
        --model-name CNN \\
        --n-classes 5 \\
        --grid-resolution 50
    """
    parser = argparse.ArgumentParser(
        description='モデルの判別困難領域を可視化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python class5-v9.py --analyze-confusion --predictions output/CNN/all_predictions.csv --output output/CNN --model-name CNN

出力ファイル:
  - spatial_accuracy_analysis.png: 正答率の空間マップ
  - class_wise_error_distribution.png: クラス別誤分類分布
  - confusion_hotspots.csv: 判別困難なホットスポット
  - {model_name}_accuracy.tif: 正答率のGeoTIFF
        """
    )
    
    parser.add_argument('--analyze-confusion', action='store_true',
                       help='判別困難領域の可視化を実行')
    parser.add_argument('--predictions', '-p', type=str,
                       help='予測結果のCSVファイルパス')
    parser.add_argument('--output', '-o', type=str,
                       help='出力ディレクトリ')
    parser.add_argument('--model-name', '-m', type=str, default='Model',
                       help='モデル名（デフォルト: Model）')
    parser.add_argument('--n-classes', '-n', type=int, default=5,
                       help='クラス数（デフォルト: 5）')
    parser.add_argument('--grid-resolution', '-g', type=int, default=50,
                       help='空間グリッド解像度（デフォルト: 50）')
    
    args = parser.parse_args()
    
    if args.analyze_confusion:
        if not args.predictions or not args.output:
            parser.error("--analyze-confusion を使用する場合は --predictions と --output が必要です")
        
        analyze_confusion_from_predictions_csv(
            predictions_csv=args.predictions,
            output_dir=args.output,
            model_name=args.model_name,
            n_classes=args.n_classes,
            grid_resolution=args.grid_resolution
        )
    else:
        # 通常のmain()を実行
        main()


if __name__ == "__main__":
    # コマンドライン引数をチェック
    import sys
    if '--analyze-confusion' in sys.argv:
        run_confusion_analysis_cli()
    else:
        main()