#MLP-mixerの多クラス分類をワンストップで行う（ResNet18版）
# 混同行列、fold毎の評価、閾値最適化、ハイパーパラメータ探索機能を追加
# Balanced Spatial K-Fold実装を追加

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
from scipy.stats import pearsonr
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
    palette = ['#bec8d1', '#fffeec', '#c3e5e7', '#86cecb', '#137a7f', '#e12885']
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

        fig, ax = plt.subplots(figsize=(10, 7))

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
                cbar.set_label('Annotation ratio (0-1)')

        if polygons_gdf is not None and not polygons_gdf.empty:
            polygons_gdf.boundary.plot(ax=ax, linewidth=1.0, edgecolor='cyan')

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_aspect('equal')
        ax.set_title(f"{image_name} | Legacy ratio overlay")

        output_path = self.patch_overlay_dir / f"{image_name}_patch_overlay.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
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

                if src.crs is None or target_crs is None or src.transform is None or target_transform is None:
                    src_data = src.read(1).astype(np.float32)
                    if src_data.shape != target_shape:
                        zoom_factors = (
                            target_shape[0] / src_data.shape[0],
                            target_shape[1] / src_data.shape[1],
                        )
                        src_data = zoom(src_data, zoom_factors, order=1)
                    return src_data

                target_bounds = self._compute_bounds(target_transform, target_shape)
                if target_bounds is None:
                    src_data = src.read(1).astype(np.float32)
                    if src_data.shape != target_shape:
                        zoom_factors = (
                            target_shape[0] / src_data.shape[0],
                            target_shape[1] / src_data.shape[1],
                        )
                        src_data = zoom(src_data, zoom_factors, order=1)
                    return src_data

                if src.crs == target_crs:
                    window = src.window(*target_bounds, boundless=True)
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

                with WarpedVRT(
                    src,
                    crs=target_crs,
                    transform=target_transform,
                    width=target_shape[1],
                    height=target_shape[0],
                    resampling=resampling_method,
                ) as vrt:
                    data = vrt.read(
                        1,
                        out_shape=(target_shape[0], target_shape[1]),
                        resampling=resampling_method,
                        masked=True,
                    )
                    if np.ma.isMaskedArray(data):
                        data = data.filled(np.nan)
                    data = data.astype(np.float32, copy=False)
                    if self.enable_aux_cache:
                        self._aux_cache[cache_key] = data
                    return data
        except Exception as exc:
            warnings.warn(f"Failed to load auxiliary raster {path}: {exc}")
            return None

    def _rasterize_polygons(self, gdf, id_col, rgb_ras, rgb_transform):
        shapes = [(geom, i) for i, geom in zip(gdf[id_col], gdf.geometry)]
        return rasioftr.rasterize(shapes, out_shape=rgb_ras.shape[:2], transform=rgb_transform, fill=np.nan, all_touched=True)

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

    def extract_tips(self, tip_size_m=5, slide_ratio=0.5, collect_data=True, show_progress=True):
        tip_size_grids = int(tip_size_m / self.resolution)
        l_buf = tip_size_grids // 2
        u_buf = tip_size_grids - l_buf - 1
        n_slide = max(1, int(tip_size_grids * slide_ratio))

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

            vdvi_ras = self._load_auxiliary_raster(
                self.vdvi_paths[idx],
                aoi_size,
                rgb_transform,
                rgb_crs,
                is_vdvi=True,
            ) if self.vdvi_paths and self.vdvi_paths[idx] is not None else None

            # Clip polygons
            extent_gdf = gpd.GeoDataFrame(geometry=[box(*rgb_bounds)], crs=rgb_crs)
            landslide_aoi_gdf = gpd.clip(self.landslide_gdf, extent_gdf)
            legacy_aoi_gdf = gpd.clip(self.legacy_gdf, extent_gdf)
            # Rasterize
            landslide_ras = self._rasterize_polygons(landslide_aoi_gdf, 'landslide_id', rgb_ras, rgb_transform)
            legacy_ras = self._rasterize_polygons(legacy_aoi_gdf, 'id', rgb_ras, rgb_transform)
            legacy_ras = np.where(np.isnan(landslide_ras), np.nan, legacy_ras)
            # Create tips
            total_iterations = len(i_range) * len(j_range)
            progress = tqdm(total=total_iterations, desc="Extracting tips") if show_progress else None
            try:
                for i in i_range:
                    for j in j_range:
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
def assign_class_from_thresholds(value, thresholds, eps: float = 1e-6) -> int:
    """Helper to map a continuous label (0-1) to a class index using thresholds."""
    if isclose(value, 0.0, abs_tol=eps):
        return 0
    if isclose(value, 1.0, abs_tol=eps):
        return len(thresholds)

    for idx, threshold in enumerate(thresholds):
        if value <= threshold:
            return idx
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
# PyTorch Dataset（閾値可変対応版）
# *******************************************************************
class LandslideDataset(Dataset):
    """PyTorch用のデータセットクラス（閾値可変対応）"""
    
    def __init__(self, images, labels, pixel_locations, geo_locations, 
                 region_idx=None, transform=None, patch_size=25, 
                 thresholds=None, n_classes=5, dem_range=None, vdvi_range=None):
        self.images = images
        self.labels = labels  # 連続値ラベル（0.0～1.0）
        self.pixel_locations = pixel_locations
        self.geo_locations = geo_locations
        self.region_idx = region_idx
        self.transform = transform
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.dem_range = dem_range
        self.vdvi_range = vdvi_range
        
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

        if image_tensor.shape[0] >= 3:
            image_tensor[:3] = torch.clamp(image_tensor[:3] / 255.0, 0.0, 1.0)

        if image_tensor.shape[0] > 3 and self.dem_range is not None:
            dem_min, dem_max = self.dem_range
            if dem_max > dem_min:
                image_tensor[3] = (image_tensor[3] - dem_min) / (dem_max - dem_min + 1e-6)
                image_tensor[3] = torch.clamp(image_tensor[3], 0.0, 1.0)

        if image_tensor.shape[0] > 4 and self.vdvi_range is not None:
            vdvi_min, vdvi_max = self.vdvi_range
            if vdvi_max > vdvi_min:
                image_tensor[4] = (image_tensor[4] - vdvi_min) / (vdvi_max - vdvi_min + 1e-6)
                image_tensor[4] = torch.clamp(image_tensor[4], 0.0, 1.0)

        image = image_tensor

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
    斜面崩壊残存植生検出に特化したハイブリッドモデル
    
    設計思想:
    - RGB: 浅いCNNで空間パターンを抽出（過学習防止）
    - VDVI/DEM: MLPで直接処理（最も効果的だった手法を維持）
    - 融合: Attentionベースではなく単純な連結＋ゲーティング
    
    in_channels対応:
    - 3: RGBのみ
    - 4: RGB + VDVI or RGB + DEM
    - 5: RGB + VDVI + DEM
    """
    
    def __init__(self, num_classes=5, image_size=25, dropout=0.3, in_channels=5):
        super().__init__()
        
        self.in_channels = in_channels
        self.use_vdvi = in_channels >= 4
        self.use_dem = in_channels >= 5
        
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
        # Stream 2: VDVI/DEM数値特徴処理（MLP）
        # ========================================
        # MLPが最も効果的だった観察を活かす
        # VDVIとDEMを別々に処理してから統合
        
        # パッチ全体の統計量を使用
        # 4チャンネル（mean, std, min, max）× 利用可能なAuxチャンネル数
        n_aux_channels = int(self.use_vdvi) + int(self.use_dem)
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
    
    def extract_aux_stats(self, vdvi, dem):
        """VDVIとDEMから統計量を抽出（MLPで処理するため）"""
        stats_list = []
        batch_size = vdvi.size(0) if vdvi is not None else dem.size(0)
        
        # VDVI統計量
        if vdvi is not None and vdvi.numel() > 0:
            vdvi_flat = vdvi.view(batch_size, -1)
            if vdvi_flat.size(1) > 0:
                vdvi_mean = vdvi_flat.mean(dim=1, keepdim=True)
                vdvi_std = vdvi_flat.std(dim=1, keepdim=True)
                vdvi_min = vdvi_flat.min(dim=1, keepdim=True)[0]
                vdvi_max = vdvi_flat.max(dim=1, keepdim=True)[0]
                stats_list.extend([vdvi_mean, vdvi_std, vdvi_min, vdvi_max])
        
        # DEM統計量
        if dem is not None and dem.numel() > 0:
            dem_flat = dem.view(batch_size, -1)
            if dem_flat.size(1) > 0:
                dem_mean = dem_flat.mean(dim=1, keepdim=True)
                dem_std = dem_flat.std(dim=1, keepdim=True)
                dem_min = dem_flat.min(dim=1, keepdim=True)[0]
                dem_max = dem_flat.max(dim=1, keepdim=True)[0]
                stats_list.extend([dem_mean, dem_std, dem_min, dem_max])
        
        # 連結
        if stats_list:
            stats = torch.cat(stats_list, dim=1)
        else:
            # Auxデータがない場合はダミーのゼロテンソルを返す
            stats = torch.zeros(batch_size, 0, device=vdvi.device if vdvi is not None else dem.device)
        
        return stats
    
    def forward(self, x):
        """
        x: (B, C, H, W) where C = 3 (RGB only), 4 (RGB+VDVI), or 5 (RGB+VDVI+DEM)
        """
        batch_size = x.size(0)
        num_channels = x.size(1)
        
        # チャンネル分離
        rgb = x[:, :3, :, :]  # (B, 3, H, W)
        
        vdvi = None
        dem = None
        if num_channels >= 4 and self.use_vdvi:
            vdvi = x[:, 3:4, :, :]  # (B, 1, H, W)
        if num_channels >= 5 and self.use_dem:
            dem = x[:, 4:5, :, :]   # (B, 1, H, W)
        
        # Stream 1: RGB空間特徴
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = rgb_feat.view(batch_size, -1)  # (B, 128)
        
        features_to_combine = [rgb_feat]
        
        # Stream 2: VDVI/DEM統計量 → MLP（利用可能な場合のみ）
        if self.aux_mlp is not None and (vdvi is not None or dem is not None):
            aux_stats = self.extract_aux_stats(vdvi, dem)
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
        
        # 個々のポイントも表示
        x = np.ones(len(values)) + np.random.normal(0, 0.02, len(values))
        ax.scatter(x, values, alpha=0.7, s=50, color=palette['magenta'], zorder=3)
        
        # 平均値のライン
        ax.axhline(
            y=np.mean(values),
            color=palette['ice'],
            linestyle='--',
            label=f'Mean: {np.mean(values):.4f}',
            alpha=0.7
        )
        
        ax.set_ylabel('Score', fontsize=10, color=palette['cream'])
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=11, fontweight='bold', color=palette['cream'])
        legend = ax.legend(loc='lower right', fontsize=9)
        if legend:
            legend.get_frame().set_facecolor(palette['deep_teal'])
            legend.get_frame().set_edgecolor(palette['mist'])
            for text in legend.get_texts():
                text.set_color(palette['cream'])
        ax.set_ylim([max(0, min(values) - 0.05), min(1, max(values) + 0.05)])
    
    # 最後のサブプロットに統計情報のテーブルを追加
    ax = axes[5]
    ax.axis('off')
    ax.set_facecolor('black')
    
    # 統計テーブル
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
        loc='center',
        colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
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
    
    # 2. 各foldのメトリクス推移
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('black')
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors=palette['cream'])
        for spine in ax.spines.values():
            spine.set_color(palette['cream'])
        ax.grid(True, alpha=0.3, color=palette['mist'])
    
    folds = list(range(1, n_folds + 1))
    
    # F1とAccuracyの推移
    ax = axes[0]
    ax.plot(folds, metrics_data['f1_macro'], 'o-', label='F1 Macro', 
           markersize=8, linewidth=2, color=palette['magenta'])
    ax.plot(folds, metrics_data['accuracy'], 's-', label='Accuracy', 
           markersize=8, linewidth=2, color=palette['teal'])
    ax.plot(folds, metrics_data['balanced_accuracy'], '^-', label='Balanced Acc', 
           markersize=8, linewidth=2, color=palette['ice'])
    
    ax.set_xlabel('Fold', fontsize=12, color=palette['cream'])
    ax.set_ylabel('Score', fontsize=12, color=palette['cream'])
    ax.set_title('Performance Metrics by Fold', fontsize=13, fontweight='bold', color=palette['cream'])
    legend = ax.legend(loc='best', fontsize=10)
    if legend:
        legend.get_frame().set_facecolor(palette['deep_teal'])
        legend.get_frame().set_edgecolor(palette['mist'])
        for text in legend.get_texts():
            text.set_color(palette['cream'])
    ax.set_xticks(folds)
    ax.set_ylim([min(min(metrics_data['f1_macro']), 
                    min(metrics_data['accuracy']), 
                    min(metrics_data['balanced_accuracy'])) - 0.02, 
                max(max(metrics_data['f1_macro']), 
                    max(metrics_data['accuracy']), 
                    max(metrics_data['balanced_accuracy'])) + 0.02])
    
    # Precision/Recallの推移
    ax = axes[1]
    ax.plot(folds, metrics_data['precision_macro'], 'D-', label='Precision Macro', 
           markersize=8, linewidth=2, color=palette['magenta'])
    ax.plot(folds, metrics_data['recall_macro'], 'p-', label='Recall Macro', 
           markersize=8, linewidth=2, color=palette['teal'])
    
    ax.set_xlabel('Fold', fontsize=12, color=palette['cream'])
    ax.set_ylabel('Score', fontsize=12, color=palette['cream'])
    ax.set_title('Precision and Recall by Fold', fontsize=13, fontweight='bold', color=palette['cream'])
    legend = ax.legend(loc='best', fontsize=10)
    if legend:
        legend.get_frame().set_facecolor(palette['deep_teal'])
        legend.get_frame().set_edgecolor(palette['mist'])
        for text in legend.get_texts():
            text.set_color(palette['cream'])
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
                        ha="center", va="center", color=palette['deep_teal'], fontsize=10, fontweight='bold')
        
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

    ax.set_title(title, color=palette['cream'], fontsize=14, fontweight='bold')
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
    
    if model_name == 'ResNet18':
        model = ResNet18Custom(
            num_classes=n_classes,
            pretrained=hyperparams.get('pretrained', True),
            dropout=hyperparams.get('dropout', 0.3),  # デフォルト0.3
            input_size=hyperparams.get('input_size', 25),
            in_channels=in_channels,
        )
    elif model_name == 'MLP-Mixer':
        model = MLPMixer(
            in_channels=in_channels,
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
            in_channels=in_channels,
            num_classes=n_classes,
            hidden_dim=hyperparams.get('hidden_dim', 128),
            depth=hyperparams.get('depth', 4),
            kernel_size=hyperparams.get('kernel_size', 3)
        )
    elif model_name == 'ViT':
        model = ViT(
            image_size=25,
            patch_size=hyperparams.get('patch_size', 5),
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
            image_size=hyperparams.get('image_size', 25),
            dropout=hyperparams.get('dropout', 0.3),
            in_channels=in_channels
        )
    elif model_name == 'TransformerV1':
        # 基本Transformer（RGB only）+ MLP（Aux）
        model = LandslideTransformerV1(
            image_size=hyperparams.get('image_size', 25),
            patch_size=hyperparams.get('patch_size', 5),
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
            image_size=hyperparams.get('image_size', 25),
            patch_size=hyperparams.get('patch_size', 5),
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
            image_size=hyperparams.get('image_size', 25),
            patch_size=hyperparams.get('patch_size', 5),
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
            image_size=hyperparams.get('image_size', 25),
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
        
        # メモリクリア
        del model
        torch.cuda.empty_cache()
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
    
    # 結果保存用ディレクトリの作成
    os.makedirs(save_dir, exist_ok=True)
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    curves_dir = os.path.join(model_save_dir, 'training_curves')
    os.makedirs(curves_dir, exist_ok=True)
    
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
        
        # 学習率とweight_decayを取得
        learning_rate = hyperparams.get('learning_rate', 1e-4)
        weight_decay = hyperparams.get('weight_decay', 1e-4)
        
        class_weights, class_counts = compute_inverse_frequency_weights(
            train_dataset.labels, train_dataset.thresholds, n_classes
        )
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"  Class counts (fold {fold + 1}): {class_counts.tolist()} -> weights: {[round(w, 4) for w in class_weights]}")
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
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu()
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
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch.to(device))
                    val_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim=1).cpu()
                    preds = torch.argmax(outputs, dim=1).cpu()
                    
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
                    normalize=False
                )
                # 正規化版も保存
                plot_confusion_matrix(
                    epoch_cm, class_names_template,
                    title=f'{model_name} - Fold {fold + 1} Epoch {epoch + 1} Normalized CM',
                    save_path=os.path.join(epoch_cm_dir, f'cm_fold_{fold+1}_epoch_{epoch+1}_normalized.png'),
                    normalize=True
                )
                print(f"  📊 Saved confusion matrix for epoch {epoch + 1}")
            
            # ベストモデルの保存
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_model_state = model.state_dict().copy()
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
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu()
                preds = torch.argmax(outputs, dim=1).cpu()
                
                val_probs.append(probs.numpy())
                val_preds.extend(preds.numpy())
                val_labels.extend(labels_batch.numpy())
                val_raw_labels.extend(raw_labels_batch.numpy())
        
        val_probs = np.vstack(val_probs)
        
        # 予測結果をDataFrameに保存
        fold_predictions_df = pd.DataFrame({
            'fold': fold,
            'true_label': val_labels,
            'predicted_label': val_preds,
            'raw_label': val_raw_labels,
            'val_index': val_idx[:len(val_labels)]
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
        
        # 混同行列の可視化と保存
        class_names = class_names_template
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
        fold_time = (datetime.now() - fold_start_time).total_seconds() / 60
        print(f"\n✅ Fold {fold + 1} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  F1 per class: {[f'{f:.3f}' for f in metrics['f1']]}")
        print(f"  Time taken: {fold_time:.1f} minutes")
        
        # メモリクリア
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # 全予測結果を保存
    all_predictions_df = pd.concat(all_predictions_df, ignore_index=True)
    predictions_path = os.path.join(model_save_dir, 'all_predictions.csv')
    all_predictions_df.to_csv(predictions_path, index=False)
    print(f"\n💾 All predictions saved to: {predictions_path}")
    
    # クロスバリデーション結果の可視化
    plot_cv_scores_enhanced(fold_results, model_save_dir, model_name, class_names=class_names_template)
    
    # 全体の混同行列（平均）
    avg_cm = np.mean(all_cms, axis=0).astype(int)
    plot_confusion_matrix(
        avg_cm, class_names_template,
        title=f'{model_name} - Average Confusion Matrix across all folds',
        save_path=os.path.join(model_save_dir, 'confusion_matrix_average.png'),
        normalize=False
    )
    
    # 正規化版も保存
    plot_confusion_matrix(
        avg_cm, class_names_template,
        title=f'{model_name} - Average Normalized Confusion Matrix across all folds',
        save_path=os.path.join(model_save_dir, 'confusion_matrix_average_normalized.png'),
        normalize=True
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
        'predictions_path': predictions_path
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
    plot_cv_scores_enhanced(fold_results, ensemble_dir, 'Ensemble', class_names=class_names)
    for fold_result in fold_results:
        fold_idx = fold_result['fold']
        cm = fold_result['confusion_matrix']
        plot_confusion_matrix(
            cm, class_names,
            title=f'Ensemble - Fold {fold_idx} Confusion Matrix',
            save_path=ensemble_dir / f'confusion_matrix_fold_{fold_idx}.png',
            normalize=False
        )
        plot_confusion_matrix(
            cm, class_names,
            title=f'Ensemble - Fold {fold_idx} Normalized Confusion Matrix',
            save_path=ensemble_dir / f'confusion_matrix_fold_{fold_idx}_normalized.png',
            normalize=True
        )

    avg_cm = np.mean([fr['confusion_matrix'] for fr in fold_results], axis=0)
    plot_confusion_matrix(
        avg_cm.astype(int), class_names,
        title='Ensemble - Average Confusion Matrix',
        save_path=ensemble_dir / 'confusion_matrix_average.png',
        normalize=False
    )

    summary = {
        'model': 'Ensemble',
        'base_models': processed_models,
        'n_folds': len(fold_results),
        'n_classes': config['n_classes'],
        'fold_results': fold_results,
        'average_confusion_matrix': avg_cm.tolist(),
        'predictions_path': str(predictions_path)
    }

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

    return summary


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
                             tip_size_m=5, slide_ratio=0.5, cache_dir=None,
                             required_channels: Optional[int] = None,
                             alignment_preview_dir=None,
                             patch_overlay_dir=None):
    """
    複数のデータセットを処理して統合
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
                epsg_code=epsg_code,
                resolution=0.2,
                alignment_preview_dir=alignment_preview_dir,
                patch_overlay_dir=patch_overlay_dir,
            )

        if not cached:
            assert processor is not None
            tips, labels, pixel_locs, geo_locs, landslide_ids = processor.extract_tips(
                tip_size_m=tip_size_m,
                slide_ratio=slide_ratio
            )
            if preview_needed:
                processor.generate_alignment_preview()
        else:
            if overlay_needed and processor is not None:
                processor.extract_tips(
                    tip_size_m=tip_size_m,
                    slide_ratio=slide_ratio,
                    collect_data=False,
                    show_progress=False,
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
    model_names = ["HybridModel", "TransformerV1", "TransformerV2", "TransformerV3", "HybridTransformer"]
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
        'n_epochs': 40,  # エポック数を40に変更
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
                'batch_size': 64,
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
    print(f"\n{'='*80}")
    print(f"All results saved to {config['output_dir']}")
    print("Pipeline completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()