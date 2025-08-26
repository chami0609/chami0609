"""
3つのモデル（ViT, CNN, MLP-Mixer）の訓練、アンサンブル、可視化を行うスクリプト
BalancedSpatialKFoldクラスを統合したバージョン
"""

# utils
from pyprojroot import here
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle
from tqdm import tqdm
import itertools
# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
# グラフの日本語対応
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# Machine learning
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# --- Spatial K-Fold Imports ---
import geopandas as gpd
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import hashlib
from datetime import datetime
from shapely.geometry import Point

# 既存のコードから必要な部分をインポート
from legacy_detection_system_v3 import (
    LandslideTipProcessor, LandslideDataset, 
    init_model, train_one_epoch, validate_one_epoch,
    ViT, LeNet, MLPMixer
)


# --- Balanced Spatial K-Fold Implementation ---

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


def apply_balanced_spatial_kfold(patches, labels, geo_locations, 
                                 landslide_ids, region_idx, config):
    """
    バランスの取れた空間的K分割を適用する関数
    """
    # 入力データの長さを確認
    n_samples = len(labels)
    print(f"\nDebug - Data lengths:")
    print(f"  labels: {len(labels)}")
    print(f"  geo_locations: {len(geo_locations)}")
    print(f"  landslide_ids: {len(landslide_ids)}")
    print(f"  region_idx: {len(region_idx) if region_idx is not None else 'None'}")
    
    # データの長さを統一
    if len(geo_locations) != n_samples:
        raise ValueError(f"geo_locations length ({len(geo_locations)}) doesn't match labels length ({n_samples})")
    if len(landslide_ids) != n_samples:
        raise ValueError(f"landslide_ids length ({len(landslide_ids)}) doesn't match labels length ({n_samples})")
    
    # GeoDataFrameの作成
    geometry = [Point(lon, lat) for lon, lat in geo_locations]
    
    # DataFrameを作成する際、すべての列が同じ長さであることを確認
    data = {
        'id': list(range(n_samples)),
        'label': list(labels),
        'landslide_id': list(landslide_ids),
        'geometry': geometry
    }
    
    # region_idxが存在する場合のみ追加
    if region_idx is not None and len(region_idx) == n_samples:
        data['old_region'] = list(region_idx)
    
    gdf = gpd.GeoDataFrame(data, crs=f'EPSG:{config["epsg_code"]}')
    
    # UTM座標系に変換
    gdf_prj = gdf.to_crs(gdf.estimate_utm_crs())
    coords = np.array([[geom.x, geom.y] for geom in gdf_prj.geometry])
    
    # パラメータの設定
    n_splits = config['n_outer_cv']
    min_samples = config.get('min_points_per_fold', 4000)
    
    # 自動的にフォールド数を調整
    total_samples = len(gdf_prj)
    max_possible_folds = total_samples // min_samples
    
    if n_splits > max_possible_folds:
        print(f"\nWarning: Requested {n_splits} folds, but with {min_samples} min samples per fold,")
        print(f"         maximum possible is {max_possible_folds}. Adjusting to {max_possible_folds}.")
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
            
            print(f"\nAttempt {attempt + 1}: CV = {cv:.3f}, fold sizes = {fold_counts}")
            
            if cv < best_balance_score:
                best_balance_score = cv
                best_fold_assignments = fold_assignments
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    if best_fold_assignments is None:
        raise ValueError("Could not create valid spatial folds")
    
    # 最良の分割を使用
    gdf_prj['folds'] = best_fold_assignments
    spatial_folds = best_fold_assignments
    
    print(f"\n{'='*60}")
    print(f"Final configuration: {n_splits} folds with CV = {best_balance_score:.3f}")
    print(f"{'='*60}")
    
    # 統計情報を表示
    print_fold_statistics_improved(spatial_folds, landslide_ids, labels)
    
    return spatial_folds, gdf_prj


def print_fold_statistics_improved(fold_assignments, landslide_ids, labels):
    """改善された統計情報の表示"""
    n_folds = len(np.unique(fold_assignments))
    fold_counts = np.bincount(fold_assignments)
    
    print("\nBalanced Spatial K-Fold Statistics:")
    print(f"Total samples: {len(fold_assignments):,}")
    print(f"Number of folds: {n_folds}")
    print(f"Target samples per fold: {len(fold_assignments) // n_folds:,}")
    
    print("\nSamples per fold:")
    for fold in range(n_folds):
        count = fold_counts[fold]
        percentage = count / len(fold_assignments) * 100
        deviation = (count - len(fold_assignments) / n_folds) / (len(fold_assignments) / n_folds) * 100
        print(f"  Fold {fold}: {count:6,} ({percentage:5.1f}%) [{deviation:+5.1f}% from target]")
    
    # バランス評価指標
    mean_count = np.mean(fold_counts)
    std_count = np.std(fold_counts)
    cv = std_count / mean_count
    max_min_ratio = np.max(fold_counts) / np.min(fold_counts) if np.min(fold_counts) > 0 else float('inf')
    
    print(f"\nBalance metrics:")
    print(f"  Mean: {mean_count:.1f}, Std: {std_count:.1f}")
    print(f"  Coefficient of Variation: {cv:.3f}")
    print(f"  Max/Min ratio: {max_min_ratio:.2f}")
    
    # 評価
    if cv < 0.1:
        print("  ✓ Excellent balance!")
    elif cv < 0.2:
        print("  ✓ Good balance")
    else:
        print("  ⚠ Consider adjusting parameters for better balance")
    
    # 地すべりIDの分割チェック
    print("\nLandslide distribution check:")
    landslide_split_check = {}
    for fold in range(n_folds):
        fold_mask = fold_assignments == fold
        fold_landslides = set(landslide_ids[fold_mask])
        for lid in fold_landslides:
            if lid not in landslide_split_check:
                landslide_split_check[lid] = []
            landslide_split_check[lid].append(fold)
    
    split_landslides = [(lid, folds) for lid, folds in landslide_split_check.items() if len(folds) > 1]
    
    if split_landslides:
        print(f"  ⚠ Warning: {len(split_landslides)} landslides split across folds")
        if len(split_landslides) <= 5:
            for lid, folds in split_landslides[:5]:
                print(f"    - Landslide {lid}: in folds {folds}")
    else:
        print("  ✓ Good: No landslides split across folds")
    
    # ラベル分布
    print("\nLabel distribution per fold:")
    for fold in range(n_folds):
        fold_labels = labels[fold_assignments == fold]
        if len(fold_labels) > 0:
            print(f"  Fold {fold}: mean={np.mean(fold_labels):.3f}, std={np.std(fold_labels):.3f}, "
                  f"positive ratio={np.mean(fold_labels > 0.5):.3f}")


def visualize_balanced_spatial_kfold(gdf_with_folds, config, comparison_gdf=None):
    """バランスの取れた空間K分割の可視化"""
    n_folds = len(gdf_with_folds['folds'].unique())
    cols_tab = cm.get_cmap('tab20', n_folds)
    color_ramp = ListedColormap([cols_tab(i) for i in range(n_folds)])
    
    n_cols = 2 if comparison_gdf is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(10*n_cols, 10))
    
    if n_cols == 1:
        axes = [axes]
    
    # 新しい手法の表示
    gdf_with_folds.plot(column='folds', ax=axes[0], cmap=color_ramp, 
                       markersize=1, legend=True, legend_kwds={'label': 'Fold'})
    axes[0].set_title('Balanced Spatial K-Fold', fontsize=14)
    axes[0].set_xlabel('Easting (m)', fontsize=12)
    axes[0].set_ylabel('Northing (m)', fontsize=12)
    
    # 統計情報をテキストボックスに追加
    fold_counts = gdf_with_folds['folds'].value_counts().sort_index()
    stats_text = 'Samples per fold:\n'
    stats_text += '\n'.join([f'Fold {f}: {c:,}' for f, c in fold_counts.items()])
    stats_text += f'\n\nCV: {np.std(fold_counts) / np.mean(fold_counts):.3f}'
    
    axes[0].text(1.02, 0.5, stats_text, transform=axes[0].transAxes, 
                verticalalignment='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 比較用の既存手法の表示
    if comparison_gdf is not None and n_cols > 1:
        comp_gdf_prj = comparison_gdf.to_crs(gdf_with_folds.crs)
        comp_gdf_prj.plot(column='old_region', ax=axes[1], cmap=color_ramp, 
                         markersize=1, legend=True, legend_kwds={'label': 'Region'})
        axes[1].set_title('Previous Method (KMeans)', fontsize=14)
        axes[1].set_xlabel('Easting (m)', fontsize=12)
        axes[1].set_ylabel('Northing (m)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(config['fig_dir'] / 'balanced_spatial_kfold.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# --- End Balanced Spatial K-Fold Implementation ---


# Visualize regions
def plot_regions(region_idx, point, Y, config):
    # Y
    sns.scatterplot(x = list(point[:, 1]), y = list(point[:, 0]), hue = Y, s = 1, linewidth=0)
    plt.ylim(point[:, 0].max(),0)
    plt.savefig(config['fig_dir'] / 'region-y_gt.png')
    plt.close()
    # Region
    sns.scatterplot(x = list(point[:, 1]), y = list(point[:, 0]), hue = region_idx, palette="colorblind", s = 1, linewidth=0)
    plt.ylim(point[:, 0].max(),0)
    plt.savefig(config['fig_dir'] / 'region-cluster_id.png')
    plt.close()


def plot_learning_curve(train_losses, test_losses, model_name, fold, config):
    """学習曲線を保存"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{model_name} - Fold {fold + 1}')
    plt.legend()
    plt.savefig(config['fig_dir'] / f'learning_curve_{model_name}_fold{fold}.png')
    plt.close()

def plot_yhat_y(model_name, yhat, y, name, cv, epoch, loss, config):
    plt.scatter(yhat, y, s=1, alpha=0.1)
    plt.title(f'{model_name}: CV-{cv}, best train epoch-{epoch}, mode-{name}, loss-{loss:.2f}')
    plt.xlabel('Output of the model')
    plt.ylabel('True label')
    plt.savefig(config['fig_dir'] / f'y-yhat_{name}_cv-{cv}_{model_name}_best-epoch-{epoch}.png')
    plt.close()


def plot_cv_scores(all_results, fig_dir):
    """交差検証スコアの可視化"""
    
    # 1. RMSEの比較（箱ひげ図）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    rmse_data = []
    corr_data = []
    model_names = []
    
    for model_name, results in all_results.items():
        scores_df = pd.DataFrame(results['cv_scores'])
        for rmse in scores_df['rmse']:
            rmse_data.append(rmse)
            model_names.append(model_name)
        for corr in scores_df['correlation']:
            corr_data.append(corr)
    
    # RMSE箱ひげ図
    rmse_df = pd.DataFrame({'Model': model_names, 'RMSE': rmse_data})
    rmse_df['Model'] = pd.Categorical(rmse_df['Model'], categories=['CNN', 'ViT', 'MLP-Mixer'])
    rmse_df = rmse_df.sort_values('Model')
    
    sns.boxplot(data=rmse_df, x='Model', y='RMSE', ax=ax1, palette='Set2')
    ax1.set_title('Cross-Validation RMSE by Model', fontsize=14)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 相関係数箱ひげ図
    corr_df = pd.DataFrame({
        'Model': model_names[:len(corr_data)], 
        'Correlation': corr_data
    })
    corr_df['Model'] = pd.Categorical(corr_df['Model'], categories=['CNN', 'ViT', 'MLP-Mixer'])
    corr_df = corr_df.sort_values('Model')
    
    sns.boxplot(data=corr_df, x='Model', y='Correlation', ax=ax2, palette='Set2')
    ax2.set_title('Cross-Validation Correlation by Model', fontsize=14)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'cv_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 各モデルのフォールドごとのスコア推移
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        scores_df = pd.DataFrame(results['cv_scores'])
        
        # RMSE推移
        ax = axes[idx, 0]
        ax.plot(scores_df['fold'] + 1, scores_df['rmse'], 'o-', color='red', markersize=8)
        ax.set_title(f'{model_name} - RMSE by Fold', fontsize=12)
        ax.set_xlabel('Fold', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, len(scores_df) + 1))
        
        # 相関係数推移
        ax = axes[idx, 1]
        ax.plot(scores_df['fold'] + 1, scores_df['correlation'], 'o-', color='blue', markersize=8)
        ax.set_title(f'{model_name} - Correlation by Fold', fontsize=12)
        ax.set_xlabel('Fold', fontsize=10)
        ax.set_ylabel('Correlation', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, len(scores_df) + 1))
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'cv_scores_by_fold.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. スコアのサマリーテーブル
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    for model_name, results in all_results.items():
        scores_df = pd.DataFrame(results['cv_scores'])
        summary_data.append([
            model_name,
            f"{scores_df['rmse'].mean():.4f} ± {scores_df['rmse'].std():.4f}",
            f"{scores_df['correlation'].mean():.4f} ± {scores_df['correlation'].std():.4f}"
        ])
    
    table = ax.table(cellText=summary_data,
                     colLabels=['Model', 'RMSE (mean ± std)', 'Correlation (mean ± std)'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # ヘッダーの背景色を設定
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Cross-Validation Score Summary', fontsize=16, pad=20)
    plt.savefig(fig_dir / 'cv_scores_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ensemble_results(ensemble_predictions, fig_dir):
    """アンサンブル結果の可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 予測値 vs 実測値の散布図
    ax = axes[0, 0]
    ax.scatter(ensemble_predictions['labels'], 
               ensemble_predictions['ensemble_predictions'], 
               alpha=0.5, s=30)
    ax.plot([0, 1], [0, 1], 'r--', lw=2)
    ax.set_xlabel('True Legacy Ratio', fontsize=12)
    ax.set_ylabel('Ensemble Predicted Legacy Ratio', fontsize=12)
    ax.set_title('Ensemble: Predicted vs True Values', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # 相関係数とRMSEを追加
    corr, _ = pearsonr(ensemble_predictions['labels'], 
                      ensemble_predictions['ensemble_predictions'])
    rmse = np.sqrt(mean_squared_error(ensemble_predictions['labels'], 
                                     ensemble_predictions['ensemble_predictions']))
    ax.text(0.05, 0.95, f'Correlation: {corr:.4f}\nRMSE: {rmse:.4f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 残差プロット
    ax = axes[0, 1]
    residuals = np.array(ensemble_predictions['ensemble_predictions']) - np.array(ensemble_predictions['labels'])
    ax.scatter(ensemble_predictions['labels'], residuals, alpha=0.5, s=30)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('True Legacy Ratio', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 3. 予測値のヒストグラム
    ax = axes[1, 0]
    ax.hist(ensemble_predictions['ensemble_predictions'], bins=30, alpha=0.7, 
            color='blue', label='Ensemble Predictions', density=True)
    ax.hist(ensemble_predictions['labels'], bins=30, alpha=0.7, 
            color='red', label='True Values', density=True)
    ax.set_xlabel('Legacy Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Predictions vs True Values', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 各モデルの重み寄与度
    ax = axes[1, 1]
    model_names = ['CNN', 'ViT', 'MLP-Mixer']
    weights = ensemble_predictions.get('weights', [1/3, 1/3, 1/3])
    bars = ax.bar(model_names, weights, color=['#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Model Weights in Ensemble', fontsize=14)
    ax.set_ylim(0, 1)
    
    # 重みの値を棒グラフの上に表示
    for bar, weight in zip(bars, weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'ensemble_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def train_all_models_with_cv(dataset, region_idx, config, device):
    """すべてのモデルで交差検証を実行"""
    
    # ハイパーパラメータの定義（厚真のベストモデルから）
    best_hyperparams = {
        'ViT': {
            'PATCH_SIZE': 5, 
            'N_DIM': 80, 
            'N_HEADS': 8, 
            'MLP_DIM': 86, 
            'DROPOUT': 0.01
        },
        'CNN': {
            'NUM_FILTER1': 22,
            'KERNEL_SIZE1': 4,
            'NUM_FILTER2': 25,
            'KERNEL_SIZE2': 5,
            'DROPOUT': 0.09
        },
        'MLP-Mixer': {
            'PATCH_SIZE': 5,
            'TOKEN_DIM': 198,
            'CHANNEL_DIM': 176,
            'MLP_DIM': 256,
            'N_BLOCK': 5
        }
    }
    
    # データ拡張の定義
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
    ])
    
    all_results = {}
    all_fold_predictions = defaultdict(lambda: defaultdict(list))
    
    for model_name in config['model_names']:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # 入れ子交差検証の実行
        cv_scores = []
        cv_predictions = []
        for outer_cv in range(config['n_outer_cv']):
            print(f"\nFold {outer_cv + 1}/{config['n_outer_cv']}")
            
            # TODO: Optunaでハイパーパラメータを探索する
            train_idx, val_idx = np.where(region_idx!=outer_cv)[0], np.where(region_idx==outer_cv)[0]

            # データローダーの作成
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            # import pdb; pdb.set_trace()
            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
            
            # モデルの初期化と訓練
            model = init_model(model_name, best_hyperparams[model_name]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()
            
            train_losses = []
            val_losses = []
            
            best_train_loss = float('inf')
            best_model_state = None
            
            for epoch in range(config['n_epochs']):
                train_loss, train_yhat, train_y = train_one_epoch(
                    epoch, model, train_loader, optimizer, loss_fn, 
                    transform, device
                )
                val_loss, val_yhat, val_y = validate_one_epoch(
                    epoch, model, val_loader, loss_fn, device
                )
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # ベストモデルの保存
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    # import pdb; pdb.set_trace()
                    best_train_yhat = train_yhat.copy()
                    best_train_y = train_y.copy()
                    best_val_yhat = val_yhat.copy()
                    best_val_y = val_y.copy()

                
                if epoch % 20 == 0:
                    print(f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {val_loss:.4f}")
            
            # 学習曲線をプロット
            plot_learning_curve(train_losses, val_losses, model_name, outer_cv, config)
            # ベストモデルでのtrue vs. predictedの可視化
            plot_yhat_y(model_name, best_train_yhat, best_train_y, 'train', outer_cv, epoch, best_train_loss, config)
            plot_yhat_y(model_name, best_val_yhat, best_val_y, 'val', outer_cv, epoch, best_val_loss, config)


            # ベストモデルをロード
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # 最終的な評価
            model.eval()
            all_preds = []
            all_labels = []
            val_indices = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    inputs = batch['image'].to(device)
                    labels = batch['label'].cpu().numpy()
                    outputs = model(inputs).cpu().numpy()[:, 0]
                    
                    all_preds.extend(outputs.tolist())
                    all_labels.extend(labels.tolist())
                    
                    # テストインデックスを記録（アンサンブル用）
                    batch_size = len(labels)
                    start_idx = batch_idx * config['batch_size']
                    val_indices.extend(val_idx[start_idx:start_idx + batch_size].tolist())
            
            # アンサンブル用に予測を保存
            for idx, pred in zip(val_indices, all_preds):
                all_fold_predictions[outer_cv][idx].append(pred)
            
            # メトリクスの計算
            rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
            correlation, _ = pearsonr(all_labels, all_preds)
            
            cv_scores.append({
                'fold': outer_cv,
                'rmse': rmse,
                'correlation': correlation,
                'best_test_loss': best_train_loss
            })
            
            cv_predictions.append({
                'fold': outer_cv,
                'predictions': all_preds,
                'labels': all_labels,
                'test_indices': val_indices
            })
            
            print(f"  Fold {outer_cv + 1} Results: RMSE = {rmse:.4f}, Correlation = {correlation:.4f}")
            
            # 各フォールドのモデルを保存
            torch.save(model.state_dict(), config['output_dir'] / f'model_{model_name}_fold{outer_cv}.pth')
        
        # 結果の保存
        all_results[model_name] = {
            'best_hyperparams': best_hyperparams[model_name],
            'cv_scores': cv_scores,
            'cv_predictions': cv_predictions
        }
        
        # CSVとして保存
        scores_df = pd.DataFrame(cv_scores)
        scores_df.to_csv(config['output_dir'] / f'cv_scores_{model_name}.csv', index=False)
        
        # 予測結果を保存
        predictions_data = []
        for pred_dict in cv_predictions:
            outer_cv = pred_dict['fold']
            for pred, label in zip(pred_dict['predictions'], pred_dict['labels']):
                predictions_data.append({
                    'fold': outer_cv,
                    'prediction': pred,
                    'label': label
                })
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(config['output_dir'] / f'cv_predictions_{model_name}.csv', index=False)
        
        # ハイパーパラメータを保存
        with open(config['output_dir'] / f'best_hyperparams_{model_name}.json', 'w') as f:
            json.dump(best_hyperparams[model_name], f, indent=2)
        
        # 結果の表示
        print(f"\n{model_name} Cross-Validation Results:")
        print(f"Average RMSE: {scores_df['rmse'].mean():.4f} ± {scores_df['rmse'].std():.4f}")
        print(f"Average Correlation: {scores_df['correlation'].mean():.4f} ± {scores_df['correlation'].std():.4f}")
    
    return all_results, all_fold_predictions


def create_ensemble_predictions(all_fold_predictions, dataset, weights=None):
    """アンサンブル予測を作成"""
    
    if weights is None:
        weights = [1/3, 1/3, 1/3]  # 等重み
    
    ensemble_predictions = []
    true_labels = []
    
    # 各フォールドの予測を集計
    for fold in sorted(all_fold_predictions.keys()):
        fold_data = all_fold_predictions[fold]
        
        for idx in sorted(fold_data.keys()):
            if len(fold_data[idx]) == 3:  # 3つのモデルすべてが予測を持っている場合
                # 重み付き平均
                ensemble_pred = np.average(fold_data[idx], weights=weights)
                ensemble_predictions.append(ensemble_pred)
                
                # 真のラベルを取得
                true_label = dataset[idx]['label'].item()
                true_labels.append(true_label)
    
    return {
        'ensemble_predictions': ensemble_predictions,
        'labels': true_labels,
        'weights': weights
    }


def train_final_models(dataset, model_names, best_hyperparams, config, device):
    """最終モデルの訓練（全データ使用）"""
    
    final_models = {}
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
    ])
    
    for model_name in model_names:
        print(f"\nTraining final {model_name} model on all data...")
        
        model = init_model(model_name, best_hyperparams[model_name]).to(device)
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        
        for epoch in range(config['n_epochs']):
            train_loss, _, _ = train_one_epoch(
                epoch, model, train_loader, optimizer, loss_fn,
                transform, device
            )
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
        
        # 最終モデルの保存
        torch.save(model.state_dict(), config['output_dir'] / f'final_model_{model_name}.pth')
        final_models[model_name] = model
        print(f"Final model saved to {config['output_dir'] / f'final_model_{model_name}.pth'}")
    
    return final_models


def main():
    """メイン実行関数"""

    trial_name = "noto-v07-april-balanced"
    root_dir = here()
    data_dir = root_dir / 'data'
    data_annotation_dir = data_dir / "annotation"
    data_Noto_April_dir = data_dir / "Noto_April"
    data_sample_dir = data_Noto_April_dir / "annotation_sample"
    output_dir = root_dir / "output" / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    
    # 設定
    config = {
        'rgb_paths': data_sample_dir / '07fd041.jpg',
        'landslide_path': data_dir / 'noto_landslide/hokai-all.geojson',
        'legacy_path': data_annotation_dir / 'Porigon_april_fix.geojson',
        'output_dir': output_dir,
        'fig_dir': fig_dir,
        'epsg_code': 6675,
        'tip_size_m': 5, #5x5mの画像チップ (25x25pixel)
        'slide_ratio': 0.5, # 50%のスライド率
        'n_outer_cv': 10,
        'n_epochs': 64,
        'batch_size': 64,
        'model_names': ['CNN', 'ViT', 'MLP-Mixer'],
        'min_points_per_fold': 4000,  # 各フォールドの最小ポイント数
        'base_seed': 42  # 再現性のためのシード
    }
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データ処理（既に作成されている場合はロード）
    cache_file = output_dir / 'cached_data.pkl'
    if cache_file.exists():
        print("Loading cached data...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        patches = cached_data['patches']
        labels = cached_data['labels']
        pixel_locations = cached_data['pixel_locations']
        geo_locations = cached_data['geo_locations']
        landslide_ids = cached_data['landslide_ids']
        # 既存のregion_idxを比較用にロード
        region_idx_old = cached_data.get('region_idx', None)
    else:
        print("Processing data...")
        processor = LandslideTipProcessor(
            rgb_paths=config['rgb_paths'],
            landslide_path=config['landslide_path'],
            legacy_path=config['legacy_path'],
            epsg_code=config['epsg_code']
        )
        
        patches, labels, pixel_locations, geo_locations, landslide_ids = processor.extract_tips(
            tip_size_m=config['tip_size_m'],
            slide_ratio=config['slide_ratio']
        )

        # 比較のためにKMeansベースの分割も生成
        print("Generating KMeans-based clusters for comparison...")
        region_idx_old = KMeans(n_clusters=config['n_outer_cv'], random_state=0).fit(geo_locations).labels_
        plot_regions(region_idx_old, geo_locations, labels, config)
        
        # データをキャッシュ
        print("Caching data for future use...")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'patches': patches,
                'labels': labels,
                'pixel_locations': pixel_locations,
                'geo_locations': geo_locations,
                'landslide_ids': landslide_ids,
                'region_idx': region_idx_old # 比較用にKMeansの結果を保存
            }, f)
    
    # バランスの取れた空間的K-fold分割を適用
    print("\nApplying Balanced Spatial K-Fold for cross-validation...")
    
    # 比較用のGeoDataFrameを作成
    comparison_gdf = None
    if region_idx_old is not None:
        geometry = [Point(lon, lat) for lon, lat in geo_locations]
        comparison_gdf = gpd.GeoDataFrame({
            'id': range(len(labels)),
            'label': labels,
            'landslide_id': landslide_ids,
            'old_region': region_idx_old,
            'geometry': geometry
        }, crs=f'EPSG:{config["epsg_code"]}')

    # バランスの取れた空間K分割を適用
    region_idx, gdf_with_folds = apply_balanced_spatial_kfold(
        patches=patches,
        labels=labels,
        geo_locations=geo_locations,
        landslide_ids=landslide_ids,
        region_idx=region_idx_old,
        config=config
    )
    
    # 可視化
    print("\nVisualizing Balanced Spatial K-Fold splits...")
    visualize_balanced_spatial_kfold(gdf_with_folds, config, comparison_gdf=comparison_gdf)
    
    print(f"\nLoaded {len(patches)} patches")
    print(f"Legacy ratio statistics: mean={labels.mean():.3f}, std={labels.std():.3f}")
    
    # データセットの作成
    dataset = LandslideDataset(patches, labels, pixel_locations, geo_locations, region_idx)
    
    # すべてのモデルで交差検証を実行
    all_results, all_fold_predictions = train_all_models_with_cv(
        dataset, region_idx, config, device
    )
    
    # 交差検証結果の可視化
    print("\nPlotting cross-validation results...")
    plot_cv_scores(all_results, fig_dir)
    
    # アンサンブル予測の作成
    print("\nCreating ensemble predictions...")
    
    # 各モデルのCV平均スコアから重みを計算（相関係数ベース）
    model_correlations = []
    for model_name in config['model_names']:
        scores_df = pd.DataFrame(all_results[model_name]['cv_scores'])
        model_correlations.append(scores_df['correlation'].mean())
    
    # 重みを正規化
    weights = np.array(model_correlations) / np.sum(model_correlations)
    print(f"Ensemble weights based on CV correlation: {dict(zip(config['model_names'], weights))}")
    
    ensemble_predictions = create_ensemble_predictions(all_fold_predictions, dataset, weights)
    
    # アンサンブル結果の評価
    ensemble_rmse = np.sqrt(mean_squared_error(
        ensemble_predictions['labels'], 
        ensemble_predictions['ensemble_predictions']
    ))
    ensemble_corr, _ = pearsonr(
        ensemble_predictions['labels'], 
        ensemble_predictions['ensemble_predictions']
    )
    
    print(f"\nEnsemble Results:")
    print(f"RMSE: {ensemble_rmse:.4f}")
    print(f"Correlation: {ensemble_corr:.4f}")
    
    # アンサンブル結果の可視化
    print("\nPlotting ensemble results...")
    plot_ensemble_results(ensemble_predictions, fig_dir)
    
    # 最終モデルの訓練
    print("\nTraining final models on all data...")
    best_hyperparams = {model: all_results[model]['best_hyperparams'] 
                       for model in config['model_names']}
    
    final_models = train_final_models(
        dataset, config['model_names'], best_hyperparams, config, device
    )
    
    # 最終的な結果サマリーの作成
    summary = {
        'individual_models': {
            model: {
                'mean_rmse': pd.DataFrame(all_results[model]['cv_scores'])['rmse'].mean(),
                'std_rmse': pd.DataFrame(all_results[model]['cv_scores'])['rmse'].std(),
                'mean_correlation': pd.DataFrame(all_results[model]['cv_scores'])['correlation'].mean(),
                'std_correlation': pd.DataFrame(all_results[model]['cv_scores'])['correlation'].std(),
            }
            for model in config['model_names']
        },
        'ensemble': {
            'rmse': ensemble_rmse,
            'correlation': ensemble_corr,
            'weights': dict(zip(config['model_names'], weights))
        }
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    print("\nResults saved to:")
    print(f"- Cross-validation scores: {output_dir}/cv_scores_*.csv")
    print(f"- Visualizations: {fig_dir}/*.png")
    print(f"- Final models: {output_dir}/final_model_*.pth")
    print(f"- Training summary: {output_dir}/training_summary.json")
    
    return all_results, ensemble_predictions, final_models


if __name__ == "__main__":
    import torchvision.transforms as transforms
    all_results, ensemble_predictions, final_models = main()