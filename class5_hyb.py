"""
5クラス分類モデルの学習・評価スクリプト
地すべり地域の植生回復度（旧崩壊地比率）を予測するためのディープラーニングモデルを実装
"""

# =============================================================================
# 標準ライブラリ
# =============================================================================
import argparse
import gc
import hashlib
import itertools
import json
import os
import pickle
import random
import re
import warnings
from collections import defaultdict
from datetime import datetime
from glob import glob
from math import floor, isclose
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# サードパーティライブラリ
# =============================================================================
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm, colormaps
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from affine import Affine

import geopandas as gpd
from shapely.geometry import box, Point
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score, balanced_accuracy_score
)
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, gaussian_kde
from scipy.ndimage import zoom
from tqdm import tqdm

# =============================================================================
# 機械学習ライブラリ
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.multiprocessing as mp
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

# =============================================================================
# CPU並列処理設定とPyTorch最適化（48コアマシン用最適化）
# =============================================================================

def setup_torch_optimizations():
    """PyTorchの最適化設定を適用"""
    
    # CPU コア数を動的に取得
    NUM_CPU_CORES = os.cpu_count() or 48
    
    # DataLoader用のワーカー数
    NUM_WORKERS = min(24, NUM_CPU_CORES // 2)
    
    # PyTorchスレッド設定
    torch.set_num_threads(NUM_CPU_CORES // 2)
    torch.set_num_interop_threads(NUM_CPU_CORES // 4)
    
    # CUDA最適化
    if torch.cuda.is_available():
        # cuDNNベンチマーク（入力サイズ固定時に高速化）
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # TF32の有効化（Ampere以降のGPU）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        

    
    return NUM_WORKERS

# グローバルで呼び出し
NUM_WORKERS = setup_torch_optimizations()

# マルチプロセッシングのスタート方法を設定
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
# =============================================================================
# Windows DLLパス設定（GDAL/rasterio用）
# =============================================================================
def _configure_dll_search_paths() -> List[Path]:
    """
    Windows環境でGDAL/PROJ等のDLL検索パスを設定する。
    DLLロードエラーを防ぐための前処理。
    """
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
            os.add_dll_directory(str(candidate))
        except (AttributeError, FileNotFoundError, OSError):
            # add_dll_directoryが使えない場合はPATHに追加
            current_path = os.environ.get('PATH', '')
            candidate_str = str(candidate)
            if candidate_str not in current_path.split(os.pathsep):
                os.environ['PATH'] = candidate_str + os.pathsep + current_path
        configured_paths.append(candidate)

    return configured_paths

_CONFIGURED_DLL_DIRS = _configure_dll_search_paths()

# =============================================================================
# 地理空間ライブラリ
# =============================================================================
try:
    import rasterio
    import rasterio.features as rasioftr
    from rasterio.warp import reproject, Resampling
    from rasterio.merge import merge as rio_merge
    from rasterio.vrt import WarpedVRT
except OSError as exc:
    conda_prefix = os.environ.get('CONDA_PREFIX', '未検出')
    hint = (
        "rasterio / GDAL のDLLロードに失敗しました。"
        "Windows環境の場合、Conda環境の 'Library\\bin' および 'Library\\lib' を"
        " PATH に追加するか、`RASTERIO_DLL_PATH` 環境変数を設定してください。"
        f" 現在のConda環境: {conda_prefix}. エラー: {exc}"
    )
    raise RuntimeError(hint) from exc

# =============================================================================
# 警告の抑制
# =============================================================================
warnings.filterwarnings('ignore')

# =============================================================================
# 日本語フォント設定
# =============================================================================
try:
    jp_fonts = [f.name for f in fm.fontManager.ttflist 
                if 'Japan' in f.name or 'Noto' in f.name]
    if jp_fonts:
        plt.rcParams['font.sans-serif'] = [jp_fonts[0], 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 定数定義
# =============================================================================
# 分類クラス数の設定（3または5）
# 使用する閾値と対応するクラス数を選択してください

# 3クラス分類の閾値（植生被覆率: 0%, 50%）
# クラス0: 0% (完全崩壊)
# クラス1: 0%超～50%以下 (部分回復)
# クラス2: 50%超～100% (高回復)
THREE_CLASS_FIXED_THRESHOLDS = [0.0, 0.5]

# 5クラス分類の閾値（植生被覆率: 0%, 25%, 50%, 75%）
# クラス0: 0% (完全崩壊)
# クラス1: 0%超～25%以下 (低回復)
# クラス2: 25%超～50%以下 (中回復)
# クラス3: 50%超～75%以下 (高回復)
# クラス4: 75%超～100% (ほぼ完全回復)
FIVE_CLASS_FIXED_THRESHOLDS = [0.0, 0.25, 0.5, 0.75]

# デフォルトで使用する分類設定（"3class" または "5class"）
# main関数内で変更可能
DEFAULT_CLASSIFICATION_MODE = "5class"

def get_classification_config(mode: str = "5class") -> Tuple[List[float], int]:
    """
    分類モードに応じた閾値とクラス数を返す
    
    Parameters:
    -----------
    mode : str
        "3class" または "5class"
    
    Returns:
    --------
    Tuple[List[float], int]
        (閾値リスト, クラス数)
    """
    if mode == "3class":
        return THREE_CLASS_FIXED_THRESHOLDS.copy(), 3
    elif mode == "5class":
        return FIVE_CLASS_FIXED_THRESHOLDS.copy(), 5
    else:
        raise ValueError(f"Unknown classification mode: {mode}. Use '3class' or '5class'.")

from class5_v9 import (_has_overviews, # オーバービュー確認
                       _default_overview_levels, # デフォルトオーバービューレベル
                       ensure_raster_overviews, # ラスターオーバービュー作成
                       BalancedSpatialKFold, # バランス空間KFoldクラス
                       apply_balanced_spatial_kfold,# バランス空間KFold適用
                       print_fold_statistics_improved,# 改良版フォールド統計表示
                       visualize_balanced_spatial_kfold,# バランス空間KFold可視化
                       ensure_dem_mosaic,# DEMモザイク作成
                       ensure_dem_reprojected,# DEM再投影
                       LandslideTipProcessor,# 斜面崩壊地チップ処理クラス
                       LandslideDataset,# データセットクラス
                       GeometricAugmentation,# 幾何学的データ拡張
                       PseudoLabelGenerator,# Pseudo-Label生成クラス
                       _count_classes_from_labels,# クラス分布カウント
                       compute_inverse_frequency_weights,# 逆頻度重み
                       calculate_metrics,# 指標計算
                       plot_training_curves,#学習曲線
                       plot_confusion_matrix,#混同行列
                       cross_validate_with_confusion_matrix,# 交差検証
                       evaluate_and_save_ensemble,# アンサンブル評価保存
                       process_multiple_datasets,# 複数データセット処理
                       analyze_misclassification_patterns,# 誤分類パターン解析
                       generate_class_names_from_thresholds,# クラス名生成
                       init_model,# モデル初期化
                       )
#model関連のインポート
from class5_v9 import (CNN,
                       ViT,
                       FeedForward,
                       MixerBlock,
                       MLPMixer,
                       HybridModel,
                       HybridModelV2,  # Phase 2-1: 柔軟なチャンネル構成対応
                       )

# Attention関連のインポート
from class5_v9 import (ChannelAttention,  # Phase 2-3: CBAM
                       SpatialAttention,
                       CBAM,
                       MultiScaleFeatureExtractor,  # Phase 5-2: ASPP風特徴抽出
                       )

# データ拡張関連のインポート
from class5_v9 import (AdvancedAugmentation,  # Phase 3-1: RGB限定ColorJitter
                       GridMaskAugmentation,   # Phase 3-2: GridMask
                       CombinedAugmentation,   # Phase 3-2: 複合拡張
                       )


# =============================================================================
# 新機能の使用方法（コメント）
# =============================================================================
"""
【新機能の使用方法】

1. HybridModelV2（柔軟なチャンネル構成対応モデル）
   ------------------------------------------------
   main()関数内で以下のように設定:
   
   model_names = ["HybridModelV2"]
   
   チャンネル構成は自動検出されますが、明示的に指定も可能:
   - RGB: [0, 1, 2]
   - DEM: [3]
   - Slope: [4]
   - VDVI: [5]
   - GLCM: [6]
   - EXG: [7]
   
   CBAMアテンションの有効/無効もハイパーパラメータ探索で自動選択されます。


2. データ拡張の切り替え
   ------------------------------------------------
   main()関数内のconfig設定で切り替え:
   
   'augmentation_type': 'geometric'  # 基本（デフォルト）
   'augmentation_type': 'advanced'   # RGB限定ColorJitter追加
   'augmentation_type': 'gridmask'   # GridMask追加
   'augmentation_type': 'combined'   # すべて組み合わせ
   
   - geometric: 回転・反転のみ（全チャンネル対応）
   - advanced: 上記 + RGB限定の色調整（DEM/Slopeには影響なし）
   - gridmask: 上記 + ランダムグリッドマスク（正則化効果）
   - combined: すべての拡張を組み合わせ（最強だが学習時間増）


3. Test-Time Augmentation (TTA)
   ------------------------------------------------
   推論時の精度向上:
   
   from class5_hyb import TestTimeAugmentation, evaluate_with_tta
   
   tta = TestTimeAugmentation(use_tta=True)
   results = evaluate_with_tta(model, dataloader, device, use_tta=True)
   
   5種類の変換（オリジナル+4変換）で予測を平均化


4. Fold Ensemble（K-Fold アンサンブル推論）
   ------------------------------------------------
   複数Foldのモデルを組み合わせて予測:
   
   ensemble = FoldEnsemble(
       model_paths=['fold0.pth', 'fold1.pth', 'fold2.pth'],
       model_name='HybridModelV2',
       model_config=hyperparams,
       n_classes=5
   )
   probs, preds = ensemble.predict(inputs)


5. Temperature Scaling（確率キャリブレーション）
   ------------------------------------------------
   予測確率の信頼性を向上:
   
   temp_scale = TemperatureScaling()
   optimal_temp = temp_scale.calibrate(model, val_loader, device)
   calibrated_probs = temp_scale.get_calibrated_probs(logits)


6. Knowledge Distillation（蒸留学習）
   ------------------------------------------------
   複数モデルの知識を1つのモデルに集約:
   
   student = train_with_distillation(
       student_model=CNN(...),
       teacher_ensemble=FoldEnsemble(...),
       train_loader=train_loader,
       val_loader=val_loader,
       n_epochs=50,
       device='cuda'
   )
"""


# *******************************************************************
# Phase 4-1: Fold Ensemble推論
# *******************************************************************
class FoldEnsemble:
    """
    K-Fold学習で得られた複数モデルのアンサンブル推論
    
    複数のFoldモデルの予測を平均して安定性を向上させる
    """
    
    def __init__(
        self,
        model_paths: List[str],
        model_name: str,
        model_config: Dict,
        n_classes: int = 5,
        device: str = 'cuda'
    ):
        """
        Parameters:
        -----------
        model_paths : List[str]
            各Foldの学習済みモデルパスのリスト
        model_name : str
            モデル名（init_model用）
        model_config : Dict
            モデルのハイパーパラメータ
        n_classes : int
            クラス数
        device : str
            推論デバイス
        """
        self.models = []
        self.device = device
        self.n_classes = n_classes
        
        for path in model_paths:
            model = init_model(model_name, model_config, n_classes)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            model.eval()
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        アンサンブル予測（確率を平均）
        
        Parameters:
        -----------
        x : torch.Tensor
            入力テンソル (B, C, H, W)
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (平均確率, 予測クラス)
        """
        x = x.to(self.device)
        all_probs = []
        
        for model in self.models:
            outputs = model(x)
            # タプル出力モデルの場合
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)
        
        # 平均確率
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        predictions = avg_probs.argmax(dim=1)
        
        return avg_probs, predictions
    
    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        不確実性も含めたアンサンブル予測
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (平均確率, 予測クラス, 予測の不一致度)
        """
        x = x.to(self.device)
        all_probs = []
        all_preds = []
        
        for model in self.models:
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs)
            all_preds.append(probs.argmax(dim=1))
        
        # 平均確率
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        predictions = avg_probs.argmax(dim=1)
        
        # 不一致度（モデル間の予測の標準偏差）
        probs_stack = torch.stack(all_probs, dim=0)  # (n_models, B, n_classes)
        disagreement = probs_stack.std(dim=0).mean(dim=1)  # (B,)
        
        return avg_probs, predictions, disagreement


# *******************************************************************
# Phase 4-2: Test-Time Augmentation (TTA)
# *******************************************************************
class TestTimeAugmentation:
    """
    推論時のデータ拡張による予測安定化
    
    5種類の変換（オリジナル + 4変換）で予測を平均
    """
    
    def __init__(self, use_tta: bool = True):
        self.use_tta = use_tta
    
    @torch.no_grad()
    def predict(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        TTA適用予測
        
        Parameters:
        -----------
        model : nn.Module
            推論モデル
        x : torch.Tensor
            入力テンソル (B, C, H, W)
        
        Returns:
        --------
        torch.Tensor
            平均確率 (B, n_classes)
        """
        if not self.use_tta:
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            return F.softmax(outputs, dim=1)
        
        all_probs = []
        
        # オリジナル
        outputs = model(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        all_probs.append(F.softmax(outputs, dim=1))
        
        # 水平反転
        x_hflip = torch.flip(x, dims=[3])
        outputs = model(x_hflip)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        all_probs.append(F.softmax(outputs, dim=1))
        
        # 垂直反転
        x_vflip = torch.flip(x, dims=[2])
        outputs = model(x_vflip)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        all_probs.append(F.softmax(outputs, dim=1))
        
        # 90度回転
        x_rot90 = torch.rot90(x, k=1, dims=[2, 3])
        outputs = model(x_rot90)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        all_probs.append(F.softmax(outputs, dim=1))
        
        # 180度回転
        x_rot180 = torch.rot90(x, k=2, dims=[2, 3])
        outputs = model(x_rot180)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        all_probs.append(F.softmax(outputs, dim=1))
        
        # 平均
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        return avg_probs
    
    @torch.no_grad()
    def predict_with_std(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TTA適用予測（標準偏差も返す）
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (平均確率, 標準偏差)
        """
        if not self.use_tta:
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = F.softmax(outputs, dim=1)
            return probs, torch.zeros_like(probs)
        
        all_probs = []
        
        for transform in [
            lambda t: t,
            lambda t: torch.flip(t, dims=[3]),
            lambda t: torch.flip(t, dims=[2]),
            lambda t: torch.rot90(t, k=1, dims=[2, 3]),
            lambda t: torch.rot90(t, k=2, dims=[2, 3])
        ]:
            x_aug = transform(x)
            outputs = model(x_aug)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            all_probs.append(F.softmax(outputs, dim=1))
        
        probs_stack = torch.stack(all_probs, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        
        return mean_probs, std_probs


def evaluate_with_tta(model: nn.Module, dataloader: DataLoader, device: str, 
                      use_tta: bool = True) -> Dict:
    """TTAを使用した評価関数"""
    tta = TestTimeAugmentation(use_tta=use_tta)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    for batch in tqdm(dataloader, desc="Evaluating with TTA"):
        inputs = batch['image'].to(device)
        labels = batch['label'].numpy()
        
        probs = tta.predict(model, inputs)
        preds = probs.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.append(probs.cpu())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': torch.cat(all_probs, dim=0).numpy(),
    }


# *******************************************************************
# Phase 4-3: Temperature Scaling
# *******************************************************************
class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for probability calibration
    
    検証データで最適なtemperatureを学習し、予測確率をキャリブレート
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logitsをtemperatureでスケーリング"""
        return logits / self.temperature
    
    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str,
        max_iter: int = 50
    ) -> float:
        """
        検証データで最適なtemperatureを学習
        
        Parameters:
        -----------
        model : nn.Module
            学習済みモデル
        val_loader : DataLoader
            検証データローダー
        device : str
            デバイス
        max_iter : int
            最大イテレーション数
        
        Returns:
        --------
        float
            最適化後のtemperature
        """
        model.eval()
        self.to(device)
        
        # 検証データのlogitsとラベルを収集
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                all_logits.append(outputs)
                all_labels.append(labels)
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # NLLを最小化するtemperatureを探索
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        final_temp = self.temperature.item()
        print(f"  ✅ Temperature Scaling calibrated: T = {final_temp:.4f}")
        
        return final_temp
    
    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """キャリブレートされた確率を返す"""
        with torch.no_grad():
            scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=1)


# *******************************************************************
# Phase 5-1: Knowledge Distillation
# *******************************************************************
class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    
    Teacher: 複数モデルのアンサンブル出力
    Student: 単一の軽量モデル
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Parameters:
        -----------
        temperature : float
            蒸留温度（高いほどソフトラベルが滑らかに）
        alpha : float
            ソフトラベル損失の重み（1-alphaがハードラベル損失の重み）
        class_weights : torch.Tensor, optional
            ハードラベル損失のクラス重み
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        student_logits : torch.Tensor
            生徒モデルの出力logits (B, n_classes)
        teacher_probs : torch.Tensor
            教師モデルの確率出力 (B, n_classes)
        targets : torch.Tensor
            真のラベル (B,)
        
        Returns:
        --------
        torch.Tensor
            統合損失
        """
        # Hard label loss（通常のCE）
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Soft label loss（KL divergence with temperature）
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_probs / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def train_with_distillation(
    student_model: nn.Module,
    teacher_ensemble: 'FoldEnsemble',
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    device: str,
    learning_rate: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.7
) -> nn.Module:
    """
    Knowledge Distillationによる学習
    
    Parameters:
    -----------
    student_model : nn.Module
        生徒モデル
    teacher_ensemble : FoldEnsemble
        教師アンサンブル
    train_loader : DataLoader
        訓練データローダー
    val_loader : DataLoader
        検証データローダー
    n_epochs : int
        エポック数
    device : str
        デバイス
    learning_rate : float
        学習率
    temperature : float
        蒸留温度
    alpha : float
        ソフトラベル損失の重み
    
    Returns:
    --------
    nn.Module
        学習済み生徒モデル
    """
    student_model.to(device)
    student_model.train()
    
    distill_loss = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        student_model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # 教師の予測（勾配なし）
            with torch.no_grad():
                teacher_probs, _ = teacher_ensemble.predict(inputs)
            
            # 生徒の予測
            student_logits = student_model(inputs)
            if isinstance(student_logits, tuple):
                student_logits = student_logits[0]
            
            # 蒸留損失
            loss = distill_loss(student_logits, teacher_probs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # 検証
        student_model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device)
                labels = batch['label'].numpy()
                
                outputs = student_model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_labels.extend(labels)
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.detach().cpu().clone() for k, v in student_model.state_dict().items()}
        
        print(f"Epoch {epoch+1}/{n_epochs}: Loss={epoch_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")
    
    # 最良モデルを復元
    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)
    
    return student_model


# *******************************************************************
# Phase 5-3: Confidence-based Sample Weighting
# *******************************************************************
class ConfidenceWeightedLoss(nn.Module):
    """
    予測確信度に基づく重み付け損失
    
    低確信度サンプルにより大きな重みを与える
    """
    
    def __init__(
        self,
        base_criterion: nn.Module = None,
        beta: float = 1.0,
        max_weight: float = 3.0
    ):
        """
        Parameters:
        -----------
        base_criterion : nn.Module
            基底の損失関数（デフォルト: CrossEntropyLoss）
        beta : float
            重み調整パラメータ（大きいほど低確信度への重みが増加）
        max_weight : float
            最大重み
        """
        super().__init__()
        self.base_criterion = base_criterion or nn.CrossEntropyLoss(reduction='none')
        self.beta = beta
        self.max_weight = max_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        logits : torch.Tensor
            モデル出力 (B, n_classes)
        targets : torch.Tensor
            真のラベル (B,)
        
        Returns:
        --------
        torch.Tensor
            重み付け損失
        """
        # 各サンプルの損失
        base_loss = self.base_criterion(logits, targets)
        
        # 予測確信度（最大確率）
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]
        
        # 重み計算: 低確信度ほど高い重み
        # w = 1 + beta * (1 - confidence)
        weights = 1 + self.beta * (1 - confidence)
        weights = torch.clamp(weights, max=self.max_weight)
        
        # 重み付け平均
        weighted_loss = (base_loss * weights).mean()
        
        return weighted_loss


# *******************************************************************
# 事前正規化済みデータセット（高速化版）
# *******************************************************************
class PreprocessedLandslideDataset(Dataset):
    """
    事前に正規化・テンソル化を完了したデータセット（高速版）
    
    __init__時に全データを一括で正規化・テンソル化してメモリに保持。
    __getitem__では単純なインデックスアクセスのみで、毎回の変換処理を排除。
    これにより学習ループが数倍高速化される。
    """
    
    def __init__(self, images, labels, pixel_locations, geo_locations, 
                 region_idx=None, transform=None, patch_size=100, 
                 thresholds=None, n_classes=5, dem_range=None, vdvi_range=None,
                 glcm_range=None, exg_range=None, force_labels=None):
        """
        Parameters:
        -----------
        images : np.ndarray
            画像パッチ配列 (N, H, W, C)
        labels : np.ndarray
            連続値ラベル (0.0〜1.0)
        transform : callable, optional
            データ拡張（幾何学的変換など）
        patch_size : int
            パッチサイズ（ピクセル）
        dem_range : tuple, optional
            DEM正規化範囲 (min, max)
        vdvi_range : tuple, optional  
            VDVI正規化範囲 (min, max)
        glcm_range : tuple, optional
            GLCM正規化範囲 (min, max)
        exg_range : tuple, optional
            EXG正規化範囲 (min, max)
        """
        self.pixel_locations = pixel_locations
        self.geo_locations = geo_locations
        self.region_idx = region_idx
        self.transform = transform
        self.patch_size = patch_size
        self.n_classes = n_classes
        
        # ラベル設定
        if force_labels is not None:
            self.labels = np.array(force_labels, dtype=np.float32)
        else:
            self.labels = np.array(labels, dtype=np.float32)
        
        # 閾値設定
        if thresholds is None:
            if n_classes == 5:
                self.thresholds = [0.0, 0.25, 0.5, 0.75]
            elif n_classes == 3:
                self.thresholds = [0.0, 0.5]
            else:
                self.thresholds = [i / n_classes for i in range(1, n_classes)]
        else:
            self.thresholds = thresholds
        
        # ====================================
        # 一括正規化・テンソル化（初期化時に1回だけ実行）
        # CPU並列処理で高速化
        # ====================================
        print(f"  📦 Preprocessing {len(images)} patches (one-time normalization with {NUM_WORKERS} workers)...")
        
        n_samples = len(images)
        n_channels = images.shape[-1] if images.ndim == 4 else 1
        
        # 正規化済みテンソルを格納
        self.preprocessed_images = torch.zeros(
            (n_samples, n_channels, patch_size, patch_size), 
            dtype=torch.float32
        )
        
        # クラスラベルを事前計算
        self.class_labels = torch.zeros(n_samples, dtype=torch.long)
        
        # 並列処理用の正規化関数
        def normalize_single_image(idx):
            """単一画像の正規化処理（並列実行用）"""
            return self._normalize_image(
                images[idx], self.labels[idx], patch_size,
                dem_range, vdvi_range, glcm_range, exg_range
            )
        
        # 並列処理でバッチ正規化（48コア活用）
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(normalize_single_image, range(n_samples)),
                total=n_samples,
                desc="  Normalizing",
                ncols=80
            ))
        
        # 結果を格納
        for idx, (image_tensor, class_label) in enumerate(results):
            self.preprocessed_images[idx] = image_tensor
            self.class_labels[idx] = class_label
        
        print(f"  ✅ Preprocessing complete: {self.preprocessed_images.shape}")
    
    def _normalize_image(self, image, label, patch_size, dem_range, vdvi_range, glcm_range, exg_range):
        """単一画像の正規化処理（並列実行用の静的メソッド）"""
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        # NumPy → Tensor変換 (H, W, C) → (C, H, W)
        image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        
        # リサイズ（必要な場合）
        if image_tensor.shape[1] != patch_size or image_tensor.shape[2] != patch_size:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # NaN処理
        image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # チャンネルごとの正規化
        # Index 0-2 (RGB): 255.0で除算し [0, 1] にクリップ
        if image_tensor.shape[0] >= 3:
            image_tensor[:3] = torch.clamp(image_tensor[:3] / 255.0, 0.0, 1.0)
        
        # Index 3 (DEM): dem_range を使用して正規化
        if image_tensor.shape[0] > 3 and dem_range is not None:
            dem_min, dem_max = dem_range
            if dem_max > dem_min:
                image_tensor[3] = (image_tensor[3] - dem_min) / (dem_max - dem_min + 1e-6)
                image_tensor[3] = torch.clamp(image_tensor[3], 0.0, 1.0)
        
        # Index 4 (Slope): 0.0〜90.0度の範囲を 90.0 で除算して正規化
        if image_tensor.shape[0] > 4:
            image_tensor[4] = torch.clamp(image_tensor[4] / 90.0, 0.0, 1.0)
        
        # Index 5 (VDVI): vdvi_range を使用して正規化
        if image_tensor.shape[0] > 5 and vdvi_range is not None:
            vdvi_min, vdvi_max = vdvi_range
            if vdvi_max > vdvi_min:
                image_tensor[5] = (image_tensor[5] - vdvi_min) / (vdvi_max - vdvi_min + 1e-6)
                image_tensor[5] = torch.clamp(image_tensor[5], 0.0, 1.0)
        
        # Index 6 (GLCM): glcm_range を使用して正規化
        if image_tensor.shape[0] > 6 and glcm_range is not None:
            glcm_min, glcm_max = glcm_range
            if glcm_max > glcm_min:
                image_tensor[6] = (image_tensor[6] - glcm_min) / (glcm_max - glcm_min + 1e-6)
                image_tensor[6] = torch.clamp(image_tensor[6], 0.0, 1.0)
        
        # Index 7 (EXG): exg_range を使用して正規化
        if image_tensor.shape[0] > 7 and exg_range is not None:
            exg_min, exg_max = exg_range
            if exg_max > exg_min:
                image_tensor[7] = (image_tensor[7] - exg_min) / (exg_max - exg_min + 1e-6)
                image_tensor[7] = torch.clamp(image_tensor[7], 0.0, 1.0)
        
        # クラスラベルの計算（修正版：正確な閾値処理）
        raw_label = label
        class_label = self._assign_class_label(raw_label)
        
        return image_tensor, class_label
    
    def _assign_class_label(self, value: float, eps: float = 1e-6) -> int:
        """
        連続値ラベル(0-1)をクラスインデックスに変換する
        
        thresholds = [0.0, 0.25, 0.5, 0.75] の場合:
        - Class 0: 0%（value ≈ 0.0）
        - Class 1: 0%超〜25%以下（0.0 < value <= 0.25）
        - Class 2: 25%超〜50%以下（0.25 < value <= 0.50）
        - Class 3: 50%超〜75%以下（0.50 < value <= 0.75）
        - Class 4: 75%超〜100%（0.75 < value <= 1.0）
        """
        # 0% の特別処理（完全崩壊）
        if abs(value) < eps:
            return 0
        
        # 100% の特別処理（ほぼ完全回復）
        if abs(value - 1.0) < eps:
            return len(self.thresholds)
        
        # 閾値に基づいて判定
        # thresholds[0] = 0.0 は Class 0 用なのでスキップし、thresholds[1:] から判定
        for i in range(1, len(self.thresholds)):
            if value <= self.thresholds[i] + eps:
                return i
        
        # 最後の閾値を超えた場合は最大クラス
        return len(self.thresholds)
    
    def __len__(self):
        return len(self.preprocessed_images)
    
    def __getitem__(self, idx):
        # 事前正規化済みテンソルを直接返す（高速）
        image = self.preprocessed_images[idx]
        
        # データ拡張（幾何学的変換のみ）
        if self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'label': self.class_labels[idx],
            'raw_label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'i': self.pixel_locations[idx][0],
            'j': self.pixel_locations[idx][1],
            'lon': self.geo_locations[idx][0],
            'lat': self.geo_locations[idx][1],
        }
        if self.region_idx is not None:
            result['region_idx'] = self.region_idx[idx]
        return result


# *******************************************************************
# Focal Loss（難しいサンプルに重点を置く損失関数）
# *******************************************************************
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    難しいサンプル（予測確率が低いサンプル）に対してより大きな損失を与える。
    gamma=0でCrossEntropyLossと同等、gamma>0で難しいサンプルに重点。
    
    Parameters:
    -----------
    alpha : torch.Tensor, optional
        クラスごとの重み（不均衡データ対策）
    gamma : float
        Focusing parameter（デフォルト: 2.0）
        gamma=0: CrossEntropyLoss
        gamma>0: 難しいサンプルに重点
    reduction : str
        損失の集約方法（'mean', 'sum', 'none'）
    """
    
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, 
                 reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        inputs : torch.Tensor
            モデルの出力（logits）(batch_size, n_classes)
        targets : torch.Tensor
            正解ラベル (batch_size,)
        
        Returns:
        --------
        torch.Tensor
            Focal Loss値
        """
        n_classes = inputs.size(1)
        
        # Label smoothing対応
        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_smooth = torch.zeros_like(inputs)
                targets_smooth.fill_(self.label_smoothing / (n_classes - 1))
                targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Softmax確率を計算
        p = F.softmax(inputs, dim=1)
        
        # 正解クラスの確率を取得
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weightの計算
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weightの適用
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Focal Lossの計算
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# *******************************************************************
# Early Stopping（改良版：Patience + Delta）
# *******************************************************************
class EarlyStopping:
    """
    早期終了のためのクラス（Patience + Min Delta対応）
    
    Parameters:
    -----------
    patience : int
        改善がない場合に待つエポック数
    min_delta : float
        改善とみなす最小の変化量
    mode : str
        'min' (損失最小化) or 'max' (指標最大化)
    restore_best_weights : bool
        Trueの場合、最良の重みを復元
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 mode: str = 'max', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float, model: nn.Module = None) -> bool:
        """
        スコアをチェックし、早期終了すべきかを返す
        
        Parameters:
        -----------
        score : float
            現在のエポックのスコア
        model : nn.Module, optional
            重みを保存するモデル
        
        Returns:
        --------
        bool
            Trueなら早期終了すべき
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_weights(self, model: nn.Module):
        """最良の重みを復元"""
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(model.device if hasattr(model, 'device') else 'cuda') 
                                   for k, v in self.best_weights.items()})


# *******************************************************************
# 高速化ユーティリティ関数群
# *******************************************************************

def clear_memory(verbose: bool = False):
    """GPUメモリとPythonメモリをクリア"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str = 'cuda',
    max_batch_size: int = 256,
) -> int:
    """GPUメモリに基づいて最適なバッチサイズを推定"""
    if device != 'cuda':
        return 64
    
    try:
        # メモリクリア
        clear_memory()
        
        # GPU空きメモリを取得
        free_memory = torch.cuda.get_device_properties(0).total_memory - \
                      torch.cuda.memory_allocated(0)
        
        # モデルパラメータサイズを概算（勾配とオプティマイザ状態も考慮）
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        model_memory = param_size * 3  # パラメータ + 勾配 + オプティマイザ状態
        
        # 利用可能メモリ（安全マージン50%）
        available = (free_memory - model_memory) * 0.5
        
        # 1サンプルあたりのメモリ概算（forward + backward）
        c, h, w = input_shape[1], input_shape[2], input_shape[3]
        sample_memory = c * h * w * 4 * 4  # float32 * forward/backward/activation
        
        optimal_bs = int(available // sample_memory)
        optimal_bs = min(max(optimal_bs, 32), max_batch_size)
        
        # 2の累乗に丸める
        power = int(np.log2(optimal_bs))
        optimal_bs = 2 ** power
        
        return optimal_bs
    except Exception as e:
        print(f"⚠️ Failed to estimate batch size: {e}")
        return 64


def compile_model_if_available(model: nn.Module, device: str = 'cuda') -> nn.Module:
    """PyTorch 2.0+でモデルをコンパイル"""
    if hasattr(torch, 'compile') and device == 'cuda':
        try:
            compiled_model = torch.compile(
                model, 
                mode='reduce-overhead',  # トライアルが多い場合に最適
                # mode='max-autotune',   # 1回だけ長時間学習する場合
            )
            print("✅ Model compiled with torch.compile()")
            return compiled_model
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")
            return model
    return model


def create_fast_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    sampler=None,
    num_workers: int = None,
    device: str = 'cuda'
) -> Tuple[DataLoader, DataLoader]:
    """高速化されたDataLoaderペアを作成"""
    
    if num_workers is None:
        num_workers = NUM_WORKERS
    
    # 共通設定
    common_kwargs = {
        'num_workers': num_workers,
        'pin_memory': (device == 'cuda'),
        'persistent_workers': (num_workers > 0),
        'prefetch_factor': 4 if num_workers > 0 else None,
    }
    
    # 訓練用DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        **common_kwargs,
    )
    
    # 検証用DataLoader（バッチサイズ8倍）
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 8,  # 勾配計算不要なので大きくできる
        shuffle=False,
        **common_kwargs,
    )
    
    return train_loader, val_loader


def create_optimized_study(study_name: str = "landslide_classification"):
    """最適化されたOptuna Studyを作成"""
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=TPESampler(
            seed=42,
            n_startup_trials=5,      # 最初の5トライアルはランダム
            multivariate=True,       # パラメータ間の相関を考慮
            constant_liar=True,      # 並列実行時の効率化
        ),
        pruner=HyperbandPruner(
            min_resource=3,          # 最低3エポック実行
            max_resource=20,         # 最大20エポック
            reduction_factor=3,      # 1/3のトライアルを残す
        ),
    )
    
    return study


def get_efficient_search_space(model_name: str, trial) -> Dict:
    """効率的な探索範囲を定義"""
    
    # 共通パラメータ（カテゴリカルで効率化）
    common_params = {
        'learning_rate': trial.suggest_categorical(
            'learning_rate', 
            [1e-4, 2e-4, 5e-4, 1e-3]  # ログスケールをカテゴリカルに
        ),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'weight_decay': trial.suggest_categorical(
            'weight_decay',
            [1e-5, 1e-4, 5e-4, 1e-3]
        ),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.1, step=0.02),
    }
    
    # モデル固有のパラメータ
    if model_name == 'CNN':
        model_params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'depth': trial.suggest_int('depth', 3, 6),
            'kernel_size': 3,  # 固定
        }
    elif model_name == 'HybridModelV2':
        model_params = {
            'dropout': trial.suggest_float('dropout', 0.1, 0.4, step=0.05),
            'use_cbam': trial.suggest_categorical('use_cbam', [True, False]),
        }
    elif model_name in ['ViT', 'MLP-Mixer']:
        model_params = {
            'patch_size': trial.suggest_categorical('patch_size', [5, 10]),
            'dim': trial.suggest_categorical('dim', [128, 256, 512]),
            'depth': trial.suggest_int('depth', 3, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4, step=0.05),
        }
    else:
        model_params = {}
    
    return {**common_params, **model_params}


# *******************************************************************
# Mixup / CutMix データ拡張
# *******************************************************************
class MixupCutmixAugmentation:
    """
    Mixup と CutMix のデータ拡張を適用するクラス
    
    Parameters:
    -----------
    mixup_alpha : float
        MixupのBeta分布パラメータ（0で無効）
    cutmix_alpha : float
        CutMixのBeta分布パラメータ（0で無効）
    prob : float
        拡張を適用する確率
    switch_prob : float
        MixupとCutMixを切り替える確率（0.5で50%ずつ）
    """
    
    def __init__(self, mixup_alpha: float = 0.4, cutmix_alpha: float = 1.0,
                 prob: float = 0.5, switch_prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor,
                 continuous_labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        バッチに対してMixup/CutMixを適用
        
        Parameters:
        -----------
        images : torch.Tensor
            画像バッチ (batch_size, channels, height, width)
        labels : torch.Tensor
            クラスラベル (batch_size,)
        continuous_labels : torch.Tensor, optional
            連続値ラベル (batch_size,)
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
            (混合画像, 混合ラベル1, 混合ラベル2, lambda値)
        """
        if np.random.random() > self.prob:
            # 拡張を適用しない
            return images, labels, labels, 1.0
        
        batch_size = images.size(0)
        
        # MixupかCutMixかを選択
        use_cutmix = np.random.random() < self.switch_prob and self.cutmix_alpha > 0
        
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        
        # シャッフルしたインデックス
        index = torch.randperm(batch_size, device=images.device)
        
        if use_cutmix:
            # CutMix: 画像の一部を置換
            mixed_images = images.clone()
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            # Lambda値を面積比で調整
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        else:
            # Mixup: 画像を線形補間
            mixed_images = lam * images + (1 - lam) * images[index]
        
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam
    
    def _rand_bbox(self, size, lam):
        """CutMix用のランダム矩形を生成"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 中心座標
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """Mixup用の損失計算"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# *******************************************************************
# 高速トレーニングクラス（AMP対応）
# *******************************************************************
class FastTrainer:
    """AMP対応の高速トレーニングクラス"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # GradScaler（AMP用）
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
    
    def train_epoch(self, train_loader, mixup_cutmix=None) -> float:
        """1エポックの訓練を高速実行"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Mixup/CutMix適用
            if mixup_cutmix is not None:
                inputs, labels_a, labels_b, lam = mixup_cutmix(inputs, labels)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
            
            # Forward pass with AMP
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    if lam < 1.0:
                        loss = lam * self.criterion(outputs, labels_a) + \
                               (1 - lam) * self.criterion(outputs, labels_b)
                    else:
                        loss = self.criterion(outputs, labels)
                    
                    loss = loss / self.accumulation_steps
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if lam < 1.0:
                    loss = lam * self.criterion(outputs, labels_a) + \
                           (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)
                
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * self.accumulation_steps
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, np.ndarray, np.ndarray]:
        """高速な検証実行"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        for batch in val_loader:
            inputs = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label']
            
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, labels.to(self.device))
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, labels.to(self.device))
            
            total_loss += loss.item()
            
            # CPU転送を最後にまとめる
            all_preds.append(outputs.argmax(dim=1))
            all_labels.append(labels)
        
        # 一括でCPU転送・numpy変換
        preds = torch.cat(all_preds).cpu().numpy()
        labels = torch.cat(all_labels).numpy()
        
        return total_loss / len(val_loader), preds, labels


# *******************************************************************
# 傾斜（Slope）計算ユーティリティ
# *******************************************************************
def calculate_slope_from_dem(dem_array: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
    """
    DEMから傾斜角（degrees）を計算
    
    Parameters:
    -----------
    dem_array : np.ndarray
        DEM配列 (height, width)
    pixel_size : float
        ピクセルサイズ（メートル）
    
    Returns:
    --------
    np.ndarray
        傾斜角（度）の配列
    """
    # 勾配を計算
    dy, dx = np.gradient(dem_array, pixel_size)
    
    # 傾斜角（ラジアン→度）
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # NaNやInfを処理
    slope_deg = np.nan_to_num(slope_deg, nan=0.0, posinf=90.0, neginf=0.0)
    
    return slope_deg


def add_slope_channel_to_tips(tips: np.ndarray, dem_channel_idx: int = 3, 
                               pixel_size: float = 5.0) -> np.ndarray:
    """
    チップ配列にDEMから計算した傾斜チャンネルを追加
    
    Parameters:
    -----------
    tips : np.ndarray
        チップ配列 (n_samples, height, width, channels)
    dem_channel_idx : int
        DEMチャンネルのインデックス（通常3: RGB後）
    pixel_size : float
        ピクセルサイズ（メートル）
    
    Returns:
    --------
    np.ndarray
        傾斜チャンネルが追加されたチップ配列
    """
    n_samples, h, w, n_channels = tips.shape
    
    # 傾斜チャンネルを計算
    slope_channel = np.zeros((n_samples, h, w, 1), dtype=np.float32)
    
    for i in range(n_samples):
        dem_patch = tips[i, :, :, dem_channel_idx]
        slope = calculate_slope_from_dem(dem_patch, pixel_size)
        slope_channel[i, :, :, 0] = slope
    
    # チャンネルを結合（DEMの後に傾斜を挿入）
    # 元: [RGB, DEM, VDVI] → [RGB, DEM, Slope, VDVI]
    tips_with_slope = np.concatenate([
        tips[:, :, :, :dem_channel_idx+1],  # RGB + DEM
        slope_channel,                        # Slope
        tips[:, :, :, dem_channel_idx+1:]   # VDVI（もしあれば）
    ], axis=-1)
    
    return tips_with_slope


# *******************************************************************
# Train/Val分布診断関数（過学習原因の特定用）
# *******************************************************************
def diagnose_train_val_distribution(
    train_tips: np.ndarray, 
    val_tips: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    thresholds: List[float],
    n_classes: int,
    channel_names: List[str] = None,
    fold_num: int = 1
) -> Dict:
    """
    Train/Valデータの分布差を診断し、過学習の原因を特定する
    
    Parameters:
    -----------
    train_tips, val_tips : np.ndarray
        訓練/検証データ (N, H, W, C)
    train_labels, val_labels : np.ndarray
        訓練/検証ラベル
    thresholds : List[float]
        クラス閾値
    n_classes : int
        クラス数
    channel_names : List[str]
        チャンネル名リスト
    fold_num : int
        Fold番号
    
    Returns:
    --------
    Dict : 診断結果
    """
    print(f"\n{'='*70}")
    print(f"🔍 Train/Val Distribution Diagnosis (Fold {fold_num})")
    print(f"{'='*70}")
    
    if channel_names is None:
        channel_names = [f'Ch{i}' for i in range(train_tips.shape[-1])]
    
    # 1. クラス分布の比較
    train_class_labels = np.array([assign_class_from_thresholds(l, thresholds) for l in train_labels])
    val_class_labels = np.array([assign_class_from_thresholds(l, thresholds) for l in val_labels])
    
    train_class_counts = np.bincount(train_class_labels, minlength=n_classes)
    val_class_counts = np.bincount(val_class_labels, minlength=n_classes)
    
    train_class_ratio = train_class_counts / len(train_labels) * 100
    val_class_ratio = val_class_counts / len(val_labels) * 100
    
    print(f"\n📊 Class Distribution:")
    print(f"  {'Class':<10} {'Train':>12} {'Val':>12} {'Diff':>10}")
    print(f"  {'-'*44}")
    max_diff = 0
    for c in range(n_classes):
        diff = abs(train_class_ratio[c] - val_class_ratio[c])
        max_diff = max(max_diff, diff)
        marker = "⚠️" if diff > 10 else ""
        print(f"  Class {c:<4} {train_class_ratio[c]:>10.1f}% {val_class_ratio[c]:>10.1f}% {diff:>8.1f}% {marker}")
    
    # 2. 各チャンネルの統計量比較
    print(f"\n📈 Channel Statistics (mean ± std):")
    print(f"  {'Channel':<10} {'Train':>20} {'Val':>20} {'Mean Diff':>12}")
    print(f"  {'-'*62}")
    
    channel_diffs = []
    for c, name in enumerate(channel_names):
        train_ch = train_tips[:, :, :, c].flatten()
        val_ch = val_tips[:, :, :, c].flatten()
        
        train_mean, train_std = train_ch.mean(), train_ch.std()
        val_mean, val_std = val_ch.mean(), val_ch.std()
        mean_diff = abs(train_mean - val_mean)
        channel_diffs.append(mean_diff)
        
        # 相対的な差の大きさで警告
        relative_diff = mean_diff / (abs(train_mean) + 1e-6) * 100
        marker = "⚠️" if relative_diff > 20 else ""
        
        print(f"  {name:<10} {train_mean:>8.3f} ± {train_std:<8.3f} {val_mean:>8.3f} ± {val_std:<8.3f} {mean_diff:>10.3f} {marker}")
    
    # 3. 診断結果のサマリー
    print(f"\n📋 Diagnosis Summary:")
    
    issues = []
    
    if max_diff > 15:
        issues.append(f"  ❌ Class distribution imbalance: max diff = {max_diff:.1f}%")
        issues.append(f"     → Train/Valで特定クラスの比率が大きく異なります")
    
    # DEMチャンネル（通常ch3）の差をチェック
    if len(channel_diffs) > 3 and channel_diffs[3] > 50:
        issues.append(f"  ❌ DEM values differ significantly between Train/Val")
        issues.append(f"     → 地形が異なる地域がTrain/Valに分かれている可能性")
    
    # Slopeチャンネル（通常ch4）の差をチェック
    if len(channel_diffs) > 4 and channel_diffs[4] > 5:
        issues.append(f"  ❌ Slope values differ significantly between Train/Val")
        issues.append(f"     → 傾斜特性が異なる地域がTrain/Valに分かれている可能性")
    
    if not issues:
        print(f"  ✅ No significant distribution differences detected")
    else:
        for issue in issues:
            print(issue)
        print(f"\n  💡 推奨対策:")
        print(f"     - use_spatial_kfold=False に変更してStratified K-Foldを試す")
        print(f"     - データ拡張（Mixup/CutMix）を有効にして分布差を軽減")
        print(f"     - 正則化を強化（Dropout増加、Weight Decay増加）")
    
    print(f"{'='*70}")
    
    return {
        'class_distribution_diff': max_diff,
        'channel_mean_diffs': channel_diffs,
        'issues': issues
    }


# *******************************************************************
# 難しいパッチの抽出・保存関数
# *******************************************************************
def extract_and_save_difficult_patches(
    images: np.ndarray,
    predictions_df: pd.DataFrame,
    save_dir: str,
    n_classes: int = 5,
    patch_size: int = 25,
    max_patches_per_type: int = 100,
    confidence_threshold: float = 0.5
) -> Dict:
    """
    判別が難しかったパッチ（誤分類・低確信度）を抽出して保存
    
    Parameters:
    -----------
    images : np.ndarray
        全パッチ画像 (N, H, W, C)
    predictions_df : pd.DataFrame
        予測結果を含むDataFrame（val_index, true_label, predicted_label, prob_class_*列を含む）
    save_dir : str
        保存先ディレクトリ
    n_classes : int
        クラス数
    patch_size : int
        抽出するパッチサイズ（中心からのクロップ）
    max_patches_per_type : int
        各タイプごとの最大保存パッチ数
    confidence_threshold : float
        低確信度とみなす閾値（これ以下を低確信度とする）
    
    Returns:
    --------
    Dict : 抽出結果のサマリー
    """
    import cv2
    
    difficult_patches_dir = os.path.join(save_dir, 'difficult_patches')
    os.makedirs(difficult_patches_dir, exist_ok=True)
    
    # サブディレクトリの作成
    misclassified_dir = os.path.join(difficult_patches_dir, 'misclassified')
    low_confidence_dir = os.path.join(difficult_patches_dir, 'low_confidence')
    boundary_cases_dir = os.path.join(difficult_patches_dir, 'boundary_cases')
    os.makedirs(misclassified_dir, exist_ok=True)
    os.makedirs(low_confidence_dir, exist_ok=True)
    os.makedirs(boundary_cases_dir, exist_ok=True)
    
    # クラスごとのサブディレクトリ
    for i in range(n_classes):
        os.makedirs(os.path.join(misclassified_dir, f'true_{i}'), exist_ok=True)
        os.makedirs(os.path.join(low_confidence_dir, f'class_{i}'), exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🔍 Extracting Difficult Patches")
    print(f"{'='*70}")
    
    # 確率列の取得
    prob_cols = [f'prob_class_{i}' for i in range(n_classes)]
    
    results = {
        'misclassified': {'count': 0, 'saved': 0},
        'low_confidence': {'count': 0, 'saved': 0},
        'boundary_cases': {'count': 0, 'saved': 0}
    }
    
    # 画像のサイズ情報
    _, h, w, c = images.shape
    crop_start = (h - patch_size) // 2
    crop_end = crop_start + patch_size
    
    # 1. 誤分類パッチの抽出
    misclassified_mask = predictions_df['true_label'] != predictions_df['predicted_label']
    misclassified_df = predictions_df[misclassified_mask].copy()
    results['misclassified']['count'] = len(misclassified_df)
    
    print(f"\n📊 Misclassified patches: {len(misclassified_df)}")
    
    # 誤分類を確信度でソート（確信度が高いのに間違えたものが重要）
    misclassified_df['max_prob'] = misclassified_df[prob_cols].max(axis=1)
    misclassified_df = misclassified_df.sort_values('max_prob', ascending=False)
    
    saved_misclassified = 0
    for _, row in misclassified_df.iterrows():
        if saved_misclassified >= max_patches_per_type:
            break
        
        val_idx = int(row['val_index'])
        true_label = int(row['true_label'])
        pred_label = int(row['predicted_label'])
        max_prob = row['max_prob']
        
        # パッチの抽出（中心25x25）
        patch = images[val_idx, crop_start:crop_end, crop_start:crop_end, :3]  # RGBのみ
        
        # 0-255にスケール
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
        else:
            patch = patch.astype(np.uint8)
        
        # ファイル名: true{真のラベル}_pred{予測ラベル}_conf{確信度}_idx{インデックス}.png
        filename = f"true{true_label}_pred{pred_label}_conf{max_prob:.3f}_idx{val_idx}.png"
        save_path = os.path.join(misclassified_dir, f'true_{true_label}', filename)
        
        # BGRに変換して保存
        cv2.imwrite(save_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        saved_misclassified += 1
    
    results['misclassified']['saved'] = saved_misclassified
    print(f"  Saved: {saved_misclassified} patches")
    
    # 2. 低確信度パッチの抽出（正解していても確信度が低いもの）
    predictions_df['max_prob'] = predictions_df[prob_cols].max(axis=1)
    low_conf_mask = predictions_df['max_prob'] < confidence_threshold
    low_conf_df = predictions_df[low_conf_mask].copy()
    low_conf_df = low_conf_df.sort_values('max_prob', ascending=True)  # 確信度が低い順
    results['low_confidence']['count'] = len(low_conf_df)
    
    print(f"\n📊 Low confidence patches (< {confidence_threshold}): {len(low_conf_df)}")
    
    saved_low_conf = 0
    for _, row in low_conf_df.iterrows():
        if saved_low_conf >= max_patches_per_type:
            break
        
        val_idx = int(row['val_index'])
        true_label = int(row['true_label'])
        pred_label = int(row['predicted_label'])
        max_prob = row['max_prob']
        
        patch = images[val_idx, crop_start:crop_end, crop_start:crop_end, :3]
        
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
        else:
            patch = patch.astype(np.uint8)
        
        correct_str = "correct" if true_label == pred_label else "wrong"
        filename = f"{correct_str}_true{true_label}_pred{pred_label}_conf{max_prob:.3f}_idx{val_idx}.png"
        save_path = os.path.join(low_confidence_dir, f'class_{pred_label}', filename)
        
        cv2.imwrite(save_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        saved_low_conf += 1
    
    results['low_confidence']['saved'] = saved_low_conf
    print(f"  Saved: {saved_low_conf} patches")
    
    # 3. 境界ケースの抽出（2つのクラスの確率が近いもの）
    predictions_df['prob_diff'] = predictions_df[prob_cols].apply(
        lambda x: sorted(x, reverse=True)[0] - sorted(x, reverse=True)[1] 
        if len(x) > 1 else 1.0, axis=1
    )
    boundary_mask = predictions_df['prob_diff'] < 0.2  # 上位2クラスの確率差が0.2未満
    boundary_df = predictions_df[boundary_mask].copy()
    boundary_df = boundary_df.sort_values('prob_diff', ascending=True)
    results['boundary_cases']['count'] = len(boundary_df)
    
    print(f"\n📊 Boundary cases (prob diff < 0.2): {len(boundary_df)}")
    
    saved_boundary = 0
    for _, row in boundary_df.iterrows():
        if saved_boundary >= max_patches_per_type:
            break
        
        val_idx = int(row['val_index'])
        true_label = int(row['true_label'])
        pred_label = int(row['predicted_label'])
        prob_diff = row['prob_diff']
        
        patch = images[val_idx, crop_start:crop_end, crop_start:crop_end, :3]
        
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
        else:
            patch = patch.astype(np.uint8)
        
        correct_str = "correct" if true_label == pred_label else "wrong"
        filename = f"{correct_str}_true{true_label}_pred{pred_label}_diff{prob_diff:.3f}_idx{val_idx}.png"
        save_path = os.path.join(boundary_cases_dir, filename)
        
        cv2.imwrite(save_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        saved_boundary += 1
    
    results['boundary_cases']['saved'] = saved_boundary
    print(f"  Saved: {saved_boundary} patches")
    
    # サマリーの保存
    summary_path = os.path.join(difficult_patches_dir, 'extraction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'patch_size': patch_size,
            'confidence_threshold': confidence_threshold,
            'max_patches_per_type': max_patches_per_type,
            'results': results
        }, f, indent=2)
    
    print(f"\n✅ Difficult patches saved to: {difficult_patches_dir}")
    print(f"  - Misclassified: {results['misclassified']['saved']}/{results['misclassified']['count']}")
    print(f"  - Low confidence: {results['low_confidence']['saved']}/{results['low_confidence']['count']}")
    print(f"  - Boundary cases: {results['boundary_cases']['saved']}/{results['boundary_cases']['count']}")
    print(f"{'='*70}")
    
    return results


# *******************************************************************
# 改善版クロスバリデーション関数（Mixup/CutMix、Focal Loss、Early Stopping対応）
# *******************************************************************
def cross_validate_with_improvements(
    model_name: str,
    hyperparams: Dict,
    images: np.ndarray,
    labels: np.ndarray,
    pixel_locations: np.ndarray,
    geo_locations: np.ndarray,
    landslide_ids: np.ndarray,
    n_classes: int = 5,
    n_folds: int = 5,
    n_epochs: int = 60,
    batch_size: int = 48,
    device: str = 'cuda',
    save_dir: str = 'results',
    optimize_threshold: bool = True,
    use_spatial_kfold: bool = True,
    config: Dict = None,
    pseudo_labels: Optional[np.ndarray] = None,
    patch_size: int = 100
) -> Dict:
    """
    改善版クロスバリデーション
    
    以下の改善を含む：
    - Focal Loss（難しいサンプルに重点）
    - Mixup/CutMix（データ拡張）
    - オーバーサンプリング（クラス不均衡対策）
    - Early Stopping（過学習防止）
    - CosineAnnealingWarmRestarts（学習率スケジューラ）
    
    Parameters:
    -----------
    pseudo_labels : np.ndarray, optional
        Self-Training用の擬似ラベル配列。指定された場合、学習データにはこの
        擬似ラベルが適用され、検証データには元のlabels（Ground Truth）が
        使用される。これにより、Validation Leakを防止し、正しい評価が行われる。
        Noneの場合は通常の学習フローとなり、全てにlabelsが使用される。
    """
    from class5_v9 import (
        init_model, compute_inverse_frequency_weights,
        calculate_metrics, plot_confusion_matrix, plot_training_curves_enhanced,
        apply_balanced_spatial_kfold, find_optimal_thresholds, plot_cv_scores_enhanced,
        is_intermediate_model, get_loss_fn_for_model
    )
    # LandslideDataset は class5_hyb.py 内の最適化版 PreprocessedLandslideDataset を使用
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    
    # 設定パラメータの取得
    fixed_thresholds = config.get('fixed_thresholds') if config else None
    thresholds_for_split = fixed_thresholds if fixed_thresholds is not None else [i / n_classes for i in range(1, n_classes)]
    class_names_template = generate_class_names_from_thresholds(fixed_thresholds, n_classes)
    
    # 改善機能の設定
    use_focal_loss = config.get('use_focal_loss', False) if config else False
    focal_gamma = config.get('focal_gamma', 2.0) if config else 2.0
    use_mixup_cutmix = config.get('use_mixup_cutmix', False) if config else False
    mixup_alpha = config.get('mixup_alpha', 0.4) if config else 0.4
    cutmix_alpha = config.get('cutmix_alpha', 1.0) if config else 1.0
    use_oversampling = config.get('use_oversampling', False) if config else False
    early_stopping_patience = config.get('early_stopping_patience', 10) if config else 10
    early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001) if config else 0.001
    dem_range = config.get('dem_range') if config else None
    vdvi_range = config.get('vdvi_range') if config else None
    glcm_range = config.get('glcm_range') if config else None
    exg_range = config.get('exg_range') if config else None
    
    # 改善機能の表示
    print(f"\n{'='*70}")
    print(f"🚀 Improved Cross-Validation Settings")
    print(f"{'='*70}")
    print(f"  Focal Loss: {'ON (γ=' + str(focal_gamma) + ')' if use_focal_loss else 'OFF'}")
    print(f"  Mixup/CutMix: {'ON (α=' + str(mixup_alpha) + '/' + str(cutmix_alpha) + ')' if use_mixup_cutmix else 'OFF'}")
    print(f"  Oversampling: {'ON' if use_oversampling else 'OFF'}")
    print(f"  Early Stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print(f"{'='*70}")
    
    # Spatial K-Foldまたは通常のStratified K-Foldを選択
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
    
    # Mixup/CutMix拡張の初期化
    mixup_cutmix = None
    if use_mixup_cutmix:
        mixup_cutmix = MixupCutmixAugmentation(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=0.5
        )
        print(f"  ✅ Mixup/CutMix augmentation initialized")
    
    fold_results = []
    all_cms = []
    best_thresholds_per_fold = []
    all_predictions_df = []
    
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
    
    total_folds = n_folds
    
    # Foldごとの処理
    for fold in tqdm(range(n_folds), desc="Folds", ncols=100):
        fold_start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"📁 Fold {fold + 1}/{n_folds} - {(fold/total_folds)*100:.1f}% Complete")
        print(f"{'='*70}")
        
        # インデックスの取得
        if use_spatial_kfold:
            val_idx = np.where(spatial_folds == fold)[0]
            train_idx = np.where(spatial_folds != fold)[0]
        else:
            train_val_splits = list(kf.split(images, class_labels))
            train_idx, val_idx = train_val_splits[fold]
        
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # =================================================================
        # データ拡張の選択（config設定に応じて切り替え）
        # =================================================================
        augmentation_type = config.get('augmentation_type', 'geometric')
        
        if augmentation_type == 'advanced':
            # AdvancedAugmentation: RGB限定ColorJitter
            train_transform = AdvancedAugmentation(
                color_jitter_prob=0.5,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                rgb_channels=[0, 1, 2]  # RGBチャンネルのみに適用
            )
            print(f"  🔄 Using AdvancedAugmentation (ColorJitter on RGB only)")
        elif augmentation_type == 'gridmask':
            # GridMaskAugmentation: ランダムグリッドマスク
            train_transform = GridMaskAugmentation(
                d_range=(96, 224),
                ratio=0.6,
                prob=0.5
            )
            print(f"  🔄 Using GridMaskAugmentation")
        elif augmentation_type == 'combined':
            # CombinedAugmentation: 全て組み合わせ
            train_transform = CombinedAugmentation(
                geometric_aug=GeometricAugmentation(
                    h_flip_prob=0.5,
                    v_flip_prob=0.5,
                    rotate90_prob=0.5
                ),
                advanced_aug=AdvancedAugmentation(
                    color_jitter_prob=0.5,
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    rgb_channels=[0, 1, 2]
                ),
                gridmask_aug=GridMaskAugmentation(
                    d_range=(96, 224),
                    ratio=0.6,
                    prob=0.5
                )
            )
            print(f"  🔄 Using CombinedAugmentation (Geometric + ColorJitter + GridMask)")
        else:
            # デフォルト: GeometricAugmentation（基本）
            train_transform = GeometricAugmentation(
                h_flip_prob=0.5,
                v_flip_prob=0.5,
                rotate90_prob=0.5
            )
            print(f"  🔄 Using GeometricAugmentation (basic)")
        
        # ========================================
        # データセットの作成（Pseudo-Label対応）
        # - 学習データ: pseudo_labelsが提供されている場合はそれを使用
        # - 検証データ: 常にオリジナルのlabels（Ground Truth）を使用
        # ========================================
        
        # 学習用ラベルの決定（Pseudo-Labelまたはオリジナル）
        if pseudo_labels is not None:
            train_force_labels = pseudo_labels[train_idx]
            print(f"  📌 Training set: Using PSEUDO-LABELS (force_labels applied)")
        else:
            train_force_labels = None
            print(f"  📌 Training set: Using ORIGINAL labels")
        
        # 検証用は常にオリジナルラベル（Validation Leak防止）
        print(f"  📌 Validation set: Using ORIGINAL labels (Ground Truth)")
        
        train_dataset = PreprocessedLandslideDataset(
            images[train_idx], labels[train_idx],  # labelsはraw_label用に保持
            pixel_locations[train_idx], geo_locations[train_idx],
            n_classes=n_classes,
            thresholds=fixed_thresholds,
            patch_size=patch_size,  # tip_size_mから計算したパッチサイズ
            dem_range=dem_range,
            vdvi_range=vdvi_range,
            glcm_range=glcm_range,
            exg_range=exg_range,
            transform=train_transform,  # 幾何学的データ拡張を適用
            force_labels=train_force_labels  # Pseudo-Label適用
        )
        val_dataset = PreprocessedLandslideDataset(
            images[val_idx], labels[val_idx],
            pixel_locations[val_idx], geo_locations[val_idx],
            n_classes=n_classes,
            thresholds=fixed_thresholds,
            patch_size=patch_size,  # tip_size_mから計算したパッチサイズ
            dem_range=dem_range,
            vdvi_range=vdvi_range,
            glcm_range=glcm_range,
            exg_range=exg_range,
            force_labels=None  # 検証データは常にGround Truthを使用
            # 検証データには拡張を適用しない
        )
        
        actual_batch_size = hyperparams.get('batch_size', batch_size)
        
        # データローダーの作成（オーバーサンプリング対応）
        if use_oversampling:
            # オーバーサンプリング用のラベル：pseudo_labelsがある場合はそれを使用
            # これにより、学習時のサンプリング重みとラベルが整合する
            if pseudo_labels is not None:
                sampler_labels = pseudo_labels[train_idx]
                print(f"  ℹ️  Oversampling weights calculated from PSEUDO-LABELS")
            else:
                sampler_labels = labels[train_idx]
            sampler = create_weighted_sampler(
                sampler_labels, fixed_thresholds, n_classes
            )
            train_loader = DataLoader(
                train_dataset, batch_size=actual_batch_size, 
                sampler=sampler, num_workers=NUM_WORKERS,
                pin_memory=True, persistent_workers=False, prefetch_factor=2
            )
            print(f"  ✅ Using oversampling (WeightedRandomSampler)")
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=actual_batch_size, 
                shuffle=True, num_workers=NUM_WORKERS,
                pin_memory=True, persistent_workers=False, prefetch_factor=2
            )
        
        val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False,
                                        num_workers=NUM_WORKERS, pin_memory=True,
                                        persistent_workers=False, prefetch_factor=2)
        
        # モデルの初期化
        model = init_model(model_name, hyperparams, n_classes)
        model = model.to(device)
        
        use_intermediate_loss = is_intermediate_model(model_name)
        
        learning_rate = hyperparams.get('learning_rate', 1e-4)
        weight_decay = hyperparams.get('weight_decay', 1e-4)
        
        # クラス重みの計算
        class_weights, class_counts = compute_inverse_frequency_weights(
            train_dataset.labels, train_dataset.thresholds, n_classes
        )
        print(f"  Class counts: {class_counts.tolist()} -> weights: {[round(w, 4) for w in class_weights]}")
        
        # 🔧 修正: use_oversamplingの場合はclass_weightsをLossに渡さない（二重適用防止）
        # WeightedRandomSamplerで既にクラスバランスを取っているため
        if use_oversampling:
            class_weight_tensor = None
            print(f"  ℹ️  Oversampling enabled → class_weights NOT applied to loss (avoiding double correction)")
            if use_focal_loss:
                import warnings
                warnings.warn(
                    "⚠️  Both Oversampling and Focal Loss are enabled. "
                    "Focal Loss will use uniform weights (alpha=None) since Oversampling already handles class imbalance. "
                    "This is intentional to avoid double correction.",
                    UserWarning
                )
        else:
            class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            print(f"  ℹ️  Class weights applied to loss: {[f'{w:.4f}' for w in class_weights]}")
        
        # 損失関数の設定
        if use_intermediate_loss:
            intermediate_loss_fn = get_loss_fn_for_model(model_name, n_classes, fixed_thresholds)
            if intermediate_loss_fn is not None:
                intermediate_loss_fn = intermediate_loss_fn.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
            print(f"  Using specialized loss function for {model_name}")
        else:
            intermediate_loss_fn = None
            if use_focal_loss:
                criterion = FocalLoss(
                    alpha=class_weight_tensor,  # Noneの場合、FocalLossは重み付けなしで動作
                    gamma=focal_gamma,
                    reduction='mean'
                )
                print(f"  ✅ Using Focal Loss (γ={focal_gamma})")
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # CosineAnnealingWarmRestarts スケジューラ
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Early Stopping の初期化
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='max'
        )
        
        # 学習曲線用のリスト
        train_losses, val_losses = [], []
        train_f1s, val_f1s = [], []
        train_accs, val_accs = [], []
        
        print(f"\n📈 Training for up to {n_epochs} epochs (Early Stopping enabled)...")
        print(f"{'─'*70}")
        
        best_model_state = None
        best_val_f1 = 0
        best_epoch = 0
        best_val_preds = None
        best_val_labels = None
        best_val_probs = None
        
        # エポック進捗バーを作成
        epoch_pbar = tqdm(range(n_epochs), desc=f"  Fold {fold+1} Training", ncols=100, leave=True)
        
        # デバッグ用：時間計測とリソース監視
        import time
        import psutil
        debug_times = {'data_load': [], 'forward': [], 'backward': [], 'optimizer': [], 'metrics': []}
        process = psutil.Process(os.getpid())
        
        # 初期メモリ使用量
        initial_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        tqdm.write(f"\\n  [DEBUG] System Information:")
        tqdm.write(f"    CPU cores available: {psutil.cpu_count()}")
        tqdm.write(f"    Initial CPU memory: {initial_memory:.2f} GB")
        if device == 'cuda' and torch.cuda.is_available():
            tqdm.write(f"    GPU Device: {torch.cuda.get_device_name(0)}")
            tqdm.write(f"    Initial GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            tqdm.write(f"    GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        else:
            tqdm.write(f"    ⚠️  Running on CPU (no GPU detected)")
        
        for epoch in epoch_pbar:
            epoch_start = datetime.now()
            
            # === Training ===
            model.train()
            train_loss = 0
            train_preds = []
            train_labels_epoch = []
            
            # デバッグ：バッチ処理時間計測
            batch_times = []
            
            for batch_idx, batch in enumerate(train_loader):
                t0 = time.time()
                
                inputs = batch['image'].to(device)
                labels_batch = batch['label'].to(device)
                raw_labels_batch = batch['raw_label'].to(device)
                t1 = time.time()
                debug_times['data_load'].append(t1 - t0)
                
                optimizer.zero_grad()
                
                # Mixup/CutMix適用（訓練時のみ）
                if mixup_cutmix is not None:
                    inputs, labels_a, labels_b, lam = mixup_cutmix(inputs, labels_batch)
                else:
                    labels_a, labels_b, lam = labels_batch, labels_batch, 1.0
                
                # 中間値予測モデルの場合
                t2 = time.time()
                if use_intermediate_loss:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        class_probs, model_outputs = outputs
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
                    # Mixup/CutMix用の損失計算
                    if mixup_cutmix is not None and lam < 1.0:
                        loss = MixupCutmixAugmentation.mixup_criterion(
                            criterion, outputs, labels_a, labels_b, lam
                        )
                    else:
                        loss = criterion(outputs, labels_batch)
                    preds = torch.argmax(outputs, dim=1).cpu()
                
                t3 = time.time()
                debug_times['forward'].append(t3 - t2)
                
                loss.backward()
                t4 = time.time()
                debug_times['backward'].append(t4 - t3)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                t5 = time.time()
                debug_times['optimizer'].append(t5 - t4)
                
                train_loss += loss.item()
                train_preds.extend(preds.numpy())
                train_labels_epoch.extend(labels_batch.cpu().numpy())
                
                # 最初の数バッチの詳細情報を表示
                if epoch == 0 and batch_idx < 5:
                    tqdm.write(f"  [DEBUG Batch {batch_idx}] DataLoad:{(t1-t0)*1000:.1f}ms Forward:{(t3-t2)*1000:.1f}ms Backward:{(t4-t3)*1000:.1f}ms Optim:{(t5-t4)*1000:.1f}ms Total:{(t5-t0)*1000:.1f}ms")
            
            t_metrics_start = time.time()
            avg_train_loss = train_loss / len(train_loader)
            train_f1 = f1_score(train_labels_epoch, train_preds, average='macro', zero_division=0)
            train_acc = accuracy_score(train_labels_epoch, train_preds)
            t_metrics_end = time.time()
            debug_times['metrics'].append(t_metrics_end - t_metrics_start)
            
            # エポック0のデバッグサマリー
            if epoch == 0:
                current_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
                cpu_percent = process.cpu_percent()
                
                tqdm.write(f"\n  [DEBUG] Epoch 0 Timing Summary (Training Phase):")
                tqdm.write(f"    DataLoad:  avg={np.mean(debug_times['data_load'])*1000:.1f}ms, total={np.sum(debug_times['data_load']):.1f}s")
                tqdm.write(f"    Forward:   avg={np.mean(debug_times['forward'])*1000:.1f}ms, total={np.sum(debug_times['forward']):.1f}s")
                tqdm.write(f"    Backward:  avg={np.mean(debug_times['backward'])*1000:.1f}ms, total={np.sum(debug_times['backward']):.1f}s")
                tqdm.write(f"    Optimizer: avg={np.mean(debug_times['optimizer'])*1000:.1f}ms, total={np.sum(debug_times['optimizer']):.1f}s")
                tqdm.write(f"    Metrics:   {debug_times['metrics'][-1]:.2f}s")
                total_train_time = sum([np.sum(debug_times[k]) for k in ['data_load', 'forward', 'backward', 'optimizer']]) + debug_times['metrics'][-1]
                tqdm.write(f"    Total Training: {total_train_time:.1f}s ({len(train_loader)} batches)")
                tqdm.write(f"    Batch size: {actual_batch_size}, Num workers: {NUM_WORKERS}")
                tqdm.write(f"  [DEBUG] Resource Usage:")
                tqdm.write(f"    Memory: {current_memory:.2f} GB (delta: {current_memory-initial_memory:.2f} GB)")
                tqdm.write(f"    CPU: {cpu_percent:.1f}%")
                
                # GPU情報（CUDAが使える場合）
                if device == 'cuda' and torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
                    gpu_memory_max = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024  # GB
                    tqdm.write(f"    GPU Memory: {gpu_memory:.2f} GB (peak: {gpu_memory_max:.2f} GB)")
                tqdm.write("")
            
            # === Validation ===
            t_val_start = time.time()
            model.eval()
            val_loss = 0
            val_preds_epoch = []      # GPU tensors
            val_labels_epoch = []     # CPU tensors
            val_probs_epoch = []      # GPU tensors
            
            val_batch_count = 0
            val_losses_batch = []  # loss テンソルをリストに蓄積
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['image'].to(device, non_blocking=True)
                    labels_batch = batch['label']
                    raw_labels_batch = batch['raw_label'].to(device, non_blocking=True)
                    
                    if use_intermediate_loss:
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            class_probs, model_outputs = outputs
                            if intermediate_loss_fn is not None:
                                loss, _ = intermediate_loss_fn(model_outputs, class_probs, labels_batch.to(device), raw_labels_batch)
                            else:
                                loss = criterion(torch.log(class_probs + 1e-7), labels_batch.to(device))
                            # GPU上に保持（.cpu()を呼ばない）
                            val_probs_epoch.append(class_probs)
                            val_preds_epoch.append(class_probs.argmax(dim=1))
                        else:
                            loss = criterion(outputs, labels_batch.to(device))
                            val_probs_epoch.append(torch.softmax(outputs, dim=1))
                            val_preds_epoch.append(outputs.argmax(dim=1))
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels_batch.to(device))
                        # GPU上に保持（.cpu()を呼ばない）
                        val_probs_epoch.append(torch.softmax(outputs, dim=1))
                        val_preds_epoch.append(outputs.argmax(dim=1))
                    
                    val_losses_batch.append(loss)  # .item() を呼ばない
                    val_labels_epoch.append(labels_batch)
                    val_batch_count += 1
            
            # ループ終了後に一括でCPU転送（1回だけ）
            val_loss = torch.stack(val_losses_batch).sum().item()  # ここで1回だけ .item()
            val_preds_epoch = torch.cat(val_preds_epoch).cpu().numpy()
            val_probs_epoch = torch.cat(val_probs_epoch).cpu().numpy()
            val_labels_epoch = torch.cat(val_labels_epoch).numpy()
            
            t_val_end = time.time()
            val_time = t_val_end - t_val_start
            
            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(val_labels_epoch, val_preds_epoch, average='macro', zero_division=0)
            val_acc = accuracy_score(val_labels_epoch, val_preds_epoch)
            
            # エポック0のValidation時間も表示
            if epoch == 0:
                tqdm.write(f"  [DEBUG] Validation: {val_time:.1f}s ({val_batch_count} batches, avg={val_time/val_batch_count*1000:.1f}ms/batch)\n")
            
            # 学習曲線データの保存
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # ベストモデルの保存
            improvement = ""
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_preds = val_preds_epoch.copy()
                best_val_labels = val_labels_epoch.copy()
                best_val_probs = np.vstack(val_probs_epoch).copy()
                improvement = " ⭐ NEW BEST!"
            
            # tqdmの進捗表示を更新
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            current_lr = optimizer.param_groups[0]['lr']
            epoch_pbar.set_postfix({
                'TrLoss': f'{avg_train_loss:.4f}',
                'VaLoss': f'{avg_val_loss:.4f}',
                'TrF1': f'{train_f1:.4f}',
                'VaF1': f'{val_f1:.4f}',
                'Best': f'{best_val_f1:.4f}',
                'LR': f'{current_lr:.2e}',
                'Time': f'{epoch_time:.1f}s'
            })
            
            # 重要な改善時には明示的に表示
            if improvement:
                tqdm.write(f"  {'='*90}")
                tqdm.write(f"  🌟 Epoch {epoch+1}/{n_epochs}: NEW BEST F1 = {val_f1:.4f} (prev: {best_val_f1-val_f1+best_val_f1:.4f})")
                tqdm.write(f"  {'='*90}")
            
            scheduler.step()
            
            # Early Stopping チェック
            early_stopping(val_f1)
            if early_stopping.early_stop:
                epoch_pbar.close()
                tqdm.write(f"\n  ⏹️  Early stopping triggered at epoch {epoch + 1}")
                tqdm.write(f"      Best F1 was {early_stopping.best_score:.4f} at epoch {best_epoch}")
                break
        
        # 学習曲線の保存
        curve_path = os.path.join(curves_dir, f'fold_{fold+1}_curves.png')
        curve_stats = plot_training_curves_enhanced(
            train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs,
            save_path=curve_path, fold_num=fold+1
        )
        print(f"\n📊 Training Summary:")
        print(f"  Best F1: {curve_stats['best_val_f1']:.4f} at epoch {curve_stats['best_epoch']}")
        print(f"  Best Acc: {curve_stats['best_val_acc']:.4f}")
        print(f"  Final F1: {curve_stats['final_val_f1']:.4f}")
        print(f"  Actual epochs: {len(train_losses)}/{n_epochs}")
        print(f"  Learning curves saved to: {curve_path}")
        
        # 最良モデルをロード
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最終評価
        model.eval()
        val_preds = []          # GPU tensors
        val_labels_final = []   # CPU tensors
        val_probs = []          # GPU tensors
        val_raw_labels = []     # CPU tensors
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['image'].to(device, non_blocking=True)
                labels_batch = batch['label']
                raw_labels_batch = batch['raw_label']
                
                if use_intermediate_loss:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        class_probs, _ = outputs
                        val_probs.append(class_probs)
                        val_preds.append(class_probs.argmax(dim=1))
                    else:
                        val_probs.append(torch.softmax(outputs, dim=1))
                        val_preds.append(outputs.argmax(dim=1))
                else:
                    outputs = model(inputs)
                    val_probs.append(torch.softmax(outputs, dim=1))
                    val_preds.append(outputs.argmax(dim=1))
                
                val_labels_final.append(labels_batch)
                val_raw_labels.append(raw_labels_batch)
        
        # 一括でCPU転送・numpy変換
        val_preds = torch.cat(val_preds).cpu().numpy()
        val_probs = torch.cat(val_probs).cpu().numpy()
        val_labels_final = torch.cat(val_labels_final).numpy()
        val_raw_labels = torch.cat(val_raw_labels).numpy()
        
        # 座標情報を取得
        val_geo_locs = geo_locations[val_idx[:len(val_labels_final)]]
        val_pixel_locs = pixel_locations[val_idx[:len(val_labels_final)]]
        val_landslide_ids = landslide_ids[val_idx[:len(val_labels_final)]]
        
        # 予測結果をDataFrameに保存
        fold_predictions_df = pd.DataFrame({
            'fold': fold,
            'true_label': val_labels_final,
            'predicted_label': val_preds,
            'raw_label': val_raw_labels,
            'val_index': val_idx[:len(val_labels_final)],
            'geo_x': val_geo_locs[:, 0],
            'geo_y': val_geo_locs[:, 1],
            'pixel_x': val_pixel_locs[:, 0],
            'pixel_y': val_pixel_locs[:, 1],
            'landslide_id': val_landslide_ids,  # 画像ID（崩壊地ID）を追加
        })
        
        for i in range(n_classes):
            fold_predictions_df[f'prob_class_{i}'] = val_probs[:, i]
        
        all_predictions_df.append(fold_predictions_df)
        
        # 閾値の最適化
        if optimize_threshold and fold == 0:
            print(f"\n🔍 Optimizing thresholds...")
            best_thresholds, best_score, _, class_dist = find_optimal_thresholds(
                np.array(val_raw_labels), val_probs, n_classes=n_classes, metric='f1_macro'
            )
            best_thresholds_per_fold.append(best_thresholds)
            print(f"  Optimized thresholds: {[f'{t:.3f}' for t in best_thresholds]}")
            print(f"  Optimized F1 score: {best_score:.4f}")
        
        # メトリクスと混同行列の計算
        metrics, cm = calculate_metrics(val_labels_final, val_preds, val_probs)
        
        fold_result = {
            'fold': fold + 1,
            'metrics': metrics,
            'confusion_matrix': cm,
            'training_stats': curve_stats,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'actual_epochs': len(train_losses)
        }
        fold_results.append(fold_result)
        all_cms.append(cm)
        
        # 混同行列の可視化
        plot_confusion_matrix(
            cm, class_names_template,
            title=f'{model_name} - Fold {fold + 1} Final Confusion Matrix',
            save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}_final.png'),
            normalize=False,
            thresholds=fixed_thresholds
        )
        plot_confusion_matrix(
            cm, class_names_template,
            title=f'{model_name} - Fold {fold + 1} Final Normalized CM',
            save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}_final_normalized.png'),
            normalize=True,
            thresholds=fixed_thresholds
        )
        
        # ベストエポック時点の混同行列
        if best_val_preds is not None:
            best_cm = confusion_matrix(best_val_labels, best_val_preds)
            plot_confusion_matrix(
                best_cm, class_names_template,
                title=f'{model_name} - Fold {fold + 1} Best (Epoch {best_epoch}) CM',
                save_path=os.path.join(model_save_dir, f'confusion_matrix_fold_{fold+1}_best.png'),
                normalize=False,
                thresholds=fixed_thresholds
            )
        
        # foldごとの結果を表示
        fold_time = (datetime.now() - fold_start_time).total_seconds() / 60
        print(f"\n✅ Fold {fold + 1} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  F1 per class: {[f'{f:.3f}' for f in metrics['f1']]}")
        print(f"  Time taken: {fold_time:.1f} minutes")
        
        # 全体の最良モデルを追跡
        if best_model_state is not None and metrics['f1_macro'] > best_overall_f1:
            best_overall_f1 = metrics['f1_macro']
            best_overall_fold = fold + 1
            best_overall_model_state = {k: v.clone() for k, v in best_model_state.items()}
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
    
    # 最良モデルを保存
    if best_overall_model_state is not None:
        best_model_path = os.path.join(models_dir, 'best_model.pth')
        torch.save({
            'model_state_dict': best_overall_model_state,
            'model_name': model_name,
            'hyperparams': hyperparams,
            'n_classes': n_classes,
            'best_fold': best_overall_fold,
            'best_epoch': best_overall_epoch,
            'best_f1_macro': best_overall_f1,
            'metrics': best_overall_metrics,
            'thresholds': fixed_thresholds,
            'improvements': {
                'focal_loss': use_focal_loss,
                'mixup_cutmix': use_mixup_cutmix,
                'oversampling': use_oversampling,
                'early_stopping': True
            }
        }, best_model_path)
        print(f"🏆 Best model saved (Fold {best_overall_fold}, F1={best_overall_f1:.4f}): {best_model_path}")
    
    # CV結果の可視化
    plot_cv_scores_enhanced(fold_results, model_save_dir, model_name, class_names=class_names_template)
    
    # 全体の混同行列（平均）
    avg_cm = np.mean(all_cms, axis=0).astype(int)
    plot_confusion_matrix(
        avg_cm, class_names_template,
        title=f'{model_name} - Average Confusion Matrix',
        save_path=os.path.join(model_save_dir, 'confusion_matrix_average.png'),
        normalize=False,
        thresholds=fixed_thresholds
    )
    plot_confusion_matrix(
        avg_cm, class_names_template,
        title=f'{model_name} - Average Normalized CM',
        save_path=os.path.join(model_save_dir, 'confusion_matrix_average_normalized.png'),
        normalize=True,
        thresholds=fixed_thresholds
    )
    
    # サマリーを作成
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
        'best_model_path': os.path.join(models_dir, 'best_model.pth') if best_overall_model_state else None,
        'improvements_used': {
            'focal_loss': use_focal_loss,
            'focal_gamma': focal_gamma,
            'mixup_cutmix': use_mixup_cutmix,
            'mixup_alpha': mixup_alpha,
            'cutmix_alpha': cutmix_alpha,
            'oversampling': use_oversampling,
            'early_stopping_patience': early_stopping_patience
        }
    }
    
    # JSONファイルに保存
    with open(os.path.join(model_save_dir, 'cv_results.json'), 'w') as f:
        json_summary = summary.copy()
        for fold_result in json_summary['fold_results']:
            fold_result['confusion_matrix'] = fold_result['confusion_matrix'].tolist()
            for key, value in fold_result['metrics'].items():
                if isinstance(value, np.ndarray):
                    fold_result['metrics'][key] = value.tolist()
        json.dump(json_summary, f, indent=2)
    
    # 平均メトリクスの計算と表示
    print(f"\n{'='*70}")
    print(f"📊 {model_name} - Cross-Validation Summary (with Improvements)")
    print(f"{'='*70}")
    
    avg_metrics = {}
    for key in fold_results[0]['metrics'].keys():
        if key in ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            values = [fold['metrics'][key] for fold in fold_results]
            avg_metrics[key] = {'mean': np.mean(values), 'std': np.std(values)}
            print(f"{key}: {avg_metrics[key]['mean']:.4f} ± {avg_metrics[key]['std']:.4f}")
    
    # 実際に使用した改善機能を表示
    print(f"\n🚀 Improvements Applied:")
    print(f"  - Focal Loss: {'✅' if use_focal_loss else '❌'}")
    print(f"  - Mixup/CutMix: {'✅' if use_mixup_cutmix else '❌'}")
    print(f"  - Oversampling: {'✅' if use_oversampling else '❌'}")
    print(f"  - Early Stopping: ✅")
    print(f"  - CosineAnnealingWarmRestarts: ✅")
    
    # 難しいパッチの抽出・保存
    save_difficult_patches = config.get('save_difficult_patches', True) if config else True
    if save_difficult_patches:
        try:
            difficult_patch_results = extract_and_save_difficult_patches(
                images=images,
                predictions_df=all_predictions_df,
                save_dir=model_save_dir,
                n_classes=n_classes,
                patch_size=config.get('difficult_patch_size', 25) if config else 25,
                max_patches_per_type=config.get('max_difficult_patches', 100) if config else 100,
                confidence_threshold=config.get('difficult_confidence_threshold', 0.5) if config else 0.5
            )
            summary['difficult_patches'] = difficult_patch_results
        except Exception as e:
            print(f"⚠️ Failed to extract difficult patches: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n💾 All results saved to: {model_save_dir}")
    print(f"{'='*70}")
    
    return summary


# *******************************************************************
# オーバーサンプリング用のWeightedRandomSampler作成
# *******************************************************************
def create_weighted_sampler(labels: np.ndarray, thresholds: List[float], 
                            n_classes: int) -> WeightedRandomSampler:
    """
    クラス不均衡に対応するWeightedRandomSamplerを作成
    
    Parameters:
    -----------
    labels : np.ndarray
        連続値ラベル (0-1)
    thresholds : List[float]
        クラス閾値
    n_classes : int
        クラス数
    
    Returns:
    --------
    WeightedRandomSampler
        オーバーサンプリング用のサンプラー
    """
    # クラスラベルに変換
    class_labels = np.zeros(len(labels), dtype=int)
    for i, label in enumerate(labels):
        class_labels[i] = assign_class_from_thresholds(label, thresholds)
    
    # クラスごとのカウント
    class_counts = np.bincount(class_labels, minlength=n_classes)
    
    # サンプルごとの重み（逆頻度）
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[class_labels]
    
    # WeightedRandomSamplerを作成
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(labels),
        replacement=True
    )
    
    return sampler


# *******************************************************************
# 距離ペナルティ付き損失関数（Ordinal-Aware CrossEntropyLoss）
# *******************************************************************
class OrdinalCrossEntropyLoss(nn.Module):
    """
    順序を考慮した交差エントロピー損失関数
    
    中間クラスを飛ばした誤分類（例: Class 0 → Class 4）に対して
    より大きなペナルティを与える。
    
    Parameters:
    -----------
    n_classes : int
        クラス数
    class_weights : torch.Tensor, optional
        クラスごとの重み（不均衡データ対策）
    distance_weight : float
        距離ペナルティの強さ（0.0で無効、1.0で最大）
    label_smoothing : float
        ラベルスムージングの強さ
    """
    
    def __init__(self, n_classes: int, class_weights: torch.Tensor = None,
                 distance_weight: float = 0.5, label_smoothing: float = 0.0):
        super().__init__()
        self.n_classes = n_classes
        self.distance_weight = distance_weight
        self.label_smoothing = label_smoothing
        
        # 基本のCrossEntropyLoss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=label_smoothing,
            reduction='none'  # 個別のロスを取得
        )
        
        # 距離行列を作成（クラス間の距離）
        # distance_matrix[i][j] = |i - j| / (n_classes - 1)  (0〜1に正規化)
        distance_matrix = torch.zeros(n_classes, n_classes)
        for i in range(n_classes):
            for j in range(n_classes):
                distance_matrix[i][j] = abs(i - j) / (n_classes - 1)
        self.register_buffer('distance_matrix', distance_matrix)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        logits : torch.Tensor
            モデルの出力 (batch_size, n_classes)
        targets : torch.Tensor
            正解ラベル (batch_size,)
        
        Returns:
        --------
        torch.Tensor
            損失値（スカラー）
        """
        # 基本のCrossEntropyLoss
        ce_loss = self.ce_loss(logits, targets)
        
        if self.distance_weight <= 0:
            return ce_loss.mean()
        
        # 予測クラスを取得
        pred_classes = torch.argmax(logits, dim=1)
        
        # 距離ペナルティを計算
        # distance_penalty[i] = distance_matrix[targets[i], pred_classes[i]]
        distance_penalty = self.distance_matrix[targets, pred_classes]
        
        # 最終的な損失 = CE損失 * (1 + distance_weight * distance_penalty)
        # 距離が大きいほどペナルティが大きくなる
        weighted_loss = ce_loss * (1.0 + self.distance_weight * distance_penalty)
        
        return weighted_loss.mean()


def create_loss_function(n_classes: int, class_weights: torch.Tensor = None,
                        use_distance_penalty: bool = False, distance_weight: float = 0.5,
                        use_focal_loss: bool = False, focal_gamma: float = 2.0,
                        label_smoothing: float = 0.0, device: str = 'cuda') -> nn.Module:
    """
    損失関数を作成するファクトリー関数
    
    Parameters:
    -----------
    n_classes : int
        クラス数
    class_weights : torch.Tensor, optional
        クラスごとの重み
    use_distance_penalty : bool
        距離ペナルティを使用するか（デフォルト: False）
    distance_weight : float
        距離ペナルティの強さ（0.0〜1.0、デフォルト: 0.5）
    use_focal_loss : bool
        Focal Lossを使用するか（デフォルト: False）
    focal_gamma : float
        Focal Lossのgammaパラメータ（デフォルト: 2.0）
    label_smoothing : float
        ラベルスムージングの強さ
    device : str
        デバイス
    
    Returns:
    --------
    nn.Module
        損失関数
    """
    if use_focal_loss:
        print(f"  🎯 Using FocalLoss (gamma={focal_gamma})")
        loss_fn = FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        return loss_fn
    elif use_distance_penalty:
        print(f"  📏 Using OrdinalCrossEntropyLoss (distance_weight={distance_weight})")
        loss_fn = OrdinalCrossEntropyLoss(
            n_classes=n_classes,
            class_weights=class_weights,
            distance_weight=distance_weight,
            label_smoothing=label_smoothing
        )
        return loss_fn.to(device)
    else:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


# *******************************************************************
# 予測結果のLegacy Ratio空間可視化関数
# *******************************************************************
def visualize_prediction_legacy_overlay(
    predictions_df: pd.DataFrame,
    landslide_gdf: gpd.GeoDataFrame,
    save_dir: Path,
    model_name: str,
    n_classes: int = 5,
    fixed_thresholds: Optional[List[float]] = None,
    crs: str = 'EPSG:6675',
    dataset_name: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 9),
) -> Dict:
    """
    モデルの予測結果をLegacy ratio overlayとして空間的に可視化する
    
    添付画像のように、予測されたクラスを連続的なlegacy ratio（0〜1）に変換して
    plasmaカラーマップで描画し、崩壊地ポリゴンの境界をシアン色で重ねる。
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        予測結果のDataFrame（x, y, pred_label等を含む）
    landslide_gdf : gpd.GeoDataFrame
        崩壊地ポリゴンのGeoDataFrame（境界線描画用）
    save_dir : Path
        出力先ディレクトリ
    model_name : str
        モデル名（ファイル名・タイトルに使用）
    n_classes : int
        クラス数（デフォルト: 5）
    fixed_thresholds : List[float], optional
        クラス閾値リスト（例: [0.0, 0.25, 0.5, 0.75]）
    crs : str
        座標参照系
    dataset_name : str, optional
        データセット名（タイトル用）
    figsize : Tuple[int, int]
        出力図のサイズ
    
    Returns:
    --------
    Dict : 出力ファイルパス等の情報
    """
    save_dir = Path(save_dir)
    overlay_dir = save_dir / 'legacy_prediction_overlay'
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🗺️  Legacy Prediction Overlay for {model_name}")
    print(f"{'='*70}")
    
    # 閾値からクラス中央値を計算（クラス→legacy ratio変換用）
    if fixed_thresholds is None:
        fixed_thresholds = [0.0, 0.25, 0.5, 0.75]
    
    # 各クラスの代表legacy ratio値を計算
    # Class 0: [0, 0.0) → 0.0 (完全植生回復)
    # Class 1: [0.0, 0.25) → 0.125
    # Class 2: [0.25, 0.5) → 0.375
    # Class 3: [0.5, 0.75) → 0.625
    # Class 4: [0.75, 1.0] → 0.875 (崩壊地のまま)
    class_to_ratio = {}
    bounds = [0.0] + fixed_thresholds + [1.0]
    for i in range(n_classes):
        class_to_ratio[i] = (bounds[i] + bounds[i+1]) / 2
    
    # 座標とラベルの取得
    if 'x' in predictions_df.columns:
        x_coords = np.asarray(predictions_df['x'].values).copy()
        y_coords = np.asarray(predictions_df['y'].values).copy()
    elif 'geo_x' in predictions_df.columns:
        x_coords = np.asarray(predictions_df['geo_x'].values).copy()
        y_coords = np.asarray(predictions_df['geo_y'].values).copy()
    else:
        print(f"  ⚠️ 座標列が見つかりません")
        return {'error': 'No coordinate columns found'}
    
    # 予測ラベルの取得
    if 'pred_label' in predictions_df.columns:
        pred_labels = np.asarray(predictions_df['pred_label'].values).copy()
    elif 'predicted_label' in predictions_df.columns:
        pred_labels = np.asarray(predictions_df['predicted_label'].values).copy()
    elif 'predicted_class' in predictions_df.columns:
        pred_labels = np.asarray(predictions_df['predicted_class'].values).copy()
    else:
        print(f"  ⚠️ 予測ラベル列が見つかりません")
        return {'error': 'No prediction label column found'}
    
    # 正解ラベルの取得（比較用）
    if 'true_label' in predictions_df.columns:
        true_labels = np.asarray(predictions_df['true_label'].values).copy()
    elif 'true_class' in predictions_df.columns:
        true_labels = np.asarray(predictions_df['true_class'].values).copy()
    else:
        true_labels = None
    
    # 予測クラスをlegacy ratioに変換
    pred_ratios = np.array([class_to_ratio[int(p)] for p in pred_labels])
    
    # 正解ラベルもratioに変換（あれば）
    if true_labels is not None:
        true_ratios = np.array([class_to_ratio[int(t)] for t in true_labels])
    
    print(f"  📊 Total samples: {len(predictions_df):,}")
    print(f"  📍 Coordinate range: X[{x_coords.min():.1f}, {x_coords.max():.1f}], Y[{y_coords.min():.1f}, {y_coords.max():.1f}]")
    
    results = {
        'total_samples': len(predictions_df),
        'output_files': [],
    }
    
    # データセット別に処理
    if 'dataset_name' in predictions_df.columns:
        datasets = predictions_df['dataset_name'].unique()
    else:
        datasets = [dataset_name if dataset_name else 'all']
    
    for ds_name in datasets:
        if 'dataset_name' in predictions_df.columns:
            mask = predictions_df['dataset_name'] == ds_name
            ds_x = x_coords[mask]
            ds_y = y_coords[mask]
            ds_pred_ratios = pred_ratios[mask]
            ds_true_ratios = true_ratios[mask] if true_labels is not None else None
        else:
            ds_x = x_coords
            ds_y = y_coords
            ds_pred_ratios = pred_ratios
            ds_true_ratios = true_ratios if true_labels is not None else None
        
        if len(ds_x) == 0:
            continue
        
        # 範囲を計算
        x_min, x_max = ds_x.min(), ds_x.max()
        y_min, y_max = ds_y.min(), ds_y.max()
        extent = (x_min, x_max, y_min, y_max)
        
        # ============================================================
        # 1. 予測結果のLegacy Ratio Overlay
        # ============================================================
        fig, ax = plt.subplots(figsize=figsize, facecolor='black')
        ax.set_facecolor('black')
        
        # 散布図でプロット（plasmaカラーマップ）
        scatter = ax.scatter(
            ds_x, ds_y,
            c=ds_pred_ratios,
            cmap='plasma',
            s=2,  # マーカーサイズ
            alpha=0.8,
            vmin=0.0,
            vmax=1.0,
        )
        
        # カラーバー追加
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.02)
        cbar.set_label('Annotation ratio (0-1)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # 崩壊地ポリゴンの境界を重ねる
        if landslide_gdf is not None and not landslide_gdf.empty:
            # 表示範囲内のポリゴンのみ抽出
            try:
                view_box = box(x_min, y_min, x_max, y_max)
                clipped_gdf = landslide_gdf[landslide_gdf.geometry.intersects(view_box)]
                if not clipped_gdf.empty:
                    clipped_gdf.boundary.plot(ax=ax, linewidth=1.0, edgecolor='cyan')
            except Exception as e:
                print(f"    ⚠️ ポリゴン描画エラー: {e}")
        
        # 軸設定
        ax.set_xlabel('Easting', color='white')
        ax.set_ylabel('Northing', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_aspect('equal')
        ax.set_title(f"{ds_name} | {model_name} Predicted Legacy ratio overlay", color='white')
        
        # 保存
        output_path = overlay_dir / f"{ds_name}_{model_name}_prediction_overlay.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"    💾 Saved: {output_path.name}")
        results['output_files'].append(str(output_path))
        
        # ============================================================
        # 2. 正解ラベルのLegacy Ratio Overlay（比較用）
        # ============================================================
        if ds_true_ratios is not None:
            fig, ax = plt.subplots(figsize=figsize, facecolor='black')
            ax.set_facecolor('black')
            
            scatter = ax.scatter(
                ds_x, ds_y,
                c=ds_true_ratios,
                cmap='plasma',
                s=2,
                alpha=0.8,
                vmin=0.0,
                vmax=1.0,
            )
            
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.02)
            cbar.set_label('Annotation ratio (0-1)', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            if landslide_gdf is not None and not landslide_gdf.empty:
                try:
                    view_box = box(x_min, y_min, x_max, y_max)
                    clipped_gdf = landslide_gdf[landslide_gdf.geometry.intersects(view_box)]
                    if not clipped_gdf.empty:
                        clipped_gdf.boundary.plot(ax=ax, linewidth=1.0, edgecolor='cyan')
                except Exception as e:
                    pass
            
            ax.set_xlabel('Easting', color='white')
            ax.set_ylabel('Northing', color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_aspect('equal')
            ax.set_title(f"{ds_name} | Ground Truth Legacy ratio overlay", color='white')
            
            output_path = overlay_dir / f"{ds_name}_ground_truth_overlay.png"
            fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            print(f"    💾 Saved: {output_path.name}")
            results['output_files'].append(str(output_path))
        
        # ============================================================
        # 3. 予測と正解の比較図（2パネル）
        # ============================================================
        if ds_true_ratios is not None:
            fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]), facecolor='black')
            
            for ax_idx, (ax, ratios, title) in enumerate([
                (axes[0], ds_true_ratios, 'Ground Truth'),
                (axes[1], ds_pred_ratios, f'{model_name} Prediction')
            ]):
                ax.set_facecolor('black')
                
                scatter = ax.scatter(
                    ds_x, ds_y,
                    c=ratios,
                    cmap='plasma',
                    s=1,
                    alpha=0.8,
                    vmin=0.0,
                    vmax=1.0,
                )
                
                if landslide_gdf is not None and not landslide_gdf.empty:
                    try:
                        view_box = box(x_min, y_min, x_max, y_max)
                        clipped_gdf = landslide_gdf[landslide_gdf.geometry.intersects(view_box)]
                        if not clipped_gdf.empty:
                            clipped_gdf.boundary.plot(ax=ax, linewidth=0.8, edgecolor='cyan')
                    except:
                        pass
                
                ax.set_xlabel('Easting', color='white')
                ax.set_ylabel('Northing', color='white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.set_aspect('equal')
                ax.set_title(f"{ds_name} | {title}", color='white')
                
                # 最後のパネルにカラーバー
                if ax_idx == 1:
                    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
                    cbar.set_label('Annotation ratio (0-1)', color='white')
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            output_path = overlay_dir / f"{ds_name}_{model_name}_comparison.png"
            fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            print(f"    💾 Saved: {output_path.name}")
            results['output_files'].append(str(output_path))
    
    # ============================================================
    # 4. 全データセット統合ビュー
    # ============================================================
    if len(datasets) > 1:
        fig, ax = plt.subplots(figsize=(figsize[0]*1.2, figsize[1]*1.2), facecolor='black')
        ax.set_facecolor('black')
        
        scatter = ax.scatter(
            x_coords, y_coords,
            c=pred_ratios,
            cmap='plasma',
            s=1,
            alpha=0.8,
            vmin=0.0,
            vmax=1.0,
        )
        
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.02)
        cbar.set_label('Annotation ratio (0-1)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        if landslide_gdf is not None and not landslide_gdf.empty:
            try:
                landslide_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='cyan')
            except:
                pass
        
        ax.set_xlabel('Easting', color='white')
        ax.set_ylabel('Northing', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_aspect('equal')
        ax.set_title(f"All Datasets | {model_name} Predicted Legacy ratio overlay", color='white')
        
        output_path = overlay_dir / f"all_datasets_{model_name}_prediction_overlay.png"
        fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"    💾 Saved: {output_path.name}")
        results['output_files'].append(str(output_path))
    
    # ============================================================
    # 5. 画像ID（landslide_id）ごとの個別出力
    # ============================================================
    if 'landslide_id' in predictions_df.columns:
        per_image_dir = overlay_dir / 'per_image'
        per_image_dir.mkdir(parents=True, exist_ok=True)
        
        unique_ids = predictions_df['landslide_id'].unique()
        print(f"\n📷 Generating per-image overlays for {len(unique_ids)} landslide regions...")
        
        for img_id in tqdm(unique_ids, desc="Per-image overlays"):
            img_mask = predictions_df['landslide_id'] == img_id
            img_df = predictions_df[img_mask]
            
            if len(img_df) < 5:  # サンプルが少なすぎる場合はスキップ
                continue
            
            # 座標とratioの取得
            if 'x' in img_df.columns:
                img_x = np.asarray(img_df['x'].values)
                img_y = np.asarray(img_df['y'].values)
            elif 'geo_x' in img_df.columns:
                img_x = np.asarray(img_df['geo_x'].values)
                img_y = np.asarray(img_df['geo_y'].values)
            else:
                continue
            
            # 予測ラベルの取得
            if 'pred_label' in img_df.columns:
                img_pred_labels = np.asarray(img_df['pred_label'].values)
            elif 'predicted_label' in img_df.columns:
                img_pred_labels = np.asarray(img_df['predicted_label'].values)
            elif 'predicted_class' in img_df.columns:
                img_pred_labels = np.asarray(img_df['predicted_class'].values)
            else:
                continue
            
            # 予測クラスをlegacy ratioに変換
            img_pred_ratios = np.array([class_to_ratio[int(p)] for p in img_pred_labels])
            
            # 正解ラベルの取得（あれば）
            img_true_ratios = None
            if 'true_label' in img_df.columns:
                img_true_labels = np.asarray(img_df['true_label'].values)
                img_true_ratios = np.array([class_to_ratio[int(t)] for t in img_true_labels])
            elif 'true_class' in img_df.columns:
                img_true_labels = np.asarray(img_df['true_class'].values)
                img_true_ratios = np.array([class_to_ratio[int(t)] for t in img_true_labels])
            
            # 範囲を計算
            x_min, x_max = img_x.min(), img_x.max()
            y_min, y_max = img_y.min(), img_y.max()
            
            # 予測結果のプロット
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
            ax.set_facecolor('black')
            
            scatter = ax.scatter(
                img_x, img_y,
                c=img_pred_ratios,
                cmap='plasma',
                s=8,  # 個別画像では少し大きめのマーカー
                alpha=0.9,
                vmin=0.0,
                vmax=1.0,
            )
            
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.02)
            cbar.set_label('Predicted ratio (0-1)', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            # 対応する崩壊地ポリゴンの境界を描画
            if landslide_gdf is not None and not landslide_gdf.empty:
                try:
                    view_box = box(x_min - 10, y_min - 10, x_max + 10, y_max + 10)
                    clipped_gdf = landslide_gdf[landslide_gdf.geometry.intersects(view_box)]
                    if not clipped_gdf.empty:
                        clipped_gdf.boundary.plot(ax=ax, linewidth=1.5, edgecolor='cyan')
                except Exception:
                    pass
            
            ax.set_xlabel('Easting', color='white')
            ax.set_ylabel('Northing', color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_aspect('equal')
            ax.set_title(f"ID: {img_id} | {model_name} Prediction (n={len(img_df)})", color='white')
            
            # ファイル名にはIDを使用（安全な文字列に変換）
            safe_id = str(img_id).replace('/', '_').replace('\\', '_').replace(' ', '_')
            output_path = per_image_dir / f"{safe_id}_{model_name}_prediction.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            results['output_files'].append(str(output_path))
            
            # 比較図（Ground Truth vs Prediction）も出力
            if img_true_ratios is not None:
                fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='black')
                
                for ax_idx, (ax, ratios, title) in enumerate([
                    (axes[0], img_true_ratios, 'Ground Truth'),
                    (axes[1], img_pred_ratios, f'{model_name} Prediction')
                ]):
                    ax.set_facecolor('black')
                    
                    scatter = ax.scatter(
                        img_x, img_y,
                        c=ratios,
                        cmap='plasma',
                        s=6,
                        alpha=0.9,
                        vmin=0.0,
                        vmax=1.0,
                    )
                    
                    if landslide_gdf is not None and not landslide_gdf.empty:
                        try:
                            view_box = box(x_min - 10, y_min - 10, x_max + 10, y_max + 10)
                            clipped_gdf = landslide_gdf[landslide_gdf.geometry.intersects(view_box)]
                            if not clipped_gdf.empty:
                                clipped_gdf.boundary.plot(ax=ax, linewidth=1.0, edgecolor='cyan')
                        except:
                            pass
                    
                    ax.set_xlabel('Easting', color='white')
                    ax.set_ylabel('Northing', color='white')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.set_aspect('equal')
                    ax.set_title(f"ID: {img_id} | {title}", color='white')
                    
                    if ax_idx == 1:
                        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
                        cbar.set_label('Ratio (0-1)', color='white')
                        cbar.ax.yaxis.set_tick_params(color='white')
                        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                
                output_path = per_image_dir / f"{safe_id}_{model_name}_comparison.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
                plt.close(fig)
                results['output_files'].append(str(output_path))
        
        print(f"    📁 Per-image overlays saved to: {per_image_dir}")
    
    print(f"\n{'='*70}")
    print(f"✅ Legacy prediction overlay complete!")
    print(f"   Output directory: {overlay_dir}")
    print(f"{'='*70}")
    
    return results


# *******************************************************************
# 誤分類領域の可視化関数（元画像オーバーレイ + QGIS用データ出力）
# *******************************************************************
def visualize_misclassification_on_image(
    predictions_df: pd.DataFrame,
    rgb_images: Dict[str, Tuple[np.ndarray, Tuple[float, float, float, float]]],
    save_dir: Path,
    model_name: str,
    class_names: Optional[List[str]] = None,
    crs: str = 'EPSG:6675',
    output_gpkg: bool = True,
    output_geotiff: bool = True,
    grid_resolution: int = 100
) -> Dict:
    """
    元画像上に誤分類領域をオーバーレイして可視化し、QGIS用データも出力する
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        予測結果のDataFrame（x, y, true_label, pred_label, dataset_name等を含む）
    rgb_images : Dict[str, Tuple[np.ndarray, Tuple[float, float, float, float]]]
        データセット名をキーとした元画像とその範囲(extent)の辞書
        {dataset_name: (rgb_array, (xmin, xmax, ymin, ymax))}
    save_dir : Path
        出力先ディレクトリ
    model_name : str
        モデル名（ファイル名に使用）
    class_names : List[str], optional
        クラス名リスト
    crs : str
        座標参照系
    output_gpkg : bool
        GeoPackage出力を行うか
    output_geotiff : bool
        GeoTIFF出力を行うか
    grid_resolution : int
        ヒートマップのグリッド解像度
    
    Returns:
    --------
    Dict : 解析結果のサマリー
    """
    save_dir = Path(save_dir)
    misclass_dir = save_dir / 'misclassification_overlay'
    misclass_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🗺️  Misclassification Overlay Visualization for {model_name}")
    print(f"{'='*70}")
    
    # カラーパレット
    palette = {
        'correct': '#2ecc71',      # 緑: 正解
        'wrong': '#e74c3c',        # 赤: 誤分類
        'boundary': '#f39c12',     # オレンジ: 境界
        'dark': '#1a1a2e',
        'cream': '#fffeec',
    }
    
    # 座標とラベルの取得（NumPy 2.0対応: 明示的にコピー）
    if 'x' in predictions_df.columns:
        x_coords = np.asarray(predictions_df['x'].values).copy()
        y_coords = np.asarray(predictions_df['y'].values).copy()
    elif 'geo_x' in predictions_df.columns:
        x_coords = np.asarray(predictions_df['geo_x'].values).copy()
        y_coords = np.asarray(predictions_df['geo_y'].values).copy()
    else:
        print("  ⚠️ 座標列が見つかりません")
        return {}
    
    y_true = np.asarray(predictions_df['true_label'].values if 'true_label' in predictions_df.columns else predictions_df['true_class'].values).copy()
    y_pred = np.asarray(predictions_df['pred_label'].values if 'pred_label' in predictions_df.columns 
              else predictions_df['predicted_label'].values if 'predicted_label' in predictions_df.columns
              else predictions_df['predicted_class'].values).copy()
    
    is_correct = np.asarray(y_true == y_pred)
    
    n_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    print(f"  📊 Total samples: {len(predictions_df):,}")
    print(f"  ✅ Correct: {np.sum(is_correct):,} ({100*np.mean(is_correct):.1f}%)")
    print(f"  ❌ Misclassified: {np.sum(~is_correct):,} ({100*np.mean(~is_correct):.1f}%)")
    
    results = {
        'total_samples': len(predictions_df),
        'correct_count': int(np.sum(is_correct)),
        'misclassified_count': int(np.sum(~is_correct)),
        'accuracy': float(np.mean(is_correct)),
        'output_files': [],
    }
    
    # ============================================================
    # 1. GeoPackage出力（QGIS用）
    # ============================================================
    if output_gpkg:
        print(f"\n  📦 Creating GeoPackage for QGIS...")
        
        # まずDataFrameを作成してからGeoDataFrameに変換
        # 🔧 修正: NumPy型をPython標準型に明示的に変換（QGIS互換性のため）
        gdf_data = pd.DataFrame({
            'true_label': [int(t) for t in y_true],  # int64 -> int
            'pred_label': [int(p) for p in y_pred],  # int64 -> int
            'true_class_name': [class_names[int(t)] for t in y_true],
            'pred_class_name': [class_names[int(p)] for p in y_pred],
            'is_correct': [int(c) for c in is_correct],  # bool -> int (0/1)
            'error_type': ['correct' if c else f'{class_names[int(t)]}->{class_names[int(p)]}' 
                          for c, t, p in zip(is_correct, y_true, y_pred)],
            'x': [float(x) for x in x_coords],  # float64 -> float
            'y': [float(y) for y in y_coords],  # float64 -> float
        })
        
        # 確率情報があれば追加（Python標準型に変換）
        prob_cols = [col for col in predictions_df.columns if col.startswith('prob_class_')]
        if prob_cols:
            for col in prob_cols:
                gdf_data[col] = [float(v) for v in predictions_df[col].values]  # float64 -> float
            # 最大確率（信頼度）
            probs = predictions_df[prob_cols].values
            gdf_data['confidence'] = [float(v) for v in np.max(probs, axis=1)]  # float64 -> float
            # エントロピー
            epsilon = 1e-10
            entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
            gdf_data['entropy'] = [float(v) for v in entropy]  # float64 -> float
        
        # dataset_nameがあれば追加（文字列はそのまま）
        if 'dataset_name' in predictions_df.columns:
            gdf_data['dataset_name'] = [str(v) for v in predictions_df['dataset_name'].values]
        
        # DataFrameからGeoDataFrameに変換（points_from_xyを使用）
        gdf = gpd.GeoDataFrame(
            gdf_data, 
            geometry=gpd.points_from_xy(gdf_data['x'], gdf_data['y']),
            crs=crs
        )
        # x, y列は削除（geometryに含まれるため）
        gdf = gdf.drop(columns=['x', 'y'])
        
        # 全ポイントを保存
        gpkg_path = misclass_dir / f'{model_name}_predictions.gpkg'
        gdf.to_file(gpkg_path, driver='GPKG', layer='all_predictions')
        
        # 誤分類ポイントのみを別レイヤーとして保存
        gdf_errors = gdf[gdf['is_correct'] == 0].copy()
        if len(gdf_errors) > 0:
            gdf_errors.to_file(gpkg_path, driver='GPKG', layer='misclassifications')
        
        # クラス別レイヤー
        for c in range(n_classes):
            gdf_class = gdf[gdf['true_label'] == c].copy()
            if len(gdf_class) > 0:
                gdf_class.to_file(gpkg_path, driver='GPKG', layer=f'class_{c}_{class_names[c]}')
        
        print(f"    💾 Saved: {gpkg_path}")
        results['output_files'].append(str(gpkg_path))
        
        # ============================================================
        # 空間的な正答率グリッドをポリゴンとして出力
        # ============================================================
        print(f"  📐 Creating accuracy grid polygons...")
        
        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())
        
        x_edges = np.linspace(x_min, x_max, grid_resolution + 1)
        y_edges = np.linspace(y_min, y_max, grid_resolution + 1)
        
        grid_polygons = []
        grid_data = []
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                # セル内のサンプル
                mask = (
                    (x_coords >= x_edges[j]) & (x_coords < x_edges[j+1]) &
                    (y_coords >= y_edges[i]) & (y_coords < y_edges[i+1])
                )
                count = int(np.sum(mask))  # int64 -> int
                
                if count > 0:
                    cell_correct = is_correct[mask]
                    accuracy = float(np.mean(cell_correct))  # float64 -> float
                    
                    # ポリゴン作成
                    poly = box(float(x_edges[j]), float(y_edges[i]), 
                              float(x_edges[j+1]), float(y_edges[i+1]))
                    grid_polygons.append(poly)
                    # 🔧 修正: すべてPython標準型に変換（QGIS互換性のため）
                    grid_data.append({
                        'accuracy': accuracy,
                        'error_rate': float(1 - accuracy),
                        'sample_count': count,
                        'correct_count': int(np.sum(cell_correct)),
                        'error_count': int(np.sum(~cell_correct)),
                    })
        
        if grid_polygons:
            # DataFrameを作成してからGeoDataFrameに変換
            grid_df = pd.DataFrame(grid_data)
            gdf_grid = gpd.GeoDataFrame(grid_df, geometry=grid_polygons, crs=crs)
            gdf_grid.to_file(gpkg_path, driver='GPKG', layer='accuracy_grid')
            print(f"    💾 Added accuracy_grid layer ({len(grid_polygons)} cells)")
    
    # ============================================================
    # 2. 元画像上への誤分類オーバーレイ
    # ============================================================
    print(f"\n  🎨 Creating overlay images...")
    
    # データセット別に処理
    if 'dataset_name' in predictions_df.columns:
        datasets = predictions_df['dataset_name'].unique()
    else:
        datasets = ['all']
    
    for dataset_name in datasets:
        if dataset_name == 'all':
            df_subset = predictions_df
            x_sub = x_coords
            y_sub = y_coords
            correct_sub = is_correct
            true_sub = y_true
            pred_sub = y_pred
        else:
            mask = predictions_df['dataset_name'] == dataset_name
            df_subset = predictions_df[mask]
            x_sub = x_coords[mask]
            y_sub = y_coords[mask]
            correct_sub = is_correct[mask]
            true_sub = y_true[mask]
            pred_sub = y_pred[mask]
        
        if len(df_subset) == 0:
            continue
        
        print(f"\n    📍 Processing {dataset_name}...")
        
        # 範囲の計算
        x_min_ds, x_max_ds = x_sub.min(), x_sub.max()
        y_min_ds, y_max_ds = y_sub.min(), y_sub.max()
        extent_ds = (x_min_ds, x_max_ds, y_min_ds, y_max_ds)
        
        # Figure作成
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.patch.set_facecolor('white')
        
        # (0,0) 全ポイント（正解/誤分類）
        ax1 = axes[0, 0]
        ax1.scatter(x_sub[correct_sub], y_sub[correct_sub], 
                   c=palette['correct'], s=8, alpha=0.5, label='Correct')
        ax1.scatter(x_sub[~correct_sub], y_sub[~correct_sub], 
                   c=palette['wrong'], s=20, alpha=0.8, marker='x', label='Misclassified')
        ax1.set_xlim(x_min_ds - 10, x_max_ds + 10)
        ax1.set_ylim(y_min_ds - 10, y_max_ds + 10)
        ax1.set_aspect('equal')
        ax1.set_title(f'{dataset_name}\nPrediction Results (n={len(df_subset):,})', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        
        # (0,1) 正答率ヒートマップ
        ax2 = axes[0, 1]
        
        # グリッド計算
        grid_res_local = min(50, int(np.sqrt(len(df_subset) / 10)))
        grid_res_local = max(10, grid_res_local)
        
        x_edges_local = np.linspace(x_min_ds, x_max_ds, grid_res_local + 1)
        y_edges_local = np.linspace(y_min_ds, y_max_ds, grid_res_local + 1)
        accuracy_grid_local = np.full((grid_res_local, grid_res_local), np.nan)
        
        for i in range(grid_res_local):
            for j in range(grid_res_local):
                mask_cell = (
                    (x_sub >= x_edges_local[j]) & (x_sub < x_edges_local[j+1]) &
                    (y_sub >= y_edges_local[i]) & (y_sub < y_edges_local[i+1])
                )
                if np.sum(mask_cell) > 0:
                    accuracy_grid_local[i, j] = np.mean(correct_sub[mask_cell])
        
        im2 = ax2.imshow(accuracy_grid_local, extent=[x_min_ds, x_max_ds, y_min_ds, y_max_ds],
                        origin='lower', cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax2.set_title(f'{dataset_name}\nSpatial Accuracy Heatmap', fontsize=12)
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        plt.colorbar(im2, ax=ax2, label='Accuracy', shrink=0.8)
        
        # (1,0) クラス別誤分類
        ax3 = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for c in range(n_classes):
            class_mask = (true_sub == c) & (~correct_sub)
            if np.sum(class_mask) > 0:
                ax3.scatter(x_sub[class_mask], y_sub[class_mask], 
                           c=[colors[c]], s=30, alpha=0.7, label=f'{class_names[c]} errors', marker='x')
        
        ax3.set_xlim(x_min_ds - 10, x_max_ds + 10)
        ax3.set_ylim(y_min_ds - 10, y_max_ds + 10)
        ax3.set_aspect('equal')
        ax3.set_title(f'{dataset_name}\nMisclassifications by True Class', fontsize=12)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        
        # (1,1) 誤分類パターン別
        ax4 = axes[1, 1]
        
        # 誤分類パターンを集計
        error_patterns = {}
        for t, p, x, y in zip(true_sub[~correct_sub], pred_sub[~correct_sub], 
                              x_sub[~correct_sub], y_sub[~correct_sub]):
            key = (int(t), int(p))
            if key not in error_patterns:
                error_patterns[key] = {'x': [], 'y': []}
            error_patterns[key]['x'].append(x)
            error_patterns[key]['y'].append(y)
        
        # 上位パターンをプロット
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: len(x[1]['x']), reverse=True)[:6]
        pattern_colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_patterns)))
        
        for idx, ((t, p), coords) in enumerate(sorted_patterns):
            label = f'{class_names[t]}→{class_names[p]} (n={len(coords["x"])})'
            ax4.scatter(coords['x'], coords['y'], c=[pattern_colors[idx]], 
                       s=25, alpha=0.7, label=label, marker='o')
        
        ax4.set_xlim(x_min_ds - 10, x_max_ds + 10)
        ax4.set_ylim(y_min_ds - 10, y_max_ds + 10)
        ax4.set_aspect('equal')
        ax4.set_title(f'{dataset_name}\nTop Confusion Patterns', fontsize=12)
        ax4.legend(loc='upper right', fontsize=7)
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        
        plt.suptitle(f'{model_name} - Misclassification Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = misclass_dir / f'{model_name}_{dataset_name}_misclassification_overlay.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"      💾 Saved: {output_path.name}")
        results['output_files'].append(str(output_path))
    
    # ============================================================
    # 3. 誤分類統計サマリー
    # ============================================================
    print(f"\n  📊 Creating summary statistics...")
    
    # 誤分類パターンの統計
    confusion_summary = []
    for (t, p), coords in error_patterns.items():
        confusion_summary.append({
            'true_class': class_names[t],
            'pred_class': class_names[p],
            'count': len(coords['x']),
            'pattern': f'{class_names[t]} → {class_names[p]}'
        })
    
    confusion_summary_df = pd.DataFrame(confusion_summary)
    confusion_summary_df = confusion_summary_df.sort_values('count', ascending=False)
    summary_csv_path = misclass_dir / f'{model_name}_confusion_patterns.csv'
    confusion_summary_df.to_csv(summary_csv_path, index=False)
    print(f"    💾 Saved: {summary_csv_path.name}")
    results['confusion_patterns'] = confusion_summary
    
    # ============================================================
    # 4. GeoTIFF出力（ラスター形式）
    # ============================================================
    if output_geotiff:
        print(f"\n  🗺️  Creating GeoTIFF rasters...")
        
        try:
            # 正答率ラスター
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            x_edges = np.linspace(x_min, x_max, grid_resolution + 1)
            y_edges = np.linspace(y_min, y_max, grid_resolution + 1)
            
            accuracy_raster = np.full((grid_resolution, grid_resolution), np.nan, dtype=np.float32)
            error_count_raster = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
            
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    mask = (
                        (x_coords >= x_edges[j]) & (x_coords < x_edges[j+1]) &
                        (y_coords >= y_edges[i]) & (y_coords < y_edges[i+1])
                    )
                    if np.sum(mask) > 0:
                        accuracy_raster[grid_resolution - 1 - i, j] = np.mean(is_correct[mask])
                        error_count_raster[grid_resolution - 1 - i, j] = np.sum(~is_correct[mask])
            
            # アフィン変換の計算
            pixel_width = (x_max - x_min) / grid_resolution
            pixel_height = (y_max - y_min) / grid_resolution
            transform = Affine.translation(x_min, y_max) * Affine.scale(pixel_width, -pixel_height)
            
            # 正答率GeoTIFF
            accuracy_tif_path = misclass_dir / f'{model_name}_accuracy_map.tif'
            with rasterio.open(
                accuracy_tif_path, 'w',
                driver='GTiff',
                height=grid_resolution,
                width=grid_resolution,
                count=1,
                dtype='float32',
                crs=crs,
                transform=transform,
                nodata=np.nan
            ) as dst:
                dst.write(accuracy_raster, 1)
            print(f"    💾 Saved: {accuracy_tif_path.name}")
            results['output_files'].append(str(accuracy_tif_path))
            
            # 誤分類数GeoTIFF
            error_tif_path = misclass_dir / f'{model_name}_error_count_map.tif'
            with rasterio.open(
                error_tif_path, 'w',
                driver='GTiff',
                height=grid_resolution,
                width=grid_resolution,
                count=1,
                dtype='float32',
                crs=crs,
                transform=transform,
                nodata=0
            ) as dst:
                dst.write(error_count_raster, 1)
            print(f"    💾 Saved: {error_tif_path.name}")
            results['output_files'].append(str(error_tif_path))
            
        except Exception as e:
            print(f"    ⚠️ GeoTIFF creation failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ Misclassification overlay analysis complete!")
    print(f"   Output directory: {misclass_dir}")
    print(f"{'='*70}")
    
    return results


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
    """クラス分類の閾値を最適化するクラス（クラス1の閾値は0で固定）"""
    
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.thresholds = None
        self.best_thresholds = None
        self.best_score = -np.inf
        self.optimization_metric = None
    
    def label_to_class(self, labels, thresholds):
        """連続値ラベルをクラスラベルに変換"""
        classes = np.zeros(len(labels), dtype=int)
        for i, label in enumerate(labels):
            classes[i] = assign_class_from_thresholds(label, thresholds)
        return classes
    
    def optimize_thresholds(self, y_true_continuous, y_pred_probs, metric='f1_macro'):
        """
        閾値を最適化（最初の閾値は0で固定）
        y_true_continuous: 連続値の真のラベル（0.0～1.0）
        y_pred_probs: モデルの予測確率 (n_samples, n_classes)
        metric: 最適化する指標 ('f1_macro', 'accuracy', 'balanced_accuracy')
        
        Note:
            5クラス分類の場合、閾値は [0.0, t2, t3, t4] の形式
            - 最初の閾値は0.0で固定（クラス0は値が0のサンプルのみ）
            - 残り3つの閾値 (t2, t3, t4) を最適化
        """
        from scipy.optimize import differential_evolution
        
        self.optimization_metric = metric
        
        # 最適化する閾値の数（最初の0は固定なので n_classes - 2）
        n_optimizable = self.n_classes - 2
        
        def objective(thresholds_to_optimize):
            # 最初の閾値を0で固定し、残りを昇順ソート
            full_thresholds = [0.0] + sorted(thresholds_to_optimize)
            
            # 連続値ラベルをクラスラベルに変換
            y_true_classes = self.label_to_class(y_true_continuous, full_thresholds)
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
        
        # 最適化する閾値の範囲（0より大きく1以下）
        bounds = [(0.01, 1.0) for _ in range(n_optimizable)]
        
        # 差分進化法で最適化
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            seed=42,
            disp=False
        )
        
        # 最終的な閾値：[0.0, 最適化された閾値...]
        self.best_thresholds = [0.0] + sorted(result.x.tolist())
        self.best_score = -result.fun
        
        return self.best_thresholds, self.best_score
    
    def save_thresholds_to_csv(self, save_path: Path, model_name: str = None, 
                                fold_num: int = None, additional_info: dict = None):
        """
        最適化された閾値をCSVファイルに保存
        
        Parameters:
        -----------
        save_path : Path
            保存先のパス（ディレクトリまたはCSVファイル）
        model_name : str, optional
            モデル名
        fold_num : int, optional
            Foldの番号（Cross-Validationの場合）
        additional_info : dict, optional
            追加情報（辞書形式）
        
        Returns:
        --------
        Path : 保存されたCSVファイルのパス
        """
        if self.best_thresholds is None:
            raise ValueError("Thresholds have not been optimized yet. Call optimize_thresholds() first.")
        
        save_path = Path(save_path)
        
        # ディレクトリが指定された場合、ファイル名を自動生成
        if save_path.is_dir() or save_path.suffix == '':
            save_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optimized_thresholds_{timestamp}.csv"
            if model_name:
                filename = f"{model_name}_optimized_thresholds_{timestamp}.csv"
            save_path = save_path / filename
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 閾値データの作成
        threshold_data = {
            'threshold_index': list(range(len(self.best_thresholds))),
            'threshold_value': self.best_thresholds,
            'threshold_percent': [t * 100 for t in self.best_thresholds],
        }
        
        # クラス範囲の説明を追加
        class_ranges = []
        for i in range(self.n_classes):
            if i == 0:
                class_ranges.append(f"0% (exactly)")
            elif i == self.n_classes - 1:
                lower = self.best_thresholds[i-1] * 100
                class_ranges.append(f">{lower:.1f}% to 100%")
            else:
                lower = self.best_thresholds[i-1] * 100
                upper = self.best_thresholds[i] * 100
                class_ranges.append(f">{lower:.1f}% to <={upper:.1f}%")
        
        # 閾値とクラスの対応をDataFrameに
        df_thresholds = pd.DataFrame(threshold_data)
        
        # メタ情報を追加
        meta_rows = []
        meta_rows.append({'threshold_index': 'metric', 'threshold_value': self.optimization_metric, 'threshold_percent': ''})
        meta_rows.append({'threshold_index': 'best_score', 'threshold_value': self.best_score, 'threshold_percent': ''})
        meta_rows.append({'threshold_index': 'n_classes', 'threshold_value': self.n_classes, 'threshold_percent': ''})
        meta_rows.append({'threshold_index': 'timestamp', 'threshold_value': datetime.now().isoformat(), 'threshold_percent': ''})
        
        if model_name:
            meta_rows.append({'threshold_index': 'model_name', 'threshold_value': model_name, 'threshold_percent': ''})
        if fold_num is not None:
            meta_rows.append({'threshold_index': 'fold_num', 'threshold_value': fold_num, 'threshold_percent': ''})
        if additional_info:
            for key, value in additional_info.items():
                meta_rows.append({'threshold_index': key, 'threshold_value': str(value), 'threshold_percent': ''})
        
        # クラス範囲情報
        df_classes = pd.DataFrame({
            'class_index': list(range(self.n_classes)),
            'class_range': class_ranges
        })
        
        # CSVに保存（閾値情報）
        df_thresholds.to_csv(save_path, index=False)
        
        # メタ情報とクラス範囲を追記
        with open(save_path, 'a') as f:
            f.write('\n# Metadata\n')
            for row in meta_rows:
                f.write(f"{row['threshold_index']},{row['threshold_value']},{row['threshold_percent']}\n")
            f.write('\n# Class Ranges\n')
            f.write('class_index,class_range\n')
            for i, range_str in enumerate(class_ranges):
                f.write(f"{i},{range_str}\n")
        
        print(f"    💾 Saved optimized thresholds to {save_path}")
        
        return save_path


# *******************************************************************
# モデル初期化関数
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
    
    if model_name == 'MLP-Mixer':
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
        # 【新モデル】柔軟なチャンネル構成対応 + CBAM Attention
        # チャンネル構成を明示的に指定可能（RGB, DEM, Slope, VDVI, GLCM, EXG）
        # channel_configの例:
        #   - RGB(3ch): [0, 1, 2]
        #   - DEM(1ch): [3]
        #   - Slope(1ch): [4]
        #   - VDVI(1ch): [5]
        #   - GLCM(1ch): [6]
        #   - EXG(1ch): [7]
        model = HybridModelV2(
            channel_config=hyperparams.get('channel_config', {
                'rgb': [0, 1, 2],
                'dem': [3] if in_channels > 3 else [],
                'slope': [4] if in_channels > 4 else [],
                'vdvi': [5] if in_channels > 5 else [],
                'glcm': [6] if in_channels > 6 else [],
                'exg': [7] if in_channels > 7 else [],
            }),
            n_classes=n_classes,
            use_cbam=hyperparams.get('use_cbam', True),  # CBAMアテンション使用
            dropout=hyperparams.get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# *******************************************************************
# ハイパーパラメータ探索のためのObjective関数
# *******************************************************************
def create_objective(model_name, train_dataset, val_dataset, n_classes, device, n_epochs=20, 
                     use_amp=True, use_distance_penalty=False, distance_weight=0.5,
                     use_focal_loss=False, focal_gamma=2.0,
                     use_mixup_cutmix=False, mixup_alpha=0.4, cutmix_alpha=1.0,
                     use_oversampling=False,
                     early_stopping_patience=10, early_stopping_min_delta=0.001):
    """Optunaのobjective関数を作成
    
    注意: train_dataset, val_datasetにはPreprocessedLandslideDatasetを渡すことで
    大幅な高速化が可能。従来のLandslideDatasetも互換性のため使用可能。
    
    Parameters:
    -----------
    use_amp : bool
        Mixed Precision Training (AMP) を使用するか（CUDA環境で約1.5-2倍高速化）
    use_distance_penalty : bool
        距離ペナルティを使用するか（中間クラスを飛ばした誤分類に大きなペナルティ）
    distance_weight : float
        距離ペナルティの強さ（0.0〜1.0）
    use_focal_loss : bool
        Focal Lossを使用するか（難しいサンプルに重点）
    focal_gamma : float
        Focal Lossのgammaパラメータ（デフォルト: 2.0）
    use_mixup_cutmix : bool
        Mixup/CutMixを使用するか（データ拡張）
    mixup_alpha : float
        MixupのBeta分布パラメータ
    cutmix_alpha : float
        CutMixのBeta分布パラメータ
    use_oversampling : bool
        WeightedRandomSamplerによるオーバーサンプリングを使用するか
    early_stopping_patience : int
        Early Stoppingの忍耐回数
    early_stopping_min_delta : float
        改善とみなす最小変化量
    """

    sample = train_dataset[0]['image']
    inferred_in_channels = int(sample.shape[0])
    
    # AMPの有効化判定
    amp_enabled = use_amp and device == 'cuda' and torch.cuda.is_available()
    if amp_enabled:
        print(f"⚡ Mixed Precision Training (AMP) enabled for faster training")
    
    # Mixup/CutMix拡張の準備
    mixup_cutmix = None
    if use_mixup_cutmix:
        mixup_cutmix = MixupCutmixAugmentation(
            mixup_alpha=mixup_alpha, 
            cutmix_alpha=cutmix_alpha,
            prob=0.5,
            switch_prob=0.5
        )
        print(f"🔀 Mixup/CutMix augmentation enabled (alpha={mixup_alpha}/{cutmix_alpha})")
    
    trial_count = [0]  # トライアル番号を追跡するためのカウンター
    
    def objective(trial):
        trial_count[0] += 1
        print(f"\n{'='*70}")
        print(f"Trial #{trial_count[0]} (Optuna Trial #{trial.number}) Starting...")
        print(f"{'='*70}")
        
        # モデル別のハイパーパラメータ探索空間（安定動作範囲に調整）
        if  model_name == 'MLP-Mixer':
            hyperparams = {
                'image_size': 100,  # 20m / 0.2m/px = 100px
                'patch_size': 10,   # 100÷10=10で割り切れる
                'dim': trial.suggest_categorical('dim', [128, 256, 384, 512]),
                'depth': trial.suggest_int('depth', 4, 12),
                'token_dim': trial.suggest_categorical('token_dim', [64, 128, 256]),
                'channel_dim': trial.suggest_categorical('channel_dim', [512, 1024, 2048]),
            }
        elif model_name == 'CNN':
            hyperparams = {
                'image_size': 100,
                'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
                'depth': trial.suggest_int('depth', 2, 6),  # depth=8でCUDAエラーのため制限
                'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
            }
        elif model_name == 'ViT':
            hyperparams = {
                'image_size': 100,  # 20m / 0.2m/px = 100px
                'patch_size': 10,   # 100÷10=10で割り切れる
                'dim': trial.suggest_categorical('dim', [64, 128, 256, 384]),
                'depth': trial.suggest_int('depth', 2, 8),
                'heads': trial.suggest_categorical('heads', [2, 4, 8]),
                'mlp_dim': trial.suggest_categorical('mlp_dim', [128, 256, 512]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            }
        elif model_name == 'HybridModel':
            hyperparams = {
                'image_size': 100,
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            }
        elif model_name == 'HybridModelV2':
            # =================================================================
            # 【新モデル】HybridModelV2のハイパーパラメータ探索範囲
            # =================================================================
            # 探索対象:
            #   - dropout: 0.1～0.5（連続値）
            #   - use_cbam: True/False（CBAM Attention使用有無）
            # 固定値:
            #   - image_size: 100（パッチサイズに応じて設定）
            #   - channel_config: 入力データに応じて自動設定
            # 共通探索対象（全モデル共通）:
            #   - learning_rate: 1e-5～5e-3 (log scale)
            #   - batch_size: [32, 64, 128]
            #   - weight_decay: 1e-6～1e-3 (log scale)
            #   - label_smoothing: 0.0～0.15
            hyperparams = {
                'image_size': 100,
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'use_cbam': trial.suggest_categorical('use_cbam', [True, False]),
                # channel_configはデータに応じて自動設定（init_modelで処理）
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")
        hyperparams['in_channels'] = inferred_in_channels
        
        # 共通のハイパーパラメータ（安定動作範囲）
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
        #batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        batch_size = trial.suggest_categorical('batch_size', [128])  # メモリ制約を考慮して128を除外
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.15)
        
        # 選択されたハイパーパラメータを表示（簡略化）
        print(f"  📊 LR={learning_rate:.5f}, BS={batch_size}, WD={weight_decay:.5f}, LS={label_smoothing:.2f}")
        trial.set_user_attr('in_channels', inferred_in_channels)
        
        # モデルの初期化
        model = init_model(model_name, hyperparams, n_classes)
        model = model.to(device)
        
        # オーバーサンプリング用のサンプラーを作成
        sampler = None
        shuffle = True
        if use_oversampling:
            sampler = create_weighted_sampler(
                train_dataset.labels, 
                train_dataset.thresholds, 
                n_classes
            )
            shuffle = False

        # 🔧 修正: persistent_workers=False に変更（メモリリーク対策）
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=(device == 'cuda'), 
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 4, shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device == 'cuda'),
            persistent_workers=False,
            prefetch_factor=2
        )
        
        # 損失関数とオプティマイザ
        class_weights, class_counts = compute_inverse_frequency_weights(
            train_dataset.labels, train_dataset.thresholds, n_classes
        )
        
        # 🔧 修正: use_oversamplingの場合はclass_weightsをLossに渡さない（二重適用防止）
        if use_oversampling:
            class_weight_tensor = None
        else:
            class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        
        # 距離ペナルティのON/OFFに応じて損失関数を選択
        criterion = create_loss_function(
            n_classes=n_classes,
            class_weights=class_weight_tensor,
            use_distance_penalty=use_distance_penalty,
            distance_weight=distance_weight,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            device=device
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Cosine Annealing with Warm Restarts（より効果的な学習率スケジューリング）
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        
        # AMP用のGradScaler
        scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
        
        # Early Stopping（改善版）
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='max',
            restore_best_weights=True
        )
        
        # 訓練
        best_val_f1 = 0
        best_epoch = 0
        
        # 中間値予測モデル（タプル出力）かどうかを判定
        is_tuple_output_model = model_name in ['IntermediateValueTransformer', 'IntermediateFocusedCNN']
        
        # エポック進捗バーを作成（簡潔版）
        epoch_pbar = tqdm(range(n_epochs), desc=f"  Trial {trial.number}", ncols=80, leave=False)
        
        for epoch in epoch_pbar:
            # Training with AMP and Mixup/CutMix
            model.train()
            train_loss = 0
            for batch in train_loader:
                inputs = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                # Mixup/CutMix を適用
                if mixup_cutmix is not None:
                    inputs, labels_a, labels_b, lam = mixup_cutmix(inputs, labels)
                
                optimizer.zero_grad(set_to_none=True)  # より効率的なzero_grad
                
                if amp_enabled:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        # タプル出力モデルの場合はclass_probsを取り出す
                        if is_tuple_output_model and isinstance(outputs, tuple):
                            class_probs, _ = outputs
                            logits = torch.log(class_probs + 1e-7)
                        else:
                            logits = outputs
                        if mixup_cutmix is not None and lam < 1.0:
                            loss = MixupCutmixAugmentation.mixup_criterion(
                                criterion, logits, labels_a, labels_b, lam
                            )
                        else:
                            loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    # タプル出力モデルの場合はclass_probsを取り出す
                    if is_tuple_output_model and isinstance(outputs, tuple):
                        class_probs, _ = outputs
                        logits = torch.log(class_probs + 1e-7)
                    else:
                        logits = outputs
                    if mixup_cutmix is not None and lam < 1.0:
                        loss = MixupCutmixAugmentation.mixup_criterion(
                            criterion, logits, labels_a, labels_b, lam
                        )
                    else:
                        loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation with AMP
            model.eval()
            val_preds = []       # GPU tensors
            val_labels_list = [] # CPU tensors
            val_losses_batch = []  # loss テンソルをリストに蓄積
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['image'].to(device, non_blocking=True)
                    labels = batch['label']
                    
                    if amp_enabled:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            # タプル出力モデルの場合はclass_probsを取り出す
                            if is_tuple_output_model and isinstance(outputs, tuple):
                                class_probs, _ = outputs
                                logits = torch.log(class_probs + 1e-7)
                            else:
                                logits = outputs
                            loss = criterion(logits, labels.to(device))
                    else:
                        outputs = model(inputs)
                        # タプル出力モデルの場合はclass_probsを取り出す
                        if is_tuple_output_model and isinstance(outputs, tuple):
                            class_probs, _ = outputs
                            logits = torch.log(class_probs + 1e-7)
                        else:
                            logits = outputs
                        loss = criterion(logits, labels.to(device))
                    
                    val_losses_batch.append(loss)  # .item() を呼ばない
                    # GPU上に保持（.cpu()を呼ばない）
                    val_preds.append(logits.argmax(dim=1))
                    val_labels_list.append(labels)
            
            # 一括でCPU転送・numpy変換
            val_loss = torch.stack(val_losses_batch).sum().item()  # ここで1回だけ .item()
            val_preds = torch.cat(val_preds).cpu().numpy()
            val_labels_list = torch.cat(val_labels_list).numpy()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # F1スコアと精度の計算
            val_f1 = f1_score(val_labels_list, val_preds, average='macro')
            val_acc = accuracy_score(val_labels_list, val_preds)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
            
            # tqdmの進捗表示を更新
            epoch_pbar.set_postfix({
                'F1': f'{val_f1:.4f}',
                'Best': f'{best_val_f1:.4f}',
                'Acc': f'{val_acc:.4f}'
            })
            
            # Cosine Annealing with Warm Restarts用の学習率更新
            scheduler.step(epoch + 1)
            
            # Aggressive Pruning（早期終了）
            trial.report(val_f1, epoch)
            if trial.should_prune():
                epoch_pbar.close()
                tqdm.write(f"  ⚠️ Pruned at E{epoch+1} (F1: {val_f1:.4f})")
                raise optuna.TrialPruned()
            
            # 改善版Early Stopping
            if early_stopping(val_f1, model):
                epoch_pbar.close()
                tqdm.write(f"  ⏹️ Early stop at E{epoch+1} (patience={early_stopping_patience}, delta={early_stopping_min_delta})")
                # 最良の重みを復元
                early_stopping.restore_weights(model)
                break
        
        # トライアル終了時のサマリー（簡略化）
        epoch_pbar.close()
        tqdm.write(f"  ✅ Trial {trial.number}: Best F1={best_val_f1:.4f} (E{best_epoch})")
        
        # ========================================
        # 🔧 修正: 完全なメモリクリーンアップ
        # ========================================
        
        # 1. DataLoaderのワーカーを明示的にシャットダウン
        #    persistent_workers=True の場合、これが必須
        if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
            train_loader._iterator._shutdown_workers()
        if hasattr(val_loader, '_iterator') and val_loader._iterator is not None:
            val_loader._iterator._shutdown_workers()
        
        # 2. EarlyStoppingの重みを解放
        if early_stopping.best_weights is not None:
            del early_stopping.best_weights
            early_stopping.best_weights = None
        
        # 3. GradScalerを解放
        if scaler is not None:
            del scaler
        
        # 4. Optimizerの状態を解放（大きなモデルでは重要）
        del optimizer
        
        # 5. Schedulerを解放
        del scheduler
        
        # 6. Criterionを解放（GPU上のclass_weightsを含む）
        del criterion
        
        # 7. DataLoaderとモデルを解放
        del train_loader
        del val_loader
        del model
        
        # 8. CUDA同期とキャッシュクリア
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # 9. Python GCを複数回実行（循環参照の解消）
        gc.collect()
        gc.collect()
        
        return best_val_f1
    
    return objective


# *******************************************************************
# Main Training Pipeline with Hyperparameter Search
# *******************************************************************
def train_with_hyperparameter_search(model_name, images, labels, pixel_locations, 
                                    geo_locations, n_classes=5, n_trials=50, 
                                    device='cuda', save_dir='results',
                                    dem_range=None, vdvi_range=None, glcm_range=None,
                                    fixed_thresholds=None, hp_search_epochs=20,
                                    use_distance_penalty=False, distance_weight=0.5,
                                    use_focal_loss=False, focal_gamma=2.0,
                                    use_mixup_cutmix=False, mixup_alpha=0.4, cutmix_alpha=1.0,
                                    use_oversampling=False,
                                    early_stopping_patience=10, early_stopping_min_delta=0.001,
                                    patch_size=100):
    """
    ハイパーパラメータ探索を含む訓練パイプライン
    
    Parameters:
    -----------
    hp_search_epochs : int
        ハイパーパラメータ探索時の各トライアルのエポック数（デフォルト: 20）
    use_distance_penalty : bool
        距離ペナルティを使用するか（デフォルト: False）
    distance_weight : float
        距離ペナルティの強さ（0.0〜1.0、デフォルト: 0.5）
    use_focal_loss : bool
        Focal Lossを使用するか（難しいサンプルに重点）
    focal_gamma : float
        Focal Lossのgammaパラメータ（デフォルト: 2.0）
    use_mixup_cutmix : bool
        Mixup/CutMixを使用するか（データ拡張）
    mixup_alpha : float
        MixupのBeta分布パラメータ
    cutmix_alpha : float
        CutMixのBeta分布パラメータ
    use_oversampling : bool
        WeightedRandomSamplerによるオーバーサンプリングを使用するか
    early_stopping_patience : int
        Early Stoppingの忍耐回数（デフォルト: 10）
    early_stopping_min_delta : float
        改善とみなす最小変化量（デフォルト: 0.001）
    patch_size : int
        パッチサイズ（ピクセル単位、デフォルト: 100）
    """
    print(f"\n{'='*80}")
    print(f"🔍 Hyperparameter Search for {model_name}")
    print(f"{'='*80}")
    print(f"Total trials to run: {n_trials}")
    print(f"Epochs per trial: {hp_search_epochs}")
    print(f"Device: {device}")
    print(f"Number of classes: {n_classes}")
    print(f"Distance penalty: {'ON' if use_distance_penalty else 'OFF'} (weight={distance_weight})")
    print(f"Focal Loss: {'ON' if use_focal_loss else 'OFF'} (gamma={focal_gamma})")
    print(f"Mixup/CutMix: {'ON' if use_mixup_cutmix else 'OFF'} (alpha={mixup_alpha}/{cutmix_alpha})")
    print(f"Oversampling: {'ON' if use_oversampling else 'OFF'}")
    print(f"Early Stopping: patience={early_stopping_patience}, delta={early_stopping_min_delta}")
    print(f"{'='*80}")
    
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
    
    # データ拡張の選択（ハイパーパラメータ探索では基本的なGeometric拡張を使用）
    # 本格的な学習時にはconfig設定で切り替え可能
    train_transform = GeometricAugmentation(
        h_flip_prob=0.5,
        v_flip_prob=0.5,
        rotate90_prob=0.5
    )
    print(f"  🔄 Using GeometricAugmentation for hyperparameter search")
    
    # データセットの作成（事前正規化済み高速版を使用）
    print(f"\n⚡ Using PreprocessedLandslideDataset for faster training...")
    train_dataset = PreprocessedLandslideDataset(
        images[train_idx], labels[train_idx],
        pixel_locations[train_idx], geo_locations[train_idx],
        n_classes=n_classes,
        thresholds=fixed_thresholds,
        patch_size=patch_size,
        dem_range=dem_range,
        vdvi_range=vdvi_range,
        glcm_range=glcm_range,
        transform=train_transform  # 幾何学的データ拡張を適用
    )
    val_dataset = PreprocessedLandslideDataset(
        images[val_idx], labels[val_idx],
        pixel_locations[val_idx], geo_locations[val_idx],
        n_classes=n_classes,
        thresholds=fixed_thresholds,
        patch_size=patch_size,
        dem_range=dem_range,
        vdvi_range=vdvi_range,
        glcm_range=glcm_range
        # 検証データには拡張を適用しない
    )
    
    # Optunaでハイパーパラメータ探索（最適化設定）
    study = create_optimized_study(study_name=f'{model_name}_optimization')
    
    # カスタムコールバック関数（簡略化）
    def print_trial_summary(study, trial):
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"\n{'─'*50}")
        print(f"📊 Progress: {completed} completed, {pruned} pruned | Best: {study.best_value:.4f}")
        print(f"{'─'*50}\n")
    
    objective = create_objective(
        model_name, train_dataset, val_dataset, 
        n_classes, device, n_epochs=hp_search_epochs, use_amp=True,
        use_distance_penalty=use_distance_penalty, distance_weight=distance_weight,
        use_focal_loss=use_focal_loss, focal_gamma=focal_gamma,
        use_mixup_cutmix=use_mixup_cutmix, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
        use_oversampling=use_oversampling,
        early_stopping_patience=early_stopping_patience, 
        early_stopping_min_delta=early_stopping_min_delta
    )
    
    print(f"\n🚀 Starting {n_trials} optimization trials (parallel execution with {NUM_WORKERS//4} jobs)...")
    print(f"{'='*80}")
    
    # Optunaの最適化を実行（48コアマシンで並列化）
    # n_jobs=-1で全CPUコア使用、または明示的に並列ジョブ数を指定
    study.optimize(
        objective, 
        n_trials=n_trials, 
        #n_jobs=min(8, NUM_WORKERS//2),  # 最大8並列ジョブ（メモリを考慮）
        n_jobs=1,  # 48コアマシンで12並列ジョブ
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
# Self-Training Pipeline（Pseudo-Labeling）
# *******************************************************************
def visualize_pseudo_label_changes(
    images: np.ndarray,
    change_details: List[Dict],
    save_dir: Path,
    n_top: int = 20,
    class_names: Optional[List[str]] = None,
):
    """
    Pseudo-Labelで変更されたサンプルのトップN件を可視化
    
    Parameters:
    -----------
    images : np.ndarray
        画像データ（N, H, W, C）
    change_details : List[Dict]
        変更詳細リスト（PseudoLabelGenerator.generateの出力）
    save_dir : Path
        保存先ディレクトリ
    n_top : int
        表示する上位件数（確信度順）
    class_names : List[str], optional
        クラス名リスト
    """
    if not change_details:
        print("  ⚠️ No pseudo-label changes to visualize")
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 確信度順にソート
    sorted_details = sorted(change_details, key=lambda x: x['confidence'], reverse=True)
    top_changes = sorted_details[:n_top]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(5)]
    
    n_cols = 5
    n_rows = (len(top_changes) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
    
    for i, detail in enumerate(top_changes):
        ax = axes[i]
        idx = detail['index']
        
        # RGB画像の取得（最初の3チャンネル）
        img = images[idx]
        if img.shape[-1] >= 3:
            rgb = img[:, :, :3]
            # 正規化されていない場合は正規化
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0, 1)
        else:
            rgb = np.stack([img[:, :, 0]] * 3, axis=-1)
            rgb = np.clip(rgb, 0, 1)
        
        ax.imshow(rgb)
        
        old_class_name = class_names[detail['old_class']] if detail['old_class'] < len(class_names) else f"C{detail['old_class']}"
        new_class_name = class_names[detail['new_class']] if detail['new_class'] < len(class_names) else f"C{detail['new_class']}"
        
        title = f"Conf: {detail['confidence']:.3f}\n{old_class_name} → {new_class_name}"
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    # 余ったaxesを非表示
    for i in range(len(top_changes), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Top {len(top_changes)} Pseudo-Label Changes (by confidence)", fontsize=12)
    plt.tight_layout()
    
    output_path = save_dir / "pseudo_label_top_changes.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Pseudo-label visualization saved to: {output_path}")


def run_self_training_pipeline(
    images: np.ndarray,
    labels: np.ndarray,
    pixel_locations: np.ndarray,
    geo_locations: np.ndarray,
    landslide_ids: np.ndarray,
    config: Dict,
    teacher_model_path: Optional[str] = None,
    pseudo_label_threshold: float = 0.9,
) -> Dict:
    """
    Self-Training（Pseudo-Labeling）パイプラインを実行
    
    Step 1: Teacherモデルのロード（または学習）
    Step 2: 全データに対してPseudo-Labelを生成し、ラベルをリファイン
    Step 3: リファインされたラベルでStudentモデルを学習（CV）
    
    Parameters:
    -----------
    images : np.ndarray
        画像データ
    labels : np.ndarray
        元のラベル（連続値）
    pixel_locations : np.ndarray
        ピクセル座標
    geo_locations : np.ndarray
        地理座標
    landslide_ids : np.ndarray
        崩壊地ID
    config : Dict
        設定辞書
    teacher_model_path : str, optional
        学習済みTeacherモデルのパス。Noneの場合は新規学習
    pseudo_label_threshold : float
        Pseudo-Label採用の確信度閾値
    
    Returns:
    --------
    Dict : 結果情報（refined_labels, change_stats, student_results等）
    """
    print(f"\n{'='*80}")
    print(f"🎓 Self-Training Pipeline (Pseudo-Labeling)")
    print(f"{'='*80}")
    print(f"  Pseudo-label threshold: {pseudo_label_threshold}")
    print(f"  Total samples: {len(images)}")
    
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = config.get('n_classes', 5)
    fixed_thresholds = config.get('fixed_thresholds', FIVE_CLASS_FIXED_THRESHOLDS.copy())
    
    # 出力ディレクトリ
    output_dir = Path(config.get('output_dir', 'output'))
    self_training_dir = output_dir / 'self_training'
    self_training_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = self_training_dir / 'self_training_viz'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'original_labels': labels.copy(),
        'refined_labels': None,
        'change_stats': None,
        'teacher_model_path': teacher_model_path,
        'student_results': None,
    }
    
    # ========================================
    # Step 1: Teacherモデルのロード/学習
    # ========================================
    print(f"\n📚 Step 1: Preparing Teacher Model...")
    
    teacher_model_name = config.get('teacher_model_name', 'HybridModel')
    teacher_hyperparams = config.get('predefined_hyperparams', {}).get(teacher_model_name, {})
    teacher_hyperparams['in_channels'] = config.get('expected_in_channels', 6)
    
    teacher_model = init_model(teacher_model_name, teacher_hyperparams, n_classes)
    teacher_model = teacher_model.to(device)
    
    if teacher_model_path and Path(teacher_model_path).exists():
        print(f"  Loading pre-trained teacher from: {teacher_model_path}")
        checkpoint = torch.load(teacher_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            teacher_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            teacher_model.load_state_dict(checkpoint)
        print(f"  ✅ Teacher model loaded successfully")
    else:
        print(f"  Training new teacher model ({teacher_model_name})...")
        
        # 一時的にTeacherを学習（全データの一部で）
        from sklearn.model_selection import train_test_split
        
        train_idx, _ = train_test_split(
            np.arange(len(images)), test_size=0.2, random_state=42
        )
        
        train_dataset = PreprocessedLandslideDataset(
            images[train_idx], labels[train_idx],
            pixel_locations[train_idx], geo_locations[train_idx],
            n_classes=n_classes,
            thresholds=fixed_thresholds,
            patch_size=config.get('patch_size', 100),  # tip_size_mから計算したパッチサイズ
            dem_range=config.get('dem_range'),
            vdvi_range=config.get('vdvi_range'),
            glcm_range=config.get('glcm_range'),
            exg_range=config.get('exg_range'),
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                          num_workers=NUM_WORKERS, pin_memory=True,
                                          persistent_workers=True, prefetch_factor=4)
        
        optimizer = torch.optim.AdamW(
            teacher_model.parameters(), 
            lr=teacher_hyperparams.get('learning_rate', 1e-3),
            weight_decay=teacher_hyperparams.get('weight_decay', 1e-4)
        )
        criterion = nn.CrossEntropyLoss()
        
        teacher_model.train()
        n_teacher_epochs = config.get('teacher_epochs', 20)
        
        # Teacher訓練の進捗バー
        epoch_pbar = tqdm(range(n_teacher_epochs), desc="  Teacher Training", ncols=80)
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            for batch in train_loader:
                inputs = batch['image'].to(device)
                targets = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = teacher_model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # tqdmの進捗表示を更新
            avg_loss = epoch_loss / len(train_loader)
            epoch_pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        epoch_pbar.close()
        
        # Teacherモデルを保存
        teacher_save_path = self_training_dir / f"teacher_{teacher_model_name}.pth"
        torch.save(teacher_model.state_dict(), teacher_save_path)
        results['teacher_model_path'] = str(teacher_save_path)
        print(f"  ✅ Teacher model trained and saved to: {teacher_save_path}")
    
    # ========================================
    # Step 2: Pseudo-Label生成
    # ========================================
    print(f"\n🏷️  Step 2: Generating Pseudo-Labels...")
    
    # 全データに対してPseudo-Labelを生成
    full_dataset = PreprocessedLandslideDataset(
        images, labels,
        pixel_locations, geo_locations,
        n_classes=n_classes,
        thresholds=fixed_thresholds,
        patch_size=config.get('patch_size', 100),  # tip_size_mから計算したパッチサイズ
        dem_range=config.get('dem_range'),
        vdvi_range=config.get('vdvi_range'),
        glcm_range=config.get('glcm_range'),
        exg_range=config.get('exg_range'),
    )
    
    generator = PseudoLabelGenerator(
        model=teacher_model,
        device=device,
        n_classes=n_classes,
        thresholds=fixed_thresholds
    )
    
    refined_labels, change_indices, change_stats = generator.generate(
        dataset=full_dataset,
        batch_size=64,
        threshold=pseudo_label_threshold,
        require_class_change=True  # 現在のラベルと異なる予測のみ採用
    )
    
    results['refined_labels'] = refined_labels
    results['change_stats'] = change_stats
    
    print(f"\n📊 Pseudo-Label Statistics:")
    print(f"  Total samples: {change_stats['total_samples']}")
    print(f"  Changed samples: {change_stats['changed_samples']} ({change_stats['change_ratio']*100:.2f}%)")
    print(f"  Average confidence: {change_stats['avg_confidence']:.4f}")
    
    # クラス間変更マトリクスを表示
    print(f"\n  Class Change Matrix (rows=old, cols=new):")
    change_matrix = np.array(change_stats['class_change_matrix'])
    class_names = generate_class_names_from_thresholds(fixed_thresholds, n_classes)
    
    header = "      " + "  ".join([f"{name[:6]:>6}" for name in class_names])
    print(header)
    for i, row in enumerate(change_matrix):
        row_str = f"{class_names[i][:6]:>6}" + "  ".join([f"{v:>6}" for v in row])
        print(row_str)
    
    # ========================================
    # Step 2.5: 変更サンプルの可視化
    # ========================================
    print(f"\n📷 Visualizing top pseudo-label changes...")
    visualize_pseudo_label_changes(
        images=images,
        change_details=change_stats['details'],
        save_dir=viz_dir,
        n_top=20,
        class_names=class_names,
    )
    
    # 変更統計をJSONで保存（NumPy型をPython型に変換）
    stats_to_save = {
        'total_samples': int(change_stats['total_samples']),
        'changed_samples': int(change_stats['changed_samples']),
        'change_ratio': float(change_stats['change_ratio']),
        'threshold': float(change_stats['threshold']),
        'avg_confidence': float(change_stats['avg_confidence']),
        'class_change_matrix': [[int(v) for v in row] for row in change_stats['class_change_matrix']],
    }
    with open(self_training_dir / 'pseudo_label_stats.json', 'w') as f:
        json.dump(stats_to_save, f, indent=2)
    
    # ========================================
    # Step 3: リファインされたラベルでStudent学習（CV）
    # ========================================
    print(f"\n🎓 Step 3: Training Student Model with Refined Labels...")
    print(f"  ⚠️  IMPORTANT: Training uses PSEUDO-LABELS, Validation uses GROUND TRUTH")
    
    # configをコピーしてStudent用に調整
    student_config = config.copy()
    student_config['output_dir'] = self_training_dir / 'student'
    Path(student_config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Student学習用のモデル名を取得
    student_model_names = config.get('model_names', ['HybridModel'])
    
    student_results = {}
    
    for model_name in student_model_names:
        print(f"\n  Training {model_name} with refined labels...")
        
        hyperparams = config.get('predefined_hyperparams', {}).get(model_name, {})
        hyperparams['in_channels'] = config.get('expected_in_channels', 6)
        
        # ========================================
        # cross_validate_with_improvements呼び出し（Validation Leak防止）
        # - labels: オリジナルの正解ラベル（検証用およびStratified K-Fold用）
        # - pseudo_labels: リファインされた擬似ラベル（学習データ用）
        # ========================================
        cv_result = cross_validate_with_improvements(
            model_name=model_name,
            hyperparams=hyperparams,
            images=images,
            labels=labels,  # オリジナルラベル（検証用・層化抽出用）
            pixel_locations=pixel_locations,
            geo_locations=geo_locations,
            landslide_ids=landslide_ids,
            n_classes=n_classes,
            n_folds=config.get('n_outer_cv', 5),
            n_epochs=config.get('n_epochs', 40),
            batch_size=hyperparams.get('batch_size', 64),
            device=device,
            save_dir=str(student_config['output_dir']),
            optimize_threshold=config.get('optimize_threshold', False),
            use_spatial_kfold=config.get('use_spatial_kfold', True),
            config=student_config,
            pseudo_labels=refined_labels,  # 学習データ用の擬似ラベル
            patch_size=config.get('patch_size', 100)
        )
        
        student_results[model_name] = cv_result
    
    results['student_results'] = student_results
    
    print(f"\n{'='*80}")
    print(f"✅ Self-Training Pipeline Complete!")
    print(f"   Output directory: {self_training_dir}")
    print(f"{'='*80}")
    
    return results


# *******************************************************************
# Main Function
# *******************************************************************
def compose_trial_name(
    output_root: Path,
    base_prefix: str,
    model_names: List[str],
    use_dem: bool,
    use_vdvi: bool,
    use_slope: bool = False,
    use_glcm: bool = False,
    use_exg: bool = False,
    n_classes: int = 5,
) -> str:
    """Build versioned trial name reflecting model usage and auxiliary rasters."""
    # base_prefixのclass部分を実際のn_classesで置き換え
    if "class5" in base_prefix:
        base_prefix = base_prefix.replace("class5", f"class{n_classes}")
    elif "class3" in base_prefix:
        base_prefix = base_prefix.replace("class3", f"class{n_classes}")
    
    tags = []
    if model_names:
        tags.append("+".join(model_names))
    if use_dem:
        tags.append("dem")
    if use_slope:
        tags.append("slope")
    if use_vdvi:
        tags.append("vdvi")
    if use_glcm:
        tags.append("glcm")
    if use_exg:
        tags.append("exg")

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
    
    # =================================================================
    # 分類モードの設定（"3class" または "5class"）
    # =================================================================
    # "3class": 3値分類（0%, 0-50%, 50-100%）
    # "5class": 5値分類（0%, 0-25%, 25-50%, 50-75%, 75-100%）
    classification_mode = "3class"  # ← ここを "3class" に変更すると3値分類になります
    
    # 分類設定の取得
    fixed_thresholds, n_classes = get_classification_config(classification_mode)
    print(f"\n{'='*80}")
    print(f"🎯 Classification Mode: {classification_mode}")
    print(f"   - Number of classes: {n_classes}")
    print(f"   - Thresholds: {fixed_thresholds}")
    print(f"   - Class boundaries: {[0] + [int(t*100) for t in fixed_thresholds] + [100]}%")
    print(f"{'='*80}\n")
    
    base_trial_prefix = f"Noto-class{n_classes}"
    
    # =================================================================
    # モデル選択（複数選択可能）
    # =================================================================
    # 利用可能なモデル一覧:
    # 【基本モデル】
    #   - "CNN": 基本的な畳み込みニューラルネットワーク
    #   - "MLP-Mixer": MLP-Mixerアーキテクチャ
    #   - "ViT": Vision Transformer
    #   - "HybridModel": CNN+ViTハイブリッドモデル（RGB+VDVI/DEM別処理）
    # 【新規追加モデル】
    #   - "HybridModelV2": 柔軟なチャンネル構成対応（GLCM/EXG明示的処理、CBAM対応）
    #
    # 使用例:
    #   model_names = ["CNN", "ViT"]  # 複数モデルで比較
    #   model_names = ["HybridModelV2"]  # 新モデルのみ使用
    #
    model_names = ["HybridModel", "HybridModelV2"]  # 使用するモデル名のリスト
    # model_names = ["HybridModelV2"]  # ← 新モデルを使う場合はこちらをコメント解除
    use_dem = True
    use_slope = True  # DEMから傾斜（Slope）を計算して追加（use_dem=Trueが必要）
    use_vdvi = True  # VDVIを使用
    use_glcm = False  # GLCMテクスチャ特徴量を使用
    use_exg = False  # EXG（Excess Green Index）を使用

    output_root = root_dir / "output"
    trial_name = compose_trial_name(output_root, base_trial_prefix, model_names, use_dem, use_vdvi, use_slope, use_glcm, use_exg, n_classes)

    # ディレクトリの設定
    data_dir = root_dir / 'data'
    data_annotation_dir = data_dir / "annotation"
    data_Noto_April_dir = data_dir / "Noto_April"
    data_sample_dir = data_Noto_April_dir / "annotation_sample"
    #dem_dir = data_dir / 'dem'
    dem_dir = Path("/home/ge/Documents/DEM_Noto")#6000マシンで回す場合こっち
    vdvi_root = data_dir / 'vdvi'
    glcm_root = Path("/home/ge/Documents/GLCM")  # GLCMファイルのルートディレクトリ
    exg_root = data_dir / 'exg'  # EXGファイルのルートディレクトリ
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / trial_name
    fig_dir = output_dir / "figs"

    # チャンネル数の計算
    # use_slopeはuse_demがTrueの時のみ有効
    use_slope_effective = use_slope and use_dem
    
    # 🔧 修正: データ読み込み時のチャンネル数を正確に計算
    # RGB(3) + DEM(1 if use_dem) + VDVI(1 if use_vdvi) + GLCM(1 if use_glcm) + EXG(1 if use_exg)
    load_channels = 3  # RGB is always present
    if use_dem:
        load_channels += 1
    if use_vdvi:
        load_channels += 1
    if use_glcm:
        load_channels += 1
    if use_exg:
        load_channels += 1
    
    # 最終的なチャンネル数（Slope追加後）
    # Slopeは読み込み後にDEMから計算されて追加されるため、ここで加算
    expected_in_channels = load_channels
    if use_slope_effective:
        expected_in_channels += 1
    
    print(f"📊 Channel configuration:")
    print(f"  RGB: 3 channels (always)")
    if use_dem:
        print(f"  DEM: 1 channel")
    if use_slope_effective:
        print(f"  Slope: 1 channel (calculated from DEM)")
    if use_vdvi:
        print(f"  VDVI: 1 channel")
    if use_glcm:
        print(f"  GLCM: 1 channel")
    if use_exg:
        print(f"  EXG: 1 channel")
    print(f"  Total load channels (before Slope): {load_channels}")
    print(f"  Expected final channels (after Slope): {expected_in_channels}")
    
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
        ('07fd041', data_sample_dir / '07fd041.jpg', data_annotation_dir / '07fd041' / '07fd041.geojson'),
        ('07fd111', data_sample_dir / '07fd111.jpg', data_annotation_dir / '07fd111' / '07fd111.geojson'),
        ('07ed684', data_sample_dir / '07ed684.jpg', data_annotation_dir / '07ed684' / '07ed684.geojson'),
        ('07ed582', data_sample_dir / '07ed582.jpg', data_annotation_dir / '07ed582' / '07ed582.geojson'),
    ]

    shared_dem_path = None
    if use_dem:
        #preferred_dem_path = Path(r"C:\Users\GE\pj24_legacy-kaken-chami\Noto\data\dem\mergedem.tif")
        preferred_dem_path = Path("/home/ge/Documents/DEM_Noto/merge/mergedem.tif")#6000マシンで回す場合こっち
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

        exg_candidate = None
        exg_warning = None
        if use_exg:
            candidate = exg_root / f'{dataset_name}.npy'
            if candidate.exists():
                exg_candidate = candidate
            else:
                exg_warning = f"EXG file not found for {dataset_name} at {candidate}."

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
        if exg_warning:
            print(f"⚠️  {exg_warning}")

        dataset_configs.append({
            'rgb_path': rgb_path,
            'dataset_name': dataset_name,
            'legacy_path': legacy_path,
            'dem_path': dem_candidate,
            'vdvi_path': vdvi_candidate,
            'exg_path': exg_candidate
        })
    
    # 設定パラメータ
    resolution = 0.2  # m/px
    tip_size_m = 20  # 入力画角を20mに拡大（周辺コンテキスト含む）
    patch_size = int(tip_size_m / resolution)  # 100px (20m / 0.2m/px)
    
    config = {
        'dataset_configs': dataset_configs,
        'landslide_path': data_dir / 'noto_landslide' / 'hokai-all.geojson',
        'output_dir': output_dir,
        'fig_dir': fig_dir,
        'epsg_code': 6675,
        'resolution': resolution,  # m/px
        'patch_size': patch_size,  # パッチサイズ（ピクセル単位）
        # Center-Labeling戦略: 入力サイズ20m（100px）、評価中心5m（25px）
        'tip_size_m': tip_size_m,  # 入力画角を20mに拡大（周辺コンテキスト含む）
        'target_size_m': 5,  # 評価中心を5mに維持（ラベル計算領域）
        
        # =================================================================
        # データ拡張設定（切り替え可能）
        # =================================================================
        # 【基本拡張】GeometricAugmentation: 回転・反転のみ
        # 【新規拡張】以下のクラスが利用可能:
        #   - AdvancedAugmentation: RGB限定ColorJitter（DEM/Slope等に影響なし）
        #   - GridMaskAugmentation: ランダムグリッドマスク
        #   - CombinedAugmentation: 上記を組み合わせた拡張
        #
        # 使用例:
        #   'augmentation_type': 'geometric'  # 基本（デフォルト）
        #   'augmentation_type': 'advanced'   # ColorJitter追加
        #   'augmentation_type': 'gridmask'   # GridMask
        #   'augmentation_type': 'combined'   # すべて組み合わせ
        #
        'augmentation_type': 'geometric',  # データ拡張タイプ（geometric/advanced/gridmask/combined）
        'slide_ratio': 0.5,
        'n_outer_cv': 8,  # クロスバリデーションのfold数
        'hp_search_epochs': 15,  # ハイパーパラメータ探索時のエポック数（高速化: 20→12）
        'n_epochs': 40,  # 本学習のエポック数
        'batch_size': 64,  # バッチサイズ（CVでは大きめに調整）
        'model_names': model_names,  # 使用するモデル
        'n_trials': 32,  # ハイパーパラメータ探索のトライアル数（高速化: 32→20）
        'n_classes': n_classes,  # クラス数（3または5、classification_modeで決定）
        'classification_mode': classification_mode,  # 分類モード（"3class" or "5class"）
        'optimize_threshold': False,  # 閾値最適化を行うか
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # ハイパーパラメータ探索のON/OFF
        'hyperparameter_search': True,  # Falseで探索をスキップ
        
        # 距離ペナルティ設定（中間クラスを飛ばした誤分類に大きなペナルティ）
        'use_distance_penalty': False,  # True: 距離ペナルティON, False: OFF
        'distance_weight': 0.3,  # ペナルティの強さ（0.0〜1.0）
        
        # Focal Loss設定（難しいサンプルに重点を置く）
        'use_focal_loss': True,  # True: Focal Loss ON, False: OFF（use_distance_penaltyと併用不可）
        'focal_gamma': 2.0,  # Focusing parameter（0で通常CE、大きいほど難しいサンプルに重点）
        
        # Mixup/CutMix設定（データ拡張）
        # 🔧 修正: まずベースラインを確認するためデフォルトでOFF
        'use_mixup_cutmix': True,  # True: Mixup/CutMix ON, False: OFF（推奨: まずFalseで検証）
        'mixup_alpha': 0.2,  # MixupのBeta分布パラメータ
        'cutmix_alpha': 1.0,  # CutMixのBeta分布パラメータ
        
        # オーバーサンプリング設定（クラス不均衡対策）
        'use_oversampling': True,  # True: WeightedRandomSampler ON
        
        # Early Stopping設定（改善版）
        'early_stopping_patience': 10,  # 改善がない場合に待つエポック数
        'early_stopping_min_delta': 0.001,  # 改善とみなす最小変化量
        
        # 傾斜（Slope）計算設定
        'use_slope': use_slope_effective,  # DEMから傾斜を計算して追加（use_dem=True時のみ有効）
        'slope_pixel_size': 0.2,  # 傾斜計算用のピクセルサイズ（メートル）- DEMの解像度に合わせて修正
        
        # GLCM（テクスチャ特徴量）設定
        'use_glcm': use_glcm,  # GLCMテクスチャ特徴量を使用
        'glcm_root': glcm_root if use_glcm else None,  # GLCMファイルのルートディレクトリ
        'glcm_range': (0.0, 100.0) if use_glcm else None,  # GLCM Contrast値の正規化範囲
        
        # EXG（Excess Green Index）設定
        'use_exg': use_exg,  # EXG植生指標を使用
        'exg_root': exg_root if use_exg else None,  # EXGファイルのルートディレクトリ
        'exg_range': (-0.3, 0.3) if use_exg else None,  # EXG値の正規化範囲（VDVIと同様）
        
        # Spatial K-Fold設定
        'use_spatial_kfold': True,  # Balanced Spatial K-Foldを使用
        'min_points_per_fold': 1000,  # 各foldの最小サンプル数
        'visualize_folds': True,  # foldの空間分布を可視化
        'diagnose_distribution': True,  # CV前にTrain/Val分布を診断（過学習分析用）
        'base_seed': 42,  # ランダムシード
        'dem_dir': dem_dir,
        'vdvi_root': vdvi_root,
        'exg_root': exg_root,
        'use_dem': use_dem,
        'use_vdvi': use_vdvi,
        'use_exg': use_exg,
        'dem_range': (-100.0, 1500.0) if use_dem else None,
        'vdvi_range': (-0.3, 0.3) if use_vdvi else None,
        'load_channels': load_channels,  # データ読み込み時のチャンネル数（Slope追加前）
        'expected_in_channels': expected_in_channels,  # 最終的なチャンネル数（Slope追加後）
        # キャッシュ設定: Noneで毎回パッチ抽出処理を実行、パスを指定するとpklキャッシュを使用
        'cache_dir': None,  # data_dir / 'pkl' に戻すとキャッシュを使用
        'fixed_thresholds': fixed_thresholds,  # 分類モードに応じた閾値（3class or 5class）
        'dem_alignment_dir': dem_alignment_dir,
        'patch_overlay_dir': patch_overlay_dir,
        
        # 事前定義されたハイパーパラメータ（探索をOFFにした場合に使用）
        # 入力サイズ: 100px (20m / 0.2m/px)
        'predefined_hyperparams': {
        # CNN用の定義を追加
            'CNN': {
                'image_size': 100,  # 20m / 0.2m/px = 100px
                'hidden_dim': 128,  # ResNetよりシンプルにするため 128 から減らす
                'depth': 5,
                'kernel_size': 3,
                'learning_rate': 0.00212516411052651,  # 5e-05 → 5e-4 に上げる（チャンネル数変更後は再調整が必要）
                'batch_size': 128,
                'weight_decay': 8.974835513193227e-05,
                "label_smoothing": 0.1,
                'in_channels': expected_in_channels
            },
            'MLP-Mixer': {
                'image_size': 100,  # 20m / 0.2m/px = 100px
                'patch_size': 5,  # 100÷10=10で割り切れる
                'dim': 512,
                'depth': 7,
                'token_dim': 256,
                'channel_dim': 1024,
                "learning_rate": 0.00011804711633389237,  # 学習率を上げる
                'batch_size': 64,
                'weight_decay': 0.00015527469527538596,
                "label_smoothing": 0.1,
                'in_channels': expected_in_channels
            },
            'ViT': {
                'image_size': 100,  # 20m / 0.2m/px = 100px
                'patch_size': 5,  # 100÷10=10で割り切れる
                'dim': 256,
                'depth': 4,
                'heads': 8,
                'mlp_dim': 512,
                'dropout': 0.35,
                'learning_rate': 0.00043625993625605574,
                'batch_size': 128,
                'weight_decay': 0.000788671412999048,
                "label_smoothing": 0.1,
                'in_channels': expected_in_channels
            },
            # 新規追加: ハイブリッドCNN-ViTモデル
            'HybridModel': {
                'image_size': 100,  # 20m / 0.2m/px = 100px
                'dropout': 0.4464704583099741,
                'learning_rate': 0.00041917115166952007,  # 学習率を調整
                'batch_size': 128,
                'weight_decay': 0.0003142880890840109,
                "label_smoothing": 0.1,
                'in_channels': expected_in_channels
            },
            # =================================================================
            # HybridModelV2: 柔軟なチャンネル構成対応 + CBAM Attention
            # =================================================================
            # 探索をOFFにした場合に使用されるデフォルト値
            # Optuna探索後に最適値で更新することを推奨
            'HybridModelV2': {
                'image_size': 100,  # 20m / 0.2m/px = 100px
                'dropout': 0.3,  # 探索範囲: 0.1～0.5
                'use_cbam': True,  # CBAM Attention使用（True/False）
                'learning_rate': 0.0005,  # 探索範囲: 1e-5～5e-3 (log scale)
                'batch_size': 128,  # 探索範囲: [32, 64, 128]
                'weight_decay': 0.0001,  # 探索範囲: 1e-6～1e-3 (log scale)
                'label_smoothing': 0.1,  # 探索範囲: 0.0～0.15
                'in_channels': expected_in_channels,
                # channel_config: 入力チャンネルに応じて自動設定
                # 明示的に指定する場合は以下のように:
                # 'channel_config': {
                #     'rgb': [0, 1, 2],
                #     'dem': [3],
                #     'slope': [4],
                #     'vdvi': [5],
                #     'glcm': [6],
                #     'exg': [7],
                # }
            }
        },
        
        # Self-Training（Pseudo-Labeling）設定
        'use_self_training': False,  # True: Pseudo-Labelingを実行, False: 通常学習
        'pseudo_label_threshold': 0.8,  # Pseudo-Label採用の確信度閾値
        'teacher_model_path': None,  # 学習済みTeacherモデルのパス（Noneで新規学習）
        'teacher_model_name': 'HybridModel',  # Teacherモデルの種類
        'teacher_epochs': 20,  # Teacherモデルの学習エポック数
        
        # 難しいパッチの抽出・保存設定
        'save_difficult_patches': True,  # True: 難しいパッチを抽出して保存
        'difficult_patch_size': 25,  # 抽出するパッチサイズ（中心からのクロップ）
        'max_difficult_patches': 100,  # 各タイプごとの最大保存パッチ数
        'difficult_confidence_threshold': 0.5,  # 低確信度とみなす閾値
    }
    
    print(f"{'='*80}")
    print(f"Landslide Classification Pipeline")
    print(f"{'='*80}")
    print(f"Trial name: {trial_name}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {config['device']}")
    
    # GPU情報の詳細表示
    if config['device'] == 'cuda':
        print(f"  ✅ CUDA is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  🎮 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"  📊 Current GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            print(f"  ⚠️  WARNING: CUDA device set but not available!")
    else:
        print(f"  ⚠️  WARNING: Running on CPU! Training will be VERY slow.")
        print(f"  💡 Recommend: Install CUDA and PyTorch with CUDA support")
    
    print(f"Number of datasets: {len(dataset_configs)}")
    print(f"Models to train: {config['model_names']}")
    print(f"Number of classes: {config['n_classes']}")
    print(f"Use DEM: {config['use_dem']} | Use Slope: {config['use_slope']} | Use VDVI: {config['use_vdvi']}")
    print(f"Expected input channels: {config['expected_in_channels']}")
    print(f"Input size: {config['tip_size_m']}m ({int(config['tip_size_m'] / 0.2)}px)")
    if config.get('target_size_m'):
        print(f"Target (label) size: {config['target_size_m']}m ({int(config['target_size_m'] / 0.2)}px)")
    print(f"Save difficult patches: {config['save_difficult_patches']}")
    print(f"{'='*80}")
    
    # ============================================================
    # キャッシュの整合性チェック（チャンネル数とパッチサイズの検証）
    # ============================================================
    cache_dir = config.get('cache_dir')
    if cache_dir and cache_dir.exists():
        print(f"\n🔍 Checking cache integrity...")
        import pickle
        invalid_caches = []
        expected_load_channels = config.get('load_channels', 5)
        expected_patch_size = int(config['tip_size_m'] / 0.2)  # 20m / 0.2m/px = 100px
        
        for cache_file in cache_dir.glob('*.pkl'):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                if 'tips' in cached_data:
                    cached_channels = cached_data['tips'].shape[-1]
                    cached_patch_size = cached_data['tips'].shape[1]  # (N, H, W, C)
                    
                    # チャンネル数チェック
                    if cached_channels != expected_load_channels:
                        invalid_caches.append((cache_file, f"{cached_channels}ch (expected {expected_load_channels}ch)"))
                        print(f"  ⚠️  {cache_file.name}: {cached_channels}ch (expected {expected_load_channels}ch)")
                    # パッチサイズチェック
                    elif cached_patch_size != expected_patch_size:
                        invalid_caches.append((cache_file, f"{cached_patch_size}px (expected {expected_patch_size}px)"))
                        print(f"  ⚠️  {cache_file.name}: {cached_patch_size}px (expected {expected_patch_size}px)")
            except Exception as e:
                print(f"  ⚠️  {cache_file.name}: Failed to read - {e}")
                invalid_caches.append((cache_file, None))
        
        if invalid_caches:
            print(f"\n🗑️  Removing {len(invalid_caches)} invalid cache file(s)...")
            for cache_file, _ in invalid_caches:
                try:
                    cache_file.unlink()
                    print(f"  ✅ Deleted: {cache_file.name}")
                except Exception as e:
                    print(f"  ❌ Failed to delete {cache_file.name}: {e}")
            print(f"  Cache will be regenerated with correct channel/size configuration.")
        else:
            print(f"  ✅ All cache files have correct configuration ({expected_load_channels}ch, {expected_patch_size}px)")
    
    # 複数データセットの処理と統合
    print("\nProcessing multiple datasets...")
    print(f"  Input FOV: {config['tip_size_m']}m, Target region: {config.get('target_size_m', config['tip_size_m'])}m")
    all_tips, all_labels, all_pixel_locs, all_geo_locs, all_landslide_ids, all_dataset_names = \
        process_multiple_datasets(
            dataset_configs=config['dataset_configs'],
            landslide_path=config['landslide_path'],
            epsg_code=config['epsg_code'],
            tip_size_m=config['tip_size_m'],
            target_size_m=config.get('target_size_m'),  # Center-Labeling用
            slide_ratio=config['slide_ratio'],
            cache_dir=config.get('cache_dir'),
            required_channels=config.get('load_channels'),  # Slope追加前のチャンネル数
            alignment_preview_dir=config.get('dem_alignment_dir'),
            patch_overlay_dir=config.get('patch_overlay_dir'),
            glcm_root=config.get('glcm_root'),  # GLCM特徴量ファイルのルートディレクトリ
            exg_root=config.get('exg_root'),  
            use_exg=config.get('use_exg', False)  # EXG特徴量ファイルのルートディレクトリ
        )
    
    print(f"\n{'='*60}")
    print(f"Total extracted tips: {len(all_tips)}")
    print(f"Overall label distribution:")
    print(f"  Min: {all_labels.min():.3f}")
    print(f"  Max: {all_labels.max():.3f}")
    print(f"  Mean: {all_labels.mean():.3f}")
    print(f"  Std: {all_labels.std():.3f}")
    
    # 傾斜（Slope）チャンネルの追加
    if config.get('use_slope', False):
        print(f"\n📐 Adding slope channel calculated from DEM...")
        original_shape = all_tips.shape
        dem_channel_idx = 3  # RGB(0,1,2)の次がDEM(3)
        all_tips = add_slope_channel_to_tips(
            all_tips, 
            dem_channel_idx=dem_channel_idx,
            pixel_size=config.get('slope_pixel_size', 5.0)
        )
        print(f"  Original shape: {original_shape}")
        print(f"  New shape: {all_tips.shape}")
        channel_order_str = "  Channel order: [R, G, B, DEM, Slope"
        if config.get('use_vdvi'):
            channel_order_str += ", VDVI"
        if config.get('use_glcm'):
            channel_order_str += ", GLCM"
        if config.get('use_exg'):
            channel_order_str += ", EXG"
        channel_order_str += "]"
        print(channel_order_str)
        
        # expected_in_channelsを更新（実際のデータ形状に合わせる）
        config['expected_in_channels'] = all_tips.shape[-1]
        print(f"  Updated expected_in_channels: {config['expected_in_channels']}")
    
    # チャンネル構成の表示
    channel_names = ['R', 'G', 'B']
    if config.get('use_dem'):
        channel_names.append('DEM')
    if config.get('use_slope'):
        channel_names.append('Slope')
    if config.get('use_vdvi'):
        channel_names.append('VDVI')
    if config.get('use_glcm'):
        channel_names.append('GLCM')
    if config.get('use_exg'):
        channel_names.append('EXG')
    print(f"\n📊 Final channel configuration: {channel_names}")
    print(f"  Total channels: {all_tips.shape[-1]}")
    
    # データセットごとの統計情報
    print(f"\nDataset-wise statistics:")
    for dataset_name in np.unique(all_dataset_names):
        mask = all_dataset_names == dataset_name
        dataset_labels = all_labels[mask]
        print(f"  {dataset_name}: n={len(dataset_labels)}, "
              f"mean={dataset_labels.mean():.3f}, std={dataset_labels.std():.3f}")
    
    # ============================================================
    # Self-Training（Pseudo-Labeling）モードの分岐
    # ============================================================
    if config.get('use_self_training', False):
        print(f"\n{'='*80}")
        print(f"🎓 Self-Training Mode Enabled")
        print(f"{'='*80}")
        print(f"  Pseudo-label threshold: {config.get('pseudo_label_threshold', 0.9)}")
        print(f"  Teacher model: {config.get('teacher_model_name', 'HybridModel')}")
        
        self_training_results = run_self_training_pipeline(
            images=all_tips,
            labels=all_labels,
            pixel_locations=all_pixel_locs,
            geo_locations=all_geo_locs,
            landslide_ids=all_landslide_ids,
            config=config,
            teacher_model_path=config.get('teacher_model_path'),
            pseudo_label_threshold=config.get('pseudo_label_threshold', 0.9),
        )
        
        # Self-Training結果をメイン結果に追加
        results = {'self_training': self_training_results}
        
        # Student結果を通常のresultsフォーマットにマッピング
        if self_training_results.get('student_results'):
            for model_name, cv_result in self_training_results['student_results'].items():
                results[model_name] = {
                    'cv_results': cv_result,
                    'predictions_path': str(
                        config['output_dir'] / 'self_training' / 'student' / model_name / 'all_predictions.csv'
                    ),
                }
        
        print(f"\n{'='*80}")
        print(f"✅ Self-Training Pipeline Complete!")
        print(f"{'='*80}")
        
        # ここでreturnせず、後続の評価・可視化も実行可能にする
    
    else:
        # ============================================================
        # 通常モード: 各モデルで訓練と評価
        # ============================================================
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
                    fixed_thresholds=config.get('fixed_thresholds'),
                    hp_search_epochs=config.get('hp_search_epochs', 20),
                    use_distance_penalty=config.get('use_distance_penalty', False),
                    distance_weight=config.get('distance_weight', 0.5),
                    use_focal_loss=config.get('use_focal_loss', False),
                    focal_gamma=config.get('focal_gamma', 2.0),
                    use_mixup_cutmix=config.get('use_mixup_cutmix', False),
                    mixup_alpha=config.get('mixup_alpha', 0.4),
                    cutmix_alpha=config.get('cutmix_alpha', 1.0),
                    use_oversampling=config.get('use_oversampling', False),
                    early_stopping_patience=config.get('early_stopping_patience', 10),
                    early_stopping_min_delta=config.get('early_stopping_min_delta', 0.001),
                    patch_size=config.get('patch_size', 100)
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
            
            # 診断モードが有効な場合、各Foldの分布差を事前分析
            if config.get('diagnose_distribution', True) and config['use_spatial_kfold']:
                print(f"\n🔍 Running Train/Val Distribution Diagnosis...")
                
                # チャンネル名を構築
                ch_names = ['R', 'G', 'B']
                if config.get('use_dem', True):
                    ch_names.append('DEM')
                if config.get('use_slope', False):
                    ch_names.append('Slope')
                if config.get('use_vdvi', True):
                    ch_names.append('VDVI')
                if config.get('use_glcm', False):
                    ch_names.append('GLCM')
                if config.get('use_exg', False):
                    ch_names.append('EXG')
                
                # Spatial K-Foldを適用してFold割り当てを取得
                try:
                    spatial_folds, gdf_prj = apply_balanced_spatial_kfold(
                        all_tips, all_labels, all_geo_locs, all_landslide_ids, config
                    )
                    # キャッシュに保存（CV関数内で再利用）
                    config['spatial_folds_cache'] = spatial_folds
                    config['spatial_folds_gdf'] = gdf_prj
                    
                    n_actual_folds = len(np.unique(spatial_folds))
                    for fold in range(n_actual_folds):
                        val_idx = np.where(spatial_folds == fold)[0]
                        train_idx = np.where(spatial_folds != fold)[0]
                        
                        diagnose_train_val_distribution(
                            train_tips=all_tips[train_idx],
                            val_tips=all_tips[val_idx],
                            train_labels=all_labels[train_idx],
                            val_labels=all_labels[val_idx],
                            thresholds=config['fixed_thresholds'],
                            n_classes=config['n_classes'],
                            channel_names=ch_names,
                            fold_num=fold + 1
                        )
                except Exception as e:
                    print(f"  ⚠️ Diagnosis skipped due to error: {e}")
            elif config.get('diagnose_distribution', True):
                print("  (Stratified K-Foldでは各Foldのクラス比率は均等です)")
            
            # 改善版CV関数を使用（Focal Loss、Mixup/CutMix、Early Stopping等を含む）
            cv_results = cross_validate_with_improvements(
                model_name=model_name,
                hyperparams=best_params,
                images=all_tips,
                labels=all_labels,
                pixel_locations=all_pixel_locs,
                geo_locations=all_geo_locs,
                landslide_ids=all_landslide_ids,
                n_classes=config['n_classes'],
                n_folds=config['n_outer_cv'],
                n_epochs=config['n_epochs'],
                batch_size=config['batch_size'],
                device=config['device'],
                save_dir=str(model_output_dir),
                optimize_threshold=config['optimize_threshold'],
                use_spatial_kfold=config['use_spatial_kfold'],
                config=config,
                patch_size=config.get('patch_size', 100)
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
    
    # ============================================================
    # 誤分類領域の可視化（元画像オーバーレイ + QGIS用データ）
    # ============================================================
    print(f"\n{'='*80}")
    print("🗺️  Creating Misclassification Overlay Visualizations")
    print(f"{'='*80}")
    
    # クラス名の生成
    class_names = generate_class_names_from_thresholds(
        config['fixed_thresholds'], 
        config['n_classes']
    )
    
    for model_name, result in results.items():
        predictions_path = result.get('predictions_path')
        if predictions_path and Path(predictions_path).exists():
            try:
                predictions_df = pd.read_csv(predictions_path)
                
                # 座標列の名前を統一
                if 'geo_x' in predictions_df.columns and 'x' not in predictions_df.columns:
                    predictions_df['x'] = predictions_df['geo_x']
                    predictions_df['y'] = predictions_df['geo_y']
                
                # dataset_name列の追加（なければ）
                if 'dataset_name' not in predictions_df.columns:
                    predictions_df['dataset_name'] = 'all'
                
                model_output_dir = config['output_dir'] / model_name
                
                viz_results = visualize_misclassification_on_image(
                    predictions_df=predictions_df,
                    rgb_images={},  # RGB画像は別途読み込み可能
                    save_dir=model_output_dir,
                    model_name=model_name,
                    class_names=class_names,
                    crs=f"EPSG:{config['epsg_code']}",
                    output_gpkg=True,
                    output_geotiff=True,
                    grid_resolution=100
                )
                
                print(f"  ✅ {model_name}: Visualization complete")
                
            except Exception as e:
                print(f"  ⚠️ {model_name}: Visualization failed - {e}")
        else:
            print(f"  ⚠️ {model_name}: No predictions file found")
    
    # ============================================================
    # 予測結果のLegacy Ratio Overlay可視化
    # ============================================================
    print(f"\n{'='*80}")
    print("🗺️  Creating Legacy Prediction Overlay Visualizations")
    print(f"{'='*80}")
    
    # 崩壊地ポリゴンの読み込み
    try:
        landslide_gdf = gpd.read_file(config['landslide_path'])
        landslide_gdf = landslide_gdf.to_crs(epsg=config['epsg_code'])
        print(f"  📦 Loaded {len(landslide_gdf)} landslide polygons")
    except Exception as e:
        print(f"  ⚠️ Failed to load landslide polygons: {e}")
        landslide_gdf = None
    
    for model_name, result in results.items():
        predictions_path = result.get('predictions_path')
        if predictions_path and Path(predictions_path).exists():
            try:
                predictions_df = pd.read_csv(predictions_path)
                
                # 座標列の名前を統一
                if 'geo_x' in predictions_df.columns and 'x' not in predictions_df.columns:
                    predictions_df['x'] = predictions_df['geo_x']
                    predictions_df['y'] = predictions_df['geo_y']
                
                model_output_dir = config['output_dir'] / model_name
                
                legacy_viz_results = visualize_prediction_legacy_overlay(
                    predictions_df=predictions_df,
                    landslide_gdf=landslide_gdf,
                    save_dir=model_output_dir,
                    model_name=model_name,
                    n_classes=config['n_classes'],
                    fixed_thresholds=config['fixed_thresholds'],
                    crs=f"EPSG:{config['epsg_code']}",
                )
                
                print(f"  ✅ {model_name}: Legacy overlay complete")
                
            except Exception as e:
                print(f"  ⚠️ {model_name}: Legacy overlay failed - {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ⚠️ {model_name}: No predictions file found")
    
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

# /home/ge/Documents/GLCM/xx/xx_contrast.npy
# /xx_correlation.npy
# /xx_disimilartiy.npy
# /xx_energy.npy
# /xx_homogeneity.py