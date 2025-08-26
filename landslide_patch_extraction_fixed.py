import os, zipfile, io
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm
from pyprojroot import here
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import gc
import joblib
import time
from datetime import datetime, timedelta
# geospatial
import geopandas as gpd
from osgeo import gdal
import rasterio as rio
from rasterio.windows import Window
from rasterio.plot import show
from rasterio.crs import CRS
from shapely.geometry import Point, box
# torch
import torch


# *********************************************************************
# Functions
# *********************************************************************
def jbl_saver(object, filename, compress = 3):
    with open(filename, 'wb') as web:
        joblib.dump(object , web, compress=compress)


def identify_point_within_landslide(landslide_gdf, vrtname, region_name, patch_size, chunk_size=10, profile=False):
    print(f'{region_name}, identifying points within landslide (chunked, no parallel)...')
    point_ls_name = f'{path_to_tmp}/{region_name}-point-within-landslide-mask-{patch_size}.parquet'
    if os.path.exists(point_ls_name):
        return gpd.read_parquet(point_ls_name)

    with rio.open(vrtname) as src:
        # Ensure CRS match
        if landslide_gdf.crs != src.crs:
            print(f'CRS mismatch: {landslide_gdf.crs} vs {src.crs}')
            landslide_gdf = landslide_gdf.to_crs(src.crs)

        crs = src.crs
        nodata = src.nodata
        vrt_bounds = box(*src.bounds)
        landslide_gdf = gpd.clip(landslide_gdf, gpd.GeoDataFrame(geometry=[vrt_bounds], crs=crs))

        if landslide_gdf.empty:
            print("No landslide polygons intersect with VRT.")
            return None

        # チャンクごとに処理
        from more_itertools import chunked
        chunks = list(chunked(landslide_gdf.geometry.tolist(), chunk_size))
        results, prof_records = [], [] 

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            t0 = time.perf_counter()
            try:
                from rasterio.mask import mask
                out_image, out_transform = mask(src, chunk, crop=True)
                out_image = out_image[0]  # use the first band
                nrow, ncol = out_image.shape
                rows, cols = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing='ij')

                mask_valid = out_image != nodata if nodata is not None else ~np.isnan(out_image)
                keep = ((rows % patch_size) == patch_size // 2) & ((cols % patch_size) == patch_size // 2)
                mask_final = mask_valid & keep

                row_idx = rows[mask_final]
                col_idx = cols[mask_final]
                xs = out_transform.a * col_idx + out_transform.c
                ys = out_transform.e * row_idx + out_transform.f

                if len(xs) > 0:
                    df = pd.DataFrame({'row': row_idx, 'col': col_idx})
                    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(xs, ys), crs=crs)
                    results.append(gdf)
                    if profile:
                        prof_records.append({
                            "chunk_id": i,
                            "n_polys" : len(chunk),
                            "n_points": len(gdf),
                            "rows"    : nrow,
                            "cols"    : ncol,
                            "sec"     : time.perf_counter() - t0,
                        })
                del out_image, mask_valid, mask_final, rows, cols, row_idx, col_idx, xs, ys
                gc.collect()

            except Exception as e:
                print(f"Skipping chunk {i+1} due to error: {e}")
                continue
        prof_df = pd.DataFrame(prof_records) if profile else None
        if results:
            full_gdf = pd.concat(results, ignore_index=True)
            full_gdf.to_parquet(point_ls_name, index=False)
            return full_gdf, prof_df
        else:
            print("No valid points found.")
            return None, prof_df


def identify_point_within_landslide_optimized(landslide_gdf, vrtname, region_name, patch_size, stride=None):
    """最適化版（並列処理なし）"""
    if stride is None:
        stride = patch_size
    
    point_ls_name = f'{path_to_tmp}/{region_name}-point-within-landslide-mask-{patch_size}-stride{stride}.parquet'
    
    if os.path.exists(point_ls_name):
        return gpd.read_parquet(point_ls_name)

    with rio.open(vrtname) as src:
        if landslide_gdf.crs != src.crs:
            landslide_gdf = landslide_gdf.to_crs(src.crs)

        crs = src.crs
        vrt_bounds = box(*src.bounds)
        landslide_gdf = gpd.clip(landslide_gdf, gpd.GeoDataFrame(geometry=[vrt_bounds], crs=crs))

        if landslide_gdf.empty:
            return None

        results = []
        
        # 進捗バー
        pbar = tqdm(landslide_gdf.geometry, 
                   desc=f"{region_name}: ポリゴン処理",
                   position=1,
                   leave=False)
        
        for polygon in pbar:
            try:
                minx, miny, maxx, maxy = polygon.bounds
                row_min, col_min = src.index(minx, maxy)
                row_max, col_max = src.index(maxx, miny)
                
                # ストライドを使用してサンプリング
                row_centers = np.arange(row_min + patch_size//2, row_max, stride)
                col_centers = np.arange(col_min + patch_size//2, col_max, stride)
                
                for row in row_centers:
                    for col in col_centers:
                        x, y = src.xy(row, col)
                        point = Point(x, y)
                        
                        if polygon.contains(point):
                            results.append({
                                'row': row,
                                'col': col,
                                'geometry': point
                            })
                
            except Exception as e:
                continue
        
        pbar.close()
        
        if results:
            full_gdf = gpd.GeoDataFrame(results, crs=crs)
            full_gdf.to_parquet(point_ls_name, index=False)
            return full_gdf
        else:
            return None


def make_dataset(point_ls_gdf, vrtname, region_name, patch_size, th, max_cntr):
    """修正版：実際にpickleファイルを保存する"""
    data4model = []
    total_points = point_ls_gdf.shape[0]
    
    print(f"\n{region_name}のデータセット作成開始:")
    print(f"  総ポイント数: {total_points:,}")
    print(f"  パッチサイズ: {patch_size}x{patch_size}")
    print(f"  バッチサイズ: {max_cntr:,}")
    
    with rio.open(vrtname) as src:
        data_cntr, subset_id = 0, 0
        valid_patches = 0
        skipped_patches = 0
        
        # プログレスバー
        pbar = tqdm(range(total_points), 
                   desc=f"{region_name}のパッチ抽出",
                   position=1,
                   leave=False)
        
        for i in range(total_points):
            # 画像の左上のrow, colと中心の座標を取得
            row_center, col_center = point_ls_gdf.iloc[i].row, point_ls_gdf.iloc[i].col
            row_start, col_start = row_center - patch_size // 2, col_center - patch_size // 2
            lon_center, lat_center = point_ls_gdf.iloc[i].geometry.coords[0]
            
            try:
                # 該当する場所のみデータを読み込む
                win = Window.from_slices((row_start, row_start + patch_size), (col_start, col_start + patch_size))
                w = src.read(window=win)
                
                if w.shape[1] != patch_size or w.shape[2] != patch_size:
                    skipped_patches += 1
                    pbar.update(1)
                    continue
                
                # 輝度が一定以下の影の場所はマスクして黒色にする
                w_mod = w.copy()
                brightness_mask = (np.mean(w/250, axis=0) > th)
                w_mod *= brightness_mask[np.newaxis, :, :]
                
                # 完全に白または黒の画像をスキップ
                if (w_mod.min() == 255 and w_mod.max() == 255) or (w_mod.min() == 0 and w_mod.max() == 0):
                    skipped_patches += 1
                    pbar.update(1)
                    continue
                
                # 画像データを取得
                data4model.append({'image': torch.tensor(w_mod/255, dtype=torch.float32), 
                                 'lon': lon_center, 
                                 'lat': lat_center})
                valid_patches += 1
                data_cntr += 1
                
                # バッチが満杯になったら保存
                if data_cntr >= max_cntr:
                    save_path = f'{path_to_all_data}/{region_name}-subset-{subset_id}.pickle'
                    jbl_saver(data4model, save_path)
                    print(f"\n  保存: {save_path} ({len(data4model)}パッチ)")
                    del data4model
                    gc.collect()
                    data4model = []
                    data_cntr = 0
                    subset_id += 1
                    
            except Exception as e:
                skipped_patches += 1
                pass
            
            pbar.update(1)
        
        # 最後のデータを保存する
        if data4model:
            save_path = f'{path_to_all_data}/{region_name}-subset-{subset_id}.pickle'
            jbl_saver(data4model, save_path)
            print(f"\n  保存: {save_path} ({len(data4model)}パッチ)")
            del data4model
            gc.collect()
        
        pbar.close()
        
        print(f"\n{region_name}の処理完了:")
        print(f"  有効パッチ数: {valid_patches:,}")
        print(f"  スキップ数: {skipped_patches:,}")
        print(f"  作成ファイル数: {subset_id + 1}")


def make_dataset_optimized(point_ls_gdf, vrtname, region_name, patch_size, th, max_cntr):
    """メモリ効率を改善したデータセット作成（進捗表示付き）"""
    
    print(f"\n{region_name}のデータセット作成開始（最適化版）:")
    print(f"  総ポイント数: {len(point_ls_gdf):,}")
    
    with rio.open(vrtname) as src:
        n_points = len(point_ls_gdf)
        batch_indices = np.array_split(np.arange(n_points), max((n_points // max_cntr), 1))
        
        total_valid = 0
        total_skipped = 0
        
        # 全サブセット処理の進捗バー
        pbar_subsets = tqdm(enumerate(batch_indices), 
                           total=len(batch_indices),
                           desc=f"{region_name}: サブセット処理",
                           position=1,
                           leave=False)
        
        for subset_id, indices in pbar_subsets:
            data4model = []
            
            # 各ポイント処理の進捗バー
            pbar_points = tqdm(indices, 
                              desc=f"サブセット{subset_id}",
                              position=2,
                              leave=False)
            
            for i in pbar_points:
                row_center = point_ls_gdf.iloc[i].row
                col_center = point_ls_gdf.iloc[i].col
                row_start = row_center - patch_size // 2
                col_start = col_center - patch_size // 2
                
                win = Window.from_slices(
                    (row_start, row_start + patch_size), 
                    (col_start, col_start + patch_size)
                )
                
                try:
                    w = src.read(window=win)
                    
                    if w.shape[1] != patch_size or w.shape[2] != patch_size:
                        total_skipped += 1
                        continue
                    
                    mean_brightness = np.mean(w / 250, axis=0)
                    mask = mean_brightness > th
                    
                    # 完全に暗いまたは明るい画像をスキップ
                    if not np.any(mask) or np.all(mask):
                        total_skipped += 1
                        continue
                    
                    w_mod = w * mask[np.newaxis, :, :]
                    
                    lon, lat = point_ls_gdf.iloc[i].geometry.coords[0]
                    data4model.append({
                        'image': torch.tensor(w_mod / 255, dtype=torch.float32),
                        'lon': lon,
                        'lat': lat
                    })
                    total_valid += 1
                    
                except Exception as e:
                    total_skipped += 1
                    continue
            
            pbar_points.close()
            
            # バッチ保存
            if data4model:
                save_path = f'{path_to_all_data}/{region_name}-subset-{subset_id}.pickle'
                jbl_saver(data4model, save_path)
                tqdm.write(f"  保存: {save_path} ({len(data4model)}パッチ)")
                del data4model
                gc.collect()
        
        pbar_subsets.close()
        
        print(f"\n{region_name}の処理完了:")
        print(f"  有効パッチ数: {total_valid:,}")
        print(f"  スキップ数: {total_skipped:,}")


# *********************************************************************
# Initialization
# *********************************************************************
# Paths
root_dir = here()
data_dir = root_dir / "data"
vrt_dir = data_dir / 'VRT'
vrt_dir.mkdir(exist_ok=True)

# Path to photo tips for prediction
path_to_all_data = data_dir / 'Noto_April'
path_to_tmp = data_dir / 'Noto_April' / 'tmp'
path_to_fig = data_dir / 'Noto_April' / 'fig'

# ディレクトリ作成
for path in [path_to_all_data, path_to_tmp, path_to_fig]:
    path.mkdir(parents=True, exist_ok=True)

print("作業ディレクトリ:")
print(f"  データ保存先: {path_to_all_data}")
print(f"  一時ファイル: {path_to_tmp}")
print(f"  図保存先: {path_to_fig}")

epsg_photo = 6675

# 対象ディレクトリ名リスト
target_blocks = ["noto_1_place", "noto_2_place"]

print("\nVRTファイル作成:")
for block_name in target_blocks:
    block_dir = path_to_all_data / block_name
    vrt_name = vrt_dir / f"{block_name}.vrt"
    
    if vrt_name.exists():
        print(f"  {block_name}: 既存のVRTを使用")
        continue
        
    # jpgとjgwが両方存在するjpgのみを対象
    jpgs = []
    for jpg_path in block_dir.glob("*.jpg"):
        jgw_path = jpg_path.with_suffix(".jgw")
        if jgw_path.exists():
            jpgs.append(str(jpg_path))
    
    if not jpgs:
        print(f"  {block_name}: JPEG+JGW画像なし（スキップ）")
        continue
    
    print(f"  {block_name}: {len(jpgs)}枚の画像からVRT作成")
    # JPEG + JGW をそのままモザイク
    vrt_opts = gdal.BuildVRTOptions(resampleAlg="nearest", addAlpha=True,
                                    outputSRS=f"EPSG:{epsg_photo}")
    gdal.BuildVRT(str(vrt_name), jpgs, options=vrt_opts)


# ------------------------------------------------------------------
# 1) 土砂崩れポリゴンを一つに統合する
# ------------------------------------------------------------------
print("\n土砂崩れポリゴン読み込み:")
landslide_root = data_dir / "noto_landslide"
landslide_paths = list(landslide_root.glob("**/hokai*.geojson"))
print(f"  見つかったファイル数: {len(landslide_paths)}")

if landslide_paths:
    landslide_gdfs = [
        gpd.read_file(p).to_crs(epsg_photo) for p in landslide_paths
    ]
    landslide_gdf = gpd.GeoDataFrame(pd.concat(landslide_gdfs, ignore_index=True))
    landslide_gdf = landslide_gdf[landslide_gdf.is_valid]
    print(f"  総ポリゴン数: {len(landslide_gdf)}")
    
    # ------------------------------------------------------------------
    # 2) 可視化と GeoJSON 出力
    # ------------------------------------------------------------------
    output_geojson = landslide_root / "hokai-all.geojson"
    landslide_gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"  統合ファイル保存: {output_geojson}")
else:
    print("  エラー: 土砂崩れポリゴンファイルが見つかりません")
    raise FileNotFoundError("土砂崩れポリゴンファイルが見つかりません")


# *********************************************************************
# Main処理部分（最適化版）
# *********************************************************************

# パラメータ設定
PATCH = 25
STRIDE = 50  # より大きなストライドで高速化
THRESH = 0.273
BATCH = 50000  # アンダースコアを削除

vrtnames = [vrt_dir / "noto_1_place.vrt", 
            vrt_dir / "noto_2_place.vrt"] 

# 存在するVRTのみ処理
vrtnames = [vrt for vrt in vrtnames if vrt.exists()]
print(f"\n処理対象VRT: {len(vrtnames)}ファイル")

# 処理開始
start_time = time.time()
print(f"\n処理開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"パラメータ:")
print(f"  パッチサイズ: {PATCH}x{PATCH}")
print(f"  ストライド: {STRIDE}")
print(f"  輝度閾値: {THRESH}")
print(f"  バッチサイズ: {BATCH:,}")

# 全体進捗バー
pbar_main = tqdm(enumerate(vrtnames), total=len(vrtnames), desc="全体進捗", position=0)

for region_idx, vrtname in pbar_main:
    region = vrtname.stem
    pbar_main.set_description(f"処理中: {region}")
    
    print(f"\n\n{'='*60}")
    print(f"{region}の処理開始")
    print(f"{'='*60}")
    
    # シンプルな最適化版を使用（並列処理なし）
    pts = identify_point_within_landslide_optimized(
        landslide_gdf, 
        vrtname, 
        region_name=region, 
        patch_size=PATCH,
        stride=STRIDE
    )
    
    if pts is None or pts.empty:
        tqdm.write(f"  -> {region}: 土砂崩れポリゴンと重複なし（スキップ）")
        continue
    
    tqdm.write(f"  -> {region}: {len(pts):,}点を抽出")
    
    # データセット作成（通常版を使用してデバッグ情報を出力）
    make_dataset(
        point_ls_gdf=pts,
        vrtname=str(vrtname),
        region_name=region,
        patch_size=PATCH,
        th=THRESH,
        max_cntr=BATCH
    )
    
    # 時間情報の更新
    elapsed_time = time.time() - start_time
    if region_idx < len(vrtnames) - 1:
        avg_time_per_region = elapsed_time / (region_idx + 1)
        remaining_regions = len(vrtnames) - region_idx - 1
        estimated_remaining = avg_time_per_region * remaining_regions
        eta_time = datetime.now() + timedelta(seconds=estimated_remaining)
        
        pbar_main.set_postfix({
            '経過': f'{elapsed_time/60:.1f}分',
            '残り': f'{estimated_remaining/60:.1f}分',
            '完了予定': eta_time.strftime('%H:%M')
        })

pbar_main.close()

# 処理完了
total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"総処理時間: {total_time/60:.1f}分")
print(f"{'='*60}")

# 作成されたpickleファイルの確認
print("\n作成されたpickleファイル:")
pickle_files = list(path_to_all_data.glob("*.pickle"))
if pickle_files:
    for pf in sorted(pickle_files):
        file_size = pf.stat().st_size / (1024 * 1024)  # MB
        print(f"  {pf.name}: {file_size:.1f} MB")
else:
    print("  なし（pickleファイルが作成されませんでした）")