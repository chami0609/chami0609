# *********************************************************************
# Lbraries
# *********************************************************************
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
# geospatial
import geopandas as gpd
from osgeo import gdal
import rasterio as rio
from rasterio.windows import Window # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
from rasterio.plot import show
from rasterio.crs import CRS
from shapely.geometry import Point
# torch
import torch



# *********************************************************************
# Functions
# *********************************************************************
def jbl_saver(object, filename, compress = 3):
    """安全なファイル保存関数"""
    try:
        # ディレクトリが存在することを確認
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as web:
            joblib.dump(object, web, compress=compress)
        print(f"ファイル保存成功: {filename}")
    except Exception as e:
        print(f"ファイル保存エラー: {filename}, エラー: {e}")
        raise


import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
import gc
from shapely.geometry import box
from shapely.geometry import box
from more_itertools import chunked  # pip install more-itertools
import time

def identify_point_within_landslide(landslide_gdf, vrtname, region_name, patch_size, chunk_size=10, profile=False):
    print(f'{region_name}, identifying points within landslide (chunked, no parallel)...')
    
    # パスの安全な作成
    try:
        # tmpディレクトリが存在することを確認
        os.makedirs(path_to_tmp, exist_ok=True)
        
        # ファイル名を安全に作成（日本語文字を避ける）
        safe_region_name = region_name.replace('/', '_').replace('\\', '_')
        point_ls_name = os.path.join(str(path_to_tmp), f'{safe_region_name}-point-within-landslide-mask-{patch_size}.parquet')
        
        print(f"保存予定パス: {point_ls_name}")
        
        if os.path.exists(point_ls_name):
            print("既存ファイルを読み込み中...")
            return gpd.read_parquet(point_ls_name)
    except Exception as e:
        print(f"パス作成エラー: {e}")
        # フォールバック: 現在のディレクトリに保存
        point_ls_name = f'{safe_region_name}-point-within-landslide-mask-{patch_size}.parquet'
        print(f"フォールバック保存パス: {point_ls_name}")

    with rasterio.open(vrtname) as src:
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
        chunks = list(chunked(landslide_gdf.geometry.tolist(), chunk_size))
        results, prof_records = [], [] 

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            t0 = time.perf_counter()
            try:
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
                            # "rss"  : psutil.Process().memory_info().rss / 1e6
                        })
                del out_image, mask_valid, mask_final, rows, cols, row_idx, col_idx, xs, ys
                gc.collect()

            except Exception as e:
                print(f"Skipping chunk {i+1} due to error: {e}")
                continue
        
        prof_df = pd.DataFrame(prof_records) if profile else None
        
        if results:
            full_gdf = pd.concat(results, ignore_index=True)
            try:
                full_gdf.to_parquet(point_ls_name, index=False)
                print(f"結果を保存しました: {point_ls_name}")
            except Exception as e:
                print(f"Parquet保存エラー: {e}")
                # フォールバック: CSVで保存
                csv_name = point_ls_name.replace('.parquet', '.csv')
                full_gdf.to_csv(csv_name, index=False)
                print(f"CSVで保存しました: {csv_name}")
            
            return full_gdf, prof_df
        else:
            print("No valid points found.")
            return None, prof_df
    

def make_dataset(point_ls_gdf, vrtname, region_name, patch_size, th, max_cntr):
    data4model = []
    print(f'{region_name}, making dataset....')
    
    # 安全なディレクトリ作成
    try:
        os.makedirs(path_to_all_data, exist_ok=True)
        print(f"データディレクトリ確認: {path_to_all_data}")
    except Exception as e:
        print(f"ディレクトリ作成エラー: {e}")
        return
    
    with rio.open(vrtname) as src:
        data_cntr, subset_id = 0, 0
        for i in tqdm(range(point_ls_gdf.shape[0])):
            try:
                # 画像の左上のrow, colと中心の座標を取得
                row_center, col_center = point_ls_gdf.iloc[i].row, point_ls_gdf.iloc[i].col
                row_start, col_start = row_center - patch_size // 2, col_center - patch_size // 2
                lon_center, lat_center = point_ls_gdf.iloc[i].geometry.coords[0]
                
                # 該当する場所のみデータを読み込む
                win = Window.from_slices((row_start, row_start + patch_size), (col_start, col_start + patch_size))
                w = src.read(window=win)
                
                if w.shape[1] != patch_size or w.shape[2] != patch_size:
                    continue
                
                # 輝度が一定以下の影の場所はマスクして黒色にする
                w_mod = w.copy()
                w_mod *= (np.mean(w/250, axis=0) > th)
                
                if (w_mod.min() == 255 and w_mod.max() == 255) or (w_mod.min() == 0 and w_mod.max() == 0):
                    continue
                
                # 画像データを取得
                data4model.append({'image': torch.tensor(w_mod/255), 'lon': lon_center, 'lat': lat_center}) # 0-1に正規化
                
                if data_cntr == max_cntr:
                    print(f'{region_name}, Subset-{subset_id} saving....')
                    safe_filename = os.path.join(str(path_to_all_data), f'{region_name}-subset-{subset_id}.pickle')
                    jbl_saver(np.array(data4model), safe_filename)
                    del data4model; gc.collect()
                    data4model = []
                    data_cntr = 0
                    subset_id += 1
                else:
                    data_cntr += 1
                    
            except Exception as e:
                print(f"データ処理エラー (行 {i}): {e}")
                continue
        
        # 最後のデータを保存する
        if data4model:  # 空でない場合のみ保存
            print(f'{region_name}, Subset-{subset_id} saving....')
            safe_filename = os.path.join(str(path_to_all_data), f'{region_name}-subset-{subset_id}.pickle')
            jbl_saver(np.array(data4model), safe_filename)
            del data4model; gc.collect()


# *********************************************************************
# Initialization
# *********************************************************************
# Paths
root_dir = here()
data_dir = root_dir / "data"
vrt_dir = data_dir / 'VRT'
vrt_dir.mkdir(exist_ok=True)

# Path to photo tips for prediction
#path_to_all_data = data_dir / 'Noto_April'
path_to_all_data = data_dir / 'Noto_September'
#path_to_tmp = data_dir / 'Noto_April/tmp'
path_to_tmp = path_to_all_data / 'tmp'
#path_to_fig = data_dir / 'noto_April/fig'
path_to_fig = path_to_all_data / 'fig'

# ディレクトリの安全な作成
try:
    if not path_to_all_data.exists():
        path_to_all_data.mkdir(parents=True, exist_ok=True)
        print(f"作成: {path_to_all_data}")
    if not path_to_tmp.exists():
        path_to_tmp.mkdir(parents=True, exist_ok=True)
        print(f"作成: {path_to_tmp}")
    if not path_to_fig.exists():
        path_to_fig.mkdir(parents=True, exist_ok=True)
        print(f"作成: {path_to_fig}")
except Exception as e:
    print(f"ディレクトリ作成エラー: {e}")
    print("現在のディレクトリを使用します")
    path_to_all_data = Path(".")
    path_to_tmp = Path("./tmp")
    path_to_fig = Path("./fig")
    path_to_tmp.mkdir(exist_ok=True)
    path_to_fig.mkdir(exist_ok=True)

epsg_photo = 6675

# 対象ディレクトリ名リスト
target_blocks = ["Wajima_east", "Wajima_west"]

for block_name in target_blocks:
    block_dir = path_to_all_data / block_name
    vrt_name = vrt_dir / f"{block_name}.vrt"
    if vrt_name.exists():
        continue
    # jpgとjgwが両方存在するjpgのみを対象
    jpgs = []
    for jpg_path in block_dir.glob("*.jpg"):
        jgw_path = jpg_path.with_suffix(".jgw")
        if jgw_path.exists():
            jpgs.append(str(jpg_path))
    if not jpgs:
        print(f"  -> {block_dir} にJPEG+JGW画像がありません。スキップします。")
        continue
    # JPEG + JGW をそのままモザイク
    vrt_opts = gdal.BuildVRTOptions(resampleAlg="nearest", addAlpha=True,
                                    outputSRS=f"EPSG:{epsg_photo}")
    gdal.BuildVRT(str(vrt_name), jpgs, options=vrt_opts)


# ------------------------------------------------------------------
# 1) 土砂崩れポリゴンを一つに統合する
# ------------------------------------------------------------------
landslide_root   = data_dir / "noto_landslide"          # ベースフォルダ
landslide_paths  = list(landslide_root.glob("**/hokai*.geojson"))
landslide_gdfs = [
    gpd.read_file(p).to_crs(epsg_photo) for p in landslide_paths
]
landslide_gdf  = gpd.GeoDataFrame(pd.concat(landslide_gdfs, ignore_index=True))
landslide_gdf = landslide_gdf[landslide_gdf.is_valid]

# ------------------------------------------------------------------
# 2) 可視化と GeoJSON 出力
# ------------------------------------------------------------------
landslide_gdf.plot();  plt.show()
output_geojson = landslide_root / "hokai-all.geojson"
landslide_gdf.to_file(output_geojson, driver="GeoJSON")

print(f"土砂崩れポリゴン統合完了: {len(landslide_gdf)} 個")

# 抽出パラメータ
PATCH  = 25
THRESH = 0.273
BATCH  = 50_000

vrtnames = [vrt_dir / "Wajima_east.vrt", 
            vrt_dir / "Wajima_west.vrt"] 

for vrtname in vrtnames:
    if not vrtname.exists():
        print(f"VRTファイルが見つかりません: {vrtname}")
        continue
        
    region = vrtname.stem
    print(f"\n=== {region} ===")
    
    try:
        # 1) ポリゴンと重なる中心点を抽出
        result = identify_point_within_landslide(
            landslide_gdf, str(vrtname), region_name=region, patch_size=PATCH, chunk_size=1
        )
        
        if result is None:
            print("  -> 土砂崩れポリゴンと重複なし（スキップ）")
            continue
            
        # 結果がタプルかどうか確認
        if isinstance(result, tuple):
            pts, prof_df = result
        else:
            pts = result
            prof_df = None
            
        if pts is None or pts.empty:
            print("  -> 土砂崩れポリゴンと重複なし（スキップ）")
            continue
        
        print(f"  -> 抽出された点の数: {len(pts)}")
        
        # 2) パッチを切り出し & pickle 保存
        make_dataset(
            point_ls_gdf=pts,
            vrtname=str(vrtname),
            region_name=region,
            patch_size=PATCH,
            th=THRESH,
            max_cntr=BATCH
        )
        
    except Exception as e:
        print(f"エラー発生 ({region}): {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n処理完了")