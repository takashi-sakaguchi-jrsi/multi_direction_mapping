"""
管内カメラカーシミュレーション用壁面画像生成処理

このパッケージは、魚眼レンズカメラで撮影した管内動画から
カラーマップ（管路壁面展開図）を生成する機能を提供します。

主要モジュール:
- config: 設定管理
- ocr_utils: OCR距離読み取り
- coordinate_transform: 座標変換処理
- feature_matching: 特徴点マッチング
- camera_estimation: カメラ位置・姿勢推定
- colormap_generator: カラーマップ生成
- progress_reporter: 進捗レポート
- debug_visualizer: デバッグ可視化
- main: エントリーポイント
"""

__version__ = "1.0.0"
__author__ = "CameraCarSim Development Team"

# パッケージレベルのエクスポート
__all__ = [
    "config",
    "ocr_utils",
    "coordinate_transform",
    "feature_matching",
    "camera_estimation",
    "colormap_generator",
    "progress_reporter",
    "debug_visualizer",
    "main",
]
