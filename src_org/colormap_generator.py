"""カラーマップ生成モジュール

管内カメラカーシミュレーション用のカラーマップ（管路壁面展開図）生成機能を提供します。
フレーム画像とカメラ状態から、管路壁面の展開カラーマップを生成・更新します。

主な機能:
- 動的Z範囲計算: カメラ位置に応じて描画範囲を自動調整
- η-zグリッド生成: 円筒座標系のグリッド座標を生成
- 色サンプリング: フレーム画像から指定座標の色を双線形補間で抽出
- カラーマップ更新: 既存のカラーマップに新しいフレームの色情報を上書き統合

座標系の定義:
- η（エータ）: 円周方向の角度（0 <= η < 2π）、上方向を0として時計回りに増加
- z: 管路延長方向の位置（mm）
- カラーマップ: 縦軸η、横軸zの2次元画像（RGB、plt.imsave用）
"""

import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
import scipy.ndimage

from src.coordinate_transform import CoordinateTransformer
from src.config import Config, ColormapConfig


# ============================================================================
# カスタム例外クラス
# ============================================================================

class ColorMapGenerationError(Exception):
    """カラーマップ生成関連のベース例外"""
    pass


class InvalidZRangeError(ColorMapGenerationError):
    """無効なZ範囲エラー"""
    pass


class InvalidGridSizeError(ColorMapGenerationError):
    """無効なグリッドサイズエラー"""
    pass


# ============================================================================
# ColorMapGenerator クラス
# ============================================================================

class ColorMapGenerator:
    """カラーマップ生成クラス
    
    フレーム画像とカメラ状態から、管路壁面の展開カラーマップを生成します。
    
    Attributes:
        config: カラーマップ生成設定
        transformer: 座標変換器
        pipe_radius: 管路半径（mm）
        eta_resolution: 円周方向の解像度（ピクセル数）
        pixels_per_mm: z方向の解像度（ピクセル/mm）
        min_radius_ratio: 内側トリミング半径比率
        max_radius_ratio: 外側トリミング半径比率
        alpha: Z範囲拡張係数（デフォルト2.0）
        logger: ロガー
    """
    
    def __init__(
        self,
        config: Config,
        coordinate_transformer: CoordinateTransformer
    ):
        """初期化
        
        Args:
            config: 設定情報（カラーマップサイズ、FOV等）
            coordinate_transformer: 座標変換器
        
        Raises:
            InvalidGridSizeError: eta_resolutionが不正な場合
        """
        self.config = config
        self.transformer = coordinate_transformer
        self.pipe_radius = config.pipe.diameter_mm / 2.0
        
        # カラーマップ解像度設定
        # eta_resolutionは円周方向のピクセル数（例: 1000ピクセル = 2π）
        # 円周長 = 2 * π * R、解像度 = pixels_per_mm より
        # eta_resolution = 2 * π * R * pixels_per_mm
        circumference_mm = 2 * np.pi * self.pipe_radius
        self.pixels_per_mm = config.colormap.pixels_per_mm
        self.eta_resolution = int(np.ceil(circumference_mm * self.pixels_per_mm))
        
        if self.eta_resolution <= 0:
            raise InvalidGridSizeError(
                f"eta_resolutionは正の整数である必要があります: {self.eta_resolution}"
            )
        
        # ドーナツ状のトリミング範囲（半径比率）
        self.min_radius_ratio = config.colormap.inner_radius_ratio
        self.max_radius_ratio = config.colormap.outer_radius_ratio

        # Z範囲拡張係数（Legacy実装では2.0固定）
        self.alpha = 2.0

        # 初期Z座標オフセット（カラーマップの左端を0にするため）
        self.z_offset: Optional[float] = None


        # TASK-24: max_z_length_per_frame が設定されている場合はそれを優先
        if config.colormap.max_z_length_per_frame is not None:
            self.max_z_pixels = config.colormap.max_z_length_per_frame * self.pixels_per_mm
        else:
            # 後方互換: 旧パラメータを使用
            self.max_z_pixels = config.colormap.max_z_pixels_per_frame
        # ロガー
        self.logger = logging.getLogger(__name__)
        
        # Task 2.5.2: Z長さ制限の統計情報
        self.z_limit_stats = {
            'total_frames': 0,
            'limited_frames': 0,
            'max_z_length_px': 0.0,
            'total_z_length_px': 0.0
        }
        
        self.logger.info(
            f"ColorMapGenerator initialized: eta_resolution={self.eta_resolution}, "
            f"pixels_per_mm={self.pixels_per_mm}, "
            f"radius_ratio=[{self.min_radius_ratio:.2f}, {self.max_radius_ratio:.2f}], "
            f"z_limit_enabled={self.config.colormap.enable_z_length_limit}, "
            f"max_z_pixels={self.max_z_pixels}"
        )
    
    def initialize_colormap(
        self,
        z_range: Tuple[float, float]
    ) -> np.ndarray:
        """カラーマップ初期化
        
        Args:
            z_range: z軸の範囲 (z_min, z_max) [mm]
        
        Returns:
            colormap: 初期化されたカラーマップ (height, width, 3) uint8
        
        Raises:
            InvalidZRangeError: z_rangeが不正な場合
        """
        z_min, z_max = z_range
        
        if z_max <= z_min:
            raise InvalidZRangeError(
                f"z_maxはz_minより大きい必要があります: z_min={z_min}, z_max={z_max}"
            )
        
        # z方向のピクセル数
        z_width_px = int(np.ceil((z_max - z_min) * self.pixels_per_mm))
        
        # カラーマップ生成（縦: eta_resolution、横: z_width_px、RGB）
        colormap = np.zeros((self.eta_resolution, z_width_px, 3), dtype=np.uint8)
        
        self.logger.info(
            f"Colormap initialized: shape={colormap.shape}, "
            f"z_range=[{z_min:.1f}, {z_max:.1f}] mm"
        )
        
        return colormap
    
    def calculate_dynamic_z_range(
        self,
        camera_z: float,
        camera_params: Dict[str, Any],
        camera_state: Dict[str, Any],
        max_dz: float,
        alpha: float
    ) -> Tuple[float, float]:
        """動的Z範囲計算

        カメラ位置とドーナツ外周上のピクセル座標から、フレームで描画すべき
        z範囲を動的に計算します。

        Args:
            camera_z: カメラのz座標 [mm]
            camera_params: カメラパラメータ（center, radius等）
            camera_state: カメラ状態（position, orientation）
            max_dz: 1フレーム間の最大前進距離 [mm]（motion.max_dzから取得）
            alpha: z方向変形量に対する最大増加率（例: 1.2）

        Returns:
            z_min: Z範囲の最小値 [mm]
            z_max: Z範囲の最大値 [mm]

        Raises:
            InvalidZRangeError: 計算されたZ範囲が不正な場合

        処理の流れ:
            1. ドーナツ外周上の座標（360点）を生成
            2. ピクセル→ワールド座標（z座標のみ）に変換
            3. z座標の最小値・最大値を取得
            4. 先行描画のために以下の式でz_maxを拡張:
               z_line_2_max = z_line_1_max + max_dz + (z_line_1_max - z_line_1_min + 1) * alpha
        """
        center = camera_params['center']
        radius = camera_params['radius']
        max_radius_px = radius * self.max_radius_ratio

        # ドーナツ外周上の座標（1度刻みで360点）
        angles = np.deg2rad(np.arange(0, 360))
        edge_x = center[0] + max_radius_px * np.cos(angles)
        edge_y = center[1] + max_radius_px * np.sin(angles)
        edge_pixels = np.stack((edge_x, edge_y), axis=-1)

        # ピクセル→ワールド座標（円筒面上の3D座標）
        cam_pos = camera_state['position']
        roll = camera_state['orientation'][0]
        yaw = camera_state['orientation'][1]
        pitch = camera_state['orientation'][2]

        edge_world = self.transformer.pixel_to_world(
            edge_pixels, cam_pos, roll, yaw, pitch
        )

        # z座標の範囲を取得
        z_coords = edge_world[:, 2]
        z_line_1_min = np.min(z_coords)
        z_line_1_max = np.max(z_coords)

        # 先行描画のためにz_maxを拡張
        # max_dz: カメラの最大移動量
        # (z_line_1_max - z_line_1_min + 1) * alpha: 姿勢変化による変形量の増加を吸収
        z_line_2_max = z_line_1_max + max_dz + (z_line_1_max - z_line_1_min + 1) * alpha
        z_max_actual = z_line_2_max - z_line_1_min

        if z_max_actual <= 0:
            raise InvalidZRangeError(
                f"計算されたZ範囲が不正です: z_max={z_max_actual:.1f} mm"
            )

        self.logger.debug(
            f"Dynamic Z range: z_min={z_line_1_min:.1f}, "
            f"z_max={z_line_2_max:.1f}, delta={z_max_actual:.1f} mm, "
            f"deformation={(z_line_1_max - z_line_1_min):.1f} mm"
        )

        return z_line_1_min, z_line_2_max
    
    def generate_eta_z_grid(
        self,
        z_range: Tuple[float, float],
        colormap_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """η-zグリッド生成
        
        Args:
            z_range: z軸の範囲 (z_min, z_max) [mm]
            colormap_shape: カラーマップサイズ (height, width)
        
        Returns:
            eta_grid: η座標グリッド (height, width) [ラジアン]
            z_grid: z座標グリッド (height, width) [mm]
        
        処理の流れ:
            1. z範囲を等間隔に分割（widthピクセル分）
            2. η範囲を等間隔に分割（heightピクセル分、0～2π）
            3. π/2シフトして上方向を0にする
            4. meshgridでグリッドを生成
        """
        z_min, z_max = z_range
        height, width = colormap_shape
        
        # z方向の等間隔分割
        z_range_array = np.linspace(z_min, z_max, num=width)
        
        # η方向の等間隔分割（0～2π、endpoint=Falseで2πを含まない）
        # π/2シフトして上方向（y=+R）を0にする
        eta_range_array = np.linspace(
            0, 2 * np.pi, num=height, endpoint=False
        ) + (np.pi / 2)
        
        # meshgridでグリッド生成（indexing='ij'でeta, z順）
        ETA, Z = np.meshgrid(eta_range_array, z_range_array, indexing='ij')
        
        self.logger.debug(
            f"Grid generated: shape={ETA.shape}, "
            f"eta_range=[{eta_range_array[0]:.3f}, {eta_range_array[-1]:.3f}], "
            f"z_range=[{z_min:.1f}, {z_max:.1f}]"
        )
        
        return ETA, Z
    
    def sample_colors_from_frame(
        self,
        frame: np.ndarray,
        eta_grid: np.ndarray,
        z_grid: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """フレームから色サンプリング
        
        Args:
            frame: 入力フレーム画像 (H, W, 3) BGR
            eta_grid: η座標グリッド (height, width) [ラジアン]
            z_grid: z座標グリッド (height, width) [mm]
            camera_state: カメラ状態（position, orientation）
            camera_params: カメラパラメータ（center, radius等）
        
        Returns:
            colors: サンプリングされた色 (height, width, 3) uint8 RGB
            mask: 有効ピクセルのマスク (height, width) bool
        
        処理の流れ:
            1. η-zグリッドからワールド座標（円筒面上の3D座標）を計算
            2. ワールド座標→ピクセル座標に変換
            3. フレーム画像から色を双線形補間でサンプリング
            4. ドーナツ状のマスクを適用（有効範囲の限定）
        """
        height, width = eta_grid.shape
        
        # η-z → ワールド座標（円筒面上）
        X_wall = self.pipe_radius * np.cos(eta_grid)
        Y_wall = self.pipe_radius * np.sin(eta_grid)
        world_points = np.stack((X_wall, Y_wall, z_grid), axis=-1)  # (H, W, 3)
        
        # (H, W, 3) → (N, 3) に平坦化
        points_3d_flat = world_points.reshape(-1, 3)
        
        # ワールド座標→ピクセル座標
        cam_pos = camera_state['position']
        roll = camera_state['orientation'][0]
        yaw = camera_state['orientation'][1]
        pitch = camera_state['orientation'][2]
        
        pixels = self.transformer.world_to_pixel(
            points_3d_flat, cam_pos, roll, yaw, pitch,
            apply_distortion=True
        )
        
        # 双線形補間で色をサンプリング
        map_x = pixels[:, 0].astype(np.float32)
        map_y = pixels[:, 1].astype(np.float32)
        
        # scipy.ndimage.map_coordinatesで補間（BGR → RGB変換）
        color_patch_flat = np.stack([
            scipy.ndimage.map_coordinates(
                frame[:, :, c], [map_y, map_x], order=1, mode='reflect'
            )
            for c in [2, 1, 0]  # BGR → RGB の順（plt.imsave用）
        ], axis=-1).astype(np.uint8)
        
        # (N, 3) → (H, W, 3) に復元
        colors = color_patch_flat.reshape(height, width, 3)
        
        # ドーナツ状のマスク（有効範囲の限定）
        center = camera_params['center']
        radius = camera_params['radius']
        min_radius_px = radius * self.min_radius_ratio
        max_radius_px = radius * self.max_radius_ratio
        
        radial_sq = (pixels[:, 0] - center[0])**2 + (pixels[:, 1] - center[1])**2
        valid_mask = (
            (radial_sq >= min_radius_px**2) & (radial_sq <= max_radius_px**2)
        )
        valid_mask_2d = valid_mask.reshape(height, width)
        
        valid_count = np.sum(valid_mask_2d)
        total_count = height * width
        self.logger.debug(
            f"Sampled colors: valid={valid_count}/{total_count} "
            f"({100.0*valid_count/total_count:.1f}%)"
        )
        
        return colors, valid_mask_2d
    
    def _apply_z_length_limit(
        self,
        z_min: float,
        z_max: float,
        frame_num: Optional[int] = None
    ) -> Tuple[float, float]:
        """Z方向長さ制限を適用
        
        Args:
            z_min: 元のZ範囲最小値 [mm]
            z_max: 元のZ範囲最大値 [mm]
            frame_num: フレーム番号（ログ用、オプション）
        
        Returns:
            z_min_limited: 制限後のZ範囲最小値 [mm]
            z_max_limited: 制限後のZ範囲最大値 [mm]
        
        Notes:
            - enable_z_length_limitがFalseの場合は元の値をそのまま返す
            - 制限を適用した場合はワーニングログを出力
            - 統計情報を更新
        """
        if not self.config.colormap.enable_z_length_limit:
            # 制限無効時は元の値をそのまま返す
            return z_min, z_max
        
        # Z方向長さをピクセル単位で計算
        z_length_mm = z_max - z_min
        z_length_px = z_length_mm * self.pixels_per_mm
        
        # 統計情報を更新
        self.z_limit_stats['total_frames'] += 1
        self.z_limit_stats['total_z_length_px'] += z_length_px
        self.z_limit_stats['max_z_length_px'] = max(
            self.z_limit_stats['max_z_length_px'], z_length_px
        )
        
        # 制限値（TASK-24: max_z_length_per_frame優先）
        max_z_px = self.max_z_pixels
        
        # 制限超過チェック
        if z_length_px <= max_z_px:
            # 制限内なのでそのまま返す
            return z_min, z_max
        
        # 制限を適用
        self.z_limit_stats['limited_frames'] += 1
        
        # 制限後のZ方向長さ [mm]
        z_length_limited_mm = max_z_px / self.pixels_per_mm
        
        # z_minを維持して、z_maxを縮小
        z_max_limited = z_min + z_length_limited_mm
        
        # ワーニングログを出力
        frame_info = f"frame {frame_num}" if frame_num is not None else "frame"
        self.logger.warning(
            f"Z length limit applied ({frame_info}): "
            f"original={z_length_px:.1f}px ({z_length_mm:.1f}mm), "
            f"limited={max_z_px:.1f}px ({z_length_limited_mm:.1f}mm), "
            f"reduction={100*(1-max_z_px/z_length_px):.1f}%"
        )
        
        return z_min, z_max_limited
    
    def update_colormap(
        self,
        colormap: np.ndarray,
        new_colors: np.ndarray,
        mask: np.ndarray,
        start_col: int
    ) -> np.ndarray:
        """カラーマップ更新
        
        Args:
            colormap: 既存のカラーマップ (eta_res, Z_width, 3)
            new_colors: 新しい色情報 (height, width, 3)
            mask: 有効ピクセルのマスク (height, width) bool
            start_col: カラーマップ上の書き込み開始列インデックス
        
        Returns:
            updated_colormap: 更新後のカラーマップ (eta_res, Z_width', 3)
        
        処理の流れ:
            1. 必要に応じてカラーマップを右方向に拡張
            2. マスクで有効なピクセルのみを抽出
            3. カラーマップの対応位置に上書き
        """
        height, width = new_colors.shape[:2]
        
        # カラーマップの初期化（初回のみ）
        if colormap is None:
            colormap = np.zeros(
                (self.eta_resolution, start_col + width, 3), dtype=np.uint8
            )
            self.logger.info(
                f"Colormap created: shape={colormap.shape}"
            )
        
        # 必要に応じてカラーマップを右方向に拡張
        current_width = colormap.shape[1]
        if start_col + width > current_width:
            extra_cols = start_col + width - current_width
            pad = np.zeros((self.eta_resolution, extra_cols, 3), dtype=np.uint8)
            colormap = np.concatenate([colormap, pad], axis=1)
            self.logger.debug(
                f"Colormap extended: {current_width} -> {colormap.shape[1]} px"
            )
        
        # 有効ピクセルのみを抽出
        if not np.any(mask):
            self.logger.warning("No valid pixels to update colormap")
            return colormap
        
        color_patch_filtered = new_colors[mask]
        valid_indices = np.where(mask)
        rows = valid_indices[0]
        cols = valid_indices[1] + start_col
        
        # カラーマップに上書き
        colormap[rows, cols] = color_patch_filtered
        
        self.logger.debug(
            f"Colormap updated: {len(rows)} pixels written at col {start_col}"
        )
        
        return colormap
    
    def add_frame(
        self,
        frame: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any],
        color_map_img: Optional[np.ndarray] = None,
        max_dz: float = 100.0,
        alpha: float = 1.2
    ) -> np.ndarray:
        """フレームをカラーマップに追加

        Args:
            frame: 入力フレーム画像 (H, W, 3) BGR
            camera_state: カメラ状態（position, orientation）
            camera_params: カメラパラメータ（center, radius等）
            color_map_img: 既存のカラーマップ（None可）
            max_dz: 1フレーム間の最大前進距離 [mm]（motion.max_dzから取得）
            alpha: z方向変形量に対する最大増加率（例: 1.2）

        Returns:
            color_map_img: 更新後のカラーマップ (eta_res, Z_width, 3)

        処理の流れ:
            1. 動的Z範囲を計算
            2. η-zグリッドを生成
            3. フレームから色をサンプリング
            4. カラーマップを更新

        Raises:
            InvalidZRangeError: Z範囲が不正な場合
        """
        camera_z = camera_state['position'][2]

        # 初回のみ: Z座標オフセットを記録（カラーマップの左端を0にするため）
        if self.z_offset is None:
            self.z_offset = camera_z
            self.logger.info(f"Z offset initialized: {self.z_offset:.2f} mm")

        # (1) 動的Z範囲計算
        z_min, z_max = self.calculate_dynamic_z_range(
            camera_z, camera_params, camera_state, max_dz, alpha
        )
        
        # Task 2.5.2: Z方向長さ制限を適用
        # frame_numは外部から渡されていないため、Noneを渡す（将来的に追加可能）
        z_min, z_max = self._apply_z_length_limit(z_min, z_max, frame_num=None)

        # Z範囲が不正な場合はスキップ
        z_range = z_max - z_min
        if z_range <= 0:
            self.logger.warning(
                f"Invalid Z range: z_min={z_min:.1f}, z_max={z_max:.1f}"
            )
            return color_map_img

        # フレーム幅（ピクセル）
        frame_width_px = int(np.ceil(z_range * self.pixels_per_mm))

        # (2) η-zグリッド生成
        eta_grid, z_grid = self.generate_eta_z_grid(
            (z_min, z_max), (self.eta_resolution, frame_width_px)
        )

        # (3) 色サンプリング
        colors, mask = self.sample_colors_from_frame(
            frame, eta_grid, z_grid, camera_state, camera_params
        )

        # (4) カラーマップ更新（z_offsetを基準とした相対位置で描画）
        start_col = int(np.floor((z_min - self.z_offset) * self.pixels_per_mm))
        color_map_img = self.update_colormap(
            color_map_img, colors, mask, start_col
        )

        return color_map_img
    
    def get_z_limit_stats(self) -> Dict[str, Any]:
        """Z長さ制限の統計情報を取得
        
        Returns:
            stats: 統計情報辞書
                - total_frames: 処理フレーム総数
                - limited_frames: 制限を適用したフレーム数
                - max_z_length_px: 最大Z方向長さ（ピクセル）
                - avg_z_length_px: 平均Z方向長さ（ピクセル）
                - limit_ratio: 制限適用率（%）
        """
        stats = self.z_limit_stats.copy()
        
        if stats['total_frames'] > 0:
            stats['avg_z_length_px'] = stats['total_z_length_px'] / stats['total_frames']
            stats['limit_ratio'] = 100.0 * stats['limited_frames'] / stats['total_frames']
        else:
            stats['avg_z_length_px'] = 0.0
            stats['limit_ratio'] = 0.0
        
        return stats
    
    def log_z_limit_stats(self) -> None:
        """Z長さ制限の統計情報をログ出力"""
        stats = self.get_z_limit_stats()
        
        if stats['total_frames'] == 0:
            self.logger.info("Z length limit stats: No frames processed")
            return
        
        self.logger.info(
            f"Z length limit stats: "
            f"total_frames={stats['total_frames']}, "
            f"limited_frames={stats['limited_frames']} ({stats['limit_ratio']:.1f}%), "
            f"max_z_length={stats['max_z_length_px']:.1f}px, "
            f"avg_z_length={stats['avg_z_length_px']:.1f}px"
        )

