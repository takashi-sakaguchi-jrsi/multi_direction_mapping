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
        self.keep_painted_pixels = bool(
            getattr(config.colormap, "keep_painted_pixels", False)
            or str(getattr(config.colormap, "blend_mode", "")).lower() == "keep"
        )
        self.prefer_peripheral_pixels = bool(
            getattr(config.colormap, "prefer_peripheral_pixels", False)
        )
        self.reject_z_behind_camera = bool(
            getattr(config.colormap, "reject_z_behind_camera", False)
        )
        front = getattr(config, "front", None)
        pipeline = str(getattr(config, "pipeline", "") or "").lower()
        self.overlap_paste_align = (
            pipeline == "front"
            and bool(getattr(front, "overlap_paste_align", False))
        )
        self.overlap_align_min_pixels = int(
            getattr(front, "overlap_align_min_pixels", 1500) if front else 1500
        )
        self.overlap_align_min_response = float(
            getattr(front, "overlap_align_min_response", 0.04) if front else 0.04
        )
        self.overlap_align_max_dtheta_deg = float(
            getattr(front, "overlap_align_max_dtheta_deg", 8.0) if front else 8.0
        )
        self.overlap_align_max_dz_mm = float(
            getattr(front, "overlap_align_max_dz_mm", 15.0) if front else 15.0
        )
        self.overlap_align_stats = {"applied": 0, "skipped": 0}
        self.strip_overlap_match = (
            pipeline == "front"
            and bool(getattr(front, "strip_overlap_match", False))
        )
        self.strip_overlap_correct = (
            self.strip_overlap_match
            and bool(getattr(front, "strip_overlap_correct", False))
        )
        self.strip_overlap_allow_bin_worse = (
            self.strip_overlap_correct
            and bool(getattr(front, "strip_overlap_allow_bin_worse", False))
        )
        if self.strip_overlap_match:
            self.overlap_paste_align = False
        self.strip_overlap_stats = {
            "frames": 0,
            "ok": 0,
            "corrected": 0,
            "rms_sum": 0.0,
            "records": [],
        }
        self._source_radius: Optional[np.ndarray] = None
        self._peripheral_eps_px = 0.5

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
            f"max_z_pixels={self.max_z_pixels}, "
            f"keep_painted_pixels={self.keep_painted_pixels}, "
            f"prefer_peripheral_pixels={self.prefer_peripheral_pixels}, "
            f"reject_z_behind_camera={self.reject_z_behind_camera}, "
            f"overlap_paste_align={self.overlap_paste_align}, "
            f"strip_overlap_match={self.strip_overlap_match}, "
            f"strip_overlap_correct={self.strip_overlap_correct}, "
            f"allow_bin_worse={self.strip_overlap_allow_bin_worse}"
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            radius: ソース画像中心からの半径 (height, width) [px]
        
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
        
        pixels_ideal = self.transformer.world_to_pixel(
            points_3d_flat, cam_pos, roll, yaw, pitch,
            apply_distortion=False
        )
        pixels = self.transformer.distort_ideal_pixels(pixels_ideal)
        
        # 双線形補間で色をサンプリング
        map_x = pixels[:, 0].astype(np.float32)
        map_y = pixels[:, 1].astype(np.float32)
        img_h, img_w = frame.shape[:2]
        
        # scipy.ndimage.map_coordinatesで補間（BGR → RGB変換）
        color_patch_flat = np.stack([
            scipy.ndimage.map_coordinates(
                frame[:, :, c], [map_y, map_x], order=1, mode='constant', cval=0.0
            )
            for c in [2, 1, 0]  # BGR → RGB の順（plt.imsave用）
        ], axis=-1).astype(np.uint8)
        
        # (N, 3) → (H, W, 3) に復元
        colors = color_patch_flat.reshape(height, width, 3)
        
        # ドーナツは実画像上の半径（config の 0.4–0.8）。Kannala が非単調な
        # ピーク以降は同じ画像半径へ折り畳むので、等距離 θ がピーク以下に限る。
        center = camera_params['center']
        radius = camera_params['radius']
        min_radius_px = radius * self.min_radius_ratio
        max_radius_px = radius * self.max_radius_ratio
        
        radial_sq = (
            (pixels[:, 0] - center[0])**2
            + (pixels[:, 1] - center[1])**2
        )
        sample_in_image = (
            (pixels[:, 0] >= 0.0) & (pixels[:, 0] < float(img_w) - 1e-6)
            & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < float(img_h) - 1e-6)
        )
        valid_mask = (
            (radial_sq >= min_radius_px**2)
            & (radial_sq <= max_radius_px**2)
            & sample_in_image
            & self.transformer.kannala_rising_branch_mask(pixels_ideal)
        )
        if self.reject_z_behind_camera:
            # カメラより後ろは塗らない。近い壁の後ろ側は前フレームの側視で済んでいる。
            camera_z = float(np.asarray(cam_pos, dtype=float).reshape(-1)[2])
            valid_mask = valid_mask & (z_grid.reshape(-1) >= camera_z)
        valid_mask_2d = valid_mask.reshape(height, width)
        
        valid_count = np.sum(valid_mask_2d)
        total_count = height * width
        self.logger.debug(
            f"Sampled colors: valid={valid_count}/{total_count} "
            f"({100.0*valid_count/total_count:.1f}%)"
        )
        
        radius = np.sqrt(np.maximum(radial_sq, 0.0)).reshape(height, width)
        return colors, valid_mask_2d, radius
    
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
    
    def _ensure_source_radius(self, colormap: np.ndarray) -> None:
        width = colormap.shape[1]
        if self._source_radius is None:
            self._source_radius = np.zeros(
                (self.eta_resolution, width), dtype=np.float32
            )
            return
        cur_w = self._source_radius.shape[1]
        if width > cur_w:
            pad = np.zeros(
                (self.eta_resolution, width - cur_w), dtype=np.float32
            )
            self._source_radius = np.concatenate(
                [self._source_radius, pad], axis=1
            )

    def update_colormap(
        self,
        colormap: np.ndarray,
        new_colors: np.ndarray,
        mask: np.ndarray,
        start_col: int,
        sample_radius: Optional[np.ndarray] = None,
        overwrite: bool = False,
    ) -> np.ndarray:
        """カラーマップ更新
        
        Args:
            colormap: 既存のカラーマップ (eta_res, Z_width, 3)
            new_colors: 新しい色情報 (height, width, 3)
            mask: 有効ピクセルのマスク (height, width) bool
            start_col: カラーマップ上の書き込み開始列インデックス
            sample_radius: ソース画像上の半径 [px]。prefer_peripheral 時に使用。
            overwrite: True なら塗済みも上書き（帯重なり補正の貼り位置合わせ用）。
        
        Returns:
            updated_colormap: 更新後のカラーマップ (eta_res, Z_width', 3)
        
        処理の流れ:
            1. 必要に応じてカラーマップを右方向に拡張
            2. マスクで有効なピクセルのみを抽出
            3. keep なら空画素のみ。prefer_peripheral ならより外周のソースだけ採用。
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

        if self.prefer_peripheral_pixels:
            self._ensure_source_radius(colormap)
        
        # 有効ピクセルのみを抽出
        if not np.any(mask):
            self.logger.warning("No valid pixels to update colormap")
            return colormap
        
        color_patch_filtered = new_colors[mask]
        valid_indices = np.where(mask)
        rows = valid_indices[0]
        cols = valid_indices[1] + start_col
        new_r = None
        if sample_radius is not None:
            new_r = sample_radius[mask].astype(np.float32, copy=False)

        if (not overwrite) and self.keep_painted_pixels:
            dest = colormap[rows, cols]
            empty = ~np.any(dest != 0, axis=-1)
            n_skip = int(np.size(empty) - np.count_nonzero(empty))
            if n_skip:
                self.logger.debug(
                    f"Colormap keep: skip {n_skip} already-painted pixels"
                )
            if not np.any(empty):
                return colormap
            rows = rows[empty]
            cols = cols[empty]
            color_patch_filtered = color_patch_filtered[empty]
            if new_r is not None:
                new_r = new_r[empty]
        elif (not overwrite) and self.prefer_peripheral_pixels and new_r is not None:
            old_r = self._source_radius[rows, cols]
            better = (old_r <= 0.0) | (new_r > old_r + self._peripheral_eps_px)
            n_skip = int(np.size(better) - np.count_nonzero(better))
            if n_skip:
                self.logger.debug(
                    f"Colormap prefer-peripheral: skip {n_skip} inner/equal samples"
                )
            if not np.any(better):
                return colormap
            rows = rows[better]
            cols = cols[better]
            color_patch_filtered = color_patch_filtered[better]
            new_r = new_r[better]

        colormap[rows, cols] = color_patch_filtered
        if self.prefer_peripheral_pixels and new_r is not None:
            self._source_radius[rows, cols] = new_r

        self.logger.debug(
            f"Colormap updated: {len(rows)} pixels written at col {start_col}"
        )
        
        return colormap

    def _align_patch_to_painted(
        self,
        color_map_img: np.ndarray,
        colors: np.ndarray,
        mask: np.ndarray,
        sample_radius: np.ndarray,
        eta_sample: np.ndarray,
        z_sample: np.ndarray,
        start_col: int,
        frame: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """既描画との重なりで新帯を並進合わせし、取り直しする。"""
        from src.front_overlap_align import (
            apply_overlap_shift_to_grids,
            estimate_overlap_shift_px,
        )

        max_drow = (self.overlap_align_max_dtheta_deg / 360.0) * self.eta_resolution
        max_dcol = self.overlap_align_max_dz_mm * self.pixels_per_mm
        shift = estimate_overlap_shift_px(
            color_map_img,
            colors,
            mask,
            start_col,
            min_pixels=self.overlap_align_min_pixels,
            min_response=self.overlap_align_min_response,
            max_drow=max_drow,
            max_dcol=max_dcol,
        )
        if shift is None:
            self.overlap_align_stats["skipped"] += 1
            return colors, mask, sample_radius, eta_sample, z_sample
        d_row, d_col, resp = shift
        eta_sample, z_sample = apply_overlap_shift_to_grids(
            eta_sample,
            z_sample,
            d_row,
            d_col,
            self.eta_resolution,
            self.pixels_per_mm,
        )
        colors, mask, sample_radius = self.sample_colors_from_frame(
            frame, eta_sample, z_sample, camera_state, camera_params
        )
        self.overlap_align_stats["applied"] += 1
        self.logger.info(
            "重ね合わせ接合: 適用 "
            f"dη={d_row * 360.0 / max(self.eta_resolution, 1):.3f}°, "
            f"dz={d_col / max(self.pixels_per_mm, 1e-6):.2f}mm "
            f"(response={resp:.3f})"
        )
        return colors, mask, sample_radius, eta_sample, z_sample

    def _render_strip(
        self,
        frame: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any],
        max_dz: float,
        alpha: float,
        residual_warp: Optional[Any],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """現在の絶対姿勢で帯を取り出す。(colors, mask, radius, start_col)"""
        camera_z = float(camera_state["position"][2])
        z_min, z_max = self.calculate_dynamic_z_range(
            camera_z, camera_params, camera_state, max_dz, alpha
        )
        z_min, z_max = self._apply_z_length_limit(z_min, z_max, frame_num=None)
        if self.reject_z_behind_camera:
            z_min = max(float(z_min), camera_z)
        if z_max - z_min <= 0:
            return None
        frame_width_px = int(np.ceil((z_max - z_min) * self.pixels_per_mm))
        eta_grid, z_grid = self.generate_eta_z_grid(
            (z_min, z_max), (self.eta_resolution, frame_width_px)
        )
        eta_sample, z_sample = eta_grid, z_grid
        if residual_warp is not None:
            from src.front_residual_warp import apply_residual_to_grids
            eta_sample, z_sample = apply_residual_to_grids(
                eta_grid, z_grid, residual_warp
            )
        colors, mask, sample_radius = self.sample_colors_from_frame(
            frame, eta_sample, z_sample, camera_state, camera_params
        )
        start_col = int(np.floor((z_min - self.z_offset) * self.pixels_per_mm))
        return colors, mask, sample_radius, start_col

    def _try_abs_pose(
        self,
        color_map_img: np.ndarray,
        before,
        camera_state: Dict[str, Any],
        frame: np.ndarray,
        camera_params: Dict[str, Any],
        max_dz: float,
        alpha: float,
        residual_warp: Optional[Any],
        z0: float,
        roll0: float,
        d_z: float,
        d_roll: float,
    ) -> Tuple[Optional[tuple], str]:
        """絶対 z, roll を仮に入れて取り直し、全周で改善すれば結果を返す。姿勢は戻す。

        leftover は現在姿勢の進み。カメラは z0-dz, roll0-droll のみ試す。
        """
        from src.front_strip_overlap import (
            allow_bin_worse_for_pose,
            match_unwrapped_strips,
            pose_correction_verdict,
        )

        pos = camera_state["position"]
        ori = camera_state["orientation"]
        pos[2] = z0 - d_z
        ori[0] = roll0 - d_roll
        note = (
            f"dz={-d_z:+.2f}mm "
            f"droll={float(np.degrees(-d_roll)):+.3f}°"
        )
        try:
            rendered = self._render_strip(
                frame, camera_state, camera_params, max_dz, alpha, residual_warp
            )
            if rendered is None:
                why = f"{note} 再展開失敗"
                self.logger.debug(f"帯重なり補正候補: {why}")
                return None, why
            colors2, mask2, radius2, start2 = rendered
            if start2 < 0:
                why = f"{note} start_col={start2}"
                self.logger.debug(f"帯重なり補正候補: {why}")
                return None, why
            check = match_unwrapped_strips(
                color_map_img, colors2, mask2, start2
            )
            z_only = abs(float(d_roll)) < 1e-12
            allow_bin = allow_bin_worse_for_pose(
                self.strip_overlap_allow_bin_worse, d_roll
            )
            ok, why = pose_correction_verdict(
                before, check, self.eta_resolution,
                allow_bin_worse=allow_bin,
            )
            if (
                ok
                and z_only
                and (not self.strip_overlap_allow_bin_worse)
                and "ηビン悪化許容" in why
            ):
                why = why.replace("ηビン悪化許容", "ηビン悪化許容(zのみ)")
            why = f"{note} {why}"
            self.logger.debug(f"帯重なり補正候補: {why}")
            if not ok:
                return None, why
            return (
                float(check.rms_px),
                colors2,
                mask2,
                radius2,
                int(start2),
                float(pos[2]),
                float(ori[0]),
                float(np.degrees(ori[0] - roll0)),
                float(check.med_abs_px),
            ), why
        finally:
            pos[2] = z0
            ori[0] = roll0

    def _measure_and_maybe_correct_strip(
        self,
        color_map_img: np.ndarray,
        colors: np.ndarray,
        mask: np.ndarray,
        sample_radius: np.ndarray,
        start_col: int,
        camera_state: Dict[str, Any],
        frame: np.ndarray,
        camera_params: Dict[str, Any],
        max_dz: float,
        alpha: float,
        residual_warp: Optional[Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
        """展開後画像1/2の特徴点誤差を測り、必要なら絶対 z, roll を直して取り直す。"""
        from src.front_strip_overlap import (
            draw_strip_error_vectors,
            leftover_to_pose_delta,
            match_unwrapped_strips,
            pose_dc_from_inliers,
        )

        result = match_unwrapped_strips(
            color_map_img, colors, mask, start_col
        )
        self.strip_overlap_stats["frames"] += 1
        ppm = max(float(self.pixels_per_mm), 1e-6)
        d_eta_deg = result.med_drow_px * 360.0 / max(self.eta_resolution, 1)
        d_z_mm = result.med_dcol_px / ppm
        calib_on = self.transformer.calibration is not None
        fno = camera_state.get("frame_num", self.strip_overlap_stats["frames"])
        if result.ok:
            self.strip_overlap_stats["ok"] += 1
            self.strip_overlap_stats["rms_sum"] += result.rms_px
        self.strip_overlap_stats["records"].append(
            {
                "frame": fno,
                "calib": "on" if calib_on else "off",
                "ok": int(result.ok),
                "n_raw": result.n_raw,
                "n_kept": result.n_kept,
                "n_inlier": result.n_inlier,
                "med_drow_px": result.med_drow_px,
                "med_dcol_px": result.med_dcol_px,
                "med_abs_px": result.med_abs_px,
                "rms_px": result.rms_px,
                "d_eta_deg": d_eta_deg,
                "d_z_mm": d_z_mm,
                "affine": int(result.affine is not None),
            }
        )
        self.logger.info(
            "帯重なり誤差: "
            f"frame={fno} "
            f"calib={'on' if calib_on else 'off'} "
            f"n={result.n_raw}->{result.n_kept}->{result.n_inlier} "
            f"med_dη={result.med_drow_px:.2f}px/{d_eta_deg:.3f}° "
            f"med_dz={result.med_dcol_px:.2f}px/{d_z_mm:.2f}mm "
            f"med|d|={result.med_abs_px:.2f}px rms={result.rms_px:.2f}px "
            f"affine={'ok' if result.affine is not None else 'none'}"
        )
        dbg = getattr(self.config, "debug", None)
        # デバッグの特徴点表示がオフのときだけ矢印図を出さない。
        show_vectors = True
        if dbg is not None:
            show_vectors = bool(getattr(dbg, "show_feature_points", True))
        if result.ok and show_vectors:
            vis = draw_strip_error_vectors(
                color_map_img, colors, mask, start_col, result
            )
            if vis is not None:
                from pathlib import Path
                import cv2 as _cv2
                if dbg is None:
                    out_dir = Path("debug/")
                else:
                    out_dir = Path(getattr(dbg, "debug_image_path", "debug/") or "debug/")
                out_dir = Path(out_dir) / "strip_overlap"
                out_dir.mkdir(parents=True, exist_ok=True)
                tag = "on" if calib_on else "off"
                path = out_dir / f"strip_overlap_calib-{tag}_{int(fno):06d}.png"
                _cv2.imwrite(str(path), vis)
        max_drow = min(
            16.0,
            (self.overlap_align_max_dtheta_deg / 360.0) * self.eta_resolution,
        )
        max_dcol = min(
            16.0,
            self.overlap_align_max_dz_mm * self.pixels_per_mm,
        )
        overwrite = False
        dc = (
            pose_dc_from_inliers(result, self.eta_resolution)
            if self.strip_overlap_correct and result.ok
            else None
        )
        if self.strip_overlap_correct and result.ok and dc is None:
            self.logger.info(
                "帯重なり補正: ηビン不足のためスキップ "
                f"frame={fno}"
            )
        elif (
            self.strip_overlap_correct
            and result.ok
            and dc is not None
            and abs(dc[0]) <= max_drow
            and abs(dc[1]) <= max_dcol
            and (abs(dc[0]) >= 0.75 or abs(dc[1]) >= 0.75)
        ):
            dc_drow, dc_dcol, n_bins_used = dc
            d_z, d_roll = leftover_to_pose_delta(
                dc_drow,
                dc_dcol,
                self.eta_resolution,
                self.pixels_per_mm,
            )
            pos = camera_state["position"]
            ori = camera_state["orientation"]
            z0 = float(pos[2])
            roll0 = float(ori[0])
            apply_z = abs(dc_dcol) >= 0.75
            apply_roll = abs(dc_drow) >= 0.75
            best = None
            tag = "z+roll"
            tried: list = []
            best_why = ""
            dz_try = d_z if apply_z else 0.0
            d_roll_try = d_roll if apply_roll else 0.0
            if apply_z or apply_roll:
                cand, why = self._try_abs_pose(
                    color_map_img, result, camera_state, frame,
                    camera_params, max_dz, alpha, residual_warp,
                    z0, roll0, dz_try, d_roll_try,
                )
                tried.append(why)
                if cand is not None:
                    best = cand
                    best_why = why
                    if apply_z and apply_roll:
                        tag = "z+roll"
                    elif apply_z:
                        tag = "z"
                    else:
                        tag = "roll"
            if best is None and apply_z and apply_roll:
                cand, why = self._try_abs_pose(
                    color_map_img, result, camera_state, frame,
                    camera_params, max_dz, alpha, residual_warp,
                    z0, roll0, d_z, 0.0,
                )
                tried.append(why)
                if cand is not None:
                    best = cand
                    best_why = why
                    tag = "z"
            if best is not None:
                rms2, colors, mask, sample_radius, start_col, z_new, roll_new, droll_deg, med2 = best
                pos[2] = z_new
                ori[0] = roll_new
                overwrite = True
                self.strip_overlap_stats["corrected"] += 1
                self.logger.info(
                    "帯重なり補正: 絶対座標 "
                    f"mode={tag} "
                    f"dz={z_new - z0:+.2f}mm droll={droll_deg:+.3f}° "
                    f"dc_nbin={n_bins_used} "
                    f"rms {result.rms_px:.2f}->{rms2:.2f}px "
                    f"med|d| {result.med_abs_px:.2f}->{med2:.2f}px "
                    f"frame={fno}"
                    + (
                        " allow_bin_worse"
                        if self.strip_overlap_allow_bin_worse
                        else ""
                    )
                    + (
                        " ビン悪化を採用"
                        if "ηビン悪化許容" in best_why
                        else ""
                    )
                )
            else:
                self.logger.info(
                    "帯重なり補正: 取り直し後に誤差が悪化したため破棄 "
                    f"dc_dη={dc_drow:.2f}px dc_dz={dc_dcol:.2f}px "
                    f"frame={fno} "
                    f"reason={'; '.join(tried)}"
                )
        elif self.strip_overlap_correct and result.ok:
            self.logger.info(
                "帯重なり補正: 並進が上限外または微小のためスキップ "
                f"dη={result.med_drow_px:.2f}px dcol={result.med_dcol_px:.2f}px "
                f"frame={fno}"
            )
        return colors, mask, sample_radius, int(start_col), overwrite

    def add_frame(
        self,
        frame: np.ndarray,
        camera_state: Dict[str, Any],
        camera_params: Dict[str, Any],
        color_map_img: Optional[np.ndarray] = None,
        max_dz: float = 100.0,
        alpha: float = 1.2,
        residual_warp: Optional[Any] = None,
    ) -> np.ndarray:
        """フレームをカラーマップに追加

        Args:
            frame: 入力フレーム画像 (H, W, 3) BGR
            camera_state: カメラ状態（position, orientation）
            camera_params: カメラパラメータ（center, radius等）
            color_map_img: 既存のカラーマップ（None可）
            max_dz: 1フレーム間の最大前進距離 [mm]（motion.max_dzから取得）
            alpha: z方向変形量に対する最大増加率（例: 1.2）
            residual_warp: 直視の残差フィールド。色の取り出し (η, z) だけずらす。
                書き込み位置とカメラ姿勢は変えない。

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
        if self.strip_overlap_match:
            residual_warp = None

        # 初回のみ: Z座標オフセットを記録（カラーマップの左端を0にするため）
        if self.z_offset is None:
            self.z_offset = camera_z
            self.logger.info(f"Z offset initialized: {self.z_offset:.2f} mm")

        rendered = self._render_strip(
            frame, camera_state, camera_params, max_dz, alpha, residual_warp
        )
        if rendered is None:
            self.logger.warning("Invalid Z range: strip render skipped")
            return color_map_img
        colors, mask, sample_radius, start_col = rendered

        if self.overlap_paste_align and color_map_img is not None:
            # 格子は再生成しない。既存の帯に対する位相合わせだけ使う。
            h, w = colors.shape[:2]
            z_min = self.z_offset + start_col / max(float(self.pixels_per_mm), 1e-6)
            z_max = z_min + w / max(float(self.pixels_per_mm), 1e-6)
            eta_sample, z_sample = self.generate_eta_z_grid(
                (z_min, z_max), (h, w)
            )
            colors, mask, sample_radius, eta_sample, z_sample = (
                self._align_patch_to_painted(
                    color_map_img,
                    colors,
                    mask,
                    sample_radius,
                    eta_sample,
                    z_sample,
                    start_col,
                    frame,
                    camera_state,
                    camera_params,
                )
            )

        overwrite = False
        if self.strip_overlap_match and color_map_img is not None:
            colors, mask, sample_radius, start_col, overwrite = (
                self._measure_and_maybe_correct_strip(
                    color_map_img,
                    colors,
                    mask,
                    sample_radius,
                    start_col,
                    camera_state,
                    frame,
                    camera_params,
                    max_dz,
                    alpha,
                    residual_warp,
                )
            )

        # (4) カラーマップ更新（z_offsetを基準とした相対位置で描画）
        color_map_img = self.update_colormap(
            color_map_img,
            colors,
            mask,
            start_col,
            sample_radius=sample_radius,
            overwrite=overwrite,
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

    def log_strip_overlap_stats(self) -> None:
        """帯重なり特徴点誤差の集計と CSV。"""
        if not self.strip_overlap_match:
            return
        s = self.strip_overlap_stats
        n_ok = int(s.get("ok", 0))
        mean_rms = (s.get("rms_sum", 0.0) / n_ok) if n_ok else 0.0
        records = s.get("records") or []
        abs_ok = [float(r["med_abs_px"]) for r in records if r.get("ok")]
        med_abs = float(np.median(abs_ok)) if abs_ok else 0.0
        self.logger.info(
            "帯重なり誤差集計: "
            f"frames={s.get('frames', 0)}, ok={n_ok}, "
            f"corrected={s.get('corrected', 0)}, "
            f"mean_rms={mean_rms:.2f}px, median_|d|={med_abs:.2f}px"
        )
        if not records:
            return
        import csv
        from pathlib import Path

        dbg = getattr(self.config, "debug", None)
        out_dir = Path(getattr(dbg, "debug_image_path", "debug/") if dbg else "debug/")
        out_dir = Path(out_dir) / "strip_overlap"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "on"
        if records:
            tag = str(records[0].get("calib") or "on")
        path = out_dir / f"summary_calib-{tag}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        self.logger.info(f"帯重なり誤差CSV: {path}")

