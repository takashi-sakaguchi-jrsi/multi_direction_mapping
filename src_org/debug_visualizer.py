"""デバッグ可視化モジュール

管内カメラカーシミュレーション用壁面画像生成処理のデバッグ可視化機能を提供します。

主な機能:
    - 暗部領域の可視化（消失点検出）
    - グリッド・軌跡描画
    - フレームウィンドウ表示
    - コンソール詳細出力
    - カラーマップ定期保存
    - テキスト情報のオーバーレイ

使用例:
    >>> config = DebugConfig(
    ...     enabled=True,
    ...     show_dark_region=True,
    ...     show_grid=True,
    ...     show_frame_window=True
    ... )
    >>> visualizer = DebugVisualizer(config)
    >>> 
    >>> # フレーム処理ループ内
    >>> frame = visualizer.draw_dark_region(frame, contours, center)
    >>> frame = visualizer.draw_grid(frame, camera_params, pipe_radius, cam_pos, roll, yaw, pitch)
    >>> frame = visualizer.add_text_overlay(frame, info_dict)
    >>> 
    >>> if not visualizer.show_frame("Debug View", frame):
    ...     break  # ユーザーが'q'キーを押下
    >>> 
    >>> visualizer.close()
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np


# ========================================
# カスタム例外クラス
# ========================================

class DebugVisualizerError(Exception):
    """デバッグ可視化関連のベース例外クラス"""
    pass


class InvalidDebugConfigError(DebugVisualizerError):
    """無効なデバッグ設定"""
    pass


# ========================================
# 設定データクラス
# ========================================

@dataclass
class DebugConfig:
    """デバッグ可視化の設定
    
    デバッグ可視化の各機能のON/OFF、表示設定、保存設定などを管理します。
    
    Attributes:
        enabled: デバッグモードのON/OFF（全機能の親スイッチ）
        show_dark_region: 暗部領域表示（消失点検出の可視化）
        show_grid: 円周グリッド表示（管路断面の可視化）
        show_front_indicator: 正面方向インジケータ表示
        show_colormap: カラーマッププレビュー表示（未実装）
        show_frame_window: OpenCVウィンドウ表示
        console_output: コンソール出力のON/OFF
        save_intermediate: 中間結果保存のON/OFF
        save_interval: 保存間隔（フレーム数）
        output_dir: 保存先ディレクトリ
        window_size: 表示ウィンドウサイズ (width, height)
        grid_color: グリッド線の色 (B, G, R)
        grid_thickness: グリッド線の太さ（ピクセル）
        text_color: テキストの色 (B, G, R)
        text_scale: テキストのスケール
        text_thickness: テキストの太さ（ピクセル）
        color_lens_center: レンズ中心点の色 (B, G, R)
        color_valid_area: 有効範囲円の色 (B, G, R)
        color_dark_contour: 暗部輪郭の色 (B, G, R)
        color_dark_center: 暗部重心の色 (B, G, R)
        color_front_indicator: 正面方向インジケータの色 (B, G, R)
    """
    enabled: bool = False
    
    # 機能別ON/OFF
    show_dark_region: bool = False
    show_grid: bool = False
    show_front_indicator: bool = False
    show_colormap: bool = False
    show_frame_window: bool = False
    console_output: bool = False
    
    # 保存設定
    save_intermediate: bool = False
    save_interval: int = 100
    output_dir: str = "data/output/debug"
    
    # 表示設定
    window_size: Tuple[int, int] = (1920, 1080)
    grid_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    grid_thickness: int = 1
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    text_scale: float = 0.5
    text_thickness: int = 1
    
    # 色設定（各要素）
    color_lens_center: Tuple[int, int, int] = (0, 0, 255)  # Red
    color_valid_area: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    color_dark_contour: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    color_dark_center: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_front_indicator: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    
    def validate(self) -> None:
        """設定値のバリデーション
        
        Raises:
            InvalidDebugConfigError: 設定値が不正な場合
        """
        if self.save_interval < 1:
            raise InvalidDebugConfigError(
                f"save_interval must be >= 1, got {self.save_interval}"
            )
        
        if self.window_size[0] < 1 or self.window_size[1] < 1:
            raise InvalidDebugConfigError(
                f"window_size must be > 0, got {self.window_size}"
            )
        
        if self.grid_thickness < 1:
            raise InvalidDebugConfigError(
                f"grid_thickness must be >= 1, got {self.grid_thickness}"
            )
        
        if self.text_scale <= 0:
            raise InvalidDebugConfigError(
                f"text_scale must be > 0, got {self.text_scale}"
            )


# ========================================
# デバッグ可視化クラス
# ========================================

class DebugVisualizer:
    """デバッグ可視化クラス
    
    フレーム画像にデバッグ情報を描画・表示する機能を提供します。
    
    主な機能:
        - 暗部領域の描画
        - 円周グリッドの描画
        - 正面方向インジケータの描画
        - テキスト情報のオーバーレイ
        - フレームウィンドウ表示
        - 中間結果の保存
        - コンソール出力
    
    Attributes:
        config: デバッグ設定
    """
    
    def __init__(self, config: DebugConfig):
        """初期化
        
        Args:
            config: デバッグ設定
            
        Raises:
            InvalidDebugConfigError: 設定値が不正な場合
        """
        self.config = config
        self._window_created: Dict[str, bool] = {}
        self._frame_count = 0
    
    def draw_dark_region(
        self,
        frame: np.ndarray,
        contours: List[np.ndarray],
        center: Tuple[int, int]
    ) -> np.ndarray:
        """暗部領域と重心を描画
        
        暗部輪郭とその重心をフレーム上に描画します。
        消失点検出の可視化に使用されます。
        
        Args:
            frame: 入力フレーム (H, W, 3)
            contours: 暗部輪郭リスト（各要素は (N, 1, 2) のndarray）
            center: レンズ中心座標 (x, y)
            
        Returns:
            描画済みフレーム (H, W, 3)
        """
        if not self.config.enabled or not self.config.show_dark_region:
            return frame
        
        if len(contours) == 0:
            return frame
        
        frame_copy = frame.copy()
        
        # 最大輪郭を取得
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 重心計算
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            gx = int(M['m10'] / M['m00'])
            gy = int(M['m01'] / M['m00'])
        else:
            gx, gy = center[0], center[1]
        
        # レンズ中心（赤点）
        cv2.circle(
            frame_copy,
            center,
            2,
            self.config.color_lens_center,
            -1
        )
        
        # 暗部輪郭（水色線）
        cv2.drawContours(
            frame_copy,
            [largest_contour],
            -1,
            self.config.color_dark_contour,
            2
        )
        
        # 暗部重心（緑点）
        cv2.circle(
            frame_copy,
            (gx, gy),
            2,
            self.config.color_dark_center,
            -1
        )
        
        return frame_copy
    
    def draw_grid(
        self,
        frame: np.ndarray,
        camera_params: Dict[str, Any],
        pipe_radius: float,
        cam_pos: np.ndarray,
        roll: float,
        yaw: float,
        pitch: float,
        circle_num: int = 12,
        circle_pitch: float = 1000.0
    ) -> np.ndarray:
        """円周グリッド線を描画
        
        管路の円周断面をグリッド線として描画します。
        座標変換モジュールに依存します。
        
        Args:
            frame: 入力フレーム (H, W, 3)
            camera_params: カメラパラメータ辞書
                - 'f': 焦点距離（ピクセル）
                - 'center': レンズ中心 (cx, cy)
                - 'radius': 有効範囲半径（ピクセル）
                - 'model': カメラモデル ('fisheye' or 'pinhole')
            pipe_radius: 管路半径 (mm)
            cam_pos: カメラ位置 [x, y, z] (mm)
            roll: ロール角（ラジアン）
            yaw: ヨー角（ラジアン）
            pitch: ピッチ角（ラジアン）
            circle_num: 描画する円周の数
            circle_pitch: 円周間隔 (mm)
            
        Returns:
            描画済みフレーム (H, W, 3)
        """
        if not self.config.enabled or not self.config.show_grid:
            return frame
        
        try:
            from coordinate_transform import world_to_pixel
        except ImportError:
            # 座標変換モジュールが無い場合はスキップ
            return frame
        
        frame_copy = frame.copy()
        
        f = camera_params.get('f', 1000.0)
        center = camera_params.get('center', (frame.shape[1] // 2, frame.shape[0] // 2))
        radius = camera_params.get('radius', min(frame.shape[0], frame.shape[1]) // 2)
        model = camera_params.get('model', 'fisheye')
        
        # 有効範囲円を描画
        cv2.circle(
            frame_copy,
            center,
            int(radius),
            self.config.color_valid_area,
            2
        )
        
        # レンズ中心点を描画
        cv2.circle(
            frame_copy,
            center,
            5,
            self.config.color_lens_center,
            -1
        )
        
        # 正面方向インジケータを描画（オプション）
        if self.config.show_front_indicator:
            x, y, z = 0.0, 0.0, 100000.0
            w_front = np.array([[x, y, z]], dtype=np.float32)
            
            try:
                w_front_on_camera = world_to_pixel(
                    w_front,
                    cam_pos,
                    roll,
                    yaw,
                    pitch,
                    f,
                    center,
                    model
                )
                
                if len(w_front_on_camera) > 0:
                    px = int(w_front_on_camera[0][0])
                    py = int(w_front_on_camera[0][1])
                    cv2.circle(
                        frame_copy,
                        (px, py),
                        3,
                        self.config.color_front_indicator,
                        2
                    )
            except Exception:
                # 投影失敗時はスキップ
                pass
        
        # 円周グリッド描画
        # 色マップの生成（距離に応じたグラデーション）
        color_map = []
        for n in range(circle_num):
            ratio = n / max(circle_num - 1, 1)
            # 緑→黄→赤のグラデーション
            r = int(255 * ratio)
            g = int(255 * (1 - ratio))
            b = 0
            color_map.append((b, g, r))
        
        # 各円周を描画
        for n in range(circle_num):
            z_wall = cam_pos[2] - (cam_pos[2] % circle_pitch) + (n + 1) * circle_pitch
            line_color = color_map[n]
            
            try:
                # project_objects_to_camera の呼び出し
                circles, x_lines, y_lines = self._project_objects_to_camera(
                    cam_pos,
                    roll,
                    yaw,
                    pitch,
                    pipe_radius,
                    z_wall,
                    center,
                    f,
                    model
                )
                
                # 各線分を描画
                self._draw_fisheye_lines_on_frame(
                    frame_copy,
                    circles,
                    center,
                    radius,
                    line_color,
                    self.config.grid_thickness
                )
                self._draw_fisheye_lines_on_frame(
                    frame_copy,
                    x_lines,
                    center,
                    radius,
                    line_color,
                    self.config.grid_thickness
                )
                self._draw_fisheye_lines_on_frame(
                    frame_copy,
                    y_lines,
                    center,
                    radius,
                    line_color,
                    self.config.grid_thickness
                )
            except Exception:
                # 描画失敗時はスキップ
                continue
        
        return frame_copy
    
    def _project_objects_to_camera(
        self,
        cam_pos: np.ndarray,
        roll: float,
        yaw: float,
        pitch: float,
        R: float,
        z_wall: float,
        center: Tuple[int, int],
        f: float,
        model: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """管路断面（円、X軸、Y軸）をカメラ画像座標に投影
        
        Args:
            cam_pos: カメラ位置 [x, y, z] (mm)
            roll: ロール角（ラジアン）
            yaw: ヨー角（ラジアン）
            pitch: ピッチ角（ラジアン）
            R: 管路半径 (mm)
            z_wall: 投影する断面のz座標 (mm)
            center: レンズ中心 (cx, cy)
            f: 焦点距離（ピクセル）
            model: カメラモデル
            
        Returns:
            (circles, x_lines, y_lines): 各線分のピクセル座標リスト
        """
        from coordinate_transform import world_to_pixel
        
        # 円周上の点を生成（100点）
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = R * np.cos(theta)
        y_circle = R * np.sin(theta)
        z_circle = np.full_like(theta, z_wall)
        
        w_circle = np.stack([x_circle, y_circle, z_circle], axis=-1)
        circles = world_to_pixel(w_circle, cam_pos, roll, yaw, pitch, f, center, model)
        
        # X軸ライン（-R から R まで30点）
        x_axis = np.linspace(-R, R, 30)
        y_axis_x = np.zeros_like(x_axis)
        z_axis_x = np.full_like(x_axis, z_wall)
        
        w_x_axis = np.stack([x_axis, y_axis_x, z_axis_x], axis=-1)
        x_lines = world_to_pixel(w_x_axis, cam_pos, roll, yaw, pitch, f, center, model)
        
        # Y軸ライン（-R から R まで30点）
        y_axis = np.linspace(-R, R, 30)
        x_axis_y = np.zeros_like(y_axis)
        z_axis_y = np.full_like(y_axis, z_wall)
        
        w_y_axis = np.stack([x_axis_y, y_axis, z_axis_y], axis=-1)
        y_lines = world_to_pixel(w_y_axis, cam_pos, roll, yaw, pitch, f, center, model)
        
        return circles, x_lines, y_lines
    
    def _draw_fisheye_lines_on_frame(
        self,
        frame: np.ndarray,
        pts_px: np.ndarray,
        center: Tuple[int, int],
        radius: float,
        color: Tuple[int, int, int],
        thickness: int
    ) -> None:
        """ピクセル座標列を線分として描画
        
        魚眼レンズの有効範囲内のみ描画します（円形マスク）。
        
        Args:
            frame: 描画対象フレーム（in-place更新）
            pts_px: ピクセル座標列 (N, 2)
            center: レンズ中心 (cx, cy)
            radius: 有効範囲半径（ピクセル）
            color: 線の色 (B, G, R)
            thickness: 線の太さ（ピクセル）
        """
        if len(pts_px) < 2:
            return
        
        cx, cy = center
        
        for i in range(len(pts_px) - 1):
            x1, y1 = pts_px[i]
            x2, y2 = pts_px[i + 1]
            
            # 有効範囲内かチェック
            dist1 = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
            dist2 = np.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
            
            if dist1 <= radius and dist2 <= radius:
                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    thickness
                )
    
    def add_text_overlay(
        self,
        frame: np.ndarray,
        info: Dict[str, Any]
    ) -> np.ndarray:
        """テキスト情報を重ねる
        
        フレーム左上にデバッグ情報をテキストとして表示します。
        
        Args:
            frame: 入力フレーム (H, W, 3)
            info: 表示する情報の辞書（以下のキーをサポート）:
                - frame_num: フレーム番号
                - cam_pos: カメラ位置 [x, y, z]
                - cam_orient: カメラ姿勢 [roll, yaw, pitch]（ラジアン）
                - ocr_distance: OCR読み取り距離
                - corrected_distance: 補正後距離
                - status: 処理ステータス文字列
                - movement: 移動量 [dx, dy, dz, dr, dp, dt]
                
        Returns:
            描画済みフレーム (H, W, 3)
        """
        if not self.config.enabled:
            return frame
        
        frame_copy = frame.copy()
        y_offset = 30
        line_height = 25
        
        # フレーム番号
        if 'frame_num' in info:
            text = f"Frame: {info['frame_num']}"
            cv2.putText(
                frame_copy,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale,
                self.config.text_color,
                self.config.text_thickness
            )
            y_offset += line_height
        
        # カメラ位置
        if 'cam_pos' in info:
            pos = info['cam_pos']
            text = f"Pos: x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}"
            cv2.putText(
                frame_copy,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale,
                self.config.text_color,
                self.config.text_thickness
            )
            y_offset += line_height
        
        # カメラ姿勢
        if 'cam_orient' in info:
            orient = info['cam_orient']
            text = (
                f"Orient: roll={np.degrees(orient[0]):.1f}, "
                f"yaw={np.degrees(orient[1]):.1f}, "
                f"pitch={np.degrees(orient[2]):.1f}"
            )
            cv2.putText(
                frame_copy,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale,
                self.config.text_color,
                self.config.text_thickness
            )
            y_offset += line_height
        
        # 移動量
        if 'movement' in info:
            mov = info['movement']
            text = (
                f"Movement: dx={mov[0]:.2f}, dy={mov[1]:.2f}, dz={mov[2]:.2f}, "
                f"dr={np.degrees(mov[3]):.2f}, dp={np.degrees(mov[4]):.2f}, dt={np.degrees(mov[5]):.2f}"
            )
            cv2.putText(
                frame_copy,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale,
                self.config.text_color,
                self.config.text_thickness
            )
            y_offset += line_height
        
        # OCR距離
        if 'ocr_distance' in info:
            text = f"OCR Distance: {info['ocr_distance']:.1f} mm"
            cv2.putText(
                frame_copy,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale,
                self.config.text_color,
                self.config.text_thickness
            )
            y_offset += line_height
        
        # 補正後距離
        if 'corrected_distance' in info:
            text = f"Corrected Distance: {info['corrected_distance']:.1f} mm"
            cv2.putText(
                frame_copy,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale,
                self.config.text_color,
                self.config.text_thickness
            )
            y_offset += line_height
        
        # ステータス
        if 'status' in info:
            text = f"Status: {info['status']}"
            cv2.putText(
                frame_copy,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.text_scale,
                self.config.text_color,
                self.config.text_thickness
            )
        
        return frame_copy
    
    def show_frame(self, window_name: str, frame: np.ndarray) -> bool:
        """フレームを表示
        
        OpenCVウィンドウでフレームを表示します。
        'q'キーで表示を終了できます。
        
        Args:
            window_name: ウィンドウ名
            frame: 表示するフレーム (H, W, 3)
            
        Returns:
            Trueで継続、Falseで終了（'q'キー押下時）
        """
        if not self.config.enabled or not self.config.show_frame_window:
            return True
        
        # ウィンドウ作成（初回のみ）
        if window_name not in self._window_created or not self._window_created[window_name]:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                window_name,
                self.config.window_size[0],
                self.config.window_size[1]
            )
            self._window_created[window_name] = True
        
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            self._window_created[window_name] = False
            return False
        
        return True
    
    def save_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        prefix: str = "debug_frame"
    ) -> Optional[str]:
        """フレームを保存
        
        指定された保存間隔ごとにフレームを保存します。
        
        Args:
            frame: 保存するフレーム (H, W, 3)
            frame_num: フレーム番号
            prefix: ファイル名のプレフィックス
            
        Returns:
            保存したファイルパス（保存しない場合はNone）
        """
        if not self.config.enabled or not self.config.save_intermediate:
            return None
        
        # 保存間隔チェック
        if frame_num % self.config.save_interval != 0:
            return None
        
        # ディレクトリ作成
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存
        filename = f"{prefix}_{frame_num:06d}.png"
        filepath = os.path.join(self.config.output_dir, filename)
        cv2.imwrite(filepath, frame)
        
        return filepath
    
    def save_colormap(
        self,
        colormap: np.ndarray,
        frame_num: int,
        theta_deg: float,
        timestamp: str,
        prefix: str = "color_map"
    ) -> Optional[str]:
        """カラーマップを定期保存
        
        指定された保存間隔ごとにカラーマップを保存します。
        matplotlibを使用してカラースケールを保持します。
        
        Args:
            colormap: カラーマップ画像 (H, W, 3)
            frame_num: フレーム番号
            theta_deg: 画角（度）
            timestamp: タイムスタンプ文字列
            prefix: ファイル名のプレフィックス
            
        Returns:
            保存したファイルパス（保存しない場合はNone）
        """
        if not self.config.enabled or not self.config.save_intermediate:
            return None
        
        # 保存間隔チェック
        if frame_num % self.config.save_interval != 0:
            return None
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            # matplotlibが無い場合はcv2で保存
            os.makedirs(self.config.output_dir, exist_ok=True)
            filename = f"{prefix}_{theta_deg}_{timestamp}_{frame_num:06d}.png"
            filepath = os.path.join(self.config.output_dir, filename)
            cv2.imwrite(filepath, colormap)
            return filepath
        
        # ディレクトリ作成
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存
        filename = f"{prefix}_{theta_deg}_{timestamp}_{frame_num:06d}.png"
        filepath = os.path.join(self.config.output_dir, filename)
        plt.imsave(filepath, colormap)
        
        return filepath
    
    def print_info(self, message: str) -> None:
        """コンソールに情報出力
        
        Args:
            message: 出力メッセージ
        """
        if self.config.enabled and self.config.console_output:
            print(message)
    
    def print_frame_info(
        self,
        frame_num: int,
        movement: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        ocr_distance: Optional[float] = None,
        corrected_distance: Optional[float] = None
    ) -> None:
        """フレーム処理情報をコンソール出力
        
        フレーム番号、移動量、カメラ位置・姿勢、距離情報を出力します。
        
        Args:
            frame_num: フレーム番号
            movement: 移動量 [dx, dy, dz, dr, dp, dt]
            position: カメラ位置・姿勢 [x, y, z, roll, yaw, pitch]
            ocr_distance: OCR読み取り距離 (mm)
            corrected_distance: 補正後距離 (mm)
        """
        if not self.config.enabled or not self.config.console_output:
            return
        
        # フレーム番号
        print(f"frame: {frame_num}")
        
        # 移動量
        if movement is not None and len(movement) >= 6:
            dx, dy, dz = movement[0], movement[1], movement[2]
            dr, dp, dt = movement[3], movement[4], movement[5]
            print(
                f"{frame_num}: {dx:.3f}, {dy:.3f}, {dz:.3f}, "
                f"{np.degrees(dr):.3f}, {np.degrees(dp):.3f}, {np.degrees(dt):.3f}"
            )
        
        # カメラ位置・姿勢、距離情報
        if position is not None and len(position) >= 6:
            x, y, z = position[0], position[1], position[2]
            roll, yaw, pitch = position[3], position[4], position[5]
            
            parts = [
                f"x:{x:.3f}, y:{y:.3f}, z:{z:.3f}",
                f"r:{np.degrees(roll):.3f}, p:{np.degrees(yaw):.3f}, t:{np.degrees(pitch):.3f}"
            ]
            
            if ocr_distance is not None:
                parts.append(f"z(OCR):{ocr_distance:.3f}")
            
            if corrected_distance is not None:
                parts.append(f"z*:{corrected_distance:.3f}")
            
            print(", ".join(parts))
    
    def close(self) -> None:
        """リソース解放
        
        開いているすべてのウィンドウを閉じます。
        """
        for window_name, created in self._window_created.items():
            if created:
                cv2.destroyWindow(window_name)
        
        self._window_created.clear()
        cv2.destroyAllWindows()
