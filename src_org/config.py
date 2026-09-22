"""設定管理モジュール

このモジュールは、カメラカーシミュレーション用壁面画像生成処理の
すべての設定パラメータを管理します。

主要機能:
    - JSONファイルからの設定読み込み
    - デフォルト設定の提供
    - 設定の階層的マージ
    - 設定値のバリデーション
    - 設定の保存

使用例:
    >>> config = Config.from_json("config.json")
    >>> config.validate()
    >>> print(config.camera.fov_degrees)
    185.0
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set

# 円筒座標マッチング設定をインポート（条件付き）
try:
    from src.feature_matching_cylindrical import CylindricalMatchingConfig
    CYLINDRICAL_MATCHING_AVAILABLE = True
except ImportError:
    CYLINDRICAL_MATCHING_AVAILABLE = False
    CylindricalMatchingConfig = None


# 消失点推定設定をインポート（条件付き）
try:
    from src.vanishing_point_estimator import VanishingPointConfig
    VANISHING_POINT_AVAILABLE = True
except ImportError:
    VANISHING_POINT_AVAILABLE = False
    VanishingPointConfig = None

from datetime import datetime


# ========================================
# カスタム例外クラス
# ========================================

class ConfigError(Exception):
    """設定関連のベース例外クラス"""
    pass


class ConfigValidationError(ConfigError):
    """設定値のバリデーションエラー"""
    pass


class ConfigFileNotFoundError(ConfigError):
    """設定ファイルが見つからない"""
    pass


class ConfigFormatError(ConfigError):
    """設定ファイルのフォーマットエラー"""
    pass


# ========================================
# データクラス群
# ========================================

@dataclass
class InputConfig:
    """入力設定
    
    Attributes:
        video_path: 入力動画ファイルのパス
        start_frame: 処理開始フレーム番号（0始まり）
        end_frame: 処理終了フレーム番号（含まれない、左閉右開区間 [start_frame, end_frame)）（Noneの場合は最後まで）
    """
    video_path: str = "data/input/videos/sample_video.mp4"
    start_frame: int = 1
    end_frame: Optional[int] = None


@dataclass
class OutputConfig:
    """出力設定

    Attributes:
        colormap_path: カラーマップ出力パス（{timestamp}はタイムスタンプに置換）
            補正後の最終カラーマップ（エンドユーザーが参照）
        temporary_colormap_path: 一時カラーマップ出力パス
            補正前の一時カラーマップ（内部処理用、Phase2で生成）
        report_path: レポート出力パス
        progress_path: 進捗情報出力パス（{process_id}はプロセスIDに置換）
    """
    colormap_path: str = "data/output/colormaps/colormap_{timestamp}.png"
    temporary_colormap_path: str = "data/output/colormaps/colormap_{timestamp}_temp.png"
    report_path: str = "data/output/reports/report_{timestamp}.xlsx"
    progress_path: str = "data/output/progress/progress_{process_id}.json"


@dataclass
class CorrectionConfig:
    """展開画像補正設定

    Attributes:
        enabled: 補正機能の有効/無効
            True: Phase3で補正処理を実行
            False: 補正をスキップ（temporary_colormap → colormap にコピー）
        ocr_interval_mm: 補正基準OCR距離間隔 L (mm)
            開始フレームからOCR距離がL(mm)増加するごとに基準フレームとして使用。
            例: L=100 → 0mm, 100mm, 200mm, 300mm, ... の位置のフレームを基準とする
            基準点を間引くことで、より滑らかな補正を実現する。
        max_image_size: 最大画像サイズ（OpenCV制限対応）
            画像サイズがこの値を超える場合、自動的にリサイズされる

    注意:
        - correction.enabled=True かつ use_ocr_z_constraints=False の場合、
          OCR距離推定が無効のため補正はスキップされます
        - 補正処理はPhase3で実行され、temporary_colormap → colormap に変換されます
    """
    enabled: bool = True
    ocr_interval_mm: float = 100.0
    max_image_size: int = 30000


@dataclass
class CameraConfig:
    """カメラ設定

    Attributes:
        fov_degrees: 視野角（度）- v1.0_fallback時のみ使用
            キャリブレーションファイルからfx,fy,cx,cyが有効な場合は無視される。
            fx,fy,cx,cyが未設定の場合のみ、焦点距離fの計算に使用される。
        ocr_roi_ratio: OCR処理対象領域の比率 [top, bottom, left, right]
        image_width: カメラ画像の幅（ピクセル）- 動画から自動取得、未使用
        image_height: カメラ画像の高さ（ピクセル）- 動画から自動取得、未使用
        focal_length_mm: 焦点距離（mm）- 未使用（互換性のため残存）
        center_offset_x: レンズ中心のx方向オフセット（ピクセル）
            フレーム中心からの相対位置。この値とフレーム中心からcx,cyを計算。
            fx,fy,cx,cyが有効な場合は、cx,cyが直接使用されこの値は無視される。
        center_offset_y: レンズ中心のy方向オフセット（ピクセル）
            フレーム中心からの相対位置。この値とフレーム中心からcx,cyを計算。
            fx,fy,cx,cyが有効な場合は、cx,cyが直接使用されこの値は無視される。
    """
    fov_degrees: float = 185.0
    camera_model: str = "fisheye"
    """
    キャリブレーションファイル未指定時に使用するレンズモデル
    
    値:
        "fisheye": 魚眼レンズモデル（デフォルト）
        "pinhole": ピンホールモデル
    
    用途:
        キャリブレーションファイルが設定されていない場合（v1.0_fallback）に、
        どのレンズモデルを使用するかを指定します。
        キャリブレーションファイルが指定されている場合、camera_modelフィールドが
        キャリブレーション情報から読み込まれるため、この設定は無視されます。
    """
    ocr_roi_ratio: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.0, 0.2])
    image_width: int = 1220
    image_height: int = 1080
    focal_length_mm: float = 2.8
    center_offset_x: int = 0
    center_offset_y: int = 0

    # === ⭐ Phase 3追加: LensCalibration v2.0フィールド ===
    fx: float = 0.0
    """焦点距離X (ピクセル単位)

    値:
        0.0（デフォルト）: キャリブレーション結果なし、FOVから計算
        > 0.0: キャリブレーション結果を使用

    詳細:
        cv2.fisheye.calibrate()から取得したカメラ行列のfx要素。
        ピクセル単位の焦点距離で、画像センサー上の1ピクセルあたりの
        焦点距離を表します。
    """

    fy: float = 0.0
    """焦点距離Y (ピクセル単位)

    値:
        0.0（デフォルト）: キャリブレーション結果なし、FOVから計算
        > 0.0: キャリブレーション結果を使用

    詳細:
        cv2.fisheye.calibrate()から取得したカメラ行列のfy要素。
        通常、fxとfyは類似した値になります。
    """

    cx: float = 0.0
    """主点X座標 (ピクセル単位)

    値:
        0.0（デフォルト）: キャリブレーション結果なし、中心+offsetから計算
        > 0.0: キャリブレーション結果を使用

    詳細:
        cv2.fisheye.calibrate()から取得したカメラ行列のcx要素。
        レンズの光学中心のX座標（画像左上を原点とする座標系）。
    """

    cy: float = 0.0
    """主点Y座標 (ピクセル単位)

    値:
        0.0（デフォルト）: キャリブレーション結果なし、中心+offsetから計算
        > 0.0: キャリブレーション結果を使用

    詳細:
        cv2.fisheye.calibrate()から取得したカメラ行列のcy要素。
        レンズの光学中心のY座標（画像左上を原点とする座標系）。
    """

    calibrated_image_width: int = 0
    """キャリブレーション時の画像幅 (ピクセル)

    値:
        0（デフォルト）: キャリブレーション結果なし
        > 0: キャリブレーション時の画像幅

    用途:
        実際のフレームサイズと異なる場合、自動スケール補正を行う。
        例: キャリブレーション時2000x1100、実行時1220x1080
    """

    calibrated_image_height: int = 0
    """キャリブレーション時の画像高さ (ピクセル)

    値:
        0（デフォルト）: キャリブレーション結果なし
        > 0: キャリブレーション時の画像高さ

    用途:
        実際のフレームサイズと異なる場合、自動スケール補正を行う。
    """

    # Phase 4: レンズキャリブレーションファイル読み込み
    _lens_calibration: Optional[Any] = field(default=None, repr=False)
    """LensCalibrationオブジェクトの保持（歪み補正パイプライン用）

    __post_init__でキャリブレーションファイル読み込み成功時に設定される。
    CoordinateTransformerに渡されて歪み補正に使用される。
    """

    lens_calibration_file: Optional[str] = None
    """
    レンズキャリブレーション情報を外部ファイルから読み込む場合に指定
    
    値:
        None（デフォルト）: 上記の直接指定パラメータを使用
        "path/to/calibration.json": 指定ファイルから読み込み
        "default_lens.json": 標準ディレクトリ（data/calibration/）から読み込み
    
    設定値の優先順位（高い順）:
        1. コマンドライン引数（--aov等）
        2. ユーザー設定ファイル（--config指定）
        3. キャリブレーションファイル（lens_calibration_file指定）
           / 自動チューニング設定値 / 動画最適化設定値
        4. default_config.json
        5. システムデフォルト値（本クラスのデフォルト）

    注意事項:
        - コマンドライン引数やユーザー設定ファイルは、パラメータを強制する必要が
          ある場合にのみ設定してください。
        - 基本的にはこれらを設定せずに、デフォルトの設定値および解析的に
          抽出されたパラメータ（キャリブレーションファイル等）を使用します。
        - 不必要なパラメータを不用意に設定しないよう注意してください。
    """

    def __post_init__(self):
        """初期化後の処理: キャリブレーションファイル読み込み"""
        if self.lens_calibration_file is not None:
            logger = logging.getLogger(__name__)
            try:
                from src.calibration import load_calibration
                
                # キャリブレーションファイルから読み込み
                calibration = load_calibration(self.lens_calibration_file)
                
                # ========================================
                # v1.0互換フィールドの上書き
                # ========================================
                self.fov_degrees = calibration.fov_degrees
                self.focal_length_mm = calibration.focal_length_mm
                self.center_offset_x = calibration.center_offset_x
                self.center_offset_y = calibration.center_offset_y

                # ========================================
                # ⭐ Phase 3追加: v2.0フィールドの上書き
                # ========================================
                self.fx = calibration.fx
                self.fy = calibration.fy
                self.cx = calibration.cx
                self.cy = calibration.cy
                self.calibrated_image_width = calibration.image_width
                self.calibrated_image_height = calibration.image_height

                # LensCalibrationオブジェクトを保持（歪み補正パイプライン用）
                self._lens_calibration = calibration

                # ログ出力を拡張
                logger.info(
                    f"レンズキャリブレーション読み込み完了: "
                    f"{self.lens_calibration_file} "
                    f"({calibration.lens_model})\n"
                    f"  v1.0互換: fov={self.fov_degrees}°, "
                    f"offset=({self.center_offset_x}, {self.center_offset_y})\n"
                    f"  v2.0拡張: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                    f"cx={self.cx:.1f}, cy={self.cy:.1f}, "
                    f"calib_size=({self.calibrated_image_width}x{self.calibrated_image_height})"
                )
            except Exception as e:
                logger.warning(
                    f"レンズキャリブレーション読み込み失敗: "
                    f"{self.lens_calibration_file} - {e}\n"
                    f"既存の直接指定パラメータを使用します"
                )
                # フォールバック: 既存の値のままにする



@dataclass
class PipeConfig:
    """管路設定

    Attributes:
        diameter_mm: 管路直径（mm）
        default_position_z: デフォルト位置（z座標、mm）
    """
    diameter_mm: float = 250.0
    default_position_z: float = 0.0



@dataclass
class FeatureFilteringConfig:
    """特徴点フィルタリング設定

    Attributes:
        magnitude_filter_enabled: ベクトル長さフィルタの有効/無効
        magnitude_filter_method: 外れ値検出方法 ('iqr' または 'mad')
        magnitude_iqr_multiplier: IQR倍率（1.5が標準、2.0で緩和、1.0で厳格）
        direction_filter_enabled: θ-セクター方向フィルタの有効/無効
        direction_n_sectors: セクター分割数（8 = 45°ごと）
        direction_iqr_multiplier: 方向フィルタのIQR倍率
        direction_min_points_per_sector: セクターあたりの最小点数
        vp_circle_filter_enabled: レンズ中心点距離フィルタの有効/無効
        vp_circle_radius: レンズ中心点距離の閾値（ピクセル）
        log_filtering_stats: フィルタリング統計のログ出力有効/無効

    注意:
        レンズ中心点座標はFisheyeCamera初期化時に計算されたレンズ中心座標
        (camera.cx, camera.cy)を使用します。この座標は以下のように計算されます:
          cx = image_width / 2 + center_offset_x
          cy = image_height / 2 + center_offset_y
        キャリブレーション実装時にはcenter_offset_x/yを更新することで、
        レンズ中心座標が自動的に反映されます。
    """
    magnitude_filter_enabled: bool = True
    magnitude_filter_method: str = 'iqr'
    magnitude_iqr_multiplier: float = 1.5
    magnitude_absolute_max: float = 100.0  # IQR前の絶対値上限（px）。これを超える変位は無条件で除去

    direction_filter_enabled: bool = True
    direction_n_sectors: int = 8
    direction_iqr_multiplier: float = 1.5
    direction_min_points_per_sector: int = 5

    # レンズ中心点距離フィルタ（方向フィルタの前処理として統合）
    vp_circle_filter_enabled: bool = True
    vp_circle_radius: float = 300.0  # レンズ中心点距離の閾値（ピクセル）
    # 注: レンズ中心点座標はcamera.cx, cyから自動取得（cx/cyにはcenter_offsetが含まれる）

    log_filtering_stats: bool = True


@dataclass
class StaticFrameDetectionConfig:
    """停止フレーム検出設定

    特徴点マッチング結果から停止状態を検出し、
    不安定なフレームペアをスキップする機能の設定。
    
    2段階判定方式:
        第1段階判定（特徴点マッチング直後）:
            - 判定式: 平均移動量(px) / pixels_per_mm <= movement_threshold
            - 目的: 完全同一フレームの早期検出
            - 現在実装済み
        
        第2段階判定（移動量推定後）※今後実装予定:
            - 判定式: abs(dz) <= movement_threshold
            - 目的: Z方向の実際の停止状態を検出
            - Phase 3で実装予定

    Attributes:
        enabled: 停止フレーム検出の有効/無効
        movement_threshold: 停止判定の移動量閾値(mm)
            - 単位: mm（ミリメートル）
            - 第1段階判定: 特徴点の平均移動量(px)をpixels_per_mmで除算して比較
            - 第2段階判定: dzの絶対値と比較（今後実装）
            - pixels_per_mm: colormap.pixels_per_mmから取得（デフォルト1.0）
            - デフォルト値: 2.0mm
        min_points: 停止判定に必要な最小特徴点数

    Note:
        この機能は円筒座標系マッチング・フレーム座標系マッチングの
        両方で使用されるため、estimation トップレベルに配置されています。
        （REFACTOR-002で cylindrical セクションから移動）
        
        パラメータ単位の変更:
            - TASK-4.2.11d Phase 1で単位をピクセル→mmに変更
            - デフォルト値は実質的に変わらない（pixels_per_mm=1.0の場合）
    """
    enabled: bool = True
    movement_threshold: float = 2.0
    min_points: int = 10


@dataclass
class FeatureMatchingConfig:
    """特徴点マッチング設定

    Attributes:
        method: 特徴点検出手法（ORB, SIFT, AKAZE等）
        max_features: 最大特徴点数
        match_threshold: マッチング閾値（0.0-1.0）
        min_match_count: 最小マッチ点数
        scale_factor: ORBスケールファクター（1.2推奨）
        n_levels: ORBピラミッドレベル数（8推奨）
        edge_threshold: ORBエッジ閾値（31推奨）
        first_level: ORB最初のレベル（0推奨）
        wta_k: ORB WTA_K（2推奨）
        patch_size: ORBパッチサイズ（31推奨）
        fast_threshold: ORB FAST閾値（20推奨）
        radius_min_ratio: ドーナツ状フィルタ内側半径比率（0.15推奨）
        radius_max_ratio: ドーナツ状フィルタ外側半径比率（0.85推奨）
        ratio_test_threshold: Lowe's比率テスト閾値（0.75推奨）
        max_distance_threshold: 最大距離閾値（100.0mm推奨）
        spatial_grid_theta_bins: 空間グリッドθビン数（8推奨）
        spatial_grid_phi_bins: 空間グリッドφビン数（8推奨）
        use_cylindrical_matching: 円筒座標マッチングの有効/無効
        cylindrical: 円筒座標マッチング設定
    """
    method: str = "ORB"
    max_features: int = 500
    match_threshold: float = 0.7
    min_match_count: int = 10
    scale_factor: float = 1.2
    n_levels: int = 8
    edge_threshold: int = 31
    first_level: int = 0
    wta_k: int = 2
    patch_size: int = 31
    fast_threshold: int = 20
    radius_min_ratio: float = 0.15
    radius_max_ratio: float = 0.85
    ratio_test_threshold: float = 0.75
    max_distance_threshold: float = 100.0
    spatial_grid_theta_bins: int = 8
    spatial_grid_phi_bins: int = 8
    
    # 円筒座標マッチングの有効化
    use_cylindrical_matching: bool = True  # デフォルトはTrue（推奨設定）
    
    # 円筒座標マッチング設定（use_cylindrical_matchingがTrueの場合に使用）
    cylindrical: Optional[Any] = None  # CylindricalMatchingConfigまたはNone
    
    def __post_init__(self):
        """初期化後の処理: cylindrical設定のデフォルト値設定"""
        if self.use_cylindrical_matching and self.cylindrical is None:
            if CYLINDRICAL_MATCHING_AVAILABLE and CylindricalMatchingConfig is not None:
                self.cylindrical = CylindricalMatchingConfig()
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "use_cylindrical_matching=True ですが、CylindricalMatchingConfigが"
                    "利用できません。feature_matching_cylindrical.pyが正しく実装されているか確認してください。"
                )
                self.cylindrical = None


@dataclass
class MotionConfig:
    """移動制約設定

    Attributes:
        max_dx: x方向最大移動量（mm/フレーム）
        max_dy: y方向最大移動量（mm/フレーム）
        max_dz: z方向最大移動量（mm/フレーム）
        max_droll: ロール角最大変化量（度/フレーム）
        max_dtheta: 左右方向最大姿勢変化量（度/フレーム）
        max_dphi: 上下方向最大姿勢変化量（度/フレーム）
        max_x: x座標累積値の制限範囲（mm、±max_x以内）
        max_y: y座標累積値の制限範囲（mm、±max_y以内）
        max_yaw: ヨー角累積値の制限範囲（度、±max_yaw以内）
        max_pitch: ピッチ角累積値の制限範囲（度、±max_pitch以内）
        max_roll: ロール角累積値の制限範囲（度、±max_roll以内）

        estimate_dx: 並進移動量dx, dyを推定するか
        estimate_dz: 前進移動量dzを推定するか
        estimate_droll: ロール角変化量を推定するか
        estimate_dyaw: ヨー角変化量を推定するか
        estimate_dpitch: ピッチ角変化量を推定するか
        use_vanishing_point_yaw: ヨー角の基準値として消失点推定値を使用するか
        use_vanishing_point_pitch: ピッチ角の基準値として消失点推定値を使用するか
        fixed_yaw: ヨー角固定値（estimate_dyaw=False かつ use_vanishing_point_yaw=False の場合）度
        fixed_pitch: ピッチ角固定値（estimate_dpitch=False かつ use_vanishing_point_pitch=False の場合）度
        fixed_roll: ロール角固定値（estimate_droll=False の場合）度
    """
    max_dx: float = 1.0
    max_dy: float = 1.0
    max_dz: float = 10.0
    max_droll: float = 0.2  # レガシー実装に合わせて±0.2度に制限
    max_dtheta: float = 0.2
    max_dphi: float = 0.2
    max_x: float = 2.0  # x座標累積値制限±2mm（デフォルト）
    max_y: float = 2.0  # y座標累積値制限±2mm（デフォルト）
    max_yaw: float = 15.0  # ヨー角累積値制限±15°（デフォルト）
    max_pitch: float = 15.0  # ピッチ角累積値制限±15°（デフォルト）
    max_roll: float = 1.0  # ロール角累積値制限±1°（デフォルト）

    # 段階的テスト用パラメータ
    estimate_dx: bool = True
    estimate_dz: bool = True
    estimate_droll: bool = True
    estimate_dyaw: bool = False
    estimate_dpitch: bool = False
    use_vanishing_point_yaw: bool = True
    use_vanishing_point_pitch: bool = True
    fixed_yaw: float = 0.0  # 度
    fixed_pitch: float = 0.0  # 度
    fixed_roll: float = 0.0  # 度

    # dz適応的制約: 前フレームdzに基づく初期値・探索範囲制限
    dz_adaptive_bounds_enabled: bool = True  # dz適応的制約の有効/無効
    dz_adaptive_margin: float = 2.0  # 前フレームdzからの許容マージン(mm)

    # yaw/pitch推定方法の選択
    fix_angles_to_vanishing_point: bool = True

    # ハイブリッド2段階推定
    hybrid_two_stage_enabled: bool = False  # 2段階推定の有効/無効
    hybrid_stage1_loss: str = "soft_l1"     # 第1段階の損失関数
    hybrid_stage1_dz_min: float = 0.0       # dz探索下限(mm)
    hybrid_stage1_dz_max: Optional[float] = None  # dz探索上限(mm)、Noneでmax_dz使用
    hybrid_stage1_ocr_margin_multiplier: float = 3.0  # OCR平均dz × この倍率 = Stage1 max_dz

    # VP角度乖離時のクランプ/フォールバック制御
    vp_angle_clamp_degrees: float = 2.0    # VP角度クランプ閾値（度）: これ以下はVP信頼、方向を保持して制限
    vp_angle_fallback_degrees: float = 5.0  # 6DOFフォールバック閾値（度）: VP異常と判断する極端な値
    """
    fix_angles_to_vanishing_point: yaw/pitch角を消失点に固定するか
        True: Phase1の消失点にyaw/pitch角を固定（dyaw/dpitchは消失点から計算した値に固定）
        False: 消失点を初期値・制約範囲として、最小二乗法でdyaw/dpitchを推定
    """


@dataclass
class SpatialFilterConfig:
    """空間フィルタ設定
    
    Attributes:
        enabled: 空間フィルタの有効/無効
        window_size: フィルタウィンドウサイズ
        threshold: フィルタ閾値
    """
    enabled: bool = True
    window_size: int = 5
    threshold: float = 0.5


@dataclass
class PoseConfig:
    """姿勢推定設定

    Attributes:
        use_vanishing_point: 消失点を使用するか
        vanishing_point_weight: 消失点の重み（0.0-1.0）
        pose_smoothing_window: 姿勢平滑化ウィンドウサイズ
        dark_threshold: 暗部検出の二値化閾値（0-255、推奨50-80）
            use_adaptive_threshold=Trueの場合は無視される
        dark_region_min_area: 暗部の最小面積（ピクセル数、推奨100）
        use_adaptive_threshold: 適応的閾値選択を使用するか
        target_dark_area: 目標暗部ピクセル数（適応的閾値選択用、推奨2000）
        threshold_search_range: 適応的閾値探索範囲 (min, max)
        dark_region_radius_limit_ratio: 画像幅に対する暗部検出半径制限の比率
        brightness_check_enabled: レンズ中心領域の明るさチェックを有効にするか
        brightness_check_radius: レンズ中心領域の明るさチェック半径（ピクセル）
        brightness_threshold: レンズ中心領域の明るさ閾値（0-255）
            この値を超える場合、DarkRegionNotFoundErrorを発生させる
        bright_pixel_count_threshold: 明るいピクセル数の閾値
        consecutive_brightness_detection_threshold: 明るさ検出が連続した場合にスキップモードに移行する閾値（フレーム数）
            デフォルト300。この回数以上連続で明るさ検出が発生した場合、以降は明るさチェックをスキップしてレンズ中心を消失点として使用する
        brightness_fallback_method: 明るさ検出時の消失点フォールバック方法
            "lens_center": レンズ中心（cx, cy）を使用（現行方式）
            "average": 明るさ検知前までの消失点平均値を使用
        brightness_center_comparison_enabled: 中心領域と周辺領域の明るいピクセル数比較を有効にするか
            Trueの場合、中心領域の明るいピクセル数が周辺領域より多い場合のみ明るさ検出エラーを発生させる
    """
    use_vanishing_point: bool = True
    vanishing_point_weight: float = 0.3
    pose_smoothing_window: int = 5
    dark_threshold: int = 30
    dark_region_min_area: int = 100

    # 適応的閾値選択パラメータ
    use_adaptive_threshold: bool = True
    target_dark_area: int = 2000
    threshold_search_range: Tuple[int, int] = (5, 100)
    dark_region_radius_limit_ratio: float = 0.5  # 画像幅に対する暗部検出半径制限の比率
    
    # Task 4.2.7: レンズ中心領域の明るさチェック
    brightness_check_enabled: bool = True
    brightness_check_radius: int = 30  # default_config準拠
    brightness_threshold: int = 30  # 明るいピクセル判定の閾値（0-255）default_config準拠
    bright_pixel_count_threshold: int = 50  # 明るいピクセル数の閾値
    consecutive_brightness_detection_threshold: int = 10  # 明るさ検出連続回数閾値 default_config準拠

    # TASK-31: 立坑明かりと管路出口明かりの区別
    brightness_center_comparison_enabled: bool = True
    """中心領域と周辺領域の明るいピクセル数比較を有効にするか

    Trueの場合、中心領域（brightness_check_radius以内）の明るいピクセル数が
    周辺領域（brightness_check_radius超 〜 dark_region_radius_limit_ratio以内）
    の明るいピクセル数より多い場合のみ、明るさ検出としてエラーを発生させる。

    これにより、立坑付近の全体的な明かりと管路出口の中心集中明かりを区別できる。
    """

    brightness_center_density_threshold: float = 0.10
    """中心領域の明るいピクセル密度の絶対閾値 (0.0-1.0)

    center_density（中心領域の明るいピクセル密度）がこの値を超えた場合、
    密度比較の結果に関わらず明るさ検出を発動させる。

    暗い管路内では周辺領域（管壁）の方が常に明るいため、密度比較だけでは
    管路出口からの漸増的な光侵入を検出できない。この絶対値閾値により、
    中心領域に一定以上の光が侵入した時点で検出可能となる。

    brightness_center_comparison_enabled=True の場合のみ有効。
    """
    
    # TASK-22: 明るさ検出時のフォールバック方法
    brightness_fallback_method: str = "lens_center"
    """明るさ検出時の消失点フォールバック方法
    
    値:
        "lens_center": レンズ中心（cx, cy）を使用（現行方式）
        "average": 明るさ検知前までの消失点平均値を使用
    """
    
    def __post_init__(self):
        """初期化後の処理: brightness_fallback_methodのバリデーション"""
        if self.brightness_fallback_method not in ("lens_center", "average"):
            raise ValueError(
                f"PoseConfig.brightness_fallback_method は 'lens_center' または 'average' である必要があります: "
                f"{self.brightness_fallback_method}"
            )




@dataclass
class EstimationConfig:
    """推定設定（カメラ位置・姿勢推定の統合設定）

    Attributes:
        max_abs_yaw_degrees: カメラ姿勢推定で許容するYaw角の最大絶対値（度）
            Phase 2のカメラ姿勢推定時に使用。消失点から推定されたYaw角が
            この範囲を超える場合、この範囲内にクリップされる。
            Noneの場合は制約なし。
        max_abs_pitch_degrees: カメラ姿勢推定で許容するPitch角の最大絶対値（度）
            Phase 2のカメラ姿勢推定時に使用。消失点から推定されたPitch角が
            この範囲を超える場合、この範囲内にクリップされる。
            Noneの場合は制約なし。
        feature_matching: 特徴点マッチング設定
        motion: 移動制約設定
        spatial_filter: 空間フィルタ設定
        pose: 姿勢推定設定
        vanishing_point: 消失点推定設定（Phase 1専用）
        use_vanishing_point_constraints: 消失点スムージングによる角度制約を使用するか
        use_ocr_z_constraints: OCR高精度推定によるZ位置制約を使用するか
        constraint_margin_ratio: 制約範囲のマージン比率
        vanishing_point_smoothing_factor: 消失点スプライン補間のスムージング係数
        vanishing_point_spline_order: 消失点スプラインの次数
        pitch_yaw_range_margin: ピッチ角・ヨー角範囲のマージン（ラジアン）
        z_range_tolerance: Z位置範囲の許容誤差（mm）
    """
    max_abs_yaw_degrees: Optional[float] = 10.0
    max_abs_pitch_degrees: Optional[float] = 15.0
    feature_matching: FeatureMatchingConfig = field(default_factory=FeatureMatchingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    spatial_filter: SpatialFilterConfig = field(default_factory=SpatialFilterConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    vanishing_point: Optional[Any] = None  # VanishingPointConfigまたはNone
    feature_filtering: FeatureFilteringConfig = field(default_factory=FeatureFilteringConfig)
    static_frame_detection: StaticFrameDetectionConfig = field(default_factory=StaticFrameDetectionConfig)

    def __post_init__(self):
        """初期化後の処理: vanishing_point設定のデフォルト値設定"""
        if self.vanishing_point is None:
            if VANISHING_POINT_AVAILABLE and VanishingPointConfig is not None:
                self.vanishing_point = VanishingPointConfig()
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "VanishingPointConfigが利用できません。"
                    "vanishing_point_estimator.pyが正しく実装されているか確認してください。"
                )
                self.vanishing_point = None


    
    # ========================================
    # Phase2 タスク1.3: 探索範囲絞り込みパラメータ
    # ========================================

    use_flexible_estimation: bool = False
    """柔軟なパラメータ推定（段階的テスト用）を使用するか"""

    use_vanishing_point_constraints: bool = True
    """消失点スムージングによる角度制約を使用するか"""

    use_ocr_z_constraints: bool = True
    """OCR高精度推定によるZ位置制約を使用するか"""
    
    constraint_margin_ratio: float = 0.5
    """制約範囲のマージン比率（0.5 = ±50%の余裕）"""
    
    # スムージングパラメータ
    vanishing_point_smoothing_method: str = "savgol"
    """消失点スムージング手法

    - "savgol": Savitzky-Golay平滑化（推奨）
      緩やかに変化する消失点座標のトレンドを捉える。
      個々の点にフィットさせるのではなく、滑らかな曲線を得る。
      異常値の影響を受けにくく、数値的に安定。

    - "spline": スプライン補間（旧方式）
      代表点を守るように補間。局所的な異常値の影響を受けやすい。

    設計方針:
    消失点は粗い姿勢方向推定の初期値として使用され、
    最小二乗法での精緻な推定のガイドとして機能する。
    したがって、個々の点に厳密にフィットさせる必要はなく、
    滑らかなトレンドを捉えることが重要。
    """

    # Savitzky-Golay平滑化パラメータ（method="savgol"の場合）
    savgol_window_length: int = 11
    """Savitzky-Golayウィンドウ長（奇数、デフォルト11）

    推奨値: 9-15フレーム
    - 小さい値（7-9）: 応答性高い、局所的な変化を捉える
    - 中程度（11-13）: バランス良好（推奨）
    - 大きい値（15-21）: 非常に滑らか、応答性低下
    """

    savgol_polyorder: int = 2
    """Savitzky-Golay多項式次数（デフォルト2）

    推奨値: 2（2次多項式）
    - 1: 線形（過度に平滑化）
    - 2: 2次曲線（推奨、緩やかな変化に適合）
    - 3: 3次曲線（複雑な変化、オーバーフィット注意）
    """

    # スプライン補間パラメータ（method="spline"の場合、旧方式）
    vanishing_point_smoothing_factor: float = 0.1
    """消失点スプライン補間のスムージング係数（0=補間、>0=平滑化）"""

    vanishing_point_spline_order: int = 3
    """消失点スプラインの次数（1=線形、3=3次）"""

    vanishing_point_window_size: Optional[int] = 30
    """消失点スプライン補間のウィンドウサイズ（フレーム数）

    長範囲スプライン補間の数値不安定性を防ぐためのウィンドウ分割補間設定。
    - None: ウィンドウ分割せず全フレーム一括補間（100フレーム未満推奨）
    - 20-30: ウィンドウ分割補間（100フレーム以上で推奨）

    100フレーム以上の動画では、natural spline境界条件により数値不安定性が発生し、
    フレーム87付近から消失点座標が発散する問題が確認されている。
    この問題の対策として、window_sizeを指定することで、
    短い範囲ごとに独立したスプライン補間を実行し、境界で線形ブレンドを行う。
    """

    use_moving_average_preprocess: bool = False
    """消失点スプライン補間前の移動平均前処理を使用するか

    消失点（暗部中心）がフレーム間で大きくジャンプする場合の対策。
    移動平均で生データを均してからスプライン補間を行うことで、
    ジャンプ境界での数値不安定性を防止する。

    - True: 移動平均前処理を適用（推奨）
    - False: 前処理なし（従来動作）

    暗部検出の失敗や光の反射などにより、消失点座標が局所的に大きく変化する場合、
    スプライン補間が不安定になる問題が確認されている。
    移動平均前処理により、このようなジャンプを平滑化する。
    """

    moving_average_window: int = 5
    """移動平均ウィンドウサイズ（フレーム数）

    消失点の移動平均前処理に使用するウィンドウサイズ。
    推奨値: 5-10フレーム

    - 小さい値（3-5）: ジャンプを適度に平滑化、応答性維持
    - 大きい値（10-20）: より滑らかだが、応答性低下
    """

    pitch_yaw_range_margin: float = 0.05
    """ピッチ角・ヨー角範囲のマージン（ラジアン）"""

    z_range_tolerance: float = 2.0
    """Z位置範囲の許容誤差（mm）"""

    # ========================================
    # オフセット移動平均距離推定法のパラメータ
    # ========================================

    use_offset_moving_average: bool = True
    """オフセット移動平均距離推定法を使用するか（False=既存の平均速度手法）"""

    offset_window_size: int = 10
    """オフセット最適化ウィンドウサイズ（フレーム数）"""

    offset_overlap: int = 5
    """オフセット最適化ウィンドウのオーバーラップ（フレーム数）"""

    offset_e_resolution: float = 0.1
    """オフセット探索解像度（mm）"""

@dataclass
class ColormapConfig:
    """カラーマップ生成設定

    Attributes:
        inner_radius_ratio: 内側トリミング半径比率（0.0-1.0）
        outer_radius_ratio: 外側トリミング半径比率（0.0-1.0）
        pixels_per_mm: カラーマップの解像度（ピクセル/mm）
        blend_mode: ブレンドモード（overwrite, alpha, max等）
        alpha: z方向変形量に対する最大増加率（推奨1.2）
        trim_left_px: 左側トリミングピクセル数
        trim_right_px: 右側トリミングピクセル数
        enable_z_length_limit: 1フレームあたりのZ方向長さ制限を有効化するか
        max_z_pixels_per_frame: 1フレームあたりの最大Z方向ピクセル数
        delete_temporary_images: Phase3補正完了後に一時展開画像を削除するか
            True: 一時ファイルを削除（本番運用推奨、ディスク容量節約）
            False: 一時ファイルを保持（デバッグ・検証用）
    """
    inner_radius_ratio: float = 0.6
    outer_radius_ratio: float = 0.815
    pixels_per_mm: float = 1.0
    blend_mode: str = "overwrite"
    alpha: float = 1.2  # z方向変形量に対する最大増加率
    trim_left_px: int = 0  # 左側トリミングピクセル数
    trim_right_px: int = 0  # 右側トリミングピクセル数
    
    # Task 2.5.2: 1フレームあたりの展開画像長さ制限（メモリ不足対策）
    enable_z_length_limit: bool = True
    max_z_pixels_per_frame: float = 50.0
    max_z_length_per_frame: Optional[float] = None
    """1フレームあたりの最大Z方向長さ（mm）

    値:
        None（デフォルト）: 未設定、max_z_pixels_per_frameを使用（後方互換）
        > 0.0: mm単位で長さを指定、pixels_per_mmと連動してピクセル値を自動計算

    詳細:
        設定時はpixels_per_mmと連動してピクセル値を自動計算します。
        未設定時はmax_z_pixels_per_frameを使用します（後方互換性維持）。
        
        例:
            max_z_length_per_frame=50.0, pixels_per_mm=1.0
            → 内部でmax_z_pixels=50.0に変換
            
            max_z_length_per_frame=50.0, pixels_per_mm=2.0
            → 内部でmax_z_pixels=100.0に変換
    """
    
    
    # TASK-PROD-002: Phase3完了後の一時ファイル削除
    delete_temporary_images: bool = True


@dataclass
class OCRConfig:
    """OCR設定

    Attributes:
        enabled: OCRの有効/無効
        tesseract_cmd: Tesseract実行ファイルのパス（Noneの場合は環境変数を使用）
        tesseract_config: Tesseractの設定文字列
        preprocessing_enabled: 強化前処理パイプラインの有効/無効
        confidence_threshold: 信頼度閾値（0-100）
        max_distance_increment_mm: OCR距離の1フレームあたり最大増加量(mm)
        preprocessing_method: デフォルト前処理方式（"enhanced", "fixed", "otsu"）
        target_roi_height: 動的拡大の目標ROI高さ（ピクセル）
        clahe_clip_limit: CLAHE のclipLimit
        clahe_tile_grid_size: CLAHE のtileGridSize（正方形のサイズ）
        adaptive_threshold_block_size: 適応的2値化のblockSize（奇数）
        adaptive_threshold_c: 適応的2値化の定数C
        morphology_kernel_size: モルフォロジーカーネルサイズ
        retry_thresholds: 従来方式リトライ時に試行する閾値リスト
    """
    enabled: bool = True
    tesseract_cmd: Optional[str] = None
    tesseract_config: str = "--psm 7 -c tessedit_char_whitelist=0123456789.m"
    preprocessing_enabled: bool = True
    confidence_threshold: float = 60.0
    max_distance_increment_mm: float = 10.0
    preprocessing_method: str = "enhanced"
    target_roi_height: int = 200
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: int = 2
    morphology_kernel_size: int = 2
    retry_thresholds: Optional[List[int]] = None


@dataclass
class InitialPositionConfig:
    """初期位置設定
    
    Attributes:
        x: 初期x座標（mm）
        y: 初期y座標（mm）
        z: 初期z座標（mm）
        theta: 初期左右方向姿勢（度）
        phi: 初期上下方向姿勢（度）
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    theta: float = 0.0
    phi: float = 0.0


@dataclass
class DebugConfig:
    """デバッグ設定

    Attributes:
        enabled: デバッグモードの有効/無効
        show_feature_points: 特徴点の表示
        show_vanishing_point: 消失点の表示
        show_roi: ROI領域の表示
        show_grid: グリッドの表示
        show_dark_region: 暗部領域の表示
        show_front_indicator: 正面方向インジケータ表示
        show_colormap: カラーマッププレビュー表示
        show_frame_window: フレームウィンドウの表示
        console_output: コンソール出力のON/OFF
        save_intermediate: 中間結果保存のON/OFF
        save_interval: 保存間隔（フレーム数）
        output_dir: 保存先ディレクトリ
        output_debug_images: デバッグ画像の出力
        debug_image_path: デバッグ画像出力先パス
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
    show_feature_points: bool = True
    show_vanishing_point: bool = True
    show_roi: bool = True
    show_grid: bool = True
    show_dark_region: bool = False
    show_front_indicator: bool = False
    show_colormap: bool = False
    show_frame_window: bool = False
    console_output: bool = False
    save_intermediate: bool = False
    save_interval: int = 100
    output_dir: str = "data/output/debug"
    output_debug_images: bool = False
    debug_image_path: str = "data/output/debug/"
    window_size: Tuple[int, int] = (1920, 1080)
    grid_color: Tuple[int, int, int] = (0, 255, 0)
    grid_thickness: int = 1
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_scale: float = 0.5
    text_thickness: int = 1
    color_lens_center: Tuple[int, int, int] = (0, 0, 255)
    color_valid_area: Tuple[int, int, int] = (0, 255, 255)
    color_dark_contour: Tuple[int, int, int] = (255, 255, 0)
    color_dark_center: Tuple[int, int, int] = (0, 255, 0)
    color_front_indicator: Tuple[int, int, int] = (0, 255, 255)


@dataclass
class LoggingConfig:
    """ログ設定

    Attributes:
        level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        file: ログファイルのパス
        format: ログフォーマット
        console_output: コンソール出力の有効/無効
        max_bytes: ログファイルのサイズ上限（バイト）。0で無制限
        backup_count: ローテーション時に保持するバックアップファイル数
    """
    level: str = "WARNING"
    file: str = "data/output/process.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = True
    max_bytes: int = 5_000_000  # 5MB
    backup_count: int = 3


# ========================================
# メイン設定クラス
# ========================================

@dataclass
class Config:
    """全設定を統合するメインクラス

    Attributes:
        input: 入力設定
        output: 出力設定
        camera: カメラ設定
        pipe: 管路設定
        estimation: 推定設定
        colormap: カラーマップ生成設定
        ocr: OCR設定
        initial_position: 初期位置設定
        correction: 展開画像補正設定
        video: 動画処理設定
        debug: デバッグ設定
        logging: ログ設定
    """
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    pipe: PipeConfig = field(default_factory=PipeConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    colormap: ColormapConfig = field(default_factory=ColormapConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    initial_position: InitialPositionConfig = field(default_factory=InitialPositionConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Phase 2: 自動チューニング機能
    auto_tune_enabled: bool = False
    """
    動画自動チューニング機能の有効化
    
    値:
        False（デフォルト）: 既存の設定値をそのまま使用
        True: 動画を分析してパラメータを自動調整
    
    注意:
        - auto_tune_enabled=Trueの場合、処理前に30フレーム程度の分析時間が追加される
        - 調整結果はログに出力される
    """
    
    auto_tune_sample_frames: Optional[int] = 30
    """自動チューニングで分析するフレーム数（デフォルト: 30）

    Noneの場合、処理対象範囲の全フレームをframe_skip間隔でサンプリングする。
    """
    
    auto_tune_frame_skip: int = 0
    """自動チューニングでのフレームスキップ間隔（デフォルト: 0=自動計算）

    0の場合、動画全体を均等にカバーするよう自動計算される:
        frame_skip = total_frames // auto_tune_sample_frames
    """

    auto_tune_motion_multiplier: float = 2.0
    """
    自動チューニングでのmotion分析結果に乗じる係数（デフォルト: 2.0）

    max_dz = motion_avg(mm/frame) × auto_tune_motion_multiplier

    実測値との比較で調整:
        - OCR距離差/フレーム数 と 分析結果を比較
        - 分析結果が小さめに出る場合は係数を大きくする
    """

    auto_tune_feature_matching_enabled: bool = True
    """
    特徴点マッチングパラメータ(orb_max_features, match_max_features)の
    実測ベース自動チューニング機能の有効化

    値:
        True（デフォルト）: サンプルフレームペアで実測し最小パラメータを探索
        False: 自動チューニングを無効化（既存の設定値をそのまま使用）

    注意:
        - use_cylindrical_matching=Trueの場合のみ有効
        - ユーザーが設定ファイルでorb_max_features/match_max_featuresを
          明示的に指定した場合、その値は保護される
    """

    auto_tune_feature_matching_sample_pairs: int = 4
    """サンプルフレームペア数（デフォルト: 4）"""

    auto_tune_feature_matching_target_points: int = 20
    """合格判定の最小特徴点数（全フィルタリング後、デフォルト: 20）"""

    auto_tune_feature_matching_pass_ratio: float = 0.75
    """合格判定の通過比率（デフォルト: 0.75 = 75%）"""

    auto_tune_brightness_edge_frames: int = 5
    """明るさベースライン算出時のエッジゾーンサンプル数（デフォルト: 5）

    動画の開始側と終了側にそれぞれNサンプルをエッジゾーンとし、
    中間ゾーンのみでcenter_baseline_brightnessを算出する。
    実際のフレーム範囲は frame_skip × N フレーム
    （frame_skip=100, N=5 の場合、先頭500フレーム・末尾500フレーム）。
    入口光・出口光の影響を排除するために使用。
    """

    # Phase Config-4: 有効フラグパターン（内部使用、ユーザーからは隠蔽）
    _user_defined_paths: Set[str] = field(
        default_factory=set,
        init=False,
        repr=False  # repr()に含めない
    )
    
    @classmethod
    def from_json(cls, config_path: str) -> "Config":
        """JSONファイルから設定を読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            Config: 読み込んだ設定オブジェクト
            
        Raises:
            ConfigFileNotFoundError: 設定ファイルが見つからない
            ConfigFormatError: JSONフォーマットが不正
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigFileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        try:
            with open(path, "r", encoding="utf-8-sig", newline='') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigFormatError(f"JSONフォーマットエラー: {e}")
        except Exception as e:
            raise ConfigError(f"設定ファイル読み込みエラー: {e}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """辞書から設定を構築する
        
        Args:
            config_dict: 設定辞書
            
        Returns:
            Config: 構築した設定オブジェクト
        """
        # ネストした設定を再帰的に構築
        input_cfg = InputConfig(**config_dict.get("input", {}))
        output_cfg = OutputConfig(**config_dict.get("output", {}))
        camera_cfg = CameraConfig(**config_dict.get("camera", {}))
        pipe_cfg = PipeConfig(**config_dict.get("pipe", {}))
        
        # estimation は階層構造
        estimation_dict = config_dict.get("estimation", {})
        
        # cylindrical設定の処理
        feature_matching_dict = estimation_dict.get("feature_matching", {}).copy()
        cylindrical_dict = feature_matching_dict.pop("cylindrical", None)

        # BUG-019修正: use_cylindrical_matchingのデフォルト値をTrueに修正
        # （FeatureMatchingConfigのデフォルト値と一致させる）
        use_cylindrical = feature_matching_dict.get("use_cylindrical_matching", True)

        feature_matching_cfg = FeatureMatchingConfig(**feature_matching_dict)

        # cylindrical設定を別途設定
        if use_cylindrical and cylindrical_dict and CYLINDRICAL_MATCHING_AVAILABLE:
            feature_matching_cfg.cylindrical = CylindricalMatchingConfig(**cylindrical_dict)
        elif use_cylindrical:
            # デフォルト値を使用
            if CYLINDRICAL_MATCHING_AVAILABLE and CylindricalMatchingConfig is not None:
                feature_matching_cfg.cylindrical = CylindricalMatchingConfig()
        
        motion_cfg = MotionConfig(**estimation_dict.get("motion", {}))
        spatial_filter_cfg = SpatialFilterConfig(
            **estimation_dict.get("spatial_filter", {})
        )
        pose_cfg = PoseConfig(**estimation_dict.get("pose", {}))

        # vanishing_point設定の処理
        vanishing_point_dict = estimation_dict.get("vanishing_point", None)
        vanishing_point_cfg = None

        if vanishing_point_dict and VANISHING_POINT_AVAILABLE:
            # TASK-17: max_abs_yaw_degrees, max_abs_pitch_degrees は EstimationConfig のトップレベルに移動
            # vanishing_point セクションから除外（後方互換性のため、あっても警告なしで除去）
            vanishing_point_dict = vanishing_point_dict.copy()
            vanishing_point_dict.pop("max_abs_yaw_degrees", None)
            vanishing_point_dict.pop("max_abs_pitch_degrees", None)

            vanishing_point_cfg = VanishingPointConfig(**vanishing_point_dict)
        elif VANISHING_POINT_AVAILABLE and VanishingPointConfig is not None:
            # デフォルト値を使用
            vanishing_point_cfg = VanishingPointConfig()

        # feature_filtering設定の処理（後方互換性のため、estimation配下とトップレベルの両方をサポート）
        feature_filtering_dict = estimation_dict.get("feature_filtering", config_dict.get("feature_filtering", {}))
        feature_filtering_cfg = FeatureFilteringConfig(**feature_filtering_dict)

        # REFACTOR-002: static_frame_detection設定の処理（後方互換性あり）
        static_frame_detection_dict = estimation_dict.get("static_frame_detection", None)

        # 旧セクション（cylindrical内）からの移行サポート
        if static_frame_detection_dict is None and cylindrical_dict:
            # cylindrical内にstatic_frame_*パラメータが存在するかチェック
            if any(k.startswith("static_frame_") for k in cylindrical_dict):
                static_frame_detection_dict = {
                    "enabled": cylindrical_dict.get("static_frame_detection_enabled", True),
                    "movement_threshold": cylindrical_dict.get("static_frame_movement_threshold", 2.0),
                    "min_points": cylindrical_dict.get("static_frame_min_points", 10)
                }
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "非推奨: static_frame_*パラメータはestimation.feature_matching.cylindricalから"
                    "estimation.static_frame_detectionに移動してください。"
                )

        static_frame_detection_cfg = StaticFrameDetectionConfig(**(static_frame_detection_dict or {}))

        # トップレベルのestimationフィールドを抽出（サブ設定を除く）
        estimation_top_level = {
            k: v for k, v in estimation_dict.items()
            if k not in ["feature_matching", "motion", "spatial_filter", "pose", "vanishing_point", "feature_filtering", "static_frame_detection"]
        }

        estimation_cfg = EstimationConfig(
            feature_matching=feature_matching_cfg,
            motion=motion_cfg,
            spatial_filter=spatial_filter_cfg,
            pose=pose_cfg,
            vanishing_point=vanishing_point_cfg,
            feature_filtering=feature_filtering_cfg,
            static_frame_detection=static_frame_detection_cfg,
            **estimation_top_level
        )
        
        colormap_cfg = ColormapConfig(**config_dict.get("colormap", {}))
        ocr_cfg = OCRConfig(**config_dict.get("ocr", {}))
        initial_position_cfg = InitialPositionConfig(
            **config_dict.get("initial_position", {})
        )
        debug_cfg = DebugConfig(**config_dict.get("debug", {}))
        logging_cfg = LoggingConfig(**config_dict.get("logging", {}))
        correction_cfg = CorrectionConfig(**config_dict.get("correction", {}))

        config = cls(
            input=input_cfg,
            output=output_cfg,
            camera=camera_cfg,
            pipe=pipe_cfg,
            estimation=estimation_cfg,
            colormap=colormap_cfg,
            ocr=ocr_cfg,
            initial_position=initial_position_cfg,
            debug=debug_cfg,
            logging=logging_cfg,
            correction=correction_cfg,
            auto_tune_enabled=config_dict.get("auto_tune_enabled", False),
            auto_tune_sample_frames=config_dict.get("auto_tune_sample_frames", 30),
            auto_tune_frame_skip=config_dict.get("auto_tune_frame_skip", 0),
            auto_tune_motion_multiplier=config_dict.get("auto_tune_motion_multiplier", 2.0),
            auto_tune_feature_matching_enabled=config_dict.get("auto_tune_feature_matching_enabled", True),
            auto_tune_feature_matching_sample_pairs=config_dict.get("auto_tune_feature_matching_sample_pairs", 4),
            auto_tune_feature_matching_target_points=config_dict.get("auto_tune_feature_matching_target_points", 20),
            auto_tune_feature_matching_pass_ratio=config_dict.get("auto_tune_feature_matching_pass_ratio", 0.75),
            auto_tune_brightness_edge_frames=config_dict.get("auto_tune_brightness_edge_frames", 5)
        )
        
        # Phase Config-4: ユーザー定義フィールドパスを抽出して記録
        config._user_defined_paths = cls._extract_defined_paths(config_dict)
        
        return config
    
    @classmethod
    def from_defaults(cls) -> "Config":
        """デフォルト設定を生成する
        
        Returns:
            Config: デフォルト設定オブジェクト
        """
        return cls()
    
    @staticmethod
    def _extract_defined_paths(d: Dict[str, Any], prefix: str = "") -> Set[str]:
        """辞書から定義済みパスを抽出
        
        Args:
            d: 設定辞書
            prefix: 現在のパスプレフィックス
            
        Returns:
            Set[str]: 定義済みパスのセット
            
        Examples:
            >>> Config._extract_defined_paths({"input": {"video_path": "test.mp4"}})
            {"input", "input.video_path"}
            
            >>> Config._extract_defined_paths({
            ...     "estimation": {
            ...         "feature_filtering": {
            ...             "magnitude_iqr_multiplier": 2.0
            ...         }
            ...     }
            ... })
            {
                "estimation",
                "estimation.feature_filtering",
                "estimation.feature_filtering.magnitude_iqr_multiplier"
            }
        """
        paths = set()
        for key, value in d.items():
            current_path = f"{prefix}.{key}" if prefix else key
            paths.add(current_path)
            
            if isinstance(value, dict):
                # 再帰的に子パスも抽出
                paths.update(Config._extract_defined_paths(value, current_path))
        
        return paths
    
    def merge(self, other: "Config") -> "Config":
        """2つの設定をマージする（otherのuser_definedフィールドのみ優先）

        Args:
            other: マージする設定

        Returns:
            Config: マージ後の設定オブジェクト

        Notes:
            Phase Config-4対応: otherのuser_defined_pathsに含まれるフィールドのみが
            selfを上書きします。これにより、部分的なカスタム設定でも、指定されていない
            フィールドは前の層（Layer 2等）の値を保持できます。

        Phase Config-4.1対応（BUG-002修正）: Layer 3のuser_defined_pathsのみを保持。
        Layer 2のパラメータはユーザー定義として扱わず、自動チューニングで上書き可能。

        BUG-019修正: other_dictをuser_defined_pathsに含まれるフィールドのみに絞り込む。
        """
        base_dict = self.to_dict()
        other_dict = other.to_dict()

        # BUG-019修正: user_defined_pathsに含まれるフィールドのみを抽出
        filtered_other_dict = self._filter_by_paths(other_dict, other._user_defined_paths)

        # Phase Config-4: user_definedフィールドのみを上書き
        merged_dict = self._smart_merge(
            base_dict,
            filtered_other_dict,
            other._user_defined_paths
        )

        result = Config.from_dict(merged_dict)

        # Phase Config-4.1（BUG-002修正）: Layer 3のuser_defined_pathsのみを保持
        # Layer 2のパラメータはユーザー定義として扱わない
        result._user_defined_paths = other._user_defined_paths

        return result
    
    @staticmethod
    def _filter_by_paths(d: Dict[str, Any], paths: Set[str], prefix: str = "") -> Dict[str, Any]:
        """辞書をuser_defined_pathsに含まれるフィールドのみに絞り込む

        BUG-019修正用ヘルパー関数。user_defined_pathsに含まれるフィールドのみを
        含む辞書を返します。

        Args:
            d: 元の辞書
            paths: user_defined_paths
            prefix: 現在のパスプレフィックス

        Returns:
            Dict[str, Any]: 絞り込まれた辞書
        """
        result = {}

        for key, value in d.items():
            current_path = f"{prefix}.{key}" if prefix else key

            # このパスがuser_definedか確認
            if current_path in paths:
                if isinstance(value, dict):
                    # 辞書の場合は再帰的にフィルタリング
                    result[key] = Config._filter_by_paths(value, paths, current_path)
                else:
                    result[key] = value
            elif isinstance(value, dict):
                # 親パスがuser_definedでなくても、子パスがuser_definedの可能性
                has_user_defined_child = any(
                    p.startswith(current_path + ".") for p in paths
                )
                if has_user_defined_child:
                    # 子にuser_definedがあれば、この辞書を再帰的にフィルタリング
                    filtered_child = Config._filter_by_paths(value, paths, current_path)
                    if filtered_child:  # 空でなければ追加
                        result[key] = filtered_child

        return result

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """辞書を再帰的にマージする

        Args:
            base: ベース辞書
            override: 上書き辞書

        Returns:
            Dict[str, Any]: マージ後の辞書
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def _smart_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
        user_paths: Set[str],
        prefix: str = ""
    ) -> Dict[str, Any]:
        """user_pathsに含まれるフィールドのみを上書き

        このメソッドは、overrideのuser_defined_pathsに含まれるフィールドのみを
        baseに上書きします。user_definedでないフィールドはbaseの値を保持します。

        修正(BUG-019): ネストした辞書のマージ時に、baseにあってoverrideにない
        フィールドが失われないように、deep copyを実装。

        Args:
            base: ベース辞書
            override: 上書き辞書
            user_paths: ユーザー定義パスのセット
            prefix: 現在のパスプレフィックス

        Returns:
            Dict[str, Any]: マージ後の辞書

        Examples:
            >>> base = {
            ...     "input": {"video_path": "default.mp4"},
            ...     "estimation": {"motion": {"max_dx": 1.0}}
            ... }
            >>> override = {
            ...     "input": {"video_path": "user.mp4"},
            ...     "estimation": {"motion": {"max_dx": 5.0}}
            ... }
            >>> user_paths = {"input", "input.video_path"}
            >>> Config._smart_merge(base, override, user_paths)
            {
                "input": {"video_path": "user.mp4"},  # user_definedなので上書き
                "estimation": {"motion": {"max_dx": 1.0}}  # not user_defined → base保持
            }
        """
        import copy
        result = copy.deepcopy(base)  # BUG-019修正: deepcopyでネストした辞書も確実にコピー

        for key, value in override.items():
            current_path = f"{prefix}.{key}" if prefix else key

            # このパスがuser_definedか確認
            if current_path in user_paths:
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    # 辞書の場合は再帰的にマージ
                    result[key] = Config._smart_merge(
                        result[key],
                        value,
                        user_paths,
                        current_path
                    )
                else:
                    # user_definedフィールドを上書き
                    result[key] = value
            elif isinstance(value, dict) and key in result:
                # 親パスがuser_definedでなくても、子パスがuser_definedの可能性
                # （例: "input"はuser_definedでなくても"input.video_path"がuser_defined）
                has_user_defined_child = any(
                    p.startswith(current_path + ".") for p in user_paths
                )
                if has_user_defined_child:
                    result[key] = Config._smart_merge(
                        result[key],
                        value,
                        user_paths,
                        current_path
                    )

        return result
    
    def validate(self) -> None:
        """全設定のバリデーションを実行
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        validator = ConfigValidator(self)
        validator.validate_all()
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書に変換する
        
        Note:
            _user_defined_pathsはJSON保存時に除外されます（内部管理用フィールド）
        
        Returns:
            Dict[str, Any]: 設定辞書
        """
        result = asdict(self)
        result.pop('_user_defined_paths', None)  # 内部管理用フィールドを除外
        return result
    
    def save(self, path: str) -> None:
        """設定をJSONファイルに保存する
        
        Args:
            path: 保存先パス
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8", newline='') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ========================================
# バリデーションクラス
# ========================================

class ConfigValidator:
    """設定値のバリデーションを行うクラス
    
    Attributes:
        config: バリデーション対象の設定
    """
    
    def __init__(self, config: Config):
        """初期化
        
        Args:
            config: バリデーション対象の設定
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_all(self) -> None:
        """すべての設定をバリデーションする

        Raises:
            ConfigValidationError: バリデーションエラー
        """
        self.validate_input_config()
        self.validate_output_config()
        self.validate_camera_config()
        self.validate_pipe_config()
        self.validate_estimation_config()
        self.validate_colormap_config()
        self.validate_ocr_config()
        self.validate_initial_position_config()
        self.validate_correction_config()
        self.validate_debug_config()
        self.validate_logging_config()
    
    def validate_input_config(self) -> None:
        """入力設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.input
        
        # video_pathは空でない
        if not cfg.video_path:
            raise ConfigValidationError("input.video_pathが空です")
        
        # start_frameは非負
        if cfg.start_frame < 0:
            raise ConfigValidationError(
                f"input.start_frameは0以上である必要があります: {cfg.start_frame}"
            )
        
        # end_frameが指定されている場合、start_frameより大きい
        if cfg.end_frame is not None and cfg.end_frame <= cfg.start_frame:
            raise ConfigValidationError(
                f"input.end_frameはstart_frameより大きい必要があります: "
                f"start={cfg.start_frame}, end={cfg.end_frame}"
            )
    
    def validate_output_config(self) -> None:
        """出力設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.output
        
        # すべてのパスが空でない
        if not cfg.colormap_path:
            raise ConfigValidationError("output.colormap_pathが空です")
        if not cfg.temporary_colormap_path:
            raise ConfigValidationError("output.temporary_colormap_pathが空です")
        if not cfg.report_path:
            raise ConfigValidationError("output.report_pathが空です")
        if not cfg.progress_path:
            raise ConfigValidationError("output.progress_pathが空です")
    
    def validate_camera_config(self) -> None:
        """カメラ設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.camera
        
        # camera_modelバリデーション
        valid_models = ["fisheye", "pinhole"]
        if cfg.camera_model not in valid_models:
            raise ConfigValidationError(
                f"camera.camera_modelは{valid_models}のいずれかである必要があります: {cfg.camera_model}"
            )
        
        # fov_degreesは0-360の範囲
        if not (0 < cfg.fov_degrees <= 360):
            raise ConfigValidationError(
                f"camera.fov_degreesは0-360の範囲である必要があります: {cfg.fov_degrees}"
            )
        
        # ocr_roi_ratioは4要素
        if len(cfg.ocr_roi_ratio) != 4:
            raise ConfigValidationError(
                f"camera.ocr_roi_ratioは4要素である必要があります: {cfg.ocr_roi_ratio}"
            )
        
        # すべての要素が0-1の範囲
        for i, val in enumerate(cfg.ocr_roi_ratio):
            if not (0 <= val <= 1):
                raise ConfigValidationError(
                    f"camera.ocr_roi_ratio[{i}]は0-1の範囲である必要があります: {val}"
                )
        
        # top < bottom, left < right
        top, bottom, left, right = cfg.ocr_roi_ratio
        if top >= bottom:
            raise ConfigValidationError(
                f"camera.ocr_roi_ratio: top < bottom である必要があります: "
                f"top={top}, bottom={bottom}"
            )
        if left >= right:
            raise ConfigValidationError(
                f"camera.ocr_roi_ratio: left < right である必要があります: "
                f"left={left}, right={right}"
            )
        
        # 画像サイズは正の値
        if cfg.image_width <= 0:
            raise ConfigValidationError(
                f"camera.image_widthは正の値である必要があります: {cfg.image_width}"
            )
        if cfg.image_height <= 0:
            raise ConfigValidationError(
                f"camera.image_heightは正の値である必要があります: {cfg.image_height}"
            )
        
        # 焦点距離は正の値
        if cfg.focal_length_mm <= 0:
            raise ConfigValidationError(
                f"camera.focal_length_mmは正の値である必要があります: {cfg.focal_length_mm}"
            )
    
    def validate_pipe_config(self) -> None:
        """管路設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.pipe
        
        # 直径は正の値
        if cfg.diameter_mm <= 0:
            raise ConfigValidationError(
                f"pipe.diameter_mmは正の値である必要があります: {cfg.diameter_mm}"
            )
    
    def validate_estimation_config(self) -> None:
        """推定設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        est = self.config.estimation
        
        # 特徴点マッチング
        fm = est.feature_matching
        if fm.method not in ["ORB", "SIFT", "AKAZE", "BRISK"]:
            self.logger.warning(
                f"estimation.feature_matching.method: 未知の手法です: {fm.method}"
            )
        if fm.max_features <= 0:
            raise ConfigValidationError(
                f"estimation.feature_matching.max_featuresは正の値である必要があります: "
                f"{fm.max_features}"
            )
        if not (0 <= fm.match_threshold <= 1):
            raise ConfigValidationError(
                f"estimation.feature_matching.match_thresholdは0-1の範囲である必要があります: "
                f"{fm.match_threshold}"
            )
        if fm.min_match_count <= 0:
            raise ConfigValidationError(
                f"estimation.feature_matching.min_match_countは正の値である必要があります: "
                f"{fm.min_match_count}"
            )
        
        # モーション制約
        mot = est.motion
        if mot.max_dx < 0:
            raise ConfigValidationError(
                f"estimation.motion.max_dxは非負である必要があります: {mot.max_dx}"
            )
        if mot.max_dy < 0:
            raise ConfigValidationError(
                f"estimation.motion.max_dyは非負である必要があります: {mot.max_dy}"
            )
        if mot.max_dz < 0:
            raise ConfigValidationError(
                f"estimation.motion.max_dzは非負である必要があります: {mot.max_dz}"
            )
        if mot.max_droll < 0:
            raise ConfigValidationError(
                f"estimation.motion.max_drollは非負である必要があります: {mot.max_droll}"
            )
        if mot.max_dtheta < 0:
            raise ConfigValidationError(
                f"estimation.motion.max_dthetaは非負である必要があります: {mot.max_dtheta}"
            )
        if mot.max_dphi < 0:
            raise ConfigValidationError(
                f"estimation.motion.max_dphiは非負である必要があります: {mot.max_dphi}"
            )
        
        # 空間フィルタ
        sf = est.spatial_filter
        if sf.window_size <= 0:
            raise ConfigValidationError(
                f"estimation.spatial_filter.window_sizeは正の値である必要があります: "
                f"{sf.window_size}"
            )
        if not (0 <= sf.threshold <= 1):
            raise ConfigValidationError(
                f"estimation.spatial_filter.thresholdは0-1の範囲である必要があります: "
                f"{sf.threshold}"
            )
        
        # 姿勢推定
        pose = est.pose
        if not (0 <= pose.vanishing_point_weight <= 1):
            raise ConfigValidationError(
                f"estimation.pose.vanishing_point_weightは0-1の範囲である必要があります: "
                f"{pose.vanishing_point_weight}"
            )
        if pose.pose_smoothing_window <= 0:
            raise ConfigValidationError(
                f"estimation.pose.pose_smoothing_windowは正の値である必要があります: "
                f"{pose.pose_smoothing_window}"
            )
    
    def validate_colormap_config(self) -> None:
        """カラーマップ設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.colormap
        
        # 半径比率は0-1の範囲
        if not (0 <= cfg.inner_radius_ratio <= 1):
            raise ConfigValidationError(
                f"colormap.inner_radius_ratioは0-1の範囲である必要があります: "
                f"{cfg.inner_radius_ratio}"
            )
        if not (0 <= cfg.outer_radius_ratio <= 1):
            raise ConfigValidationError(
                f"colormap.outer_radius_ratioは0-1の範囲である必要があります: "
                f"{cfg.outer_radius_ratio}"
            )
        
        # inner < outer
        if cfg.inner_radius_ratio >= cfg.outer_radius_ratio:
            raise ConfigValidationError(
                f"colormap.inner_radius_ratio < outer_radius_ratio である必要があります: "
                f"inner={cfg.inner_radius_ratio}, outer={cfg.outer_radius_ratio}"
            )
        
        # 解像度は正の値
        if cfg.pixels_per_mm <= 0:
            raise ConfigValidationError(
                f"colormap.pixels_per_mmは正の値である必要があります: {cfg.pixels_per_mm}"
            )
        
        # ブレンドモード
        if cfg.blend_mode not in ["overwrite", "alpha", "max", "average"]:
            self.logger.warning(
                f"colormap.blend_mode: 未知のモードです: {cfg.blend_mode}"
            )
        
        # Task 2.5.2: Z長さ制限パラメータ
        if cfg.max_z_pixels_per_frame <= 0:
            raise ConfigValidationError(
                f"colormap.max_z_pixels_per_frameは正の値である必要があります: "
                f"{cfg.max_z_pixels_per_frame}"
            )
        
        # TASK-24: max_z_length_per_frame
        if cfg.max_z_length_per_frame is not None and cfg.max_z_length_per_frame <= 0:
            raise ConfigValidationError(
                f"colormap.max_z_length_per_frameは正の値である必要があります: "
                f"{cfg.max_z_length_per_frame}"
            )
    
    def validate_ocr_config(self) -> None:
        """OCR設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.ocr
        
        # 信頼度閾値は0-100の範囲
        if not (0 <= cfg.confidence_threshold <= 100):
            raise ConfigValidationError(
                f"ocr.confidence_thresholdは0-100の範囲である必要があります: "
                f"{cfg.confidence_threshold}"
            )
    
    def validate_initial_position_config(self) -> None:
        """初期位置設定のバリデーション

        Raises:
            ConfigValidationError: バリデーションエラー
        """
        # 特に制約なし（すべての値が許容される）
        pass

    def validate_correction_config(self) -> None:
        """展開画像補正設定のバリデーション

        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.correction

        # ocr_interval_mmは正の値
        if cfg.ocr_interval_mm <= 0:
            raise ConfigValidationError(
                f"correction.ocr_interval_mmは正の値である必要があります: {cfg.ocr_interval_mm}"
            )

        # max_image_sizeは正の値
        if cfg.max_image_size <= 0:
            raise ConfigValidationError(
                f"correction.max_image_sizeは正の値である必要があります: {cfg.max_image_size}"
            )

    def validate_debug_config(self) -> None:
        """デバッグ設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.debug
        
        # debug_image_pathは空でない
        if not cfg.debug_image_path:
            raise ConfigValidationError("debug.debug_image_pathが空です")
    
    def validate_logging_config(self) -> None:
        """ログ設定のバリデーション
        
        Raises:
            ConfigValidationError: バリデーションエラー
        """
        cfg = self.config.logging
        
        # ログレベル
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if cfg.level not in valid_levels:
            raise ConfigValidationError(
                f"logging.levelは{valid_levels}のいずれかである必要があります: {cfg.level}"
            )
        
        # ログファイルパスは空でない
        if not cfg.file:
            raise ConfigValidationError("logging.fileが空です")


# ========================================
# ユーティリティ関数
# ========================================

def load_config(custom_config_path: Optional[str] = None) -> Config:
    """設定を階層的に読み込む（3層マージ）

    優先順位（Layer 3が最優先）:
        Layer 1: Config.from_defaults()（プログラムデフォルト値）
        Layer 2: data/config/default_config.json（カテゴリ3最適化パラメータ）
        Layer 3: custom_config_path（カテゴリ4ユーザー設定）

    設定の上書き規則:
        - Layer 2はLayer 1を上書き
        - Layer 3はLayer 2を上書き
        - 各層で指定されていないパラメータは前の層の値を保持

    Args:
        custom_config_path: カスタム設定ファイルのパス（オプション）
            Noneの場合、Layer 1とLayer 2のみが適用される

    Returns:
        Config: 読み込んだ設定オブジェクト

    Raises:
        ConfigError: 設定読み込みエラー
        ConfigValidationError: バリデーションエラー

    Examples:
        >>> # Layer 1 + Layer 2のみ（default_config.jsonの最適化値を使用）
        >>> config = load_config()
        
        >>> # Layer 1 + Layer 2 + Layer 3（カスタム設定で上書き）
        >>> config = load_config("my_config.json")

    Notes:
        - default_config.jsonが存在しない場合、警告を出してLayer 1にフォールバック
        - default_config.jsonの読み込み失敗時も、警告を出してLayer 1にフォールバック
        - Layer 3（custom_config_path）の読み込み失敗時は例外が発生（従来通り）
    """
    logger = logging.getLogger(__name__)

    # Layer 1: プログラム上のデフォルト値（最終フォールバック）
    config = Config.from_defaults()
    logger.debug("Layer 1: プログラムデフォルト値を読み込みました")

    # Layer 2: default_config.json（カテゴリ3最適化パラメータ）
    default_config_path = Path("data/config/default_config.json")
    if default_config_path.exists():
        try:
            default_config = Config.from_json(str(default_config_path))
            config = config.merge(default_config)
            # TASK-32: Layer 2はユーザー定義として扱わない（自動チューニングで上書き可能）
            # Phase Config-4.1のドキュメント通りの動作を実装
            config._user_defined_paths = set()
            logger.info(f"Layer 2: デフォルト設定を読み込みました: {default_config_path}")
        except Exception as e:
            logger.warning(
                f"Layer 2: デフォルト設定の読み込みに失敗しました: {e}\n"
                f"プログラムデフォルト値を使用します"
            )
            # プログラムデフォルト値のまま継続
    else:
        logger.debug(f"Layer 2: デフォルト設定ファイルが見つかりません: {default_config_path}")

    # Layer 3: カスタム設定（カテゴリ4ユーザー設定）
    if custom_config_path:
        custom_config = Config.from_json(custom_config_path)
        config = config.merge(custom_config)
        logger.info(f"Layer 3: カスタム設定を読み込みました: {custom_config_path}")

    # バリデーション
    config.validate()

    return config



# ========================================
# モジュールレベルの設定
# ========================================

# ロガー設定
logger = logging.getLogger(__name__)
