"""動画自動チューニング機能

このモジュールは、動画ごとの視認性・ノイズ特性に応じて、
処理パラメータを自動調整する機能を提供します。

主要機能:
    - 動画特性の分析（明るさ、ノイズ、動き量、特徴点検出可能性）
    - パラメータの自動調整（特徴点検出、マッチング、フィルタリング、VP推定）
    - 調整後の設定の保存

設計方針:
    - 後方互換性の維持（auto_tune_enabled=Falseがデフォルト）
    - 失敗時のフォールバック（デフォルト値に戻す）
    - ログ出力の充実

使用例:
    >>> config = Config.from_json("config.json")
    >>> auto_tuner = AutoTuner(config, "path/to/video.mp4")
    >>> video_stats = auto_tuner.analyze_video()
    >>> tuned_config = auto_tuner.tune_parameters(video_stats)
    >>> tuned_config.save("tuned_config.json")
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable

from src.config import Config

logger = logging.getLogger(__name__)


# ========================================
# VideoAnalyzerクラス
# ========================================


class VideoAnalyzer:
    """動画特性の分析
    
    このクラスは、動画の最初のN_FRAMESフレームを分析し、
    明るさ、ノイズレベル、動き量、特徴点検出可能性を推定します。
    
    Attributes:
        video_path: 入力動画のパス
        logger: ロガー
    """
    
    def __init__(self, video_path: str, logger: Optional[logging.Logger] = None):
        """初期化
        
        Args:
            video_path: 入力動画のパス
            logger: ロガー（Noneの場合はモジュールロガーを使用）
        
        Raises:
            FileNotFoundError: 動画ファイルが見つからない
        """
        self.video_path = video_path
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
        # 動画ファイルの存在確認
        if not Path(video_path).exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
    
    def analyze_brightness(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """明るさ分析
        
        フレームのグレースケール平均輝度を計算し、明るさレベルを推定します。
        
        Args:
            frames: 分析対象のフレームリスト
        
        Returns:
            {
                'mean_brightness': float,  # 平均輝度 [0-255]
                'std_brightness': float,   # 標準偏差
                'dark_ratio': float        # 暗部比率 [0-1]
            }
        """
        if not frames:
            self.logger.warning("明るさ分析: フレームが空です")
            return {
                'mean_brightness': 128.0,
                'std_brightness': 50.0,
                'dark_ratio': 0.5
            }
        
        brightness_values = []
        dark_pixel_counts = []
        total_pixels = frames[0].shape[0] * frames[0].shape[1]
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(gray.mean())
            dark_pixel_counts.append(np.sum(gray < 50))
        
        mean_brightness = float(np.mean(brightness_values))
        std_brightness = float(np.std(brightness_values))
        dark_ratio = float(np.mean(dark_pixel_counts) / total_pixels)
        
        self.logger.info(
            f"明るさ分析: 平均輝度={mean_brightness:.1f}, "
            f"標準偏差={std_brightness:.1f}, 暗部比率={dark_ratio:.3f}"
        )
        
        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'dark_ratio': dark_ratio
        }

    def analyze_center_brightness(
        self,
        frames: List[np.ndarray],
        cx: float,
        cy: float,
        radius: int = 30
    ) -> Dict[str, float]:
        """中心領域の輝度分析（TASK-32: brightness_threshold自動調整用）

        レンズ中心付近の輝度を分析し、直視カメラなど中心が明るい動画を検出します。

        Args:
            frames: 分析対象のフレームリスト
            cx: レンズ中心X座標
            cy: レンズ中心Y座標
            radius: 分析対象の中心領域半径（ピクセル）

        Returns:
            {
                'center_min_brightness': float,   # 中心領域の最小輝度の平均
                'center_mean_brightness': float,  # 中心領域の平均輝度の平均
                'center_max_brightness': float    # 中心領域の最大輝度の平均
            }
        """
        if not frames:
            self.logger.warning("中心領域輝度分析: フレームが空です")
            return {
                'center_min_brightness': 0.0,
                'center_mean_brightness': 128.0,
                'center_max_brightness': 255.0
            }

        min_values = []
        mean_values = []
        max_values = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # 中心領域マスクを作成
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
            center_mask = dist_sq <= radius**2

            # 中心領域のピクセルを抽出
            center_pixels = gray[center_mask]

            if len(center_pixels) > 0:
                min_values.append(float(center_pixels.min()))
                mean_values.append(float(center_pixels.mean()))
                max_values.append(float(center_pixels.max()))

        if not min_values:
            self.logger.warning("中心領域輝度分析: 有効なピクセルがありません")
            return {
                'center_min_brightness': 0.0,
                'center_mean_brightness': 128.0,
                'center_max_brightness': 255.0
            }

        center_min = float(np.mean(min_values))
        center_mean = float(np.mean(mean_values))
        center_max = float(np.mean(max_values))

        self.logger.info(
            f"中心領域輝度分析: min={center_min:.1f}, "
            f"mean={center_mean:.1f}, max={center_max:.1f} "
            f"(cx={cx:.1f}, cy={cy:.1f}, radius={radius})"
        )

        return {
            'center_min_brightness': center_min,
            'center_mean_brightness': center_mean,
            'center_max_brightness': center_max
        }

    def analyze_noise_level(self, frames: List[np.ndarray]) -> float:
        """ノイズレベル分析
        
        Laplacian分散を使用してエッジ量（ノイズとエッジが混在）を推定します。
        
        Args:
            frames: 分析対象のフレームリスト
        
        Returns:
            noise_level: float  # ノイズレベル推定値
        """
        if not frames:
            self.logger.warning("ノイズ分析: フレームが空です")
            return 100.0
        
        laplacian_variances = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_variances.append(laplacian.var())
        
        noise_level = float(np.mean(laplacian_variances))
        
        self.logger.info(f"ノイズ分析: Laplacian分散={noise_level:.1f}")
        
        return noise_level
    
    def analyze_motion(
        self,
        frame_pairs: List[Tuple[np.ndarray, np.ndarray]],
        cx: float = None,
        cy: float = None,
        inner_radius_ratio: float = 0.6,
        outer_radius_ratio: float = 0.815,
        top_n_pairs: int = 5,
        iqr_multiplier: float = 1.5
    ) -> Dict[str, float]:
        """動き量分析（連続フレームペア、ドーナツ領域限定、上位ペア採用）

        連続フレームペア間の特徴点移動量を統計して、動き量を推定します。
        特徴点検出はouter_radius_ratio付近のドーナツ領域に限定されます。
        異常値を除外した上で、上位N件のペアの平均を採用します。

        Args:
            frame_pairs: 連続フレームペアのリスト [(frame_i, frame_i+1), ...]
            cx: レンズ中心X座標（Noneの場合は画像中心）
            cy: レンズ中心Y座標（Noneの場合は画像中心）
            inner_radius_ratio: ドーナツ内側半径比率（デフォルト0.6）
            outer_radius_ratio: ドーナツ外側半径比率（デフォルト0.815）
            top_n_pairs: 上位何ペアの平均を採用するか（デフォルト5）
            iqr_multiplier: IQR法の外れ値判定倍率（デフォルト1.5）

        Returns:
            {
                'mean_motion': float,      # 平均動き量 [pixels]（上位ペアの平均）
                'max_motion': float,       # 最大動き量
                'motion_variance': float   # 動き量の分散
            }
        """
        if len(frame_pairs) < 1:
            self.logger.warning("動き量分析: フレームペア数不足")
            return {
                'mean_motion': 0.0,
                'max_motion': 0.0,
                'motion_variance': 0.0
            }

        # 最初のフレームからサイズを取得
        h, w = frame_pairs[0][0].shape[:2]
        if cx is None:
            cx = w / 2.0
        if cy is None:
            cy = h / 2.0

        # ドーナツ領域マスクを作成
        image_radius = min(w, h) / 2.0
        inner_r = int(image_radius * inner_radius_ratio)
        outer_r = int(image_radius * outer_radius_ratio)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), outer_r, 255, -1)
        cv2.circle(mask, (int(cx), int(cy)), inner_r, 0, -1)

        self.logger.info(
            f"動き量分析: ドーナツ領域マスク適用 "
            f"(inner_r={inner_r}px, outer_r={outer_r}px, center=({cx:.1f}, {cy:.1f}))"
        )

        # ORB特徴点検出器を作成
        orb = cv2.ORB_create(nfeatures=500)
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # 各ペアごとの平均移動量を記録
        pair_mean_motions = []

        for frame1, frame2 in frame_pairs:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # マスクを使用して特徴点検出（ドーナツ領域のみ）
            kp1, desc1 = orb.detectAndCompute(gray1, mask)
            kp2, desc2 = orb.detectAndCompute(gray2, mask)

            if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
                continue

            # マッチング
            matches = bf_matcher.knnMatch(desc1, desc2, k=2)

            # Lowe's比率テストでフィルタリング
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 5:
                continue

            # このペアの移動量を計算
            magnitudes = []
            for m in good_matches:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                magnitude = np.sqrt(dx**2 + dy**2)
                magnitudes.append(magnitude)

            # このペアの平均移動量を記録
            pair_mean = float(np.mean(magnitudes))
            pair_mean_motions.append(pair_mean)

        if not pair_mean_motions:
            self.logger.warning("動き量分析: マッチング失敗")
            return {
                'mean_motion': 0.0,
                'max_motion': 0.0,
                'motion_variance': 0.0
            }

        self.logger.info(
            f"動き量分析: {len(pair_mean_motions)}ペアの平均移動量を取得"
        )

        # IQR法で異常値を除外
        pair_motions_array = np.array(pair_mean_motions)
        q1 = np.percentile(pair_motions_array, 25)
        q3 = np.percentile(pair_motions_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # 異常値を除外
        valid_motions = pair_motions_array[
            (pair_motions_array >= lower_bound) & (pair_motions_array <= upper_bound)
        ]

        outlier_count = len(pair_motions_array) - len(valid_motions)
        if outlier_count > 0:
            self.logger.info(
                f"動き量分析: IQR法で{outlier_count}件の異常値を除外 "
                f"(範囲: {lower_bound:.2f}～{upper_bound:.2f}px)"
            )

        if len(valid_motions) == 0:
            self.logger.warning("動き量分析: 異常値除外後にデータがありません")
            return {
                'mean_motion': 0.0,
                'max_motion': 0.0,
                'motion_variance': 0.0
            }

        # 上位N件のペアの平均を採用（移動量が大きい方を採用）
        sorted_motions = np.sort(valid_motions)[::-1]  # 降順ソート
        actual_top_n = min(top_n_pairs, len(sorted_motions))
        top_motions = sorted_motions[:actual_top_n]

        mean_motion = float(np.mean(top_motions))
        max_motion = float(np.max(valid_motions))
        motion_variance = float(np.var(valid_motions))

        self.logger.info(
            f"動き量分析: 上位{actual_top_n}ペアの平均={mean_motion:.2f}px "
            f"(最大={max_motion:.2f}px, 分散={motion_variance:.2f})"
        )
        self.logger.info(
            f"  上位ペアの値: {[f'{v:.2f}' for v in top_motions]}"
        )

        return {
            'mean_motion': mean_motion,
            'max_motion': max_motion,
            'motion_variance': motion_variance
        }

    def analyze_motion_legacy(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """動き量分析（レガシー: 後方互換性用）

        フレーム間の特徴点移動量を統計して、動き量を推定します。
        注意: この方法はスキップされたフレーム間でマッチングするため精度が低いです。

        Args:
            frames: 分析対象のフレームリスト

        Returns:
            {
                'mean_motion': float,      # 平均動き量 [pixels]
                'max_motion': float,       # 最大動き量
                'motion_variance': float   # 動き量の分散
            }
        """
        if len(frames) < 2:
            self.logger.warning("動き量分析: フレーム数不足")
            return {
                'mean_motion': 0.0,
                'max_motion': 0.0,
                'motion_variance': 0.0
            }

        # ORB特徴点検出器を作成
        orb = cv2.ORB_create(nfeatures=500)
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        motion_magnitudes = []

        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            kp1, desc1 = orb.detectAndCompute(gray1, None)
            kp2, desc2 = orb.detectAndCompute(gray2, None)

            if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
                continue

            # マッチング
            matches = bf_matcher.knnMatch(desc1, desc2, k=2)

            # Lowe's比率テストでフィルタリング
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # 移動量計算
            for m in good_matches:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                magnitude = np.sqrt(dx**2 + dy**2)
                motion_magnitudes.append(magnitude)

        if not motion_magnitudes:
            self.logger.warning("動き量分析: マッチング失敗")
            return {
                'mean_motion': 0.0,
                'max_motion': 0.0,
                'motion_variance': 0.0
            }

        mean_motion = float(np.mean(motion_magnitudes))
        max_motion = float(np.max(motion_magnitudes))
        motion_variance = float(np.var(motion_magnitudes))

        self.logger.info(
            f"動き量分析(レガシー): 平均={mean_motion:.2f}px, "
            f"最大={max_motion:.2f}px, 分散={motion_variance:.2f}"
        )

        return {
            'mean_motion': mean_motion,
            'max_motion': max_motion,
            'motion_variance': motion_variance
        }
    
    def analyze_feature_detectability(self, frames: List[np.ndarray]) -> Dict[str, int]:
        """特徴点検出可能性分析
        
        各フレームで検出される特徴点数を統計します。
        
        Args:
            frames: 分析対象のフレームリスト
        
        Returns:
            {
                'mean_feature_count': int,  # 平均特徴点数
                'min_feature_count': int,   # 最小特徴点数
                'max_feature_count': int    # 最大特徴点数
            }
        """
        if not frames:
            self.logger.warning("特徴点検出分析: フレームが空です")
            return {
                'mean_feature_count': 0,
                'min_feature_count': 0,
                'max_feature_count': 0
            }
        
        # ORB特徴点検出器を作成
        orb = cv2.ORB_create(nfeatures=500)
        
        feature_counts = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, _ = orb.detectAndCompute(gray, None)
            feature_counts.append(len(kp) if kp else 0)
        
        mean_count = int(np.mean(feature_counts))
        min_count = int(np.min(feature_counts))
        max_count = int(np.max(feature_counts))
        
        self.logger.info(
            f"特徴点検出分析: 平均={mean_count}, 最小={min_count}, 最大={max_count}"
        )
        
        return {
            'mean_feature_count': mean_count,
            'min_feature_count': min_count,
            'max_feature_count': max_count
        }


# ========================================
# ParameterTunerクラス
# ========================================


class ParameterTuner:
    """パラメータの自動チューニング
    
    このクラスは、VideoAnalyzerが出力した動画統計に基づいて、
    Configオブジェクトのパラメータを調整します。
    
    Attributes:
        base_config: ベース設定（調整前のConfig）
        logger: ロガー
    """
    
    def __init__(self, base_config: Config, logger: Optional[logging.Logger] = None):
        """初期化（ベース設定を保持）
        
        Args:
            base_config: ベース設定
            logger: ロガー（Noneの場合はモジュールロガーを使用）
        """
        self.base_config = base_config
        self.logger = logger if logger is not None else logging.getLogger(__name__)
    
    def tune_all_parameters(self, video_stats: Dict[str, Any]) -> Config:
        """全パラメータを調整
        
        Args:
            video_stats: VideoAnalyzerが出力した統計情報
        
        Returns:
            調整済みConfig
        """
        # ベース設定をコピー（変更は新しいオブジェクトに適用）
        import copy
        tuned_config = copy.deepcopy(self.base_config)
        
        # 動画統計から主要な指標を抽出
        brightness_stats = video_stats.get('brightness', {})
        mean_brightness = brightness_stats.get('mean_brightness', 128.0)
        brightness_level = mean_brightness / 255.0  # 0.0～1.0に正規化
        
        laplacian_variance = video_stats.get('noise_level', 100.0)
        
        motion_stats = video_stats.get('motion', {})
        motion_avg = motion_stats.get('mean_motion', 0.0)
        
        # TASK-32: 中心領域輝度統計を取得
        center_brightness_stats = video_stats.get('center_brightness', {})
        center_min_brightness = center_brightness_stats.get('center_min_brightness', 0.0)
        center_baseline_brightness = center_brightness_stats.get('center_baseline_brightness', 0.0)
        center_baseline_start = center_brightness_stats.get('center_baseline_start', 0.0)
        center_baseline_middle = center_brightness_stats.get('center_baseline_middle', 0.0)
        center_baseline_end = center_brightness_stats.get('center_baseline_end', 0.0)

        self.logger.info(
            f"パラメータ調整開始: brightness_level={brightness_level:.2f}, "
            f"laplacian_variance={laplacian_variance:.1f}, motion_avg={motion_avg:.2f}, "
            f"center_min_brightness={center_min_brightness:.1f}, "
            f"center_baseline_brightness={center_baseline_brightness:.1f} "
            f"(start={center_baseline_start:.1f}, middle={center_baseline_middle:.1f}, end={center_baseline_end:.1f})"
        )

        # ルール適用
        self._apply_rule1_dark_environment(tuned_config, brightness_level)
        self._apply_rule2_noisy_environment(tuned_config, laplacian_variance)
        self._apply_rule3_fast_motion(tuned_config, motion_avg)
        self._apply_rule4_dark_and_noisy(tuned_config, brightness_level, laplacian_variance)
        self._apply_rule5_slow_motion(tuned_config, motion_avg)
        self._apply_rule8_large_pipe_brightness(tuned_config)
        self._apply_rule6_bright_center(tuned_config, center_baseline_brightness)
        self._apply_rule7_motion_based_max_dz(tuned_config, motion_avg)

        self.logger.info("パラメータ調整完了")
        
        return tuned_config
    
    def _get_nested_attr(self, obj: Any, path: str) -> Any:
        """ネストされた属性を取得
        
        Args:
            obj: 対象オブジェクト
            path: ドット区切りのパス（例: "estimation.feature_matching.max_features"）
        
        Returns:
            Any: 取得した値
        """
        parts = path.split('.')
        for part in parts:
            obj = getattr(obj, part)
        return obj
    
    def _set_nested_attr(self, obj: Any, path: str, value: Any) -> None:
        """ネストされた属性を設定
        
        Args:
            obj: 対象オブジェクト
            path: ドット区切りのパス（例: "estimation.feature_matching.max_features"）
            value: 設定する値
        """
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def _set_parameter_if_not_user_defined(
        self,
        config: Config,
        param_path: str,
        new_value: Any,
        description: str = ""
    ) -> bool:
        """ユーザー設定でない場合のみパラメータを設定
        
        ユーザーがカスタム設定ファイルで明示的に設定したパラメータは保護され、
        自動チューニングで上書きされません。
        
        Args:
            config: Config object
            param_path: パラメータパス（例: "estimation.feature_matching.max_features"）
            new_value: 新しい値
            description: パラメータ説明（ログ出力用、省略時はparam_pathを使用）
        
        Returns:
            bool: 設定を実行した場合True、ユーザー設定保護でスキップした場合False
        """
        if param_path in config._user_defined_paths:
            current_value = self._get_nested_attr(config, param_path)
            self.logger.info(f"  {description or param_path}: ユーザー設定を保持 ({current_value})")
            return False
        
        old_value = self._get_nested_attr(config, param_path)
        self._set_nested_attr(config, param_path, new_value)
        self.logger.info(f"  {description or param_path}: {old_value} → {new_value}")
        return True
    
    def _apply_rule1_dark_environment(self, config: Config, brightness_level: float) -> None:
        """ルール1: 暗い環境 (brightness_level < 0.3)
        
        Args:
            config: 調整対象のConfig
            brightness_level: 明るさレベル（0.0～1.0）
        """
        if brightness_level < 0.3:
            self.logger.info("ルール1適用: 暗い環境")
            
            # max_features: 500 → 300
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_matching.max_features",
                300,
                "max_features"
            )
            
            # fast_threshold: 20 → 15
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_matching.fast_threshold",
                15,
                "fast_threshold"
            )
            
            # dark_threshold: 30 → 20
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.pose.dark_threshold",
                20,
                "dark_threshold"
            )
            
            # match_threshold: 0.7 → 0.5
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_matching.match_threshold",
                0.5,
                "match_threshold"
            )
    
    def _apply_rule2_noisy_environment(self, config: Config, laplacian_variance: float) -> None:
        """ルール2: ノイズが多い (laplacian_variance > 300)
        
        Args:
            config: 調整対象のConfig
            laplacian_variance: Laplacian分散
        """
        if laplacian_variance > 300:
            self.logger.info("ルール2適用: ノイズが多い環境")
            
            # magnitude_iqr_multiplier: 1.5 → 2.0
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_filtering.magnitude_iqr_multiplier",
                2.0,
                "magnitude_iqr_multiplier"
            )
            
            # direction_iqr_multiplier: 1.5 → 2.0
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_filtering.direction_iqr_multiplier",
                2.0,
                "direction_iqr_multiplier"
            )
            
            # iqr_multiplier（VP用）: 1.5 → 2.0
            if hasattr(config.estimation, 'vanishing_point') and config.estimation.vanishing_point is not None:
                self._set_parameter_if_not_user_defined(
                    config,
                    "estimation.vanishing_point.iqr_multiplier",
                    2.0,
                    "iqr_multiplier（VP用）"
                )
            
            # ratio_test_threshold: 0.75 → 0.85
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_matching.ratio_test_threshold",
                0.85,
                "ratio_test_threshold"
            )
            
            # direction_n_sectors: 8 → 6
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_filtering.direction_n_sectors",
                6,
                "direction_n_sectors"
            )
    
    def _apply_rule3_fast_motion(self, config: Config, motion_avg: float) -> None:
        """ルール3: 動きが速い (motion_avg > 3.0)

        Args:
            config: 調整対象のConfig
            motion_avg: 平均動き量（ピクセル/フレーム）
        """
        if motion_avg > 3.0:
            self.logger.info(f"ルール3適用: 動きが速い環境 (motion_avg={motion_avg:.2f})")

            # direction_iqr_multiplier: 1.5 → 2.0（重複適用の可能性あり）
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_filtering.direction_iqr_multiplier",
                2.0,
                "direction_iqr_multiplier"
            )

            # max_movement_threshold: 100.0 → 150.0
            # 注: max_distance_thresholdとして実装
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_matching.max_distance_threshold",
                150.0,
                "max_distance_threshold"
            )
            # 注: max_dzはルール7で全速度範囲に対応して設定
    
    def _apply_rule4_dark_and_noisy(
        self,
        config: Config,
        brightness_level: float,
        laplacian_variance: float
    ) -> None:
        """ルール4: 暗くてノイズが多い (複合条件)
        
        Args:
            config: 調整対象のConfig
            brightness_level: 明るさレベル（0.0～1.0）
            laplacian_variance: Laplacian分散
        """
        if brightness_level < 0.3 and laplacian_variance > 300:
            self.logger.info("ルール4適用: 暗くてノイズが多い環境（複合条件）")
            
            # max_features: 500 → 200
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_matching.max_features",
                200,
                "max_features"
            )
            
            # ratio_test_threshold: 0.75 → 0.9
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_matching.ratio_test_threshold",
                0.9,
                "ratio_test_threshold"
            )
            
            # magnitude_iqr_multiplier: 1.5 → 2.5
            self._set_parameter_if_not_user_defined(
                config,
                "estimation.feature_filtering.magnitude_iqr_multiplier",
                2.5,
                "magnitude_iqr_multiplier"
            )
    
    def _apply_rule5_slow_motion(self, config: Config, motion_avg: float) -> None:
        """ルール5: 動きが少ない (motion_avg < 1.5)

        Args:
            config: 調整対象のConfig
            motion_avg: 平均動き量（ピクセル/フレーム）
        """
        if motion_avg < 1.5:
            self.logger.info(f"ルール5適用: 動きが少ない環境 (motion_avg={motion_avg:.2f})")

            # 注: max_dzはルール7で全速度範囲に対応して設定

            # vp_inlier_threshold: 5.0 → 3.0
            if hasattr(config.estimation, 'vanishing_point') and config.estimation.vanishing_point is not None:
                self._set_parameter_if_not_user_defined(
                    config,
                    "estimation.vanishing_point.vp_inlier_threshold",
                    3.0,
                    "vp_inlier_threshold"
                )

    def _apply_rule6_bright_center(
        self,
        config: Config,
        center_baseline_brightness: float
    ) -> None:
        """ルール6: 中心輝度に基づくbrightness_threshold自動調整（TASK-32）

        中立区間代表輝度（20パーセンタイル）に基づいてbrightness_thresholdを調整します。
        - 上方向調整: 明るい管路でbaselineが閾値を超える場合（閾値引き上げ）
        - 下方向調整: 暗い管路で閾値が過大な場合（閾値引き下げ）

        Args:
            config: 調整対象のConfig
            center_baseline_brightness: 中立区間代表輝度（20パーセンタイル）（0-255）
        """
        # 現在のbrightness_thresholdを取得
        current_threshold = config.estimation.pose.brightness_threshold

        # 上方向調整: 中立区間輝度が現閾値より高い場合（明るい管路）
        if center_baseline_brightness > current_threshold:
            self.logger.info(
                f"ルール6適用（上方向）: 中心が明るい動画 "
                f"(baseline={center_baseline_brightness:.1f})"
            )

            # 新しい閾値をbaselineの80%に設定（余裕を持たせる）
            new_threshold = int(center_baseline_brightness * 0.8)

            # 30-200の範囲に制限
            new_threshold = max(30, min(200, new_threshold))

            self._set_parameter_if_not_user_defined(
                config,
                "estimation.pose.brightness_threshold",
                new_threshold,
                "brightness_threshold"
            )

            self.logger.info(
                f"  brightness_threshold上方調整: "
                f"baseline({center_baseline_brightness:.1f}) × 0.8 = {new_threshold}"
            )

        # 下方向調整: 暗い管路で閾値が過大な場合
        # ガード条件: baseline >= 5（極暗管路は調整しない）
        # 条件: baseline < 現閾値の半分（= 閾値が過大）
        elif center_baseline_brightness >= 5 and center_baseline_brightness < current_threshold / 2:
            new_threshold = max(15, int(center_baseline_brightness * 3))

            self.logger.info(
                f"ルール6適用（下方向）: 暗い管路動画 "
                f"(baseline={center_baseline_brightness:.1f}, "
                f"current_threshold={current_threshold})"
            )

            self._set_parameter_if_not_user_defined(
                config,
                "estimation.pose.brightness_threshold",
                new_threshold,
                "brightness_threshold"
            )

            self.logger.info(
                f"  brightness_threshold下方調整: "
                f"baseline({center_baseline_brightness:.1f}) × 3 = {new_threshold}"
            )

    def _apply_rule7_motion_based_max_dz(self, config: Config, motion_avg: float) -> None:
        """ルール7: 動き量に基づくmax_dz調整（TASK-32: 全速度範囲対応）

        平均動き量に基づいてmax_dzを自動調整します。
        max_dz = 平均速度(mm/frame) × auto_tune_motion_multiplier（最低1.0mm保証）

        単位変換の近似方法:
            特徴点はouter_radius_ratioの円周上に分布すると仮定。
            フレーム円周（px） = 2π × image_radius × outer_radius_ratio
            管の周長（mm）     = π × diameter_mm

            mm_per_pixel = (π × diameter_mm) / (2π × image_radius × outer_radius_ratio)
                         = diameter_mm / (2 × image_radius × outer_radius_ratio)

        Args:
            config: 調整対象のConfig
            motion_avg: 平均動き量（ピクセル/フレーム）
        """
        # 画像半径を計算（短辺の半分を使用）
        image_width = config.camera.image_width
        image_height = config.camera.image_height
        image_radius = min(image_width, image_height) / 2.0

        # outer_radius_ratioを取得（colormapの値を使用）
        outer_radius_ratio = config.colormap.outer_radius_ratio

        # 管径を取得
        diameter_mm = config.pipe.diameter_mm

        # motion係数を取得（パラメータで調整可能）
        motion_multiplier = config.auto_tune_motion_multiplier

        # mm/pixel変換係数を計算
        # 円周上のピクセル数と管周長の対応から近似
        mm_per_pixel = diameter_mm / (2.0 * image_radius * outer_radius_ratio)

        # motion_avg (px/frame) → mm/frame に変換
        motion_avg_mm = motion_avg * mm_per_pixel

        # max_dz = 平均速度 × 係数（最低1.0mm保証）
        new_max_dz = max(1.0, motion_avg_mm * motion_multiplier)

        self.logger.info(
            f"ルール7適用: 動き量に基づくmax_dz調整"
        )
        self.logger.info(
            f"  [検証用] 推定平均速度: {motion_avg_mm:.2f} mm/frame "
            f"(ORB特徴点: {motion_avg:.2f}px × {mm_per_pixel:.4f}mm/px)"
        )
        self.logger.info(
            f"  [検証用] 変換係数: diameter_mm={diameter_mm}, "
            f"image_radius={image_radius:.1f}px, outer_radius_ratio={outer_radius_ratio}"
        )
        self.logger.info(
            f"  [検証用] motion_multiplier={motion_multiplier} "
            f"→ reports出力のフレーム間距離と比較してください"
        )

        self._set_parameter_if_not_user_defined(
            config,
            "estimation.motion.max_dz",
            new_max_dz,
            f"max_dz (= {motion_avg_mm:.2f}mm × {motion_multiplier})"
        )

    def _apply_rule8_large_pipe_brightness(self, config: Config) -> None:
        """ルール8: 管径に基づく明るさ検出パラメータ調整

        大口径管では照明が奥まで届くため、brightness検出パラメータを
        管径に応じてスケーリングします。
        基準管径Φ250（デフォルト値が適合）〜Φ1500（ユーザー確認済み値）間で
        線形補間します。

        Args:
            config: 調整対象のConfig
        """
        SMALL_PIPE_MM = 250.0
        LARGE_PIPE_MM = 1500.0

        diameter_mm = config.pipe.diameter_mm

        # 管径が基準範囲外の場合はクランプ
        t = (diameter_mm - SMALL_PIPE_MM) / (LARGE_PIPE_MM - SMALL_PIPE_MM)
        t = max(0.0, min(1.0, t))

        if t <= 0.0:
            self.logger.info("ルール8スキップ: 小口径管（Φ250以下）のため調整不要")
            return

        # 大口径端（Φ1500）の目標値
        LARGE_PIPE_RADIUS_LIMIT_RATIO = 0.3
        LARGE_PIPE_CHECK_RADIUS = 20
        LARGE_PIPE_BRIGHTNESS_THRESHOLD = 70
        LARGE_PIPE_PIXEL_COUNT_THRESHOLD = 300

        # default_configの値をベースとして線形補間
        base_radius_limit = config.estimation.pose.dark_region_radius_limit_ratio
        base_check_radius = config.estimation.pose.brightness_check_radius
        base_brightness_threshold = config.estimation.pose.brightness_threshold
        base_pixel_count = config.estimation.pose.bright_pixel_count_threshold

        self.logger.info(
            f"ルール8適用: 管径ベース明るさパラメータ調整 "
            f"(diameter={diameter_mm}mm, t={t:.3f})"
        )

        # dark_region_radius_limit_ratio: base → 0.3
        new_radius_limit_ratio = base_radius_limit + t * (LARGE_PIPE_RADIUS_LIMIT_RATIO - base_radius_limit)
        self._set_parameter_if_not_user_defined(
            config,
            "estimation.pose.dark_region_radius_limit_ratio",
            round(new_radius_limit_ratio, 4),
            "dark_region_radius_limit_ratio"
        )

        # brightness_check_radius: base → 20
        new_check_radius = int(base_check_radius + t * (LARGE_PIPE_CHECK_RADIUS - base_check_radius))
        self._set_parameter_if_not_user_defined(
            config,
            "estimation.pose.brightness_check_radius",
            new_check_radius,
            "brightness_check_radius"
        )

        # brightness_threshold: base → 70
        new_brightness_threshold = int(base_brightness_threshold + t * (LARGE_PIPE_BRIGHTNESS_THRESHOLD - base_brightness_threshold))
        self._set_parameter_if_not_user_defined(
            config,
            "estimation.pose.brightness_threshold",
            new_brightness_threshold,
            "brightness_threshold"
        )

        # bright_pixel_count_threshold: base → 300
        new_pixel_count_threshold = int(base_pixel_count + t * (LARGE_PIPE_PIXEL_COUNT_THRESHOLD - base_pixel_count))
        self._set_parameter_if_not_user_defined(
            config,
            "estimation.pose.bright_pixel_count_threshold",
            new_pixel_count_threshold,
            "bright_pixel_count_threshold"
        )


# ========================================
# AutoTunerクラス（オーケストレーター）
# ========================================


class AutoTuner:
    """自動チューニングのオーケストレーター
    
    このクラスは、VideoAnalyzerとParameterTunerを統合し、
    動画自動チューニング機能の全体フローを管理します。
    
    Attributes:
        config: Config オブジェクト
        video_path: 入力動画パス
        logger: ロガー
        video_analyzer: 動画分析器
        parameter_tuner: パラメータチューナー
    """
    
    def __init__(
        self,
        config: Config,
        video_path: str,
        logger: Optional[logging.Logger] = None
    ):
        """初期化
        
        Args:
            config: Config オブジェクト
            video_path: 入力動画パス
            logger: ロガー（Noneの場合はモジュールロガーを使用）
        
        Raises:
            FileNotFoundError: 動画ファイルが見つからない
        """
        self.config = config
        self.video_path = video_path
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
        # VideoAnalyzerとParameterTunerを初期化
        self.video_analyzer = VideoAnalyzer(video_path, self.logger)
        self.parameter_tuner = ParameterTuner(config, self.logger)
    
    def analyze_video(
        self,
        num_sample_frames: int = 30,
        frame_skip: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """動画を分析（ストリーミング処理でメモリ効率化）

        フレームを1つずつ読み込み、統計量を逐次計算することでメモリ使用量を削減。
        従来版（全フレームをリストに蓄積）と比較して、メモリ使用量を大幅に削減。

        Args:
            num_sample_frames: 分析フレーム数（デフォルト30）
            frame_skip: フレームスキップ間隔（デフォルト5フレーム）
            progress_callback: 進捗コールバック関数(processed_count, total_count)

        Returns:
            video_stats: 全分析結果を統合した辞書

        Raises:
            RuntimeError: 動画読み込み失敗
        """
        import gc

        self.logger.info(
            f"動画自動チューニング開始（ストリーミング処理）: {self.video_path} "
            f"(num_sample_frames={num_sample_frames}, frame_skip={frame_skip})"
        )

        # 動画を開く
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"動画フレーム数: {total_frames}")

        # 処理対象範囲を決定（config.input.start_frame / end_frame）
        start = self.config.input.start_frame
        end = total_frames
        if self.config.input.end_frame is not None:
            end = min(end, self.config.input.end_frame)
        usable_frames = end - start
        self.logger.info(f"サンプリング対象範囲: [{start}, {end}) ({usable_frames}フレーム)")

        # num_sample_frames=None の場合、全フレームをframe_skip間隔でサンプリング
        if num_sample_frames is None:
            if frame_skip <= 0:
                frame_skip = 1
            self.logger.info(f"num_sample_frames=None: 全フレームをframe_skip={frame_skip}でサンプリング")
        else:
            # frame_skip自動計算（0以下の場合、対象範囲を均等にカバー）
            if frame_skip <= 0:
                frame_skip = max(1, usable_frames // num_sample_frames)
                self.logger.info(f"frame_skip自動計算: {frame_skip}")

        # サンプリング位置を計算
        sample_positions = []
        pos = start
        while (num_sample_frames is None or len(sample_positions) < num_sample_frames) and pos < end:
            sample_positions.append(pos)
            pos += frame_skip

        self.logger.info(f"サンプリング位置数: {len(sample_positions)}")

        # レンズ中心座標を取得（最初のフレームから）
        cx, cy = None, None
        first_frame_shape = None

        # ストリーミング用の統計アキュムレータを初期化
        # Welfordのオンラインアルゴリズム用
        brightness_count = 0
        brightness_mean = 0.0
        brightness_m2 = 0.0  # 分散計算用
        dark_pixel_total = 0
        total_pixels = 0

        noise_count = 0
        noise_sum = 0.0

        feature_counts = []

        # 中心領域輝度用
        center_min_values = []
        center_mean_values = []
        center_max_values = []

        # motion分析用
        pair_mean_motions = []
        inner_radius_ratio = self.config.colormap.inner_radius_ratio
        outer_radius_ratio = self.config.colormap.outer_radius_ratio
        mask = None
        orb = cv2.ORB_create(nfeatures=500)
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        processed_count = 0

        for sample_pos in sample_positions:
            # サンプル位置のフレームを取得
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_pos)
            ret1, frame1 = cap.read()
            if not ret1:
                self.logger.warning(f"フレーム読み込み失敗（position={sample_pos}）")
                continue

            # 最初のフレームでレンズ中心とマスクを初期化
            if first_frame_shape is None:
                first_frame_shape = frame1.shape
                h, w = frame1.shape[:2]
                total_pixels = h * w
                cx, cy = self._get_lens_center(frame1)

                # motion分析用ドーナツマスクを作成
                image_radius = min(w, h) / 2.0
                inner_r = int(image_radius * inner_radius_ratio)
                outer_r = int(image_radius * outer_radius_ratio)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (int(cx), int(cy)), outer_r, 255, -1)
                cv2.circle(mask, (int(cx), int(cy)), inner_r, 0, -1)

            # === 明るさ分析（Welfordのオンラインアルゴリズム） ===
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            brightness = float(gray1.mean())
            brightness_count += 1
            delta = brightness - brightness_mean
            brightness_mean += delta / brightness_count
            delta2 = brightness - brightness_mean
            brightness_m2 += delta * delta2
            dark_pixel_total += np.sum(gray1 < 50)

            # === ノイズ分析 ===
            laplacian = cv2.Laplacian(gray1, cv2.CV_64F)
            noise_count += 1
            noise_sum += laplacian.var()

            # === 特徴点検出分析 ===
            kp1, desc1 = orb.detectAndCompute(gray1, None)
            feature_counts.append(len(kp1) if kp1 else 0)

            # === 中心領域輝度分析 ===
            brightness_radius = self.config.estimation.pose.brightness_check_radius
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
            center_mask = dist_sq <= brightness_radius**2
            center_pixels = gray1[center_mask]
            if len(center_pixels) > 0:
                center_min_values.append(float(center_pixels.min()))
                center_mean_values.append(float(center_pixels.mean()))
                center_max_values.append(float(center_pixels.max()))

            # === motion分析（連続フレームペア） ===
            ret2, frame2 = cap.read()
            if ret2:
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                kp2, desc2 = orb.detectAndCompute(gray2, mask)
                kp1_masked, desc1_masked = orb.detectAndCompute(gray1, mask)

                if (desc1_masked is not None and desc2 is not None and
                    len(kp1_masked) >= 10 and len(kp2) >= 10):

                    matches = bf_matcher.knnMatch(desc1_masked, desc2, k=2)
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)

                    if len(good_matches) >= 5:
                        magnitudes = []
                        for m in good_matches:
                            pt1 = kp1_masked[m.queryIdx].pt
                            pt2 = kp2[m.trainIdx].pt
                            dx = pt2[0] - pt1[0]
                            dy = pt2[1] - pt1[1]
                            magnitude = np.sqrt(dx**2 + dy**2)
                            magnitudes.append(magnitude)
                        pair_mean_motions.append(float(np.mean(magnitudes)))

                # frame2のメモリを解放
                del frame2, gray2

            # frame1のメモリを解放
            del frame1, gray1
            processed_count += 1

            # 進捗コールバック呼び出し
            if progress_callback is not None:
                progress_callback(processed_count, len(sample_positions))

            # 定期的にガベージコレクション
            if processed_count % 50 == 0:
                gc.collect()

        cap.release()

        self.logger.info(
            f"ストリーミング処理完了: {processed_count}フレーム処理, "
            f"{len(pair_mean_motions)}連続ペア（motion分析用）"
        )

        if processed_count == 0:
            raise RuntimeError("サンプリングフレームが取得できませんでした")

        # === 統計結果を集計 ===

        # 明るさ統計
        brightness_std = np.sqrt(brightness_m2 / brightness_count) if brightness_count > 1 else 0.0
        dark_ratio = float(dark_pixel_total) / (total_pixels * brightness_count) if brightness_count > 0 else 0.5
        brightness_stats = {
            'mean_brightness': brightness_mean,
            'std_brightness': brightness_std,
            'dark_ratio': dark_ratio
        }
        self.logger.info(
            f"明るさ分析: 平均輝度={brightness_mean:.1f}, "
            f"標準偏差={brightness_std:.1f}, 暗部比率={dark_ratio:.3f}"
        )

        # ノイズ統計
        noise_level = noise_sum / noise_count if noise_count > 0 else 100.0
        self.logger.info(f"ノイズ分析: Laplacian分散={noise_level:.1f}")

        # 特徴点統計
        feature_stats = {
            'mean_feature_count': int(np.mean(feature_counts)) if feature_counts else 0,
            'min_feature_count': int(np.min(feature_counts)) if feature_counts else 0,
            'max_feature_count': int(np.max(feature_counts)) if feature_counts else 0
        }
        self.logger.info(
            f"特徴点検出分析: 平均={feature_stats['mean_feature_count']}, "
            f"最小={feature_stats['min_feature_count']}, 最大={feature_stats['max_feature_count']}"
        )

        # motion統計
        motion_stats = self._compute_motion_stats(pair_mean_motions)

        # 中心領域輝度統計（3ゾーン分割: サンプルインデックスベース）
        # 先頭edge_nサンプル・末尾edge_nサンプルをエッジゾーンとし、
        # 中間ゾーンのbaselineを入口光・出口光の影響がない代表値とする
        # 実際のフレーム範囲: frame_skip × edge_n（例: 100×5=500フレーム）
        edge_n = self.config.auto_tune_brightness_edge_frames
        total_samples = len(center_mean_values)

        def _zone_baseline(values):
            if len(values) >= 4:
                return float(np.percentile(values, 20))
            elif values:
                return float(np.min(values))
            return 0.0

        if total_samples > 2 * edge_n and edge_n > 0:
            start_zone = center_mean_values[:edge_n]
            middle_zone = center_mean_values[edge_n:total_samples - edge_n]
            end_zone = center_mean_values[total_samples - edge_n:]

            start_baseline = _zone_baseline(start_zone)
            middle_baseline = _zone_baseline(middle_zone)
            end_baseline = _zone_baseline(end_zone)
            center_baseline = middle_baseline

            self.logger.info(
                f"中心領域輝度ゾーン分割: edge_n={edge_n}サンプル "
                f"(≈{edge_n * max(1, frame_skip)}フレーム), "
                f"start({len(start_zone)}件)={start_baseline:.1f}, "
                f"middle({len(middle_zone)}件)={middle_baseline:.1f}, "
                f"end({len(end_zone)}件)={end_baseline:.1f}"
            )
        else:
            # サンプル数不足またはedge_n=0: 全体で算出
            start_baseline = 0.0
            end_baseline = 0.0
            center_baseline = _zone_baseline(center_mean_values)
            middle_baseline = center_baseline

            self.logger.info(
                f"中心領域輝度ゾーン分割: サンプル不足 "
                f"({total_samples} <= {2 * edge_n}), "
                f"全体baselineを使用: {center_baseline:.1f}"
            )

        center_brightness_stats = {
            'center_min_brightness': float(np.mean(center_min_values)) if center_min_values else 0.0,
            'center_mean_brightness': float(np.mean(center_mean_values)) if center_mean_values else 128.0,
            'center_max_brightness': float(np.mean(center_max_values)) if center_max_values else 255.0,
            'center_baseline_brightness': center_baseline,
            'center_baseline_start': start_baseline,
            'center_baseline_middle': middle_baseline,
            'center_baseline_end': end_baseline,
        }
        self.logger.info(
            f"中心領域輝度分析: min={center_brightness_stats['center_min_brightness']:.1f}, "
            f"mean={center_brightness_stats['center_mean_brightness']:.1f}, "
            f"max={center_brightness_stats['center_max_brightness']:.1f}, "
            f"baseline={center_baseline:.1f} (middle zone)"
        )

        # 統合結果
        video_stats = {
            'brightness': brightness_stats,
            'noise_level': noise_level,
            'motion': motion_stats,
            'feature_detectability': feature_stats,
            'center_brightness': center_brightness_stats
        }

        self.logger.info("動画分析完了（ストリーミング処理）")
        gc.collect()

        return video_stats

    def _compute_motion_stats(
        self,
        pair_mean_motions: List[float],
        top_n_pairs: int = 5,
        iqr_multiplier: float = 1.5
    ) -> Dict[str, float]:
        """motion統計を計算（IQR法で異常値除外、上位N件採用）

        Args:
            pair_mean_motions: 各ペアの平均移動量リスト
            top_n_pairs: 上位何ペアの平均を採用するか
            iqr_multiplier: IQR法の外れ値判定倍率

        Returns:
            motion統計辞書
        """
        if not pair_mean_motions:
            self.logger.warning("動き量分析: マッチング失敗")
            return {
                'mean_motion': 0.0,
                'max_motion': 0.0,
                'motion_variance': 0.0
            }

        self.logger.info(
            f"動き量分析: {len(pair_mean_motions)}ペアの平均移動量を取得"
        )

        # IQR法で異常値を除外
        pair_motions_array = np.array(pair_mean_motions)
        q1 = np.percentile(pair_motions_array, 25)
        q3 = np.percentile(pair_motions_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        valid_motions = pair_motions_array[
            (pair_motions_array >= lower_bound) & (pair_motions_array <= upper_bound)
        ]

        outlier_count = len(pair_motions_array) - len(valid_motions)
        if outlier_count > 0:
            self.logger.info(
                f"動き量分析: IQR法で{outlier_count}件の異常値を除外 "
                f"(範囲: {lower_bound:.2f}～{upper_bound:.2f}px)"
            )

        if len(valid_motions) == 0:
            self.logger.warning("動き量分析: 異常値除外後にデータがありません")
            return {
                'mean_motion': 0.0,
                'max_motion': 0.0,
                'motion_variance': 0.0
            }

        # 上位N件のペアの平均を採用
        sorted_motions = np.sort(valid_motions)[::-1]
        actual_top_n = min(top_n_pairs, len(sorted_motions))
        top_motions = sorted_motions[:actual_top_n]

        mean_motion = float(np.mean(top_motions))
        max_motion = float(np.max(valid_motions))
        motion_variance = float(np.var(valid_motions))

        self.logger.info(
            f"動き量分析: 上位{actual_top_n}ペアの平均={mean_motion:.2f}px "
            f"(最大={max_motion:.2f}px, 分散={motion_variance:.2f})"
        )
        self.logger.info(
            f"  上位ペアの値: {[f'{v:.2f}' for v in top_motions]}"
        )

        return {
            'mean_motion': mean_motion,
            'max_motion': max_motion,
            'motion_variance': motion_variance
        }
    
    def tune_parameters(self, video_stats: Optional[Dict] = None) -> Config:
        """パラメータを自動調整
        
        Args:
            video_stats: 分析結果（Noneの場合は自動でanalyze_video()を実行）
        
        Returns:
            調整済みConfig
        """
        # video_statsがNoneの場合は自動分析
        if video_stats is None:
            self.logger.info("video_stats未指定のため、自動分析を実行します")
            video_stats = self.analyze_video(
                num_sample_frames=self.config.auto_tune_sample_frames,
                frame_skip=self.config.auto_tune_frame_skip
            )
        
        # パラメータ調整
        tuned_config = self.parameter_tuner.tune_all_parameters(video_stats)

        # 特徴点マッチングパラメータの実測ベース自動チューニング
        tuned_config = self._tune_feature_matching_params(tuned_config)

        return tuned_config

    def _tune_feature_matching_params(self, config: Config) -> Config:
        """特徴点マッチングパラメータの実測ベース自動チューニング

        条件:
            - auto_tune_feature_matching_enabled=True
            - use_cylindrical_matching=True
            - ユーザーが設定ファイルで明示的にパラメータを指定していない

        Args:
            config: 調整対象のConfig

        Returns:
            調整済みConfig
        """
        # 有効判定
        if not config.auto_tune_feature_matching_enabled:
            self.logger.info(
                "特徴点マッチングパラメータ自動チューニング: 無効 "
                "(auto_tune_feature_matching_enabled=False)"
            )
            return config

        if not config.estimation.feature_matching.use_cylindrical_matching:
            self.logger.info(
                "特徴点マッチングパラメータ自動チューニング: スキップ "
                "(use_cylindrical_matching=False)"
            )
            return config

        # ユーザー定義保護チェック
        orb_path = "estimation.feature_matching.cylindrical.orb_max_features"
        match_path = "estimation.feature_matching.cylindrical.match_max_features"
        orb_user_defined = orb_path in config._user_defined_paths
        match_user_defined = match_path in config._user_defined_paths

        if orb_user_defined and match_user_defined:
            self.logger.info(
                "特徴点マッチングパラメータ自動チューニング: スキップ "
                "(両パラメータともユーザー設定で保護)"
            )
            return config

        # チューニング実行
        try:
            fm_tuner = FeatureMatchingTuner(config, self.video_path, self.logger)
            best_orb, best_match = fm_tuner.tune()
        except Exception as e:
            self.logger.warning(
                f"特徴点マッチングパラメータ自動チューニング失敗: {e}、"
                f"既存の設定値を維持"
            )
            return config

        # 結果を反映（ユーザー設定保護に従う）
        if not orb_user_defined:
            old_val = config.estimation.feature_matching.cylindrical.orb_max_features
            config.estimation.feature_matching.cylindrical.orb_max_features = best_orb
            self.logger.info(
                f"  orb_max_features: {old_val} → {best_orb} (自動チューニング)"
            )
        else:
            self.logger.info(
                f"  orb_max_features: ユーザー設定を保持 "
                f"({config.estimation.feature_matching.cylindrical.orb_max_features})"
            )

        if not match_user_defined:
            old_val = config.estimation.feature_matching.cylindrical.match_max_features
            config.estimation.feature_matching.cylindrical.match_max_features = best_match
            self.logger.info(
                f"  match_max_features: {old_val} → {best_match} (自動チューニング)"
            )
        else:
            self.logger.info(
                f"  match_max_features: ユーザー設定を保持 "
                f"({config.estimation.feature_matching.cylindrical.match_max_features})"
            )

        return config

    def _get_lens_center(self, sample_frame: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """レンズ中心座標を取得（TASK-32）

        以下の優先順位でレンズ中心座標を取得します：
        1. キャリブレーションファイルがある場合、そのcx, cyを使用
        2. config.camera.fxが設定されている場合（v2.0形式）、cx, cyを使用
        3. サンプルフレームがある場合、フレーム中心を使用
        4. config.camera.image_width/heightからフレーム中心を計算

        優先順位1・2で取得したcx/cyは、キャリブレーション画像サイズと
        実フレームサイズが異なる場合にトリミング/リサイズ補正を適用します。

        Args:
            sample_frame: サンプルフレーム（オプション）

        Returns:
            (cx, cy): レンズ中心座標（フレーム座標系）
        """
        cx, cy = None, None
        needs_correction = False

        # 1. キャリブレーションファイルから取得を試みる
        if self.config.camera.lens_calibration_file:
            try:
                from src.calibration import load_calibration
                calibration = load_calibration(self.config.camera.lens_calibration_file)
                self.logger.info(
                    f"レンズ中心: キャリブレーションファイルから取得 "
                    f"(cx={calibration.cx:.1f}, cy={calibration.cy:.1f})"
                )
                cx, cy = calibration.cx, calibration.cy
                needs_correction = True
            except Exception as e:
                self.logger.warning(f"キャリブレーションファイル読み込み失敗: {e}")

        # 2. config.camera.fxが設定されている場合（v2.0形式）
        if cx is None and hasattr(self.config.camera, 'fx') and self.config.camera.fx > 0:
            cx_val = getattr(self.config.camera, 'cx', None)
            cy_val = getattr(self.config.camera, 'cy', None)
            if cx_val is not None and cy_val is not None:
                self.logger.info(
                    f"レンズ中心: config.cameraから取得 (cx={cx_val:.1f}, cy={cy_val:.1f})"
                )
                cx, cy = float(cx_val), float(cy_val)
                needs_correction = True

        # トリミング/リサイズ補正を適用（優先順位1・2で取得した場合）
        if cx is not None and needs_correction and sample_frame is not None:
            frame_h, frame_w = sample_frame.shape[:2]
            calib_w = self.config.camera.calibrated_image_width
            calib_h = self.config.camera.calibrated_image_height
            if calib_w > 0 and calib_h > 0 and (calib_w != frame_w or calib_h != frame_h):
                from src.image_size_correction import apply_camera_correction
                correction = apply_camera_correction(
                    fx=self.config.camera.fx,
                    cx=cx,
                    cy=cy,
                    calib_width=calib_w,
                    calib_height=calib_h,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    logger=self.logger
                )
                self.logger.info(
                    f"レンズ中心: トリミング/リサイズ補正適用 "
                    f"(cx={cx:.1f}→{correction.cx:.1f}, cy={cy:.1f}→{correction.cy:.1f})"
                )
                cx, cy = correction.cx, correction.cy

        if cx is not None:
            return cx, cy

        # 3. サンプルフレームから取得
        if sample_frame is not None:
            h, w = sample_frame.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            self.logger.info(
                f"レンズ中心: フレーム中心を使用 (cx={cx:.1f}, cy={cy:.1f})"
            )
            return cx, cy

        # 4. config.camera.image_width/heightから計算
        w = self.config.camera.image_width
        h = self.config.camera.image_height
        cx, cy = w / 2.0, h / 2.0
        self.logger.info(
            f"レンズ中心: config画像サイズから計算 (cx={cx:.1f}, cy={cy:.1f})"
        )
        return cx, cy

    def save_tuned_config(self, config: Config, output_path: str) -> None:
        """調整済み設定をJSON保存
        
        Args:
            config: 保存する設定
            output_path: 保存先パス
        """
        self.logger.info(f"調整済み設定を保存: {output_path}")
        config.save(output_path)
        self.logger.info("保存完了")

    def optimize(
        self,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Config:
        """
        パラメータ最適化を実行（動画分析 + パラメータ調整の統合処理）

        Args:
            progress_callback: 進捗コールバック関数 progress_callback(progress_percent)
                              - 開始時: 0%
                              - 動画分析完了時: 50%
                              - パラメータ最適化完了時: 100%

        Returns:
            調整済みConfig

        Raises:
            RuntimeError: 動画読み込み失敗
        """
        # 開始（0%）
        if progress_callback:
            try:
                progress_callback(0.0)
            except Exception as e:
                self.logger.warning(f"Progress callback failed at start (0%%): {e}")

        # 動画分析
        self.logger.info("動画分析を開始します")
        video_stats = self.analyze_video(
            num_sample_frames=self.config.auto_tune_sample_frames,
            frame_skip=self.config.auto_tune_frame_skip
        )
        self.logger.info("動画分析が完了しました")

        # 動画分析完了（50%）
        if progress_callback:
            try:
                progress_callback(50.0)
            except Exception as e:
                self.logger.warning(f"Progress callback failed at 50%: {e}")

        # パラメータ最適化
        self.logger.info("パラメータ最適化を開始します")
        tuned_config = self.parameter_tuner.tune_all_parameters(video_stats)
        self.logger.info("パラメータ最適化が完了しました")

        # パラメータ最適化完了（100%）
        if progress_callback:
            try:
                progress_callback(100.0)
            except Exception as e:
                self.logger.warning(f"Progress callback failed at 100%: {e}")

        return tuned_config


# ========================================
# FeatureMatchingTunerクラス
# ========================================


class FeatureMatchingTuner:
    """特徴点マッチングパラメータの実測ベース自動チューニング

    サンプルフレームペアで実際にCylindricalFeatureMatcherを実行し、
    全フィルタリング後の特徴点数を計測して、orb_max_features と
    match_max_features の最小値を二分探索で決定します。

    Attributes:
        config: Config オブジェクト
        video_path: 入力動画パス
        logger: ロガー
    """

    # デフォルトフォールバック値
    DEFAULT_ORB_MAX_FEATURES: int = 4000
    DEFAULT_MATCH_MAX_FEATURES: int = 200

    def __init__(
        self,
        config: Config,
        video_path: str,
        logger: Optional[logging.Logger] = None
    ):
        """初期化

        Args:
            config: Config オブジェクト
            video_path: 入力動画パス
            logger: ロガー（Noneの場合はモジュールロガーを使用）
        """
        self.config = config
        self.video_path = video_path
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def _collect_sample_frame_pairs(
        self,
        num_pairs: int,
        frame_skip: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """動画から等間隔でサンプルフレームペア(連続フレーム)を収集

        先頭付近（マンホール内で暗い）を避け、frame_skip * 2 以降からサンプリング。

        Args:
            num_pairs: 収集するペア数
            frame_skip: フレームスキップ間隔（暗い先頭を避けるために使用）

        Returns:
            [(prev_frame, curr_frame), ...] のリスト

        Raises:
            RuntimeError: 動画読み込み失敗
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 処理対象範囲を決定
        range_start = self.config.input.start_frame
        end_frame = total_frames
        if self.config.input.end_frame is not None:
            end_frame = min(end_frame, self.config.input.end_frame)
        usable_total = end_frame - range_start

        # frame_skip自動計算（0以下の場合、対象範囲を均等にカバー）
        if frame_skip <= 0:
            frame_skip = max(1, usable_total // (num_pairs + 1))
            self.logger.info(f"frame_skip自動計算: {frame_skip}")

        # 先頭の暗いフレームを避けるオフセット
        start_offset = range_start + frame_skip * 2

        usable_range = end_frame - start_offset
        if usable_range < num_pairs * 2:
            self.logger.warning(
                f"フレーム範囲が狭い(usable={usable_range})、"
                f"範囲先頭から収集します"
            )
            start_offset = range_start
            usable_range = end_frame - start_offset

        # 等間隔位置を計算
        interval = max(1, usable_range // (num_pairs + 1))
        positions = [start_offset + interval * (i + 1) for i in range(num_pairs)]

        frame_pairs = []
        for pos in positions:
            if pos >= end_frame - 1:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret1, frame1 = cap.read()
            ret2, frame2 = cap.read()
            if ret1 and ret2:
                frame_pairs.append((frame1, frame2))
            else:
                self.logger.warning(f"フレーム読み込み失敗: position={pos}")

        cap.release()

        self.logger.info(
            f"サンプルフレームペア収集: {len(frame_pairs)}/{num_pairs}ペア "
            f"(positions={positions})"
        )
        return frame_pairs

    def _build_camera_params(
        self,
        frame_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """カメラパラメータ辞書を構築

        main.pyの_initialize_camera_modelのロジックを簡略再実装。

        Args:
            frame_shape: フレーム形状 (height, width, channels)

        Returns:
            camera_params辞書
        """
        height, width = frame_shape[:2]

        has_valid_v2 = (
            self.config.camera.fx > 0 and
            self.config.camera.fy > 0 and
            self.config.camera.cx > 0 and
            self.config.camera.cy > 0
        )

        if has_valid_v2:
            calib_w = self.config.camera.calibrated_image_width
            calib_h = self.config.camera.calibrated_image_height

            if calib_w > 0 and calib_h > 0 and (width != calib_w or height != calib_h):
                try:
                    from src.image_size_correction import apply_camera_correction
                    correction = apply_camera_correction(
                        fx=self.config.camera.fx,
                        cx=self.config.camera.cx,
                        cy=self.config.camera.cy,
                        calib_width=calib_w,
                        calib_height=calib_h,
                        frame_width=width,
                        frame_height=height,
                        logger=self.logger
                    )
                    f, cx, cy = correction.f, correction.cx, correction.cy
                except Exception as e:
                    self.logger.warning(f"画像サイズ補正失敗、直接値を使用: {e}")
                    f = self.config.camera.fx
                    cx = self.config.camera.cx
                    cy = self.config.camera.cy
            else:
                f = self.config.camera.fx
                cx = self.config.camera.cx
                cy = self.config.camera.cy
        else:
            theta_max = np.radians(self.config.camera.fov_degrees / 2.0)
            cx = width // 2 + self.config.camera.center_offset_x
            cy = height // 2 + self.config.camera.center_offset_y
            radius = min(cx, cy)
            f = radius / theta_max

        return {
            'f': f,
            'center': (cx, cy),
            'radius': min(cx, cy),
            'pipe_diameter': self.config.pipe.diameter_mm,
            'image_width': width,
            'image_height': height,
            'model': 'fisheye',
        }

    def _build_matcher(
        self,
        orb_max_features: int,
        match_max_features: int,
        camera_params: Dict[str, Any]
    ) -> Any:
        """指定パラメータでCylindricalFeatureMatcherを生成

        Args:
            orb_max_features: ORB最大特徴点数
            match_max_features: マッチング後最大特徴点数
            camera_params: カメラパラメータ辞書

        Returns:
            CylindricalFeatureMatcher インスタンス
        """
        import copy
        from src.feature_matching_cylindrical import (
            CylindricalFeatureMatcher,
            CylindricalMatchingConfig,
        )
        from src.coordinate_transform import FisheyeCamera, PinholeCamera, CoordinateTransformer

        # 元のcylindrical configをコピーし、2パラメータのみ差し替え
        base_cyl_config = self.config.estimation.feature_matching.cylindrical
        if base_cyl_config is not None:
            cyl_config = copy.deepcopy(base_cyl_config)
        else:
            cyl_config = CylindricalMatchingConfig()

        cyl_config.orb_max_features = orb_max_features
        cyl_config.match_max_features = match_max_features

        # カメラモデル・transformer を構築
        f = camera_params['f']
        cx, cy = camera_params['center']
        w = camera_params['image_width']
        h = camera_params['image_height']

        camera_model = camera_params.get('model', self.config.camera.camera_model or 'fisheye')
        if camera_model == "pinhole":
            camera = PinholeCamera(f=f, cx=cx, cy=cy, image_width=w, image_height=h)
        else:
            camera = FisheyeCamera(f=f, cx=cx, cy=cy, image_width=w, image_height=h)
        pipe_radius = self.config.pipe.diameter_mm / 2.0
        transformer = CoordinateTransformer(
            camera, pipe_radius,
            calibration=self.config.camera._lens_calibration
        )

        matcher = CylindricalFeatureMatcher(
            transformer=transformer,
            config=cyl_config,
            feature_filtering_config=self.config.estimation.feature_filtering,
            static_frame_detection_config=self.config.estimation.static_frame_detection,
            pixels_per_mm=self.config.colormap.pixels_per_mm,
        )
        return matcher

    def _evaluate_params(
        self,
        frame_pairs: List[Tuple[np.ndarray, np.ndarray]],
        camera_params: Dict[str, Any],
        orb_max_features: int,
        match_max_features: int
    ) -> bool:
        """指定パラメータで合格判定

        サンプルフレームペアに対してextract_and_matchを実行し、
        pass_ratio以上のペアがtarget_points以上の特徴点を返せば合格。

        Args:
            frame_pairs: サンプルフレームペアのリスト
            camera_params: カメラパラメータ辞書
            orb_max_features: ORB最大特徴点数
            match_max_features: マッチング後最大特徴点数

        Returns:
            True: 合格、False: 不合格
        """
        target_points = self.config.auto_tune_feature_matching_target_points
        pass_ratio = self.config.auto_tune_feature_matching_pass_ratio

        if not frame_pairs:
            return False

        matcher = self._build_matcher(orb_max_features, match_max_features, camera_params)

        pass_count = 0
        point_counts = []
        for prev_frame, curr_frame in frame_pairs:
            try:
                prev_pts, curr_pts, status = matcher.extract_and_match(
                    prev_frame, curr_frame, camera_params
                )
                n_points = len(prev_pts) if prev_pts is not None else 0
                point_counts.append(n_points)
                if n_points >= target_points:
                    pass_count += 1
            except Exception as e:
                self.logger.debug(f"evaluate_params例外: {e}")
                point_counts.append(0)

        actual_ratio = pass_count / len(frame_pairs)
        passed = actual_ratio >= pass_ratio

        self.logger.debug(
            f"  evaluate(orb={orb_max_features}, match={match_max_features}): "
            f"points={point_counts}, pass={pass_count}/{len(frame_pairs)} "
            f"({actual_ratio:.0%}), {'PASS' if passed else 'FAIL'}"
        )

        return passed

    def tune(self) -> Tuple[int, int]:
        """二段階二分探索でorb_max_featuresとmatch_max_featuresの最小値を決定

        Returns:
            (orb_max_features, match_max_features) の最適値タプル
        """
        self.logger.info("=== 特徴点マッチングパラメータ自動チューニング開始 ===")

        num_pairs = self.config.auto_tune_feature_matching_sample_pairs
        frame_skip = self.config.auto_tune_frame_skip

        # サンプルフレームペア収集
        try:
            frame_pairs = self._collect_sample_frame_pairs(num_pairs, frame_skip)
        except Exception as e:
            self.logger.warning(f"サンプルフレーム収集失敗: {e}、デフォルト値を使用")
            return self.DEFAULT_ORB_MAX_FEATURES, self.DEFAULT_MATCH_MAX_FEATURES

        if len(frame_pairs) == 0:
            self.logger.warning("フレームペアが取得できません、デフォルト値を使用")
            return self.DEFAULT_ORB_MAX_FEATURES, self.DEFAULT_MATCH_MAX_FEATURES

        # カメラパラメータ構築
        camera_params = self._build_camera_params(frame_pairs[0][0].shape)

        # ================================================
        # Phase A: match_max_features の最小値を探索
        # orb_max_features=10000（十分大きい値）で固定
        # 探索範囲: [20, 500]、刻み幅10
        # ================================================
        self.logger.info("Phase A: match_max_features の最小値を探索 (orb=10000固定)")

        fixed_orb = 10000
        match_lo, match_hi = 20, 500
        match_step = 10

        # まず上限で通るか確認
        if not self._evaluate_params(frame_pairs, camera_params, fixed_orb, match_hi):
            self.logger.warning(
                f"match_max_features={match_hi}でも不合格、デフォルト値を使用"
            )
            return self.DEFAULT_ORB_MAX_FEATURES, self.DEFAULT_MATCH_MAX_FEATURES

        # 二分探索（刻み幅を考慮）
        best_match = match_hi
        lo_idx = match_lo // match_step
        hi_idx = match_hi // match_step

        while lo_idx <= hi_idx:
            mid_idx = (lo_idx + hi_idx) // 2
            mid_val = mid_idx * match_step
            mid_val = max(match_lo, min(match_hi, mid_val))

            if self._evaluate_params(frame_pairs, camera_params, fixed_orb, mid_val):
                best_match = mid_val
                hi_idx = mid_idx - 1
            else:
                lo_idx = mid_idx + 1

        self.logger.info(f"Phase A結果: match_max_features = {best_match}")

        # ================================================
        # Phase B: orb_max_features の最小値を探索
        # Phase Aで決定したmatch_max_featuresで固定
        # 探索範囲: [500, 10000]、刻み幅200
        # ================================================
        self.logger.info(
            f"Phase B: orb_max_features の最小値を探索 (match={best_match}固定)"
        )

        orb_lo, orb_hi = 500, 10000
        orb_step = 200

        # まず上限で通るか確認
        if not self._evaluate_params(frame_pairs, camera_params, orb_hi, best_match):
            self.logger.warning(
                f"orb_max_features={orb_hi}でも不合格、デフォルト値を使用"
            )
            return self.DEFAULT_ORB_MAX_FEATURES, best_match

        # 二分探索
        best_orb = orb_hi
        lo_idx = orb_lo // orb_step
        hi_idx = orb_hi // orb_step

        while lo_idx <= hi_idx:
            mid_idx = (lo_idx + hi_idx) // 2
            mid_val = mid_idx * orb_step
            mid_val = max(orb_lo, min(orb_hi, mid_val))

            if self._evaluate_params(frame_pairs, camera_params, mid_val, best_match):
                best_orb = mid_val
                hi_idx = mid_idx - 1
            else:
                lo_idx = mid_idx + 1

        self.logger.info(
            f"Phase B結果: orb_max_features = {best_orb}"
        )

        self.logger.info(
            f"=== 特徴点マッチングパラメータ自動チューニング完了: "
            f"orb_max_features={best_orb}, match_max_features={best_match} ==="
        )

        return best_orb, best_match
