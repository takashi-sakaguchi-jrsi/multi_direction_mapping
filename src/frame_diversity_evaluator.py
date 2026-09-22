"""フレーム多様性評価モジュール

このモジュールは、チェッカーボードキャリブレーションで得られたrvecs/tvecsから、
各フレームの姿勢多様性を定量評価します。

多様性特徴ベクトル（6次元）:
- ロール角、ピッチ角、ヨー角（Rodrigues変換から）
- 並進X、Y、Z（tvecsから）

Author: Claude Code
Date: 2025-11-13
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np


# ==================== カスタム例外 ====================


class DiversityEvaluationError(Exception):
    """多様性評価関連のベース例外"""
    pass


class InsufficientFramesError(DiversityEvaluationError):
    """フレーム数不足エラー
    
    Attributes:
        frame_count: 実際のフレーム数
        required_count: 最小要求フレーム数
    """
    
    def __init__(
        self,
        message: str,
        frame_count: int = 0,
        required_count: int = 0
    ):
        """
        Args:
            message: エラーメッセージ
            frame_count: 実際のフレーム数
            required_count: 最小要求フレーム数
        """
        self.frame_count = frame_count
        self.required_count = required_count
        full_message = (
            f"{message} (frame_count: {frame_count}, required: {required_count})"
        )
        super().__init__(full_message)


class InvalidShapeError(DiversityEvaluationError):
    """形状不正エラー
    
    Attributes:
        actual_shape: 実際の形状
        expected_shape: 期待される形状
    """
    
    def __init__(
        self,
        message: str,
        actual_shape: Optional[Tuple[int, ...]] = None,
        expected_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Args:
            message: エラーメッセージ
            actual_shape: 実際の形状
            expected_shape: 期待される形状
        """
        self.actual_shape = actual_shape
        self.expected_shape = expected_shape
        parts = [message]
        if actual_shape is not None:
            parts.append(f"actual_shape: {actual_shape}")
        if expected_shape is not None:
            parts.append(f"expected_shape: {expected_shape}")
        full_message = " (".join([parts[0], ", ".join(parts[1:])]) + ")"
        super().__init__(full_message)


# ==================== フレーム多様性評価器クラス ====================


class FrameDiversityEvaluator:
    """フレーム多様性評価器
    
    チェッカーボードキャリブレーションで得られたrvecs/tvecsから、
    各フレームの姿勢多様性を定量評価します。
    
    Attributes:
        logger: ロガー
    
    Example:
        >>> evaluator = FrameDiversityEvaluator()
        >>> diversity_features = evaluator.compute_diversity_features(rvecs, tvecs)
        >>> print(diversity_features.shape)
        (60, 6)  # 60フレーム × 6次元特徴
        
        >>> normalized_features = evaluator.normalize_features(diversity_features)
        >>> print(normalized_features.mean(axis=0))  # 平均0
        >>> print(normalized_features.std(axis=0))   # 標準偏差1
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """初期化
        
        Args:
            logger: ロガー（オプション）
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def rotation_vector_to_euler_angles(
        self, rvec: np.ndarray
    ) -> Tuple[float, float, float]:
        """回転ベクトルをオイラー角に変換
        
        cv2.Rodrigues()で回転ベクトルを回転行列に変換し、
        回転行列からオイラー角（ロール、ピッチ、ヨー）を抽出します。
        
        Args:
            rvec: 回転ベクトル (3x1) または (3,)
        
        Returns:
            (roll, pitch, yaw) のタプル（度単位）
            - roll: X軸周りの回転（-180° ~ 180°）
            - pitch: Y軸周りの回転（-90° ~ 90°）
            - yaw: Z軸周りの回転（-180° ~ 180°）
        
        Raises:
            InvalidShapeError: rvecの形状が不正な場合
        
        Example:
            >>> evaluator = FrameDiversityEvaluator()
            >>> rvec = np.array([0.0, 0.0, 0.0])
            >>> roll, pitch, yaw = evaluator.rotation_vector_to_euler_angles(rvec)
            >>> print(f"roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
            roll=0.00, pitch=0.00, yaw=0.00
        """
        # 形状チェック
        rvec = np.asarray(rvec)
        if rvec.shape not in [(3,), (3, 1), (1, 3)]:
            raise InvalidShapeError(
                f"rvecは(3,)、(3,1)、または(1,3)の形状である必要があります",
                actual_shape=rvec.shape,
                expected_shape=(3,)
            )
        
        # 1次元配列に変換
        rvec_flat = rvec.flatten()
        
        # Rodrigues変換: 回転ベクトル → 回転行列
        rotation_matrix, _ = cv2.Rodrigues(rvec_flat)
        
        # 回転行列からオイラー角を抽出
        # ZYX順（ヨー、ピッチ、ロール）の回転を仮定
        # rotation_matrix = Rz(yaw) * Ry(pitch) * Rx(roll)
        
        # ピッチ角（Y軸周り）: -90° ~ 90°
        # sin(pitch) = -R[2, 0]
        sin_pitch = -rotation_matrix[2, 0]
        # ジンバルロック対策: -1 <= sin_pitch <= 1
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)
        
        # ロール角（X軸周り）: -180° ~ 180°
        # cos(pitch) != 0の場合: tan(roll) = R[2,1] / R[2,2]
        cos_pitch = np.cos(pitch)
        if abs(cos_pitch) > 1e-6:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            # ジンバルロック時: roll=0と仮定
            roll = 0.0
        
        # ヨー角（Z軸周り）: -180° ~ 180°
        # cos(pitch) != 0の場合: tan(yaw) = R[1,0] / R[0,0]
        if abs(cos_pitch) > 1e-6:
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            # ジンバルロック時: yaw=0と仮定
            yaw = 0.0
        
        # ラジアンから度に変換
        roll_deg = np.rad2deg(roll)
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)
        
        return roll_deg, pitch_deg, yaw_deg
    
    def compute_translation_distance(
        self, tvec: np.ndarray
    ) -> Tuple[float, float, float]:
        """並進ベクトルから距離を算出
        
        Args:
            tvec: 並進ベクトル (3x1) または (3,)、単位はmm
        
        Returns:
            (tx, ty, tz) のタプル（mm単位）
        
        Raises:
            InvalidShapeError: tvecの形状が不正な場合
        
        Example:
            >>> evaluator = FrameDiversityEvaluator()
            >>> tvec = np.array([100.0, 200.0, 300.0])
            >>> tx, ty, tz = evaluator.compute_translation_distance(tvec)
            >>> print(f"tx={tx}, ty={ty}, tz={tz}")
            tx=100.0, ty=200.0, tz=300.0
        """
        # 形状チェック
        tvec = np.asarray(tvec)
        if tvec.shape not in [(3,), (3, 1), (1, 3)]:
            raise InvalidShapeError(
                f"tvecは(3,)、(3,1)、または(1,3)の形状である必要があります",
                actual_shape=tvec.shape,
                expected_shape=(3,)
            )
        
        # 1次元配列に変換
        tvec_flat = tvec.flatten()
        
        tx = float(tvec_flat[0])
        ty = float(tvec_flat[1])
        tz = float(tvec_flat[2])
        
        return tx, ty, tz
    
    def extract_pose_features(
        self,
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray]
    ) -> np.ndarray:
        """姿勢特徴ベクトルを抽出（旧名）
        
        注意: このメソッドはcompute_diversity_featuresのエイリアスです。
        後方互換性のために残されていますが、新規コードでは
        compute_diversity_featuresを使用してください。
        
        Args:
            rvecs: 回転ベクトルリスト
            tvecs: 並進ベクトルリスト
        
        Returns:
            多様性特徴行列 (N×6)
        
        Raises:
            InsufficientFramesError: リストが空の場合
            InvalidShapeError: リストの長さが不一致、または形状が不正な場合
        """
        return self.compute_diversity_features(rvecs, tvecs)
    
    def compute_diversity_features(
        self,
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray]
    ) -> np.ndarray:
        """多様性特徴ベクトルを算出
        
        各フレームを6次元特徴ベクトルで表現します：
        [roll, pitch, yaw, tx, ty, tz]
        
        Args:
            rvecs: 回転ベクトルリスト
            tvecs: 並進ベクトルリスト
        
        Returns:
            多様性特徴行列 (N×6)
            - N: フレーム数
            - 列: [roll, pitch, yaw, tx, ty, tz]
        
        Raises:
            InsufficientFramesError: リストが空の場合
            InvalidShapeError: リストの長さが不一致、または形状が不正な場合
        
        Example:
            >>> evaluator = FrameDiversityEvaluator()
            >>> rvecs = [np.array([0.1, 0.2, 0.3]) for _ in range(10)]
            >>> tvecs = [np.array([100.0, 200.0, 300.0]) for _ in range(10)]
            >>> features = evaluator.compute_diversity_features(rvecs, tvecs)
            >>> print(features.shape)
            (10, 6)
        """
        # 入力バリデーション: 空リスト
        if len(rvecs) == 0 or len(tvecs) == 0:
            raise InsufficientFramesError(
                "rvecs/tvecsが空です",
                frame_count=len(rvecs),
                required_count=1
            )
        
        # 入力バリデーション: 長さ不一致
        if len(rvecs) != len(tvecs):
            raise InvalidShapeError(
                f"rvecs/tvecsの長さが一致しません "
                f"(len(rvecs)={len(rvecs)}, len(tvecs)={len(tvecs)})"
            )
        
        n_frames = len(rvecs)
        diversity_features = np.zeros((n_frames, 6), dtype=np.float64)
        
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            # オイラー角を抽出（ロール、ピッチ、ヨー）
            roll, pitch, yaw = self.rotation_vector_to_euler_angles(rvec)
            
            # 並進距離を抽出（X、Y、Z）
            tx, ty, tz = self.compute_translation_distance(tvec)
            
            # 6次元特徴ベクトル: [roll, pitch, yaw, tx, ty, tz]
            diversity_features[i] = [roll, pitch, yaw, tx, ty, tz]
            
            self.logger.debug(
                f"Frame {i}: roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°, "
                f"tx={tx:.1f}mm, ty={ty:.1f}mm, tz={tz:.1f}mm"
            )
        
        self.logger.info(
            f"多様性特徴を算出しました: {n_frames}フレーム × 6次元"
        )
        
        return diversity_features
    
    def normalize_features(
        self, diversity_features: np.ndarray
    ) -> np.ndarray:
        """多様性特徴を正規化
        
        各特徴次元を平均0、標準偏差1に正規化します。
        これにより、回転角度（度）と並進距離（mm）のスケール差を解消します。
        
        Args:
            diversity_features: 多様性特徴行列 (N×6)
        
        Returns:
            正規化された多様性特徴行列 (N×6)
            - 各列の平均: 0
            - 各列の標準偏差: 1（標準偏差が0でない場合）
        
        Raises:
            InvalidShapeError: 形状が不正な場合
        
        Example:
            >>> evaluator = FrameDiversityEvaluator()
            >>> features = np.random.randn(10, 6) * 100 + 50  # 平均50、標準偏差100
            >>> normalized = evaluator.normalize_features(features)
            >>> print(normalized.mean(axis=0))  # ほぼ0
            >>> print(normalized.std(axis=0))   # ほぼ1
        
        Note:
            - 標準偏差が0の列（全フレームで同じ値）は正規化されず、0に設定されます。
            - これにより、多様性のない特徴が無視されます。
        """
        # 形状チェック
        diversity_features = np.asarray(diversity_features)
        if diversity_features.ndim != 2:
            raise InvalidShapeError(
                f"diversity_featuresは2次元配列である必要があります",
                actual_shape=diversity_features.shape,
                expected_shape=(None, 6)
            )
        
        if diversity_features.shape[1] != 6:
            raise InvalidShapeError(
                f"diversity_featuresの列数は6である必要があります",
                actual_shape=diversity_features.shape,
                expected_shape=(None, 6)
            )
        
        # 各列の平均と標準偏差を算出
        mean = diversity_features.mean(axis=0)
        std = diversity_features.std(axis=0)
        
        # 正規化: (x - mean) / std
        # 標準偏差が0の列（多様性なし）は0に設定
        normalized_features = np.zeros_like(diversity_features)
        for col in range(6):
            if std[col] > 1e-6:
                normalized_features[:, col] = (
                    diversity_features[:, col] - mean[col]
                ) / std[col]
            else:
                # 標準偏差が0の場合、全て0に設定
                normalized_features[:, col] = 0.0
                self.logger.warning(
                    f"特徴次元 {col} の標準偏差が0です。この特徴には多様性がありません。"
                )
        
        self.logger.info(
            f"多様性特徴を正規化しました: "
            f"mean={mean.round(2).tolist()}, std={std.round(2).tolist()}"
        )
        
        return normalized_features
    
    def compute_diversity_scores(
        self, pose_features: np.ndarray
    ) -> np.ndarray:
        """多様性スコアを算出
        
        各フレームと全体平均との距離（ユークリッド距離）を多様性スコアとします。
        スコアが高いほど、そのフレームは全体平均から離れた姿勢を持ちます。
        
        Args:
            pose_features: 姿勢特徴行列 (N×6)（正規化済みを推奨）
        
        Returns:
            多様性スコア配列 (N,)
            - 各要素: フレームと平均との距離（ユークリッド距離）
        
        Raises:
            InvalidShapeError: 形状が不正な場合
        
        Example:
            >>> evaluator = FrameDiversityEvaluator()
            >>> features = np.random.randn(10, 6)
            >>> scores = evaluator.compute_diversity_scores(features)
            >>> print(scores.shape)
            (10,)
            >>> print(f"最大スコア: {scores.max():.2f}")
            >>> print(f"平均スコア: {scores.mean():.2f}")
        """
        # 形状チェック
        pose_features = np.asarray(pose_features)
        if pose_features.ndim != 2:
            raise InvalidShapeError(
                f"pose_featuresは2次元配列である必要があります",
                actual_shape=pose_features.shape,
                expected_shape=(None, 6)
            )
        
        # 全体平均を算出
        mean_features = pose_features.mean(axis=0)
        
        # 各フレームと平均との距離を算出
        diversity_scores = np.linalg.norm(
            pose_features - mean_features, axis=1
        )
        
        self.logger.info(
            f"多様性スコアを算出しました: "
            f"min={diversity_scores.min():.2f}, "
            f"max={diversity_scores.max():.2f}, "
            f"mean={diversity_scores.mean():.2f}"
        )
        
        return diversity_scores
    
    def evaluate_diversity(
        self,
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray]
    ) -> Dict[str, Any]:
        """統合多様性評価メソッド
        
        rvecs/tvecsから多様性特徴を抽出し、正規化、スコア算出を一括実行します。
        
        Args:
            rvecs: 回転ベクトルリスト
            tvecs: 並進ベクトルリスト
        
        Returns:
            評価結果辞書 {
                "diversity_features": np.ndarray (N×6),  # 正規化前の特徴
                "normalized_features": np.ndarray (N×6), # 正規化後の特徴
                "diversity_scores": np.ndarray (N,),     # 多様性スコア
                "n_frames": int,                          # フレーム数
                "mean_score": float,                      # 平均スコア
                "std_score": float                        # スコアの標準偏差
            }
        
        Raises:
            InsufficientFramesError: リストが空の場合
            InvalidShapeError: リストの長さが不一致、または形状が不正な場合
        
        Example:
            >>> evaluator = FrameDiversityEvaluator()
            >>> rvecs = [np.random.randn(3) for _ in range(10)]
            >>> tvecs = [np.random.randn(3) * 100 for _ in range(10)]
            >>> result = evaluator.evaluate_diversity(rvecs, tvecs)
            >>> print(f"フレーム数: {result['n_frames']}")
            >>> print(f"平均スコア: {result['mean_score']:.2f}")
        """
        # 多様性特徴を抽出
        diversity_features = self.compute_diversity_features(rvecs, tvecs)
        
        # 正規化
        normalized_features = self.normalize_features(diversity_features)
        
        # 多様性スコアを算出
        diversity_scores = self.compute_diversity_scores(normalized_features)
        
        # 統計情報を算出
        n_frames = len(rvecs)
        mean_score = float(diversity_scores.mean())
        std_score = float(diversity_scores.std())
        
        self.logger.info(
            f"多様性評価を完了しました: "
            f"n_frames={n_frames}, mean_score={mean_score:.2f}, "
            f"std_score={std_score:.2f}"
        )
        
        return {
            "diversity_features": diversity_features,
            "normalized_features": normalized_features,
            "diversity_scores": diversity_scores,
            "n_frames": n_frames,
            "mean_score": mean_score,
            "std_score": std_score
        }
