"""フレーム品質評価モジュール

このモジュールは、レンズキャリブレーション用のフレーム品質を評価します。
ブレ、明るさ、チェッカーボード完全性を評価し、総合スコアを算出します。

Author: Claude Code
Date: 2025-11-13
"""

import logging
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np


# ==================== カスタム例外 ====================


class FrameQualityError(Exception):
    """フレーム品質評価関連のベース例外"""
    pass


class InvalidFrameError(FrameQualityError):
    """無効なフレームが渡された場合の例外"""
    
    def __init__(self, message: str, frame_shape: Optional[Tuple[int, ...]] = None):
        """
        Args:
            message: エラーメッセージ
            frame_shape: フレームの形状（オプション）
        """
        self.frame_shape = frame_shape
        full_message = message
        if frame_shape is not None:
            full_message += f" (frame_shape: {frame_shape})"
        super().__init__(full_message)


# ==================== フレーム品質評価クラス ====================


class FrameQualityEvaluator:
    """フレーム品質評価クラス
    
    ブレ、明るさ、チェッカーボード完全性を評価し、総合スコアを算出します。
    
    Attributes:
        blur_threshold: ブレ検出の閾値（Laplacian分散）
        brightness_threshold: 明るさ検出の閾値（0-255）
        pattern_size: チェッカーボードのパターンサイズ（横、縦）
        logger: ロガー
    
    Example:
        >>> evaluator = FrameQualityEvaluator(
        ...     blur_threshold=100.0,
        ...     brightness_threshold=200.0,
        ...     pattern_size=(9, 6)
        ... )
        >>> frame = cv2.imread("frame.png")
        >>> result = evaluator.evaluate_frame(frame)
        >>> print(result["is_qualified"])
        True
    """
    
    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_threshold: float = 200.0,
        pattern_size: Tuple[int, int] = (9, 6),
        logger: Optional[logging.Logger] = None
    ):
        """初期化
        
        Args:
            blur_threshold: ブレ検出の閾値（Laplacian分散）デフォルト: 100.0
            brightness_threshold: 明るさ検出の閾値（0-255）デフォルト: 200.0
            pattern_size: チェッカーボードのパターンサイズ（横、縦）デフォルト: (9, 6)
            logger: ロガー（オプション）
        
        Raises:
            ValueError: パラメータが不正な場合
        """
        # パラメータバリデーション
        if blur_threshold <= 0:
            raise ValueError(f"blur_threshold must be positive, got {blur_threshold}")
        
        if not (0 <= brightness_threshold <= 255):
            raise ValueError(
                f"brightness_threshold must be in [0, 255], got {brightness_threshold}"
            )
        
        if not (isinstance(pattern_size, tuple) and len(pattern_size) == 2):
            raise ValueError(f"pattern_size must be a tuple of 2 integers, got {pattern_size}")
        
        if pattern_size[0] <= 0 or pattern_size[1] <= 0:
            raise ValueError(
                f"pattern_size elements must be positive, got {pattern_size}"
            )
        
        self.blur_threshold = blur_threshold
        self.brightness_threshold = brightness_threshold
        self.pattern_size = pattern_size
        self.logger = logger or logging.getLogger(__name__)
    
    def _validate_frame(self, frame: np.ndarray) -> None:
        """フレームのバリデーション（内部用）
        
        Args:
            frame: 検証対象フレーム
        
        Raises:
            InvalidFrameError: フレームが無効な場合
        """
        if frame is None:
            raise InvalidFrameError("Frame is None")
        
        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError(f"Frame must be numpy.ndarray, got {type(frame)}")
        
        if frame.size == 0:
            raise InvalidFrameError("Frame is empty", frame_shape=frame.shape)
        
        if frame.ndim < 2:
            raise InvalidFrameError(
                f"Frame must have at least 2 dimensions, got {frame.ndim}",
                frame_shape=frame.shape
            )
    
    def evaluate_blur(self, frame: np.ndarray) -> float:
        """ブレ評価（Laplacian分散法）
        
        Laplacian分散法でエッジの鮮明さを評価します。
        値が大きいほど鮮明で、小さいほどブレています。
        
        Args:
            frame: 評価対象フレーム（BGR形式またはグレースケール）
        
        Returns:
            ブレスコア（Laplacian分散値、高いほど鮮明）
        
        Raises:
            InvalidFrameError: フレームが無効な場合
        """
        # フレームバリデーション
        self._validate_frame(frame)
        
        # グレースケール変換
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Laplacian分散法でブレ評価
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        self.logger.debug(f"Blur score (Laplacian variance): {variance:.2f}")
        
        return float(variance)
    
    def evaluate_brightness(self, frame: np.ndarray) -> float:
        """明るさ評価（ヒストグラム分析）
        
        ヒストグラムで明るさの分布を分析し、過度な明るさ（反射）を検出します。
        
        Args:
            frame: 評価対象フレーム（BGR形式またはグレースケール）
        
        Returns:
            明るさスコア（0: 暗すぎる/明るすぎる、1: 最適）
        
        Raises:
            InvalidFrameError: フレームが無効な場合
        """
        # フレームバリデーション
        self._validate_frame(frame)
        
        # グレースケール変換
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 平均輝度を計算
        mean_brightness = float(gray.mean())
        
        # 過度に明るい（反射）
        if mean_brightness > self.brightness_threshold:
            self.logger.debug(f"Brightness score: 0.0 (too bright: {mean_brightness:.2f})")
            return 0.0
        
        # 暗すぎる
        if mean_brightness < 50:
            self.logger.debug(f"Brightness score: 0.0 (too dark: {mean_brightness:.2f})")
            return 0.0
        
        # 適切な明るさ（128を最適値として、離れるほどスコア低下）
        score = 1.0 - abs(mean_brightness - 128) / 128
        self.logger.debug(f"Brightness score: {score:.2f} (mean: {mean_brightness:.2f})")
        
        return score
    
    def evaluate_completeness(self, frame: np.ndarray) -> bool:
        """チェッカーボード完全性評価
        
        OpenCV findChessboardCorners() でコーナー検出を試み、
        全コーナーが検出できれば完全と判定します。
        
        Args:
            frame: 評価対象フレーム（BGR形式またはグレースケール）
        
        Returns:
            完全性フラグ（True: 完全、False: 不完全）
        
        Raises:
            InvalidFrameError: フレームが無効な場合
        """
        # フレームバリデーション
        self._validate_frame(frame)
        
        # グレースケール変換
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # チェッカーボードコーナー検出
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            self.logger.debug(f"Checkerboard completeness: True ({self.pattern_size[0]}x{self.pattern_size[1]} corners detected)")
        else:
            self.logger.debug(f"Checkerboard completeness: False (corners not detected)")
        
        return ret
    
    def evaluate_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """フレーム総合評価
        
        ブレ、明るさ、完全性を総合的に評価します。
        
        Args:
            frame: 評価対象フレーム（BGR形式）
        
        Returns:
            評価結果辞書 {
                "blur_score": float,           # ブレスコア
                "brightness_score": float,      # 明るさスコア
                "is_complete": bool,            # 完全性フラグ
                "overall_score": float,         # 総合スコア（0-1）
                "is_qualified": bool            # 合否フラグ
            }
        
        Raises:
            InvalidFrameError: フレームが無効な場合
        """
        # フレームバリデーション
        self._validate_frame(frame)
        
        # 各評価を実行
        blur_score = self.evaluate_blur(frame)
        brightness_score = self.evaluate_brightness(frame)
        is_complete = self.evaluate_completeness(frame)
        
        # 総合スコア算出: ブレ60%、明るさ30%、完全性10%
        # ブレスコアを正規化（100で割る）
        normalized_blur = min(blur_score / 100.0, 1.0)
        
        overall_score = (
            0.6 * normalized_blur +
            0.3 * brightness_score +
            0.1 * (1.0 if is_complete else 0.0)
        )
        
        # 合否判定
        is_qualified = (
            blur_score >= self.blur_threshold and
            brightness_score > 0.5 and
            is_complete
        )
        
        self.logger.debug(
            f"Frame evaluation: blur={blur_score:.2f}, "
            f"brightness={brightness_score:.2f}, "
            f"complete={is_complete}, "
            f"overall={overall_score:.2f}, "
            f"qualified={is_qualified}"
        )
        
        return {
            "blur_score": blur_score,
            "brightness_score": brightness_score,
            "is_complete": is_complete,
            "overall_score": overall_score,
            "is_qualified": is_qualified
        }
