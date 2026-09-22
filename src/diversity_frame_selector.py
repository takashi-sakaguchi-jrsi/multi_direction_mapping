"""多様性ベースフレーム選択モジュール (v4)

このモジュールは、K-meansクラスタリングでカメラ姿勢空間を分割し、
各クラスタから品質最高のフレームを選択することで、
多様な姿勢のフレームセットを構築します。

v4での変更点（2025-11-16）:
- selection_strategy機能を実装（diversity_first / interval_first）
- diversity_first: K-means → 各クラスタから品質最高選択
- interval_first: 貪欲法（間隔制約） → K-means → 各クラスタから品質最高選択
- 統一インターフェースselect_frames()を追加
- 既存のselect_representative_frames()は後方互換ラッパーとして保持

v3.1修正（2025-11-16）:
- min_frame_interval > 0の場合、K-meansを使わず貪欲法で選択するように修正

v3新機能:
- 最小フレーム間隔制約（min_frame_interval）の追加
- min-max品質クライテリアの追加（quality_criteria: "overall" or "minmax"）

Author: Claude Code
Date: 2025-11-16
Version: 4.0
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.cluster import KMeans


# ==================== カスタム例外 ====================


class DiversitySelectionError(Exception):
    """多様性選択関連のベース例外"""
    pass


class InsufficientFramesError(DiversitySelectionError):
    """フレーム数不足エラー"""
    
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
        full_message = f"{message} (frame_count: {frame_count}, required: {required_count})"
        super().__init__(full_message)


class InvalidClusterCountError(DiversitySelectionError):
    """クラスタ数不正エラー"""
    
    def __init__(
        self,
        message: str,
        n_clusters: int = 0,
        frame_count: int = 0
    ):
        """
        Args:
            message: エラーメッセージ
            n_clusters: 指定されたクラスタ数
            frame_count: フレーム数
        """
        self.n_clusters = n_clusters
        self.frame_count = frame_count
        full_message = f"{message} (n_clusters: {n_clusters}, frame_count: {frame_count})"
        super().__init__(full_message)


# ==================== 多様性ベースフレーム選択クラス ====================


class DiversityFrameSelector:
    """多様性ベースフレーム選択器 (v4.0)
    
    K-meansクラスタリングで姿勢空間を分割し、
    各クラスタから品質最高のフレームを選択します。
    
    v4新機能（2025-11-16）:
    - selection_strategy機能を実装（diversity_first / interval_first）
    - diversity_first: K-means → 各クラスタから品質最高選択
    - interval_first: 貪欲法（間隔制約） → K-means → 各クラスタから品質最高選択
    
    v3.1修正内容（2025-11-16）:
    - min_frame_interval > 0の場合、K-meansを使わず貪欲法で選択
    - 貪欲法: 品質降順でフレームを走査し、間隔制約を満たすものを選択
    
    v3新機能:
    - 最小フレーム間隔制約（select_representative_frames）
    - min-max品質クライテリア（overall/minmax）
    
    Attributes:
        n_clusters: クラスタ数
        random_state: 乱数シード（再現性のため）
        logger: ロガー
    
    Example:
        >>> selector = DiversityFrameSelector(n_clusters=30)
        >>> result = selector.select_frames(
        ...     strategy="diversity_first",
        ...     frames=frames,
        ...     quality_scores=quality_scores,
        ...     pose_features=pose_features,
        ...     target_count=30
        ... )
        >>> print(result["n_selected"])
        30
    """
    
    def __init__(
        self,
        n_clusters: int = 30,
        random_state: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """初期化
        
        Args:
            n_clusters: クラスタ数（デフォルト: 30）
            random_state: 乱数シード（デフォルト: 42）
            logger: ロガー（オプション）
        
        Raises:
            ValueError: n_clusters <= 0の場合
        """
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
    
    def cluster_frames(
        self,
        pose_features: np.ndarray
    ) -> np.ndarray:
        """K-meansクラスタリング実行
        
        Args:
            pose_features: 正規化済み姿勢特徴行列 (N×6)
                           列: [roll, pitch, yaw, tx, ty, tz]
        
        Returns:
            クラスタラベル配列 (N,)
            各要素は0からn_clusters-1のクラスタID
        
        Raises:
            ValueError: pose_featuresの形状が不正な場合
            InsufficientFramesError: フレーム数 < n_clustersの場合
        """
        # 入力検証
        if pose_features.ndim != 2:
            raise ValueError(
                f"pose_features must be 2D array, got shape {pose_features.shape}"
            )
        
        n_frames = pose_features.shape[0]
        
        if n_frames < self.n_clusters:
            raise InsufficientFramesError(
                "Number of frames must be >= n_clusters",
                frame_count=n_frames,
                required_count=self.n_clusters
            )
        
        # K-meansクラスタリング実行
        self.logger.info(
            f"K-means clustering: n_frames={n_frames}, n_clusters={self.n_clusters}"
        )
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(pose_features)
        
        self.logger.debug(
            f"Clustering completed: unique_clusters={len(np.unique(cluster_labels))}"
        )
        
        return cluster_labels
    
    def compute_minmax_score(
        self,
        frame_idx: int,
        quality_data: Dict[str, List[float]]
    ) -> float:
        """min-maxスコアを計算（v3新機能）
        
        各メトリクス（blur, brightness, completeness）を正規化し、
        最小値（最悪値）を返します。この値が最大となるフレームを選択することで、
        「最悪の品質指標が最も良い」フレームを選ぶことができます。
        
        Args:
            frame_idx: フレームインデックス
            quality_data: 品質データ辞書 {
                "blur_scores": List[float],
                "brightness_scores": List[float],
                "is_complete_list": List[bool]
            }
        
        Returns:
            min-maxスコア（0-1範囲、高いほど良い）
        
        Raises:
            ValueError: 必須キーが不足している場合
        """
        # 必須キーチェック
        required_keys = ["blur_scores", "brightness_scores", "is_complete_list"]
        for key in required_keys:
            if key not in quality_data:
                raise ValueError(f"quality_data must contain '{key}'")
        
        # 各メトリクスを取得
        blur_score = quality_data["blur_scores"][frame_idx]
        brightness_score = quality_data["brightness_scores"][frame_idx]
        is_complete = quality_data["is_complete_list"][frame_idx]
        
        # 正規化（0-1範囲に変換）
        # blur_score: 0-500範囲と仮定（高いほど良い）
        normalized_blur = min(blur_score / 500.0, 1.0)
        
        # brightness_score: 0-255範囲と仮定（高いほど良い）
        normalized_brightness = min(brightness_score / 255.0, 1.0)
        
        # is_complete: True=1.0, False=0.0
        normalized_complete = 1.0 if is_complete else 0.0
        
        # 最小値を取得（最悪値が最も良いフレームを選択）
        minmax_score = min(normalized_blur, normalized_brightness, normalized_complete)
        
        return minmax_score
    
    def _select_greedy_with_interval(
        self,
        n_frames: int,
        quality_scores: List[float],
        frame_numbers: List[int],
        target_count: int,
        min_frame_interval: int
    ) -> List[int]:
        """貪欲法でフレームを選択（間隔制約付き）
        
        品質降順でフレームを走査し、間隔制約を満たすものを選択します。
        
        Args:
            n_frames: 総フレーム数
            quality_scores: 品質スコアリスト (N,)
            frame_numbers: 元動画のフレーム番号リスト (N,)
            target_count: 選択する目標フレーム数
            min_frame_interval: 最小フレーム間隔
        
        Returns:
            選択フレームインデックスリスト
        """
        # (インデックス, 品質スコア, フレーム番号)のリストを作成
        candidates = [
            (idx, score, frame_numbers[idx])
            for idx, score in enumerate(quality_scores)
        ]
        
        # 品質スコア降順でソート
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 貪欲法で選択
        selected_indices = []
        selected_frame_numbers = []
        
        for idx, quality, frame_num in candidates:
            # フレーム間隔制約をチェック
            if all(abs(frame_num - existing_num) >= min_frame_interval 
                   for existing_num in selected_frame_numbers):
                selected_indices.append(idx)
                selected_frame_numbers.append(frame_num)
                self.logger.debug(
                    f"選択: index={idx}, frame_num={frame_num}, quality={quality:.4f}"
                )
                
                # 目標数に達したら終了
                if len(selected_indices) >= target_count:
                    break
        
        return selected_indices
    
    def _select_best_from_clusters(
        self,
        n_frames: int,
        quality_scores: List[float],
        cluster_labels: np.ndarray,
        n_clusters: int,
        min_frame_interval: int = 0,
        frame_numbers: Optional[List[int]] = None
    ) -> List[int]:
        """各クラスタから品質最高のフレームを選択
        
        Args:
            n_frames: 総フレーム数
            quality_scores: 品質スコアリスト (N,)
            cluster_labels: クラスタラベル配列 (N,)
            n_clusters: クラスタ数
            min_frame_interval: 最小フレーム間隔（0の場合は制約なし）
            frame_numbers: 元動画のフレーム番号リスト (N,)（Noneの場合はインデックスを使用）
        
        Returns:
            選択フレームインデックスリスト
        """
        if frame_numbers is None:
            frame_numbers = list(range(n_frames))
        
        selected_indices = []
        selected_frame_numbers = []
        
        for cluster_id in range(n_clusters):
            # クラスタ内のフレームインデックスを取得
            cluster_mask = cluster_labels == cluster_id
            cluster_frame_indices = np.where(cluster_mask)[0]
            
            if len(cluster_frame_indices) == 0:
                self.logger.warning(f"Cluster {cluster_id} is empty, skipping")
                continue
            
            # 品質スコアを取得してソート
            cluster_quality_scores = [quality_scores[i] for i in cluster_frame_indices]
            sorted_pairs = sorted(
                zip(cluster_frame_indices, cluster_quality_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # 間隔制約を満たす最高品質のフレームを選択
            selected = False
            for frame_idx, quality in sorted_pairs:
                frame_num = frame_numbers[frame_idx]
                
                # 間隔制約チェック（min_frame_interval == 0なら制約なし）
                if min_frame_interval == 0 or all(
                    abs(frame_num - existing_num) >= min_frame_interval 
                    for existing_num in selected_frame_numbers
                ):
                    selected_indices.append(int(frame_idx))
                    selected_frame_numbers.append(frame_num)
                    self.logger.debug(
                        f"Cluster {cluster_id}: selected frame {frame_idx}, "
                        f"quality_score={quality:.4f}, "
                        f"cluster_size={len(cluster_frame_indices)}"
                    )
                    selected = True
                    break
            
            if not selected:
                self.logger.warning(
                    f"Cluster {cluster_id}: no frame satisfied interval constraint"
                )
        
        return selected_indices
    
    def select_frames_diversity_first(
        self,
        frames: List[np.ndarray],
        quality_scores: List[float],
        pose_features: np.ndarray,
        target_count: int,
        min_frame_interval: int = 0,
        quality_criteria: str = "overall",
        quality_data: Optional[Dict[str, List[float]]] = None,
        frame_numbers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Strategy A: diversity_first（姿勢多様性優先）
        
        アルゴリズム:
        1. K-meansクラスタリング（全フレーム → n_clustersクラスタ）
        2. 各クラスタから品質最高のフレームを選択（間隔制約付き）
        
        Args:
            frames: フレームリスト (N,)
            quality_scores: 品質スコアリスト (N,)
            pose_features: 正規化済み姿勢特徴行列 (N×6)
            target_count: 選択する目標フレーム数（参考値、n_clusters個選択される）
            min_frame_interval: 最小フレーム間隔（0の場合は制約なし、デフォルト: 0）
            quality_criteria: 品質評価基準（"overall" or "minmax"、デフォルト: "overall"）
            quality_data: min-maxスコア計算用の品質データ（quality_criteria="minmax"時に必須）
            frame_numbers: 元動画のフレーム番号リスト (N,)（Noneの場合はインデックスを使用）
        
        Returns:
            {
                "selected_indices": List[int],
                "cluster_labels": np.ndarray,
                "n_clusters": int,
                "n_selected": int,
                "strategy_used": str
            }
        
        Raises:
            ValueError: パラメータが不正な場合
        """
        n_frames = len(frames)
        
        # 入力検証
        if len(quality_scores) != n_frames:
            raise ValueError(
                f"Length mismatch: frames={n_frames}, quality_scores={len(quality_scores)}"
            )
        
        if quality_criteria not in ["overall", "minmax"]:
            raise ValueError(
                f"quality_criteria must be 'overall' or 'minmax', got '{quality_criteria}'"
            )
        
        if quality_criteria == "minmax" and quality_data is None:
            raise ValueError(
                "quality_data is required when quality_criteria='minmax'"
            )
        
        if frame_numbers is None:
            frame_numbers = list(range(n_frames))
        
        self.logger.info(
            f"diversity_first戦略開始: n_frames={n_frames}, n_clusters={self.n_clusters}, "
            f"min_frame_interval={min_frame_interval}, quality_criteria={quality_criteria}"
        )
        
        # ステップ1: K-meansクラスタリング
        cluster_labels = self.cluster_frames(pose_features)
        
        # 品質スコアを計算
        if quality_criteria == "overall":
            final_quality_scores = quality_scores
        else:  # minmax
            final_quality_scores = [
                self.compute_minmax_score(i, quality_data)
                for i in range(n_frames)
            ]
        
        # ステップ2: 各クラスタから品質最高のフレームを選択
        selected_indices = self._select_best_from_clusters(
            n_frames=n_frames,
            quality_scores=final_quality_scores,
            cluster_labels=cluster_labels,
            n_clusters=self.n_clusters,
            min_frame_interval=min_frame_interval,
            frame_numbers=frame_numbers
        )
        
        self.logger.info(
            f"diversity_first戦略完了: {len(selected_indices)}/{self.n_clusters}フレーム選択"
        )
        
        return {
            "selected_indices": selected_indices,
            "cluster_labels": cluster_labels,
            "n_clusters": self.n_clusters,
            "n_selected": len(selected_indices),
            "strategy_used": "diversity_first"
        }
    
    def select_frames_interval_first(
        self,
        frames: List[np.ndarray],
        quality_scores: List[float],
        pose_features: np.ndarray,
        target_count: int,
        min_frame_interval: int,
        quality_criteria: str = "overall",
        quality_data: Optional[Dict[str, List[float]]] = None,
        frame_numbers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Strategy B: interval_first（フレーム間隔優先）
        
        アルゴリズム:
        1. 貪欲法で品質上位を選択（全フレーム → target_count個、間隔制約付き）
        2. K-meansクラスタリング（target_count個 → n_clustersクラスタ）
        3. 各クラスタから品質最高のフレームを選択（最終的にn_clusters個）
        
        Args:
            frames: フレームリスト (N,)
            quality_scores: 品質スコアリスト (N,)
            pose_features: 正規化済み姿勢特徴行列 (N×6)
            target_count: 貪欲法で選択する目標フレーム数
            min_frame_interval: 最小フレーム間隔（必須）
            quality_criteria: 品質評価基準（"overall" or "minmax"、デフォルト: "overall"）
            quality_data: min-maxスコア計算用の品質データ（quality_criteria="minmax"時に必須）
            frame_numbers: 元動画のフレーム番号リスト (N,)（Noneの場合はインデックスを使用）
        
        Returns:
            {
                "selected_indices": List[int],
                "cluster_labels": np.ndarray,
                "n_clusters": int,
                "n_selected": int,
                "strategy_used": str,
                "greedy_count": int  # 貪欲法で選択したフレーム数
            }
        
        Raises:
            ValueError: パラメータが不正な場合
        """
        n_frames = len(frames)
        
        # 入力検証
        if len(quality_scores) != n_frames:
            raise ValueError(
                f"Length mismatch: frames={n_frames}, quality_scores={len(quality_scores)}"
            )
        
        if min_frame_interval <= 0:
            raise ValueError(
                f"min_frame_interval must be positive for interval_first strategy, got {min_frame_interval}"
            )
        
        if quality_criteria not in ["overall", "minmax"]:
            raise ValueError(
                f"quality_criteria must be 'overall' or 'minmax', got '{quality_criteria}'"
            )
        
        if quality_criteria == "minmax" and quality_data is None:
            raise ValueError(
                "quality_data is required when quality_criteria='minmax'"
            )
        
        if frame_numbers is None:
            frame_numbers = list(range(n_frames))
        
        self.logger.info(
            f"interval_first戦略開始: n_frames={n_frames}, target_count={target_count}, "
            f"n_clusters={self.n_clusters}, min_frame_interval={min_frame_interval}, "
            f"quality_criteria={quality_criteria}"
        )
        
        # 品質スコアを計算
        if quality_criteria == "overall":
            final_quality_scores = quality_scores
        else:  # minmax
            final_quality_scores = [
                self.compute_minmax_score(i, quality_data)
                for i in range(n_frames)
            ]
        
        # ステップ1: 貪欲法で品質上位を選択（間隔制約付き）
        self.logger.info(f"ステップ1: 貪欲法選択開始（target_count={target_count}）")
        greedy_indices = self._select_greedy_with_interval(
            n_frames=n_frames,
            quality_scores=final_quality_scores,
            frame_numbers=frame_numbers,
            target_count=target_count,
            min_frame_interval=min_frame_interval
        )
        
        self.logger.info(
            f"ステップ1完了: {len(greedy_indices)}/{target_count}フレーム選択"
        )
        
        if len(greedy_indices) < self.n_clusters:
            self.logger.warning(
                f"貪欲法で選択したフレーム数（{len(greedy_indices)}）が"
                f"n_clusters（{self.n_clusters}）より少ないため、"
                f"n_clustersを{len(greedy_indices)}に調整します"
            )
            effective_n_clusters = len(greedy_indices)
        else:
            effective_n_clusters = self.n_clusters
        
        # ステップ2: 貪欲法で選択したフレームのみでK-meansクラスタリング
        self.logger.info(
            f"ステップ2: K-meansクラスタリング開始（n_frames={len(greedy_indices)}, "
            f"n_clusters={effective_n_clusters}）"
        )
        
        greedy_pose_features = pose_features[greedy_indices]
        
        # 一時的にn_clustersを変更してクラスタリング実行
        original_n_clusters = self.n_clusters
        self.n_clusters = effective_n_clusters
        
        try:
            greedy_cluster_labels = self.cluster_frames(greedy_pose_features)
        finally:
            self.n_clusters = original_n_clusters
        
        # ステップ3: 各クラスタから品質最高のフレームを選択
        self.logger.info(
            f"ステップ3: 各クラスタから代表選択開始（n_clusters={effective_n_clusters}）"
        )
        
        # 貪欲法で選択したフレーム内でのインデックスを元のインデックスに変換
        greedy_quality_scores = [final_quality_scores[i] for i in greedy_indices]
        greedy_frame_numbers = [frame_numbers[i] for i in greedy_indices]
        
        # 各クラスタから品質最高を選択（間隔制約なし、すでに貪欲法で制約済み）
        selected_greedy_indices = self._select_best_from_clusters(
            n_frames=len(greedy_indices),
            quality_scores=greedy_quality_scores,
            cluster_labels=greedy_cluster_labels,
            n_clusters=effective_n_clusters,
            min_frame_interval=0,  # 貪欲法で既に制約済み
            frame_numbers=greedy_frame_numbers
        )
        
        # 元のフレームリストのインデックスに変換
        final_selected_indices = [greedy_indices[i] for i in selected_greedy_indices]
        
        self.logger.info(
            f"interval_first戦略完了: 貪欲法{len(greedy_indices)}→最終{len(final_selected_indices)}フレーム選択"
        )
        
        # 全フレームに対するクラスタラベルを作成（選択されなかったフレームは-1）
        full_cluster_labels = np.full(n_frames, -1, dtype=int)
        for greedy_idx, cluster_label in zip(greedy_indices, greedy_cluster_labels):
            full_cluster_labels[greedy_idx] = cluster_label
        
        return {
            "selected_indices": final_selected_indices,
            "cluster_labels": full_cluster_labels,
            "n_clusters": effective_n_clusters,
            "n_selected": len(final_selected_indices),
            "strategy_used": "interval_first",
            "greedy_count": len(greedy_indices)
        }
    
    def select_frames(
        self,
        strategy: str,
        frames: List[np.ndarray],
        quality_scores: List[float],
        pose_features: np.ndarray,
        target_count: int,
        min_frame_interval: int = 0,
        quality_criteria: str = "overall",
        quality_data: Optional[Dict[str, List[float]]] = None,
        frame_numbers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """統一フレーム選択インターフェース（v4新機能）
        
        2つの選択戦略を統一的に呼び出すインターフェースです。
        
        戦略:
        - "diversity_first": 姿勢多様性優先
          K-means → 各クラスタから品質最高選択
          
        - "interval_first": フレーム間隔優先
          貪欲法（間隔制約） → K-means → 各クラスタから品質最高選択
        
        Args:
            strategy: 選択戦略（"diversity_first" or "interval_first"）
            frames: フレームリスト (N,)
            quality_scores: 品質スコアリスト (N,)
            pose_features: 正規化済み姿勢特徴行列 (N×6)
            target_count: 選択する目標フレーム数
                         - diversity_first: 参考値（n_clusters個選択される）
                         - interval_first: 貪欲法の目標数
            min_frame_interval: 最小フレーム間隔（0の場合は制約なし、デフォルト: 0）
            quality_criteria: 品質評価基準（"overall" or "minmax"、デフォルト: "overall"）
            quality_data: min-maxスコア計算用の品質データ（quality_criteria="minmax"時に必須）
            frame_numbers: 元動画のフレーム番号リスト (N,)（Noneの場合はインデックスを使用）
        
        Returns:
            {
                "selected_indices": List[int],  # 選択フレームインデックス
                "cluster_labels": np.ndarray,   # クラスタラベル (N,)
                "n_clusters": int,              # クラスタ数
                "n_selected": int,              # 選択フレーム数
                "strategy_used": str,           # 使用した戦略
                "greedy_count": int             # 貪欲法選択数（interval_firstのみ）
            }
        
        Raises:
            ValueError: strategyが不正な場合、またはパラメータが不正な場合
        
        Example:
            >>> # diversity_first戦略
            >>> result = selector.select_frames(
            ...     strategy="diversity_first",
            ...     frames=frames,
            ...     quality_scores=quality_scores,
            ...     pose_features=pose_features,
            ...     target_count=30
            ... )
            
            >>> # interval_first戦略
            >>> result = selector.select_frames(
            ...     strategy="interval_first",
            ...     frames=frames,
            ...     quality_scores=quality_scores,
            ...     pose_features=pose_features,
            ...     target_count=60,
            ...     min_frame_interval=20
            ... )
        """
        if strategy == "diversity_first":
            return self.select_frames_diversity_first(
                frames=frames,
                quality_scores=quality_scores,
                pose_features=pose_features,
                target_count=target_count,
                min_frame_interval=min_frame_interval,
                quality_criteria=quality_criteria,
                quality_data=quality_data,
                frame_numbers=frame_numbers
            )
        elif strategy == "interval_first":
            if min_frame_interval <= 0:
                raise ValueError(
                    "interval_first strategy requires min_frame_interval > 0, "
                    f"got {min_frame_interval}"
                )
            
            return self.select_frames_interval_first(
                frames=frames,
                quality_scores=quality_scores,
                pose_features=pose_features,
                target_count=target_count,
                min_frame_interval=min_frame_interval,
                quality_criteria=quality_criteria,
                quality_data=quality_data,
                frame_numbers=frame_numbers
            )
        else:
            raise ValueError(
                f"strategy must be 'diversity_first' or 'interval_first', got '{strategy}'"
            )
    
    def select_representative_frames(
        self,
        frames: List[np.ndarray],
        quality_scores: List[float],
        cluster_labels: np.ndarray,
        target_count: int,
        min_frame_interval: int = 0,
        quality_criteria: str = "overall",
        quality_data: Optional[Dict[str, List[float]]] = None,
        frame_numbers: Optional[List[int]] = None
    ) -> List[int]:
        """代表フレームを選択（後方互換ラッパー）
        
        このメソッドは後方互換性のために残されています。
        新しいコードでは select_frames() を使用してください。
        
        動作:
        - min_frame_interval > 0: interval_first戦略を使用
        - min_frame_interval == 0: diversity_first戦略を使用
        
        Args:
            frames: フレームリスト (N,)
            quality_scores: 品質スコアリスト (N,)
            cluster_labels: クラスタラベル配列 (N,)（参考用、内部で再計算される場合あり）
            target_count: 選択する目標フレーム数
            min_frame_interval: 最小フレーム間隔（0の場合は制約なし、デフォルト: 0）
            quality_criteria: 品質評価基準（"overall" or "minmax"、デフォルト: "overall"）
            quality_data: min-maxスコア計算用の品質データ（quality_criteria="minmax"時に必須）
            frame_numbers: 元動画のフレーム番号リスト (N,)（Noneの場合はインデックスを使用）
        
        Returns:
            選択フレームインデックスリスト
        
        Raises:
            ValueError: パラメータが不正な場合
            NotImplementedError: pose_featuresが必要な場合
        
        Note:
            このメソッドは cluster_labels を受け取りますが、内部で再度
            K-meansクラスタリングを実行する必要がある場合があります。
            そのため、呼び出し側で pose_features を用意し、
            select_frames() を直接呼び出すことを推奨します。
        """
        self.logger.warning(
            "select_representative_frames() is deprecated. "
            "Use select_frames() instead."
        )
        
        # pose_featuresが必要だが提供されていないため、エラーを投げる
        raise NotImplementedError(
            "select_representative_frames() requires pose_features but it's not provided. "
            "Please use select_frames() with pose_features instead. "
            "\n\nExample:\n"
            "  result = selector.select_frames(\n"
            "      strategy='diversity_first' if min_frame_interval == 0 else 'interval_first',\n"
            "      frames=frames,\n"
            "      quality_scores=quality_scores,\n"
            "      pose_features=pose_features,  # Required!\n"
            "      target_count=target_count,\n"
            "      min_frame_interval=min_frame_interval\n"
            "  )\n"
            "  selected_indices = result['selected_indices']"
        )
    
    def select_diverse_frames(
        self,
        frames: List[np.ndarray],
        rvecs: List[np.ndarray],
        tvecs: List[np.ndarray],
        quality_scores: List[float]
    ) -> Dict[str, Any]:
        """統合フレーム選択メソッド（廃止予定）
        
        このメソッドは廃止予定です。
        代わりに select_frames() を使用してください。
        
        Raises:
            NotImplementedError: このメソッドは実装されていません
        """
        raise NotImplementedError(
            "This method is deprecated. Use select_frames() instead. "
            "\n\nExample:\n"
            "  evaluator = FrameDiversityEvaluator()\n"
            "  pose_features = evaluator.compute_diversity_features(rvecs, tvecs)\n"
            "  normalized_features = evaluator.normalize_features(pose_features)\n"
            "  result = selector.select_frames(\n"
            "      strategy='diversity_first',\n"
            "      frames=frames,\n"
            "      quality_scores=quality_scores,\n"
            "      pose_features=normalized_features,\n"
            "      target_count=30\n"
            "  )"
        )
    
    def compute_diversity_metrics(
        self,
        pose_features: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """多様性指標を算出
        
        クラスタ内分散とクラスタ間距離を算出します。
        
        Args:
            pose_features: 正規化済み姿勢特徴行列 (N×6)
            cluster_labels: クラスタラベル配列 (N,)
        
        Returns:
            多様性指標辞書 {
                "intra_cluster_variance": float,  # クラスタ内分散（低いほど良い）
                "inter_cluster_distance": float   # クラスタ間距離（高いほど良い）
            }
        
        Raises:
            ValueError: 入力配列の形状が不正な場合
        """
        # 入力検証
        if pose_features.ndim != 2:
            raise ValueError(
                f"pose_features must be 2D array, got shape {pose_features.shape}"
            )
        
        if len(cluster_labels) != pose_features.shape[0]:
            raise ValueError(
                f"Length mismatch: pose_features={pose_features.shape[0]}, "
                f"cluster_labels={len(cluster_labels)}"
            )
        
        # クラスタ内分散を算出
        intra_cluster_variances = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = pose_features[cluster_mask]
            
            if len(cluster_features) == 0:
                continue
            
            # クラスタ内の分散を算出（全特徴次元の平均）
            cluster_variance = np.mean(np.var(cluster_features, axis=0))
            intra_cluster_variances.append(cluster_variance)
        
        intra_cluster_variance = float(np.mean(intra_cluster_variances))
        
        # クラスタ間距離を算出
        # 各クラスタの中心を算出
        cluster_centers = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = pose_features[cluster_mask]
            
            if len(cluster_features) == 0:
                continue
            
            cluster_center = np.mean(cluster_features, axis=0)
            cluster_centers.append(cluster_center)
        
        if len(cluster_centers) < 2:
            inter_cluster_distance = 0.0
        else:
            # クラスタ中心間の距離の平均を算出
            cluster_centers_array = np.array(cluster_centers)
            n_centers = len(cluster_centers_array)
            
            distances = []
            for i in range(n_centers):
                for j in range(i + 1, n_centers):
                    distance = np.linalg.norm(
                        cluster_centers_array[i] - cluster_centers_array[j]
                    )
                    distances.append(distance)
            
            inter_cluster_distance = float(np.mean(distances))
        
        self.logger.debug(
            f"Diversity metrics: intra_cluster_variance={intra_cluster_variance:.4f}, "
            f"inter_cluster_distance={inter_cluster_distance:.4f}"
        )
        
        return {
            "intra_cluster_variance": intra_cluster_variance,
            "inter_cluster_distance": inter_cluster_distance
        }
