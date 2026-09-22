"""進捗状況レポートモジュール

このモジュールは、動画処理の進捗状況をJSON形式でファイルに出力します。
親プロセス側でGUI表示できるようにリアルタイム更新を行います。

主要クラス:
    ProgressReporter: 進捗状況をJSON形式でレポートするクラス

カスタム例外:
    ProgressReporterError: 進捗レポート関連のベース例外
    ProgressFileError: 進捗ファイル操作のエラー

使用例:
    >>> from pathlib import Path
    >>> reporter = ProgressReporter(
    ...     output_path=Path("progress.json"),
    ...     total_frames=1000
    ... )
    >>> reporter.initialize()
    >>> for i in range(1000):
    ...     reporter.update(i + 1)
    >>> reporter.finalize(success=True)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

# ログ設定
logger = logging.getLogger(__name__)


# ========================================
# カスタム例外
# ========================================

class ProgressReporterError(Exception):
    """進捗レポート関連のベース例外"""
    pass


class ProgressFileError(ProgressReporterError):
    """進捗ファイル操作のエラー"""
    pass


# ========================================
# データクラス
# ========================================

@dataclass
class ProgressData:
    """進捗データ構造
    
    Attributes:
        total_frames: 総フレーム数
        processed_frames: 処理済みフレーム数
        progress_percentage: 進捗率（0-100%）
        start_time: 処理開始時刻（ISO 8601形式）
        elapsed_time_seconds: 経過時間（秒）
        estimated_remaining_seconds: 推定残り時間（秒）
        current_frame_info: 現在のフレーム情報
        status: 処理ステータス（'initializing', 'processing', 'completed', 'error'）
        error_message: エラーメッセージ（エラー時のみ）
    """
    total_frames: int
    processed_frames: int
    progress_percentage: float
    start_time: str
    elapsed_time_seconds: float
    estimated_remaining_seconds: float
    current_frame_info: Optional[Dict[str, Any]]
    status: str
    error_message: Optional[str]


@dataclass
class StepInfo:
    """ステップ情報
    
    Attributes:
        step_id: ステップID（1-4）
        name_ja: ステップ名（日本語）
        name_en: ステップ名（英語）
        status: ステータス（'pending', 'in_progress', 'completed', 'error', 'skipped'）
        progress: 進捗率（0-100%）
        start_time: 開始時刻（ISO 8601形式、None=未開始）
        end_time: 終了時刻（ISO 8601形式、None=未完了）
        details: 詳細情報（オプション）
    """
    step_id: int
    name_ja: str
    name_en: str
    status: str = 'pending'
    progress: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ProcessProgress:
    """処理全体の進捗情報
    
    Attributes:
        process_id: プロセスID（タイムスタンプ形式）
        overall_progress: 全体進捗率（0-100%）
        current_step: 現在のステップID（1-5、None=未開始/完了）
        steps: 各ステップの情報リスト
    """
    process_id: str
    overall_progress: float
    current_step: Optional[int]
    steps: List[StepInfo]


# ========================================
# 進捗レポータークラス
# ========================================

class ProgressReporter:
    """進捗状況レポートクラス
    
    処理の進捗状況をJSON形式でファイルに出力します。
    親プロセス側でGUI表示できるようにリアルタイム更新を行います。
    
    Attributes:
        output_path: 進捗ファイルの出力パス
        total_frames: 総フレーム数
        start_time: 処理開始時刻（UNIX時間、秒単位）
        use_5steps: 5ステップモード使用フラグ
        process_progress: 5ステップ進捗データ
        step_weights: ステップごとの重み
        
    使用例:
        >>> reporter = ProgressReporter(
        ...     output_path=Path("progress.json"),
        ...     total_frames=1000
        ... )
        >>> reporter.initialize()
        >>> reporter.update(500)
        >>> reporter.finalize(success=True)
    """
    
    # ステップごとの重み（合計100%）
    # 4ステップ構成: 実態に合わせた重み配分
    STEP_WEIGHTS = {
        1: 10.0,  # パラメータ調整: 10%
        2: 30.0,  # フレーム解析: 30%（旧Step 2+3統合）
        3: 55.0,  # カラーマップ生成: 55%
        4: 5.0,   # カラーマップ補正: 5%
    }
    
    def __init__(self, output_path: Path, total_frames: int):
        """初期化
        
        Args:
            output_path: 進捗ファイルの出力パス
            total_frames: 総フレーム数
            
        Raises:
            ProgressReporterError: total_framesが0以下の場合
        """
        if total_frames <= 0:
            raise ProgressReporterError(
                f"total_framesは正の値である必要があります: {total_frames}"
            )
        
        self.output_path: Path = Path(output_path)
        self.total_frames: int = total_frames
        self.start_time: float = 0.0
        
        # 5ステップモード用の属性
        self.use_5steps: bool = False
        self.process_progress: Optional[ProcessProgress] = None
        self.step_weights: Dict[int, float] = self.STEP_WEIGHTS.copy()
        
        logger.info(
            f"ProgressReporter初期化: output={self.output_path}, "
            f"total_frames={self.total_frames}"
        )
    
    def initialize(self) -> None:
        """進捗レポート初期化
        
        進捗ファイルを作成し、初期状態を書き込みます。
        
        Raises:
            ProgressFileError: ファイル作成に失敗した場合
        """
        try:
            # 出力ディレクトリが存在しない場合は作成
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 開始時刻を記録
            self.start_time = time.perf_counter()
            
            # 初期データを作成
            initial_data = ProgressData(
                total_frames=self.total_frames,
                processed_frames=0,
                progress_percentage=0.0,
                start_time=datetime.now().isoformat(),
                elapsed_time_seconds=0.0,
                estimated_remaining_seconds=0.0,
                current_frame_info=None,
                status='initializing',
                error_message=None
            )
            
            # JSONファイルに書き込み
            self._write_progress(initial_data)
            
            logger.info(f"進捗レポート初期化完了: {self.output_path}")
            
        except OSError as e:
            raise ProgressFileError(
                f"進捗ファイルの作成に失敗しました: {self.output_path}"
            ) from e
    
    def update(
        self,
        processed_frames: int,
        current_frame_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """進捗状況更新
        
        Args:
            processed_frames: 処理済みフレーム数
            current_frame_info: 現在のフレーム情報（オプション）
                例: {
                    "frame_index": 250,
                    "camera_position": [0.0, 0.0, 250.0],
                    "camera_orientation": [0.0, 0.0, 0.0]
                }
        
        Raises:
            ProgressFileError: ファイル書き込みに失敗した場合
        """
        try:
            # 経過時間を計算
            elapsed_time = time.perf_counter() - self.start_time
            
            # 進捗率を計算
            progress_percentage = self.get_progress_percentage(processed_frames)
            
            # 推定残り時間を計算
            estimated_remaining = self.get_estimated_time_remaining(
                processed_frames
            )
            
            # 進捗データを作成
            progress_data = ProgressData(
                total_frames=self.total_frames,
                processed_frames=processed_frames,
                progress_percentage=progress_percentage,
                start_time=datetime.fromtimestamp(
                    time.time() - elapsed_time
                ).isoformat(),
                elapsed_time_seconds=elapsed_time,
                estimated_remaining_seconds=estimated_remaining,
                current_frame_info=current_frame_info,
                status='processing',
                error_message=None
            )
            
            # JSONファイルに書き込み
            self._write_progress(progress_data)
            
            logger.debug(
                f"進捗更新: {processed_frames}/{self.total_frames} "
                f"({progress_percentage:.1f}%)"
            )
            
        except Exception as e:
            logger.error(f"進捗更新に失敗しました: {e}")
            raise ProgressFileError(
                f"進捗ファイルの更新に失敗しました: {self.output_path}"
            ) from e
    
    def finalize(
        self,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """進捗レポート終了
        
        Args:
            success: 処理が成功したかどうか
            error_message: エラーメッセージ（失敗時）
            details: ステップの詳細情報（オプション）
            details: ステップの詳細情報（オプション）
            
        Raises:
            ProgressFileError: ファイル書き込みに失敗した場合
        """
        try:
            # 経過時間を計算
            elapsed_time = time.perf_counter() - self.start_time
            
            # 最終データを作成
            final_data = ProgressData(
                total_frames=self.total_frames,
                processed_frames=self.total_frames if success else 0,
                progress_percentage=100.0 if success else 0.0,
                start_time=datetime.fromtimestamp(
                    time.time() - elapsed_time
                ).isoformat(),
                elapsed_time_seconds=elapsed_time,
                estimated_remaining_seconds=0.0,
                current_frame_info=None,
                status='completed' if success else 'error',
                error_message=error_message
            )
            
            # JSONファイルに書き込み
            self._write_progress(final_data)
            
            if success:
                logger.info(
                    f"進捗レポート完了: {self.total_frames}フレーム処理 "
                    f"（{elapsed_time:.1f}秒）"
                )
            else:
                logger.error(
                    f"進捗レポート異常終了: {error_message}"
                )
                
        except Exception as e:
            logger.error(f"進捗終了処理に失敗しました: {e}")
            raise ProgressFileError(
                f"進捗ファイルの終了処理に失敗しました: {self.output_path}"
            ) from e
    
    def get_progress_percentage(self, processed_frames: int) -> float:
        """進捗率計算
        
        Args:
            processed_frames: 処理済みフレーム数
        
        Returns:
            progress_percentage: 進捗率（0-100%）
        """
        if self.total_frames == 0:
            return 0.0
        
        percentage = (processed_frames / self.total_frames) * 100.0
        return min(100.0, max(0.0, percentage))
    
    def get_estimated_time_remaining(self, processed_frames: int) -> float:
        """推定残り時間計算
        
        Args:
            processed_frames: 処理済みフレーム数
        
        Returns:
            estimated_seconds: 推定残り時間（秒）
        """
        if processed_frames == 0:
            # 処理が開始されていない場合は推定不可
            return 0.0
        
        # 経過時間を取得
        elapsed_time = time.perf_counter() - self.start_time
        
        # 1フレームあたりの処理時間を計算
        time_per_frame = elapsed_time / processed_frames
        
        # 残りフレーム数を計算
        remaining_frames = self.total_frames - processed_frames
        
        # 推定残り時間を計算
        estimated_remaining = time_per_frame * remaining_frames
        
        return max(0.0, estimated_remaining)
    
    def _write_progress(self, progress_data: ProgressData, retry_on_error: bool = True) -> None:
        """進捗データをJSONファイルに書き込む

        ファイルの原子性を保証するため、一時ファイル→renameの手順で書き込みます。
        
        Args:
            progress_data: 進捗データ
            retry_on_error: True の場合、PermissionError 発生時に最大5回リトライ。
                           False の場合、エラーを無視して静かに続行（debugログのみ出力）。

        Raises:
            ProgressFileError: retry_on_error=True で、ファイル書き込みに失敗した場合（5回リトライ後）

        Note:
            - retry_on_error=True時のリトライ回数: 最大5回
            - バックオフ時間: 200ms、500ms、1000ms、2000ms、4000ms（指数バックオフ）
            - 対象エラー: PermissionError のみ
            - retry_on_error=False時: PermissionError を静かに無視（処理続行）
        """
        max_retries = 5
        retry_delays = [0.2, 0.5, 1.0, 2.0, 4.0]  # 秒単位（合計約7.7秒）
        temp_path = None

        for attempt in range(max_retries + 1):  # 初回 + リトライ5回
            try:
                # 一時ファイルパスを生成（ユニークなサフィックスで競合回避）
                temp_suffix = f'.tmp.{os.getpid()}'
                temp_path = self.output_path.with_suffix(temp_suffix)

                # データを辞書に変換
                data_dict = asdict(progress_data)

                # 一時ファイルに書き込み
                with open(temp_path, 'w', encoding='utf-8', newline='') as f:
                    json.dump(data_dict, f, indent=2, ensure_ascii=False)

                # 原子的にリネーム（上書き）
                os.replace(temp_path, self.output_path)

                # 成功したらリターン
                return

            except PermissionError as e:
                # retry_on_error=False の場合は静かに無視
                if not retry_on_error:
                    logger.debug(
                        f"進捗ファイル書き込み失敗（リトライなし）: {self.output_path} (PermissionError)"
                    )
                    # 一時ファイルが残っている場合は削除
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
                    return  # 処理を続行

                # 最後の試行の場合はエラーを送出
                if attempt == max_retries:
                    # 一時ファイルが残っている場合は削除
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass

                    raise ProgressFileError(
                        f"進捗ファイルの書き込みに失敗しました（{max_retries}回リトライ後）: {self.output_path}"
                    ) from e

                # リトライを記録（WARNINGからDEBUGに変更）
                delay = retry_delays[attempt]
                logger.debug(
                    f"進捗ファイル書き込みリトライ {attempt + 1}/{max_retries}: "
                    f"{self.output_path} (PermissionError, 次回まで{delay:.1f}秒待機)"
                )

                # バックオフ
                time.sleep(delay)

            except OSError as e:
                # PermissionError以外のOSErrorは即座にエラー
                # 一時ファイルが残っている場合は削除
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

                raise ProgressFileError(
                    f"進捗ファイルの書き込みに失敗しました: {self.output_path}"
                ) from e
    
    # ========================================
    # 5ステップモード用の新規メソッド
    # ========================================
    
    def initialize_steps(self, steps: List[Dict[str, str]]) -> None:
        """4ステップ進捗管理の初期化
        
        Args:
            steps: ステップ情報の辞書リスト（4要素）
                各辞書は以下のキーを持つ:
                - step_id (int): ステップID（1-4）
                - name_ja (str): ステップ名（日本語）
                - name_en (str): ステップ名（英語）
        
        Raises:
            ProgressReporterError: ステップ数が5でない場合
            ProgressReporterError: ステップIDが重複している場合
            ProgressReporterError: 必須キーが不足している場合
        
        使用例:
            >>> steps = [
            ...     {"step_id": 1, "name_ja": "パラメータ調整", "name_en": "Parameter Tuning"},
            ...     {"step_id": 2, "name_ja": "カメラ方位抽出", "name_en": "Vanishing Point Estimation"},
            ...     {"step_id": 3, "name_ja": "フレーム距離抽出", "name_en": "OCR Distance Reading"},
            ...     {"step_id": 4, "name_ja": "カラーマップ生成", "name_en": "Colormap Generation"},
            ...     {"step_id": 5, "name_ja": "カラーマップ補正", "name_en": "Colormap Correction"},
            ... ]
            >>> reporter.initialize_steps(steps)
        """
        # 出力ディレクトリが存在しない場合は作成
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 開始時刻を記録
        self.start_time = time.perf_counter()
        
        # ステップ数の検証
        if len(steps) != 4:
            raise ProgressReporterError(
                f"ステップ数は4である必要があります: {len(steps)}"
            )
        
        # ステップIDの重複チェック
        step_ids = [s["step_id"] for s in steps]
        if len(step_ids) != len(set(step_ids)):
            raise ProgressReporterError(
                f"ステップIDが重複しています: {step_ids}"
            )
        
        # 必須キーの確認
        required_keys = {"step_id", "name_ja", "name_en"}
        for step in steps:
            if not required_keys.issubset(step.keys()):
                missing = required_keys - step.keys()
                raise ProgressReporterError(
                    f"ステップに必須キーが不足しています: {missing}"
                )
        
        # StepInfoオブジェクトのリストを作成
        step_infos = [
            StepInfo(
                step_id=step["step_id"],
                name_ja=step["name_ja"],
                name_en=step["name_en"],
                status='pending',
                progress=0.0,
                start_time=None,
                end_time=None,
                details=None
            )
            for step in steps
        ]
        
        # ProcessProgressオブジェクトを作成
        process_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.process_progress = ProcessProgress(
            process_id=process_id,
            overall_progress=0.0,
            current_step=None,
            steps=step_infos
        )
        
        # 5ステップモードを有効化
        self.use_5steps = True
        
        # JSON出力
        self._write_progress_5steps()
        
        logger.info(
            f"5ステップ進捗管理初期化完了: process_id={process_id}"
        )
    
    def start_step(self, step_id: int, start_time: Optional[str] = None) -> None:
        """ステップの開始

        Args:
            step_id: ステップID（1-4）
            start_time: 開始時刻（ISO 8601形式）。指定時はその時刻を使用。

        Raises:
            ProgressReporterError: 5ステップモードが無効の場合
            ProgressReporterError: 無効なステップIDの場合
            ProgressReporterError: ステップが既に開始されている場合

        使用例:
            >>> reporter.start_step(step_id=2)
        """
        if not self.use_5steps or self.process_progress is None:
            raise ProgressReporterError(
                "5ステップモードが初期化されていません。"
                "initialize_steps()を先に呼び出してください。"
            )
        
        # ステップIDの検証
        if not 1 <= step_id <= 4:
            raise ProgressReporterError(
                f"無効なステップIDです: {step_id}（1-4の範囲で指定してください）"
            )
        
        # ステップオブジェクトを取得
        step = self._get_step(step_id)
        
        # 既に開始済みかチェック
        if step.status == 'in_progress':
            raise ProgressReporterError(
                f"ステップ{step_id}は既に開始されています。"
            )
        
        # ステップを開始
        step.status = 'in_progress'
        step.progress = 0.0
        step.start_time = start_time if start_time else datetime.now().isoformat()
        step.end_time = None

        # current_stepを更新
        self.process_progress.current_step = step_id

        # overall_progressを再計算
        self._update_overall_progress()

        # JSON出力
        self._write_progress_5steps()

        logger.info(
            f"ステップ{step_id}開始: {step.name_ja} ({step.name_en})"
        )
    
    def update_step(
        self,
        step_id: int,
        progress: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """ステップの進捗更新
        
        Args:
            step_id: ステップID（1-4）
            progress: 進捗率（0-100%）
            details: 詳細情報（オプション）
        
        Raises:
            ProgressReporterError: 5ステップモードが無効の場合
            ProgressReporterError: 無効なステップIDの場合
            ProgressReporterError: ステップが開始されていない場合
            ProgressReporterError: 進捗率が範囲外の場合
        
        使用例:
            >>> reporter.update_step(
            ...     step_id=2,
            ...     progress=50.0,
            ...     details={"vp_success_rate": 0.95}
            ... )
        """
        if not self.use_5steps or self.process_progress is None:
            raise ProgressReporterError(
                "5ステップモードが初期化されていません。"
            )
        
        # ステップIDの検証
        if not 1 <= step_id <= 4:
            raise ProgressReporterError(
                f"無効なステップIDです: {step_id}"
            )
        
        # 進捗率の検証
        if not 0.0 <= progress <= 100.0:
            raise ProgressReporterError(
                f"進捗率は0-100の範囲で指定してください: {progress}"
            )
        
        # ステップオブジェクトを取得
        step = self._get_step(step_id)
        
        # ステップが開始されているかチェック
        if step.status != 'in_progress':
            raise ProgressReporterError(
                f"ステップ{step_id}が開始されていません。"
                f"start_step()を先に呼び出してください。"
            )
        
        # 進捗を更新
        step.progress = progress
        if details is not None:
            step.details = details
        
        # overall_progressを再計算
        self._update_overall_progress()
        
        # JSON出力（リトライなし）
        self._write_progress_5steps(retry_on_error=False)
        
        logger.debug(
            f"ステップ{step_id}進捗更新: {progress:.1f}%"
        )
    
    def complete_step(
        self,
        step_id: int,
        success: bool = True,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        end_time: Optional[str] = None
    ) -> None:
        """ステップの完了

        Args:
            step_id: ステップID（1-4）
            success: 成功時True、失敗時False
            error_message: エラーメッセージ（失敗時）
            details: 詳細情報（オプション）
            end_time: 終了時刻（ISO 8601形式）。指定時はその時刻を使用。

        Raises:
            ProgressReporterError: 5ステップモードが無効の場合
            ProgressReporterError: 無効なステップIDの場合
            ProgressReporterError: ステップが開始されていない場合

        使用例:
            >>> # 成功時
            >>> reporter.complete_step(step_id=2, success=True)
            >>>
            >>> # 失敗時
            >>> reporter.complete_step(
            ...     step_id=3,
            ...     success=False,
            ...     error_message="OCR読み取り失敗"
            ... )
        """
        if not self.use_5steps or self.process_progress is None:
            raise ProgressReporterError(
                "5ステップモードが初期化されていません。"
            )
        
        # ステップIDの検証
        if not 1 <= step_id <= 4:
            raise ProgressReporterError(
                f"無効なステップIDです: {step_id}"
            )
        
        # ステップオブジェクトを取得
        step = self._get_step(step_id)
        
        # ステップが開始されているかチェック
        if step.status != 'in_progress':
            raise ProgressReporterError(
                f"ステップ{step_id}が開始されていません。"
            )
        
        # ステップを完了
        if success:
            step.status = 'completed'
            step.progress = 100.0
            if details:
                if step.details is None:
                    step.details = {}
                step.details.update(details)
        else:
            step.status = 'error'
            if error_message:
                if step.details is None:
                    step.details = {}
                step.details['error_message'] = error_message
        
        step.end_time = end_time if end_time else datetime.now().isoformat()

        # current_stepをNoneに更新
        self.process_progress.current_step = None
        
        # overall_progressを再計算
        self._update_overall_progress()
        
        # JSON出力（リトライあり）
        self._write_progress_5steps(retry_on_error=True)
        
        if success:
            logger.info(
                f"ステップ{step_id}完了: {step.name_ja}"
            )
        else:
            logger.error(
                f"ステップ{step_id}失敗: {step.name_ja} - {error_message}"
            )
    
    def skip_step(self, step_id: int, reason: str) -> None:
        """ステップのスキップ
        
        Args:
            step_id: ステップID（1-4）
            reason: スキップ理由
        
        Raises:
            ProgressReporterError: 5ステップモードが無効の場合
            ProgressReporterError: 無効なステップIDの場合
        
        使用例:
            >>> reporter.skip_step(
            ...     step_id=1,
            ...     reason="auto_tune_enabled=false"
            ... )
        """
        if not self.use_5steps or self.process_progress is None:
            raise ProgressReporterError(
                "5ステップモードが初期化されていません。"
            )
        
        # ステップIDの検証
        if not 1 <= step_id <= 4:
            raise ProgressReporterError(
                f"無効なステップIDです: {step_id}"
            )
        
        # ステップオブジェクトを取得
        step = self._get_step(step_id)
        
        # ステップをスキップ
        step.status = 'skipped'
        step.progress = 0.0
        step.details = {'skip_reason': reason}
        
        # overall_progressを再計算
        self._update_overall_progress()
        
        # JSON出力
        self._write_progress_5steps()
        
        logger.info(
            f"ステップ{step_id}スキップ: {step.name_ja} - {reason}"
        )
    
    def _get_step(self, step_id: int) -> StepInfo:
        """ステップIDに対応するStepInfoオブジェクトを取得
        
        Args:
            step_id: ステップID（1-4）
        
        Returns:
            StepInfo: ステップ情報オブジェクト
        
        Raises:
            ProgressReporterError: ステップが見つからない場合
        """
        if self.process_progress is None:
            raise ProgressReporterError(
                "ProcessProgressが初期化されていません。"
            )
        
        for step in self.process_progress.steps:
            if step.step_id == step_id:
                return step
        
        raise ProgressReporterError(
            f"ステップID {step_id} が見つかりません。"
        )
    
    def _update_overall_progress(self) -> None:
        """全体進捗率を再計算
        
        各ステップの進捗率と重みから全体進捗率を計算します。
        スキップされたステップは100%として計算します。
        """
        if self.process_progress is None:
            return
        
        overall = 0.0
        for step in self.process_progress.steps:
            step_progress = step.progress
            
            # スキップされたステップは100%として扱う
            if step.status == 'skipped':
                step_progress = 100.0
            
            # 重み付け加算
            weight = self.step_weights.get(step.step_id, 0.0)
            overall += step_progress * weight / 100.0
        
        self.process_progress.overall_progress = overall
    
    def _write_progress_5steps(self, retry_on_error: bool = True) -> None:
        """5ステップ進捗データをJSONファイルに書き込む

        ファイルの原子性を保証するため、一時ファイル→renameの手順で書き込みます。

        Args:
            retry_on_error: True の場合、PermissionError 発生時に最大5回リトライ。
                           False の場合、エラーを無視して静かに続行（debugログのみ出力）。

        Raises:
            ProgressFileError: retry_on_error=True で、ファイル書き込みに失敗した場合（5回リトライ後）

        Note:
            - retry_on_error=True時のリトライ回数: 最大5回
            - バックオフ時間: 200ms、500ms、1000ms、2000ms、4000ms（指数バックオフ）
            - 対象エラー: PermissionError のみ
            - retry_on_error=False時: PermissionError を静かに無視（処理続行）
            - Windows環境でウイルス対策ソフトやWindows Search等がファイルをロックする場合に有効
            - Linux環境では通常リトライなしで成功（POSIXファイルシステム）
        """
        if self.process_progress is None:
            raise ProgressReporterError(
                "ProcessProgressが初期化されていません。"
            )

        max_retries = 5
        retry_delays = [0.2, 0.5, 1.0, 2.0, 4.0]  # 秒単位（合計約7.7秒）
        temp_path = None

        for attempt in range(max_retries + 1):  # 初回 + リトライ5回
            try:
                # 一時ファイルパスを生成（ユニークなサフィックスで競合回避）
                temp_suffix = f'.tmp.{os.getpid()}'
                temp_path = self.output_path.with_suffix(temp_suffix)

                # データを辞書に変換
                data_dict = asdict(self.process_progress)

                # 一時ファイルに書き込み
                with open(temp_path, 'w', encoding='utf-8', newline='') as f:
                    json.dump(data_dict, f, indent=2, ensure_ascii=False)

                # 原子的にリネーム（上書き）
                os.replace(temp_path, self.output_path)

                # 成功したらリターン
                return

            except PermissionError as e:
                # retry_on_error=False の場合は静かに無視
                if not retry_on_error:
                    logger.debug(
                        f"5ステップ進捗ファイル書き込み失敗（リトライなし）: {self.output_path} (PermissionError)"
                    )
                    # 一時ファイルが残っている場合は削除
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
                    return  # 処理を続行

                # 最後の試行の場合はエラーを送出
                if attempt == max_retries:
                    # 一時ファイルが残っている場合は削除
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            logger.debug(
                                f"一時ファイルの削除に失敗しました: {temp_path}"
                            )

                    raise ProgressFileError(
                        f"5ステップ進捗ファイルの書き込みに失敗しました（{max_retries}回リトライ後）: {self.output_path}"
                    ) from e

                # リトライを記録（WARNINGからDEBUGに変更）
                delay = retry_delays[attempt]
                logger.debug(
                    f"進捗ファイル書き込みリトライ {attempt + 1}/{max_retries}: "
                    f"{self.output_path} (PermissionError, 次回まで{delay:.1f}秒待機)"
                )

                # バックオフ
                time.sleep(delay)

            except OSError as e:
                # PermissionError以外のOSErrorは即座にエラー
                # 一時ファイルが残っている場合は削除
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        logger.debug(
                            f"一時ファイルの削除に失敗しました: {temp_path}"
                        )

                raise ProgressFileError(
                    f"5ステップ進捗ファイルの書き込みに失敗しました: {self.output_path}"
                ) from e

