"""フレームストリーム読み込みユーティリティ

メモリ効率的なビデオフレーム・画像ディレクトリ処理のためのユーティリティクラスを提供します。

主要クラス:
    - LazyFrameLoader: ビデオからフレームパスのみを保持し、必要時に読み込み
    - ImageDirectoryLoader: 画像ディレクトリから遅延読み込み
    - BatchFrameProcessor: バッチ処理対応（メモリ効率を保ちながら複数フレームを処理）

設計思想:
    - 本番プログラム（main_twopass.py）のメモリ効率設計に準拠
    - ストリーミング処理でメモリ使用量を最小化（最大2フレーム程度）
    - コンテキストマネージャで安全なリソース管理
    - エラーハンドリングとログ出力の統一

使用例:
    # ビデオストリーミング処理（メモリ最小化）
    with LazyFrameLoader(video_path, start_frame=0, end_frame=100) as loader:
        for frame_idx, frame in loader:
            process_frame(frame)  # メモリに最大1フレームのみ

    # 画像ディレクトリストリーミング処理
    loader = ImageDirectoryLoader(frames_dir, logger, pattern="frame_*.png")
    for frame_idx, frame in loader:
        process_frame(frame)  # メモリに最大1フレームのみ

    # バッチ処理（品質評価等）
    def evaluate_batch(batch):
        return [evaluate_quality(idx, frame) for idx, frame in batch]

    processor = BatchFrameProcessor(evaluate_batch, batch_size=32)
    results = processor.process_video(video_path)
"""

# 標準ライブラリ
import logging
from pathlib import Path
from typing import Optional, Tuple, Iterator, List, Callable, Any, Union

# サードパーティライブラリ
import cv2
import numpy as np


# ============================================================================
# カスタム例外クラス
# ============================================================================

class FrameReaderError(Exception):
    """フレーム読み込み関連のベース例外"""
    pass


class VideoFileError(FrameReaderError):
    """ビデオファイル関連のエラー"""
    pass


class FrameReadError(FrameReaderError):
    """フレーム読み込みエラー"""
    pass


class ImageDirectoryError(FrameReaderError):
    """画像ディレクトリ関連のエラー"""
    pass


# ============================================================================
# LazyFrameLoader: メモリ効率的なビデオフレーム読み込み
# ============================================================================

class LazyFrameLoader:
    """メモリ効率的なビデオフレーム読み込みローダー

    フレームを1つのみメモリに保持し、ストリーミング処理を実現します。
    本番プログラム（main_twopass.py）のメモリ効率設計に準拠。

    特徴:
        - コンテキストマネージャで安全なリソース管理
        - イテレータパターンでメモリ使用量を最小化（最大1フレーム）
        - __getitem__でランダムアクセス対応
        - __len__でフレーム数取得対応
        - load_selected_frames()で指定インデックスのフレームのみ読み込み

    Example:
        # ストリーミング処理
        with LazyFrameLoader(video_path, start_frame=0, end_frame=100) as loader:
            for frame_idx, frame in loader:
                process_frame(frame)

        # ランダムアクセス
        with LazyFrameLoader(video_path) as loader:
            frame_50 = loader[50]
            frame_100 = loader[100]

        # 選択的読み込み
        with LazyFrameLoader(video_path) as loader:
            selected_frames = loader.load_selected_frames([10, 20, 30, 40, 50])

    Attributes:
        video_path: ビデオファイルパス
        start_frame: 開始フレーム番号
        end_frame: 終了フレーム番号（Noneの場合は最後まで）
        cap: cv2.VideoCapture オブジェクト
        logger: ロガー
        total_frames: ビデオ総フレーム数
    """

    def __init__(
        self,
        video_path: Union[Path, str],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        """初期化

        Args:
            video_path: ビデオファイルパス
            start_frame: 開始フレーム番号（デフォルト: 0）
            end_frame: 終了フレーム番号（Noneの場合は最後まで）
            logger: ロガー（オプション）

        Raises:
            ValueError: start_frameが負の値の場合
        """
        self.video_path = Path(video_path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.cap: Optional[cv2.VideoCapture] = None
        self.logger = logger or logging.getLogger(__name__)
        self.total_frames: Optional[int] = None

        if self.start_frame < 0:
            raise ValueError(f"start_frame must be non-negative: {start_frame}")

    def __enter__(self) -> "LazyFrameLoader":
        """コンテキストマネージャ: 開始時

        Returns:
            self: LazyFrameLoaderインスタンス

        Raises:
            VideoFileError: ビデオファイルを開けない場合
            ValueError: start_frameが総フレーム数以上の場合
        """
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise VideoFileError(
                    f"Cannot open video file: {self.video_path}"
                )

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.end_frame = self.end_frame or self.total_frames

            if self.start_frame >= self.total_frames:
                raise ValueError(
                    f"start_frame ({self.start_frame}) >= "
                    f"total_frames ({self.total_frames})"
                )

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            self.logger.info(
                f"LazyFrameLoader opened: {self.video_path.name}, "
                f"frames {self.start_frame}-{self.end_frame}, "
                f"total_frames={self.total_frames}"
            )

            return self

        except Exception as e:
            if self.cap:
                self.cap.release()
            raise

    def __exit__(self, *args) -> None:
        """コンテキストマネージャ: 終了時"""
        if self.cap:
            self.cap.release()
            self.logger.debug("LazyFrameLoader closed")

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """イテレータ実装: フレーム番号とフレーム画像を返す

        Yields:
            (frame_idx, frame): フレーム番号とフレーム画像

        Raises:
            RuntimeError: コンテキストマネージャ外で呼び出された場合
        """
        if self.cap is None:
            raise RuntimeError(
                "LazyFrameLoader not opened (use 'with' statement)"
            )

        for frame_num in range(self.start_frame, self.end_frame):
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning(
                    f"Failed to read frame {frame_num}, stopping iteration"
                )
                break

            yield frame_num, frame

    def __len__(self) -> int:
        """フレーム数を取得

        Returns:
            処理対象フレーム数

        Raises:
            RuntimeError: コンテキストマネージャ外で呼び出された場合
        """
        if self.cap is None or self.end_frame is None:
            raise RuntimeError(
                "LazyFrameLoader not opened (use 'with' statement)"
            )
        return self.end_frame - self.start_frame

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """ランダムアクセス: 指定フレームを読み込み

        Args:
            frame_idx: 読み込むフレーム番号（絶対位置）

        Returns:
            フレーム画像

        Raises:
            RuntimeError: コンテキストマネージャ外で呼び出された場合
            IndexError: フレーム番号が範囲外の場合
            FrameReadError: フレーム読み込みに失敗した場合
        """
        if self.cap is None or self.total_frames is None:
            raise RuntimeError(
                "LazyFrameLoader not opened (use 'with' statement)"
            )

        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {self.total_frames})"
            )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise FrameReadError(f"Failed to read frame {frame_idx}")

        return frame

    def load_selected_frames(
        self,
        frame_indices: List[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """指定インデックスのフレームのみを読み込み

        メモリ効率的な選択的フレーム読み込みを実現します。
        品質フィルタリング後のフレーム読み込み等に使用。

        Args:
            frame_indices: 読み込むフレーム番号のリスト（絶対位置）

        Returns:
            (frames, indices): 読み込んだフレームのリストと対応するインデックス

        Raises:
            RuntimeError: コンテキストマネージャ外で呼び出された場合
            FrameReadError: フレーム読み込みに失敗した場合
        """
        if self.cap is None:
            raise RuntimeError(
                "LazyFrameLoader not opened (use 'with' statement)"
            )

        frames = []
        successful_indices = []

        for frame_idx in sorted(frame_indices):
            try:
                frame = self[frame_idx]
                frames.append(frame)
                successful_indices.append(frame_idx)
            except (IndexError, FrameReadError) as e:
                self.logger.warning(
                    f"Skipping frame {frame_idx}: {e}"
                )

        self.logger.info(
            f"Loaded {len(frames)}/{len(frame_indices)} selected frames"
        )

        return frames, successful_indices

    def get_total_frames(self) -> int:
        """ビデオ総フレーム数を取得

        Returns:
            総フレーム数

        Raises:
            RuntimeError: コンテキストマネージャ外で呼び出された場合
        """
        if self.cap is None:
            raise RuntimeError(
                "LazyFrameLoader not opened (use 'with' statement)"
            )
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self) -> float:
        """フレームレートを取得

        Returns:
            フレームレート（fps）

        Raises:
            RuntimeError: コンテキストマネージャ外で呼び出された場合
        """
        if self.cap is None:
            raise RuntimeError(
                "LazyFrameLoader not opened (use 'with' statement)"
            )
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self) -> Tuple[int, int]:
        """ビデオ解像度を取得

        Returns:
            (width, height): ビデオ解像度

        Raises:
            RuntimeError: コンテキストマネージャ外で呼び出された場合
        """
        if self.cap is None:
            raise RuntimeError(
                "LazyFrameLoader not opened (use 'with' statement)"
            )
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height


# ============================================================================
# ImageDirectoryLoader: 画像ディレクトリ用遅延読み込み
# ============================================================================

class ImageDirectoryLoader:
    """画像ディレクトリ用の遅延読み込みローダー

    メモリ効率的な画像読み込みを実現します。
    全画像を事前に読み込まず、必要な時だけ読み込みます。

    特徴:
        - イテレータパターンでメモリ使用量を最小化（最大1フレーム）
        - __getitem__でランダムアクセス対応
        - __len__でフレーム数取得対応
        - load_selected_frames()で指定インデックスのフレームのみ読み込み

    Example:
        # イテレータパターン
        loader = ImageDirectoryLoader(frames_dir, logger, pattern="frame_*.png")
        for frame_idx, frame in loader:
            process_frame(frame)

        # ランダムアクセス
        frame = loader[10]

        # 選択的読み込み
        selected_frames, indices = loader.load_selected_frames([0, 10, 20])

    Attributes:
        frames_dir: 画像ディレクトリパス
        logger: ロガー
        pattern: ファイル名パターン（デフォルト: "frame_*.png"）
        frame_files: 画像ファイルパスのリスト（ソート済み）
    """

    def __init__(
        self,
        frames_dir: Union[Path, str],
        logger: Optional[logging.Logger] = None,
        pattern: str = "frame_*.png"
    ):
        """初期化

        Args:
            frames_dir: 画像ディレクトリパス（Path or str）
            logger: ロガー（オプション）
            pattern: ファイル名パターン（デフォルト: "frame_*.png"）

        Raises:
            ImageDirectoryError: ディレクトリが存在しない、またはフレームが見つからない場合
        """
        self.frames_dir = Path(frames_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.pattern = pattern

        # ディレクトリ存在チェック
        if not self.frames_dir.exists():
            raise ImageDirectoryError(
                f"Directory does not exist: {self.frames_dir}"
            )

        if not self.frames_dir.is_dir():
            raise ImageDirectoryError(
                f"Path is not a directory: {self.frames_dir}"
            )

        # フレームファイルリストを取得（ソート済み）
        self.frame_files = sorted(self.frames_dir.glob(pattern))

        if not self.frame_files:
            raise ImageDirectoryError(
                f"No frames found in {self.frames_dir} with pattern '{pattern}'"
            )

        self.logger.info(
            f"ImageDirectoryLoader initialized: {self.frames_dir}, "
            f"pattern='{pattern}', frames={len(self.frame_files)}"
        )

    def __len__(self) -> int:
        """フレーム数を取得

        Returns:
            フレーム数
        """
        return len(self.frame_files)

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """イテレータ実装: インデックスとフレーム画像を返す

        Yields:
            (index, frame): インデックスとフレーム画像
        """
        for i, frame_file in enumerate(self.frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                self.logger.warning(
                    f"Failed to read frame {i}: {frame_file}"
                )
                continue
            yield i, frame

    def __getitem__(self, index: int) -> np.ndarray:
        """ランダムアクセス: 指定インデックスのフレームを読み込み

        Args:
            index: フレームインデックス

        Returns:
            フレーム画像

        Raises:
            IndexError: インデックスが範囲外の場合
            FrameReadError: フレーム読み込みに失敗した場合
        """
        if index < 0 or index >= len(self.frame_files):
            raise IndexError(
                f"Index {index} out of range [0, {len(self.frame_files)})"
            )

        frame = cv2.imread(str(self.frame_files[index]))
        if frame is None:
            raise FrameReadError(
                f"Failed to read frame {index}: {self.frame_files[index]}"
            )

        return frame

    def load_selected_frames(
        self,
        indices: List[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """指定インデックスのフレームのみを読み込み

        メモリ効率的な選択的フレーム読み込みを実現します。
        品質フィルタリング後のフレーム読み込み等に使用。

        Args:
            indices: 読み込むフレームインデックスのリスト

        Returns:
            (frames, successful_indices): 読み込んだフレームのリストと成功したインデックス
        """
        frames = []
        successful_indices = []

        for index in sorted(indices):
            try:
                frame = self[index]
                frames.append(frame)
                successful_indices.append(index)
            except (IndexError, FrameReadError) as e:
                self.logger.warning(
                    f"Skipping frame {index}: {e}"
                )

        self.logger.info(
            f"Loaded {len(frames)}/{len(indices)} selected frames"
        )

        return frames, successful_indices

    def get_frame_path(self, index: int) -> Path:
        """指定インデックスのフレームファイルパスを取得

        Args:
            index: フレームインデックス

        Returns:
            フレームファイルパス

        Raises:
            IndexError: インデックスが範囲外の場合
        """
        if index < 0 or index >= len(self.frame_files):
            raise IndexError(
                f"Index {index} out of range [0, {len(self.frame_files)})"
            )

        return self.frame_files[index]

    def get_original_frame_number(self, index: int) -> int:
        """インデックスから元動画のフレーム番号を取得
        
        Args:
            index: ImageDirectoryLoaderのインデックス（0, 1, 2, ...）
        
        Returns:
            元動画のフレーム番号（frame_NNNNNN.pngのNNNNNN部分）
            パターンにマッチしない場合はインデックスをそのまま返す
        
        Raises:
            IndexError: インデックスが範囲外の場合
        
        Example:
            >>> loader = ImageDirectoryLoader("frames/", pattern="frame_*.png")
            >>> loader.get_original_frame_number(0)  # frame_000080.png の場合
            80
        """
        import re
        
        if index < 0 or index >= len(self.frame_files):
            raise IndexError(
                f"Index {index} out of range [0, {len(self.frame_files)})"
            )
        
        frame_file = self.frame_files[index]
        
        # frame_NNNNNN.png から NNNNNN を抽出
        match = re.match(r'frame_(\d+)\.png', frame_file.name)
        if match:
            return int(match.group(1))
        else:
            # フォールバック: インデックスを返す
            self.logger.debug(
                f"Frame file {frame_file.name} does not match 'frame_NNNNNN.png' pattern, "
                f"using index {index} as frame number"
            )
            return index



# ============================================================================
# BatchFrameProcessor: バッチ処理向けフレームプロセッサ
# ============================================================================

class BatchFrameProcessor:
    """バッチ処理向けフレームプロセッサ

    複数フレームをバッチで処理することで、
    キャリブレーション等の複数パス処理を効率化します。

    特徴:
        - メモリ効率を保ちながらバッチ処理を実現
        - 処理関数をカスタマイズ可能
        - 進捗ログ出力

    Example:
        def evaluate_batch(batch):
            return [evaluate_quality(idx, frame)
                    for idx, frame in batch]

        processor = BatchFrameProcessor(evaluate_batch, batch_size=32)
        results = processor.process_video(video_path)

    Attributes:
        process_func: バッチ処理関数
        batch_size: バッチサイズ
        logger: ロガー
    """

    def __init__(
        self,
        process_func: Callable[[List[Tuple[int, np.ndarray]]], List[Any]],
        batch_size: int = 32,
        logger: Optional[logging.Logger] = None
    ):
        """初期化

        Args:
            process_func: バッチを処理する関数
                          入力: List[Tuple[int, np.ndarray]] (フレーム番号, フレーム)
                          出力: List[Any] (処理結果)
            batch_size: バッチサイズ（デフォルト: 32）
            logger: ロガー（オプション）

        Raises:
            ValueError: batch_sizeが1未満の場合
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1: {batch_size}")

    def process_video(
        self,
        video_path: Union[Path, str],
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[Any]:
        """ビデオをバッチ処理

        Args:
            video_path: ビデオファイルパス
            start_frame: 開始フレーム番号
            end_frame: 終了フレーム番号（Noneの場合は最後まで）

        Returns:
            処理結果のリスト

        Raises:
            VideoFileError: ビデオファイルを開けない場合
        """
        results = []
        batch = []

        with LazyFrameLoader(
            video_path, start_frame, end_frame, self.logger
        ) as loader:
            for frame_idx, frame in loader:
                batch.append((frame_idx, frame))

                if len(batch) >= self.batch_size:
                    # バッチ処理実行
                    batch_result = self.process_func(batch)
                    results.extend(batch_result)
                    batch = []

                    # 進捗ログ（100フレームごと）
                    if len(results) % 100 == 0:
                        self.logger.info(f"Processed {len(results)} frames")

        # 残りのバッチを処理
        if batch:
            batch_result = self.process_func(batch)
            results.extend(batch_result)

        self.logger.info(f"Batch processing complete: {len(results)} frames")

        return results

    def process_frames(
        self,
        frames: List[Tuple[int, np.ndarray]]
    ) -> List[Any]:
        """フレームリストをバッチ処理

        既に読み込まれたフレームリストをバッチ処理します。

        Args:
            frames: フレームリスト（フレーム番号, フレーム）

        Returns:
            処理結果のリスト
        """
        results = []
        batch = []

        for frame_idx, frame in frames:
            batch.append((frame_idx, frame))

            if len(batch) >= self.batch_size:
                batch_result = self.process_func(batch)
                results.extend(batch_result)
                batch = []

        # 残りのバッチを処理
        if batch:
            batch_result = self.process_func(batch)
            results.extend(batch_result)

        self.logger.info(f"Batch processing complete: {len(results)} frames")

        return results


# ============================================================================
# 使用例（ドキュメント用）
# ============================================================================

def _example_streaming_processing():
    """ストリーミング処理の例（メモリ最小化）

    この例では、フレーム毎に処理を行い、メモリ使用量を最小化します。
    本番プログラム（main_twopass.py）のPhase2処理に相当します。
    """
    video_path = Path("data/input/sample.mp4")

    # フレーム毎に処理（メモリ: 1フレームのみ）
    with LazyFrameLoader(video_path) as loader:
        for frame_idx, frame in loader:
            # フレーム処理（例: 品質評価）
            quality_score = _evaluate_quality(frame)
            print(f"Frame {frame_idx}: quality={quality_score:.2f}")


def _example_image_directory_processing():
    """画像ディレクトリ処理の例（メモリ最小化）

    この例では、画像ディレクトリから1フレームずつ読み込み、
    メモリ使用量を最小化します。
    """
    frames_dir = Path("data/calibration/frames")
    logger = logging.getLogger(__name__)

    # フレーム毎に処理（メモリ: 1フレームのみ）
    loader = ImageDirectoryLoader(frames_dir, logger, pattern="frame_*.png")
    for frame_idx, frame in loader:
        # フレーム処理（例: 品質評価）
        quality_score = _evaluate_quality(frame)
        print(f"Frame {frame_idx}: quality={quality_score:.2f}")


def _example_batch_processing():
    """バッチ処理の例（品質評価等）

    この例では、複数フレームをバッチで処理します。
    キャリブレーション等の複数パス処理に使用します。
    """
    video_path = Path("data/input/sample.mp4")

    def evaluate_batch(batch: List[Tuple[int, np.ndarray]]) -> List[dict]:
        """バッチ内の全フレームを評価"""
        results = []
        for frame_idx, frame in batch:
            score = _evaluate_quality(frame)
            results.append({'frame_idx': frame_idx, 'score': score})
        return results

    # バッチ処理実行（メモリ: batch_size フレームのみ）
    processor = BatchFrameProcessor(evaluate_batch, batch_size=32)
    results = processor.process_video(video_path)

    # 結果分析
    scores = [r['score'] for r in results]
    print(f"Mean quality: {np.mean(scores):.2f}")
    print(f"Max quality: {np.max(scores):.2f}")


def _example_selective_loading():
    """選択的フレーム読み込みの例

    この例では、品質フィルタリング後のフレームのみを読み込みます。
    キャリブレーション最適化に使用します。
    """
    video_path = Path("data/input/sample.mp4")

    # Phase 1: 全フレームの品質評価（ストリーミング）
    quality_scores = []
    with LazyFrameLoader(video_path) as loader:
        for frame_idx, frame in loader:
            score = _evaluate_quality(frame)
            quality_scores.append(score)

    # Phase 2: 良質フレームのインデックスを抽出
    quality_threshold = 0.7
    good_indices = [
        i for i, score in enumerate(quality_scores)
        if score >= quality_threshold
    ]

    print(f"良質フレーム数: {len(good_indices)} / {len(quality_scores)}")

    # Phase 3: 良質フレームのみ読み込み
    with LazyFrameLoader(video_path) as loader:
        good_frames, successful_indices = loader.load_selected_frames(good_indices)

    print(f"読み込みフレーム数: {len(good_frames)}")


def _evaluate_quality(frame: np.ndarray) -> float:
    """フレーム品質評価（ダミー実装）

    ブレ検出（Laplacianの分散）でフレーム品質を評価します。

    Args:
        frame: 入力フレーム（BGR）

    Returns:
        品質スコア（0.0-1.0）
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 0-1の範囲に正規化
    normalized_score = min(laplacian_variance / 100.0, 1.0)
    return float(normalized_score)
