"""OCR処理ユーティリティモジュール

管内カメラ映像から距離表示を読み取り、カメラ位置を補正する機能を提供。
"""

# 標準ライブラリ
import logging
import re
from typing import Optional, Tuple, List

# サードパーティライブラリ
import cv2
import numpy as np
import pytesseract


# ロガー設定
logger = logging.getLogger(__name__)


# ==================== カスタム例外 ====================

class OCRError(Exception):
    """OCR関連のベース例外"""
    pass


class OCRTextNotFoundError(OCRError):
    """距離テキストが検出できなかった場合の例外"""
    
    def __init__(
        self,
        ocr_text: str,
        roi_ratio: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Args:
            ocr_text: OCRで読み取られた生テキスト
            roi_ratio: 使用されたROI領域の割合 (top, bottom, left, right)
        """
        self.ocr_text = ocr_text
        self.roi_ratio = roi_ratio
        message = f"No distance found in OCR text: '{ocr_text}'"
        if roi_ratio:
            message += f" (ROI: {roi_ratio})"
        super().__init__(message)


# ==================== 設定クラス ====================

class OCRConfig:
    """OCR処理の設定パラメータ

    Attributes:
        threshold: 2値化の初期閾値
        psm_mode: TesseractのPage Segmentation Mode
        retry_thresholds: リトライ時に試行する閾値リスト
        distance_pattern_decimal: 小数点付き距離表示の正規表現パターン
        distance_pattern_integer: 整数距離表示の正規表現パターン
        preprocessing_enabled: 強化前処理パイプラインの有効/無効
    """

    def __init__(
        self,
        threshold: int = 100,
        psm_mode: int = 6,
        retry_thresholds: Optional[List[int]] = None,
        distance_pattern_decimal: str = r"(\d+\.\d{1,2})m",
        distance_pattern_integer: str = r"(\d{3,5})m",
        preprocessing_enabled: bool = False,
        preprocessing_method: str = "enhanced"
    ):
        self.threshold = threshold
        self.psm_mode = psm_mode
        self.retry_thresholds = retry_thresholds or [100, 120, 80, 150]
        self.distance_pattern_decimal = distance_pattern_decimal
        self.distance_pattern_integer = distance_pattern_integer
        self.preprocessing_enabled = preprocessing_enabled
        self.preprocessing_method = preprocessing_method

    @classmethod
    def _extract_psm_mode(cls, tesseract_config: str) -> int:
        """tesseract_config文字列からPSMモードを抽出する

        Args:
            tesseract_config: Tesseract設定文字列（例: "--psm 7 -c tessedit_char_whitelist=..."）

        Returns:
            PSMモード値（抽出できない場合はデフォルト6）
        """
        match = re.search(r'--psm\s+(\d+)', tesseract_config)
        if match:
            return int(match.group(1))
        return 6  # デフォルト

    @classmethod
    def from_config_ocr(cls, config_ocr) -> 'OCRConfig':
        """config.py側のOCRConfig dataclassからocr_utils側のOCRConfigを生成する

        config.py::OCRConfig（設定用dataclass）の値をocr_utils::OCRConfig（実行用）に
        変換する橋渡しメソッド。これによりconfig.jsonの設定が実際のOCR処理に反映される。

        Args:
            config_ocr: src.config.OCRConfig dataclassインスタンス

        Returns:
            ocr_utils.OCRConfig インスタンス
        """
        psm_mode = cls._extract_psm_mode(config_ocr.tesseract_config)
        preprocessing_enabled = getattr(config_ocr, 'preprocessing_enabled', False)
        preprocessing_method = getattr(config_ocr, 'preprocessing_method', 'enhanced')
        retry_thresholds = getattr(config_ocr, 'retry_thresholds', None)

        return cls(
            psm_mode=psm_mode,
            preprocessing_enabled=preprocessing_enabled,
            preprocessing_method=preprocessing_method,
            retry_thresholds=retry_thresholds,
        )


# ==================== 内部関数 ====================

def _extract_roi(
    frame: np.ndarray,
    roi_ratio: Tuple[float, float, float, float]
) -> np.ndarray:
    """フレームからROI領域を切り出す
    
    Args:
        frame: 入力フレーム画像（BGR）
        roi_ratio: ROI領域の割合 (top, bottom, left, right), 各値は0.0〜1.0
        
    Returns:
        切り出されたROI画像
        
    Raises:
        ValueError: roi_ratioの値が不正な場合
    """
    validate_roi_ratio(roi_ratio)
    
    height, width = frame.shape[:2]
    top, bottom, left, right = roi_ratio
    
    y1 = int(height * top)
    y2 = int(height * bottom)
    x1 = int(width * left)
    x2 = int(width * right)
    
    return frame[y1:y2, x1:x2]


# 前処理パイプライン定数
OCR_TARGET_HEIGHT = 200  # 動的拡大の目標ROI高さ（ピクセル）
OCR_MIN_ROI_HEIGHT = 60  # 従来方式の最小ROI高さ
OCR_UPSCALE_FACTOR = 3   # 従来方式の固定拡大倍率


def _preprocess_for_ocr(
    gray: np.ndarray,
    method: str = "enhanced",
    threshold: int = 100
) -> np.ndarray:
    """OCR用の前処理パイプラインを適用する

    Args:
        gray: グレースケール画像
        method: 前処理方式
            - "enhanced": 強化パイプライン（CLAHE + 適応的2値化 + モルフォロジー）
            - "fixed": 従来の固定閾値方式（後方互換）
            - "otsu": 大津の2値化方式
        threshold: 固定閾値方式で使用する閾値（method="fixed"のときのみ使用）

    Returns:
        前処理済みの2値化画像
    """
    # 動的アップスケール: 目標200pxまでLANCZOS4で拡大
    height = gray.shape[0]
    scale_factor = max(1.0, OCR_TARGET_HEIGHT / height)
    if scale_factor > 1.0:
        gray = cv2.resize(
            gray, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv2.INTER_LANCZOS4
        )

    if method == "enhanced":
        # CLAHE: 照明変化に対する適応的コントラスト強化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # ガウシアンブラー: ノイズ除去（文字を潰さない最小カーネル）
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 適応的2値化: 照明のムラに対応
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2
        )

        # モルフォロジー（クロージング）: ノイズ穴埋め
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    elif method == "otsu":
        # ガウシアンブラー: ノイズ除去
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 大津の2値化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    else:
        # "fixed": 従来の固定閾値方式
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        return binary


def _extract_with_threshold(
    frame: np.ndarray,
    roi_ratio: Tuple[float, float, float, float],
    threshold: int,
    psm_mode: int,
    preprocessing_method: str = "fixed"
) -> str:
    """指定された閾値・前処理方式でOCRを実行

    Args:
        frame: 入力フレーム画像（BGR）
        roi_ratio: ROI領域の割合
        threshold: 2値化の閾値（method="fixed"のときのみ使用）
        psm_mode: TesseractのPage Segmentation Mode
        preprocessing_method: 前処理方式（"enhanced", "fixed", "otsu"）

    Returns:
        OCRで読み取られたテキスト（小文字、スペース除去済み）
    """
    # ROI切り出し
    roi = _extract_roi(frame, roi_ratio)

    # 前処理：グレースケール化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if preprocessing_method in ("enhanced", "otsu"):
        # 新しい前処理パイプライン
        thresh = _preprocess_for_ocr(gray, method=preprocessing_method, threshold=threshold)
    else:
        # 従来方式: 固定拡大 + 固定閾値
        if gray.shape[0] < OCR_MIN_ROI_HEIGHT:
            gray = cv2.resize(
                gray, None,
                fx=OCR_UPSCALE_FACTOR, fy=OCR_UPSCALE_FACTOR,
                interpolation=cv2.INTER_CUBIC
            )
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # OCR実行（文字候補を0-9, ., mに制限）
    config = f'--psm {psm_mode} -c tessedit_char_whitelist=0123456789.m'
    text = pytesseract.image_to_string(thresh, config=config)

    # 正規化（スペース除去、小文字化）
    text_normalized = text.replace(" ", "").lower()

    return text_normalized


def _parse_distance_text(
    text: str,
    pattern_decimal: str,
    pattern_integer: str
) -> float:
    """OCRテキストから距離を抽出・解析
    
    Args:
        text: OCRで読み取られたテキスト
        pattern_decimal: 小数点付きパターン（例: "4.10m"）
        pattern_integer: 整数パターン（例: "410m"）
        
    Returns:
        距離（mm単位）
        
    Raises:
        OCRTextNotFoundError: 距離が検出できなかった場合
    """
    # パターン1: 小数点あり（例: "4.10m", "12.34m"）
    match_decimal = re.search(pattern_decimal, text)
    if match_decimal:
        distance_m = float(match_decimal.group(1))
        logger.debug(f"Decimal distance found: {distance_m}m")
        return distance_m * 1000  # mm変換
    
    # パターン2: ピリオド欠落（3〜5桁、例: "410m" → "4.10m"）
    match_integer = re.search(pattern_integer, text)
    if match_integer:
        raw_digits = match_integer.group(1)
        if len(raw_digits) >= 3:
            # 下2桁を小数部として補完
            distance_str = raw_digits[:-2] + "." + raw_digits[-2:]
            try:
                distance_m = float(distance_str)
                logger.debug(
                    f"Integer distance found and converted: "
                    f"{raw_digits}m → {distance_m}m"
                )
                return distance_m * 1000  # mm変換
            except ValueError as e:
                logger.error(
                    f"Failed to convert fallback format: "
                    f"{raw_digits} → {distance_str}"
                )
                raise OCRTextNotFoundError(text) from e
    
    # どちらのパターンにも一致しない
    raise OCRTextNotFoundError(text)


# マルチ戦略の戦略定義（preprocessing_methodに応じて切り替え）
MULTI_STRATEGY_PRESETS = {
    "enhanced": [
        ("enhanced", None),
        ("fixed", 100),
        ("fixed", 150),
        ("otsu", None),
    ],
    "fixed": [
        ("fixed", 100),
        ("fixed", 120),
        ("fixed", 150),
    ],
    "otsu": [
        ("otsu", None),
        ("fixed", 100),
        ("fixed", 150),
    ],
}


def _get_strategy_list(preprocessing_method: str) -> List[Tuple[str, Optional[int]]]:
    """preprocessing_methodに対応する戦略リストを取得する"""
    return MULTI_STRATEGY_PRESETS.get(preprocessing_method, MULTI_STRATEGY_PRESETS["enhanced"])


def _extract_with_multi_strategy(
    frame: np.ndarray,
    roi_ratio: Tuple[float, float, float, float],
    psm_mode: int,
    pattern_decimal: str,
    pattern_integer: str,
    preprocessing_method: str = "enhanced"
) -> float:
    """複数の前処理戦略で逐次OCRを実行し、合意に基づいて結果を返す

    同一フレームに対して異なる前処理方式でOCRを実行し、
    2つ以上の戦略が同じ結果を返した場合に高信頼度で採用する。

    メモリ安全性:
    - 各戦略は同一フレーム（既にメモリ上にある1枚）のROIに対して
      異なる前処理を適用するだけ。フレームの追加読み込みは発生しない。
    - ROI画像（数十KB）の一時コピーのみで、追加メモリは無視できるレベル。

    Args:
        frame: 入力フレーム画像（BGR）
        roi_ratio: ROI領域の割合
        psm_mode: TesseractのPage Segmentation Mode
        pattern_decimal: 小数点付き距離パターン
        pattern_integer: 整数距離パターン
        preprocessing_method: 戦略プリセットの選択
            - "enhanced": enhanced + fixed(100,150) + otsu の4戦略
            - "fixed": fixed(100,120,150) の3戦略
            - "otsu": otsu + fixed(100,150) の3戦略

    Returns:
        距離（mm単位）

    Raises:
        OCRTextNotFoundError: すべての戦略で距離が検出できなかった場合
    """
    strategies = _get_strategy_list(preprocessing_method)
    results = []  # (distance_mm, strategy_name)

    for method, threshold in strategies:
        try:
            text = _extract_with_threshold(
                frame, roi_ratio,
                threshold=threshold if threshold is not None else 100,
                psm_mode=psm_mode,
                preprocessing_method=method
            )
            distance_mm = _parse_distance_text(text, pattern_decimal, pattern_integer)
            strategy_name = f"{method}({threshold})" if threshold is not None else method
            results.append((distance_mm, strategy_name))
            logger.debug(f"Multi-strategy OCR: {strategy_name} → {distance_mm}mm")

            # 早期打切り: 最初の2戦略が一致したら残りをスキップ
            if len(results) >= 2:
                distances = [r[0] for r in results]
                # 最新の結果と過去の結果で一致するものがあるか
                for prev_dist in distances[:-1]:
                    if abs(prev_dist - distances[-1]) < 0.1:
                        logger.info(
                            f"Multi-strategy consensus reached: {distances[-1]}mm "
                            f"(agreed by {results[-1][1]} and earlier strategy)"
                        )
                        return distances[-1]

        except OCRTextNotFoundError:
            strategy_name = f"{method}({threshold})" if threshold is not None else method
            logger.debug(f"Multi-strategy OCR: {strategy_name} → failed")
            continue

    # 全戦略終了後の処理
    if len(results) == 0:
        raise OCRTextNotFoundError("All multi-strategy attempts failed")

    # 合意判定: 最頻値を採用
    from collections import Counter
    distance_counts = Counter()
    for dist, _ in results:
        # 浮動小数点の丸め誤差を考慮
        rounded = round(dist, 1)
        distance_counts[rounded] += 1

    most_common_dist, count = distance_counts.most_common(1)[0]
    if count >= 2:
        logger.info(
            f"Multi-strategy consensus (post-scan): {most_common_dist}mm ({count} strategies agree)"
        )
        return most_common_dist

    # 1つのみ成功 → 低信頼度で採用
    logger.warning(
        f"Multi-strategy: no consensus, using single result {results[0][0]}mm "
        f"from {results[0][1]}"
    )
    return results[0][0]


# ==================== 公開API ====================

def validate_roi_ratio(roi_ratio: Tuple[float, float, float, float]) -> None:
    """ROI割合の妥当性をチェック
    
    Args:
        roi_ratio: (top, bottom, left, right)の割合、各値は0.0〜1.0
        
    Raises:
        ValueError: 値が範囲外、または top >= bottom, left >= right の場合
    """
    top, bottom, left, right = roi_ratio
    
    if not (0.0 <= top < bottom <= 1.0):
        raise ValueError(
            f"Invalid ROI vertical range: top={top}, bottom={bottom}. "
            f"Expected: 0.0 <= top < bottom <= 1.0"
        )
    
    if not (0.0 <= left < right <= 1.0):
        raise ValueError(
            f"Invalid ROI horizontal range: left={left}, right={right}. "
            f"Expected: 0.0 <= left < right <= 1.0"
        )


def extract_distance_from_frame(
    frame: np.ndarray,
    roi_ratio: Tuple[float, float, float, float],
    config: Optional[OCRConfig] = None,
    expected_range_mm: Optional[Tuple[float, float]] = None,
    preferred_threshold: Optional[int] = None
) -> Tuple[float, Optional[int]]:
    """フレーム画像から距離表示を読み取る（検証リトライ対応）

    "xx.xx m" または "xxxx m" 形式の距離を読み取る。
    ピリオド欠落（例: "410m" → "4.10m"）にも対応。

    preprocessing_enabled=True の場合:
        マルチ戦略方式を使用。複数の前処理方式でOCRを実行し、
        合意に基づいて結果を返す（誤読検出機能あり）。

    preprocessing_enabled=False の場合:
        前回成功した閾値（preferred_threshold）を優先して試行。
        expected_range_mmが指定されている場合、結果が範囲外なら
        別の閾値でリトライし、範囲内の結果を返す。
        すべての閾値で範囲外または解析失敗の場合はエラーを送出。

    Args:
        frame: 入力フレーム画像（OpenCVで取得したBGR形式）
        roi_ratio: ROI領域の割合 (top, bottom, left, right), 各値は0.0〜1.0
        config: OCR設定パラメータ（Noneの場合はデフォルト設定を使用）
        expected_range_mm: 期待距離範囲 (min_mm, max_mm)（Noneの場合は範囲チェックなし）
        preferred_threshold: 前回成功した閾値（優先的に試行）

    Returns:
        (距離(mm単位), 成功した閾値) のタプル。
        マルチ戦略方式の場合、閾値はNone。

    Raises:
        OCRTextNotFoundError: すべてのリトライで距離が検出できなかった場合
        ValueError: ROI設定が不正な場合

    Examples:
        >>> frame = cv2.imread("test.jpg")
        >>> distance_mm, thresh = extract_distance_from_frame(
        ...     frame,
        ...     roi_ratio=(0.0, 0.05, 0.0, 0.2)
        ... )
        >>> print(f"Distance: {distance_mm}mm (threshold={thresh})")
    """
    if config is None:
        config = OCRConfig()

    validate_roi_ratio(roi_ratio)

    # マルチ戦略方式（preprocessing_enabled=True）
    if config.preprocessing_enabled:
        return _extract_with_multi_strategy(
            frame, roi_ratio, config.psm_mode,
            config.distance_pattern_decimal,
            config.distance_pattern_integer,
            preprocessing_method=config.preprocessing_method
        ), None

    # 従来方式（preprocessing_enabled=False）
    # 閾値リストを構築: preferred_thresholdを先頭に配置
    thresholds = list(config.retry_thresholds)
    if preferred_threshold is not None and preferred_threshold in thresholds:
        thresholds.remove(preferred_threshold)
        thresholds.insert(0, preferred_threshold)

    last_error = None
    first_parsed_result = None
    first_parsed_threshold = None

    for attempt, threshold in enumerate(thresholds, 1):
        try:
            logger.debug(
                f"OCR attempt {attempt}/{len(thresholds)} "
                f"with threshold={threshold}"
            )

            # OCR実行
            text = _extract_with_threshold(
                frame, roi_ratio, threshold, config.psm_mode
            )

            # 距離解析
            distance_mm = _parse_distance_text(
                text,
                config.distance_pattern_decimal,
                config.distance_pattern_integer
            )

            # 最初の解析成功結果を記録
            if first_parsed_result is None:
                first_parsed_result = distance_mm
                first_parsed_threshold = threshold

            # 範囲チェック
            if expected_range_mm is not None:
                min_mm, max_mm = expected_range_mm
                if not (min_mm <= distance_mm <= max_mm):
                    logger.debug(
                        f"OCR result {distance_mm}mm outside expected range "
                        f"[{min_mm:.0f}, {max_mm:.0f}]mm (threshold={threshold}), "
                        f"trying next threshold"
                    )
                    continue

            logger.info(
                f"Distance extracted successfully: {distance_mm}mm "
                f"(threshold={threshold}, attempt={attempt})"
            )
            return distance_mm, threshold

        except OCRTextNotFoundError as e:
            last_error = e
            logger.warning(
                f"OCR attempt {attempt} failed with threshold={threshold}: "
                f"text='{e.ocr_text}'"
            )
            continue

    # すべてのリトライが失敗
    if first_parsed_result is not None:
        # 解析はできたが全て範囲外 → 読み取り失敗
        logger.warning(
            f"All OCR results out of expected range "
            f"[{expected_range_mm[0]:.0f}, {expected_range_mm[1]:.0f}]mm. "
            f"First parsed: {first_parsed_result}mm (threshold={first_parsed_threshold})"
        )
        raise OCRTextNotFoundError(
            f"out_of_range:{first_parsed_result}mm"
        )

    logger.error(
        f"All OCR attempts failed. Last error: {last_error}"
    )
    if last_error:
        raise last_error
    else:
        raise OCRTextNotFoundError("Unknown error", roi_ratio)


def correct_camera_position_single_frame(
    curr_frame: np.ndarray,
    z1: float,
    roi_ratio: Tuple[float, float, float, float],
    z_prev_corrected: Optional[float] = None,
    prev_z2: Optional[float] = None,
    config: Optional[OCRConfig] = None
) -> Tuple[float, float]:
    """単一フレームに対してカメラ位置を補正する
    
    OCRで読み取った距離表示と解析上のカメラ位置を比較し、
    10mm単位の計測誤差を考慮して補正後のカメラ位置を算出する。
    
    補正ロジック:
    - 初回フレーム: OCR値をそのまま採用
    - 10mm誤差あり & 表示更新あり: OCR値を採用（推定範囲の最小値）
    - 10mm誤差あり & 表示未更新: 直前値と次の更新距離の中間値
    - 10mm誤差なし: 解析値を採用
    
    Args:
        curr_frame: 現在のビデオフレーム（OpenCV形式、BGR）
        z1: 現在フレームの解析上のカメラ位置（mm）
        roi_ratio: OCR用ROI領域の割合 (top, bottom, left, right)
        z_prev_corrected: 前フレームの補正後カメラ位置 z**（mm）
        prev_z2: 前フレームのOCR読み取り距離（mm）
        config: OCR設定パラメータ
        
    Returns:
        Tuple[float, float]:
            - z_star: 現在フレームの補正後カメラ位置（mm）
            - z2: OCR読み取り距離（mm、次フレームへ引き継ぎ用）
            
    Raises:
        ValueError: ROI設定が不正な場合
        
    Examples:
        >>> z_star, z2 = correct_camera_position_single_frame(
        ...     frame, z1=4150.0, roi_ratio=(0, 0.05, 0, 0.2),
        ...     z_prev_corrected=4100.0, prev_z2=4100.0
        ... )
    """
    if config is None:
        config = OCRConfig()
    
    # OCR実行
    try:
        z2, _ = extract_distance_from_frame(curr_frame, roi_ratio, config)
    except OCRTextNotFoundError as e:
        logger.warning(f"OCR failed: {e}. Using previous values.")
        # OCR失敗時は解析値と前フレームのOCR値をそのまま返す
        return z1, prev_z2 if prev_z2 is not None else z1
    
    # 初回フレーム: OCR値を採用
    if z_prev_corrected is None:
        logger.info(f"First frame: z_star={z2}mm (from OCR)")
        return z2, z2
    
    # 10mm単位の計測誤差チェック
    has_10mm_error = (z1 < z2) or (z1 >= z2 + 10)
    display_updated = z_prev_corrected < z2
    
    if has_10mm_error and display_updated:
        # 表示が更新されている → OCR値を採用
        z_star = z2
        logger.debug(
            f"10mm error detected, display updated: "
            f"z_star={z_star}mm (from OCR)"
        )
    elif has_10mm_error and not display_updated:
        # 表示未更新 → 直前値と次の更新距離の中間値
        z_star = z_prev_corrected + ((z2 + 10) - z_prev_corrected) / 2
        logger.debug(
            f"10mm error detected, display not updated: "
            f"z_star={z_star}mm (interpolated)"
        )
    else:
        # 誤差なし → 解析値を採用
        z_star = z1
        logger.debug(
            f"No 10mm error: z_star={z_star}mm (from analysis)"
        )
    
    return z_star, z2

# ==================== 高精度Z位置推定機能（フェーズ2タスク1.2） ====================

def collect_ocr_distances(
    frames: List[np.ndarray],
    roi_ratio: Optional[Tuple[float, float, float, float]] = None,
    tesseract_config: Optional[OCRConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """全フレームのOCR距離読み取りを収集
    
    全フレームに対してextract_distance_from_frame()を呼び出し、
    OCR読み取り距離と成功/失敗フラグを収集する。
    読み取り失敗時はNaNを記録し、統計情報をログ出力する。
    
    Args:
        frames: 全フレーム画像のリスト（BGR画像）
        roi_ratio: ROI領域比率 [top, bottom, left, right]
            Noneの場合はデフォルト値 (0.0, 0.05, 0.0, 0.2) を使用
        tesseract_config: Tesseract設定オブジェクト
            Noneの場合はデフォルト設定を使用
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - ocr_distances: OCR読み取り距離配列 (N,) [mm]
                読み取り失敗時はNaNを格納
            - success_flags: 読み取り成功フラグ (N,) bool配列
    
    処理の流れ:
        1. 各フレームでextract_distance_from_frame()を呼び出し
        2. 読み取り成功/失敗を記録
        3. 統計情報をログ出力
    
    Examples:
        >>> frames = [frame1, frame2, frame3]
        >>> ocr_dist, success = collect_ocr_distances(frames)
        >>> print(ocr_dist)  # [450.0, nan, 460.0]
        >>> print(success)   # [True, False, True]
    
    Note:
        - 既存のextract_distance_from_frame()を使用
        - エラーハンドリング: OCRエラー時はNaNを記録
        - ログ出力: 成功率、平均距離、距離範囲
    """
    if roi_ratio is None:
        roi_ratio = (0.0, 0.05, 0.0, 0.2)
    
    if tesseract_config is None:
        tesseract_config = OCRConfig()
    
    n_frames = len(frames)
    ocr_distances = np.full(n_frames, np.nan, dtype=float)
    success_flags = np.zeros(n_frames, dtype=bool)
    
    logger.info(f"Starting OCR distance collection for {n_frames} frames")
    
    for i, frame in enumerate(frames):
        try:
            distance, _ = extract_distance_from_frame(
                frame, roi_ratio, tesseract_config
            )
            ocr_distances[i] = distance
            success_flags[i] = True
            
            if (i + 1) % 100 == 0:
                logger.debug(
                    f"OCR progress: {i+1}/{n_frames} frames processed, "
                    f"success rate: {np.sum(success_flags[:i+1]) / (i+1) * 100:.1f}%"
                )
        
        except OCRTextNotFoundError as e:
            logger.debug(
                f"OCR failed for frame {i}: {e.ocr_text}"
            )
            # ocr_distances[i]は既にNaN
            # success_flags[i]は既にFalse
    
    # 統計情報をログ出力
    success_count = np.sum(success_flags)
    success_rate = success_count / n_frames * 100
    
    logger.info(
        f"OCR collection completed: "
        f"{success_count}/{n_frames} successful ({success_rate:.1f}%)"
    )
    
    if success_count > 0:
        valid_distances = ocr_distances[success_flags]
        logger.info(
            f"Distance statistics: "
            f"min={np.min(valid_distances):.1f}mm, "
            f"max={np.max(valid_distances):.1f}mm, "
            f"mean={np.mean(valid_distances):.1f}mm"
        )
    else:
        logger.warning("No successful OCR readings found!")
    
    return ocr_distances, success_flags


def compute_average_speed(
    ocr_distances: np.ndarray,
    success_flags: np.ndarray
) -> float:
    """カメラ車の平均速度を計算
    
    成功したOCR読み取り値から、最初と最後の値を使用して
    平均速度を計算する。
    
    Args:
        ocr_distances: OCR読み取り距離配列 (N,) [mm]
        success_flags: 読み取り成功フラグ (N,)
    
    Returns:
        average_speed: 平均速度 [mm/frame]
    
    Raises:
        ValueError: 成功したOCR読み取りが2点未満の場合
    
    処理の流れ:
        1. 成功したOCR読み取り値を抽出
        2. 最初と最後の値から総移動距離を計算
        3. フレーム数で割って平均速度を算出
    
    Examples:
        >>> ocr_dist = np.array([450.0, np.nan, 460.0, 470.0])
        >>> success = np.array([True, False, True, True])
        >>> avg_speed = compute_average_speed(ocr_dist, success)
        >>> print(avg_speed)  # (470 - 450) / 3 = 6.67 mm/frame
    
    Note:
        - 成功フラグがTrueの値のみを使用
        - 最初と最後のインデックスを取得
        - total_distance = ocr_distances[last] - ocr_distances[first]
        - n_frames = last_index - first_index
        - average_speed = total_distance / n_frames
    """
    # 成功したOCR読み取りのインデックスを取得
    success_indices = np.where(success_flags)[0]
    
    if len(success_indices) < 2:
        raise ValueError(
            f"Need at least 2 successful OCR readings, got {len(success_indices)}"
        )
    
    # 最初と最後の成功インデックス
    first_idx = success_indices[0]
    last_idx = success_indices[-1]
    
    # 総移動距離
    total_distance = ocr_distances[last_idx] - ocr_distances[first_idx]
    
    # フレーム数
    n_frames = last_idx - first_idx
    
    if n_frames == 0:
        raise ValueError(
            "First and last successful OCR readings are at the same frame"
        )
    
    # 平均速度
    average_speed = total_distance / n_frames
    
    logger.info(
        f"Average speed computed: {average_speed:.3f} mm/frame "
        f"(from frame {first_idx} to {last_idx}, "
        f"total distance: {total_distance:.1f}mm, "
        f"n_frames: {n_frames})"
    )
    
    return average_speed


def estimate_high_precision_z_positions(
    ocr_distances: np.ndarray,
    success_flags: np.ndarray,
    average_speed: float
) -> np.ndarray:
    """平均速度ベースの高精度Z位置推定（簡易等分割法）

    OCR読み取り値から高精度Z位置を推定する。
    同一OCR値が連続する場合、10mm区間を等分割して補間する。

    Args:
        ocr_distances: OCR読み取り距離配列 (N,) [mm]
        success_flags: 読み取り成功フラグ (N,)
        average_speed: 平均速度 [mm/frame]（このメソッドでは使用しない）

    Returns:
        z_positions: 高精度Z位置配列 (N,) [mm]

    処理ロジック:
        同一値Xがn個連続した場合、i番目（0-indexed）の値を X + i * 10 / n とする。

        例: OCR値が [3000, 3000, 3000, 3000] の場合（n=4）
            i=0 → 3000 + 0 * 10 / 4 = 3000.0
            i=1 → 3000 + 1 * 10 / 4 = 3002.5
            i=2 → 3000 + 2 * 10 / 4 = 3005.0
            i=3 → 3000 + 3 * 10 / 4 = 3007.5

    Examples:
        >>> ocr_dist = np.array([3000.0, 3010.0, 3010.0, 3020.0, 3020.0, 3030.0])
        >>> success = np.array([True, True, True, True, True, True])
        >>> avg_speed = 6.0
        >>> z_pos = estimate_high_precision_z_positions(ocr_dist, success, avg_speed)
        >>> print(z_pos)  # [3000.0, 3010.0, 3015.0, 3020.0, 3025.0, 3030.0]

    Note:
        - OCR失敗フレーム(NaN)は前後の値から線形補間
        - 平均速度パラメータは互換性のために残すが使用しない
    """
    n_frames = len(ocr_distances)

    # 最初の成功OCR読み取り値を基準点とする
    success_indices = np.where(success_flags)[0]
    if len(success_indices) == 0:
        raise ValueError("No successful OCR readings found")

    first_idx = success_indices[0]

    # 結果配列を初期化
    z_estimated = np.zeros(n_frames, dtype=float)

    logger.info(
        f"Starting simplified equal-division Z position estimation: "
        f"first_frame={first_idx}, n_frames={n_frames}"
    )

    # OCR成功フレームのみを処理（連続する同一値を等分割）
    i = 0
    interpolated_count = 0

    while i < n_frames:
        if not success_flags[i]:
            # OCR失敗フレームは後で補間
            i += 1
            continue

        # 連続する同一OCR値の範囲を探す
        current_ocr_value = ocr_distances[i]
        j = i + 1

        # 同一値が続く範囲を探索
        while j < n_frames and success_flags[j] and ocr_distances[j] == current_ocr_value:
            j += 1

        # [i, j) の範囲が同一OCR値
        n_consecutive = j - i

        # 等分割計算
        for k in range(n_consecutive):
            z_estimated[i + k] = current_ocr_value + k * 10.0 / n_consecutive

        if n_consecutive > 1:
            interpolated_count += n_consecutive
            logger.debug(
                f"OCR value {current_ocr_value:.1f}mm: {n_consecutive} consecutive frames "
                f"divided equally from {z_estimated[i]:.2f}mm to {z_estimated[j-1]:.2f}mm"
            )
        else:
            z_estimated[i] = current_ocr_value

        i = j

    # OCR失敗フレームを線形補間
    for i in range(n_frames):
        if not success_flags[i]:
            # 前後の成功フレームを探す
            prev_success = success_indices[success_indices < i]
            next_success = success_indices[success_indices > i]

            if len(prev_success) > 0 and len(next_success) > 0:
                # 前後の値から線形補間
                prev_idx = prev_success[-1]
                next_idx = next_success[0]

                weight = (i - prev_idx) / (next_idx - prev_idx)
                z_estimated[i] = z_estimated[prev_idx] * (1 - weight) + z_estimated[next_idx] * weight

            elif len(prev_success) > 0:
                # 前方のみ存在 → 前方の値を使用
                z_estimated[i] = z_estimated[prev_success[-1]]
            elif len(next_success) > 0:
                # 後方のみ存在 → 後方の値を使用
                z_estimated[i] = z_estimated[next_success[0]]

    logger.info(
        f"Simplified Z position estimation completed: "
        f"{interpolated_count}/{n_frames} frames interpolated"
    )

    return z_estimated


def compute_z_ranges(
    z_positions: np.ndarray,
    tolerance: float = 3.0
) -> np.ndarray:
    """Z位置の探索範囲を計算
    
    フレーム間のdz範囲を計算し、次タスク（タスク1.3）での
    カメラ位置推定の探索範囲として利用する。
    
    Args:
        z_positions: 高精度Z位置配列 (N,) [mm]
        tolerance: 許容誤差 [mm]（デフォルト±3mm）
    
    Returns:
        z_ranges: Z位置範囲 (N, 2) [z_min, z_max]
            z_ranges[i] = [dz_min, dz_max]（i番目のフレームのdz範囲）
    
    処理の流れ:
        フレーム間のdz範囲を計算:
        dz_ranges[i] = [
            (z_positions[i] - z_positions[i-1]) - tolerance,
            (z_positions[i] - z_positions[i-1]) + tolerance
        ]
    
    Examples:
        >>> z_pos = np.array([450.0, 456.67, 463.34, 470.0])
        >>> z_ranges = compute_z_ranges(z_pos, tolerance=3.0)
        >>> print(z_ranges[1])  # [3.67, 9.67]（dz = 6.67 ± 3）
    
    Note:
        - 最初のフレーム(i=0)は特別処理（広い範囲）
        - 負の値が発生する場合は0にクリップ
    """
    n_frames = len(z_positions)
    z_ranges = np.zeros((n_frames, 2), dtype=float)
    
    # 最初のフレームは広い範囲を設定
    z_ranges[0] = [0.0, 20.0]  # 0〜20mm/frameの範囲
    
    for i in range(1, n_frames):
        dz = z_positions[i] - z_positions[i-1]
        
        # dz範囲: dz ± tolerance
        dz_min = max(0.0, dz - tolerance)  # 負の値は0にクリップ
        dz_max = dz + tolerance
        
        z_ranges[i] = [dz_min, dz_max]
    
    logger.info(
        f"Z ranges computed: "
        f"mean_dz_min={np.mean(z_ranges[1:, 0]):.3f}mm/frame, "
        f"mean_dz_max={np.mean(z_ranges[1:, 1]):.3f}mm/frame, "
        f"tolerance={tolerance:.1f}mm"
    )

    return z_ranges


# ==================== オフセット移動平均距離推定法（フェーズ2タスク） ====================

def count_consistent_frames(
    ocr_values: np.ndarray,
    ocr_success: np.ndarray,
    j: int,
    j_end: int,
    e_j: float,
    e_j_end: float
) -> int:
    """制約条件を満たすフレーム数をカウント

    各フレームの推定距離がOCR値の10mm範囲内に入るかチェック。

    Args:
        ocr_values: OCR読み取り距離（mm）
        ocr_success: OCR成功フラグ
        j: ウィンドウ開始フレームインデックス
        j_end: ウィンドウ終了フレームインデックス
        e_j: フレームjのオフセット（mm）
        e_j_end: フレームj_endのオフセット（mm）

    Returns:
        制約を満たすフレーム数

    Note:
        制約条件: z_i ≤ z*_i < z_i + 10
        ここで z*_i = z_j + e_j + ave × (i - j)
             ave = (z_{j_end} - z_j) / (j_end - j) + (e_{j_end} - e_j) / (j_end - j)
    """
    # 平均移動距離を計算
    n_frames = j_end - j
    if n_frames == 0:
        return 0

    ave = ((ocr_values[j_end] - ocr_values[j]) / n_frames +
           (e_j_end - e_j) / n_frames)

    # 各フレームの推定距離をチェック
    consistent_count = 0
    for i in range(j, j_end + 1):
        if not ocr_success[i]:
            # OCR失敗フレームはスキップ
            continue

        # 推定距離
        z_star_i = ocr_values[j] + e_j + ave * (i - j)

        # 制約条件チェック: z_i ≤ z*_i < z_i + 10
        if ocr_values[i] <= z_star_i < ocr_values[i] + 10:
            consistent_count += 1

    return consistent_count


def compute_window_estimates(
    ocr_values: np.ndarray,
    j: int,
    j_end: int,
    e_j: float,
    e_j_end: float
) -> np.ndarray:
    """ウィンドウ内の各フレームの推定距離を計算

    Args:
        ocr_values: OCR読み取り距離（mm）
        j: ウィンドウ開始フレームインデックス
        j_end: ウィンドウ終了フレームインデックス
        e_j: フレームjのオフセット（mm）
        e_j_end: フレームj_endのオフセット（mm）

    Returns:
        ウィンドウ内の推定距離、shape (j_end - j + 1,)
    """
    n_frames = j_end - j
    if n_frames == 0:
        return np.array([ocr_values[j] + e_j])

    # 平均移動距離
    ave = ((ocr_values[j_end] - ocr_values[j]) / n_frames +
           (e_j_end - e_j) / n_frames)

    # 各フレームの推定距離
    estimates = np.zeros(j_end - j + 1, dtype=float)
    for i in range(j, j_end + 1):
        estimates[i - j] = ocr_values[j] + e_j + ave * (i - j)

    return estimates


def optimize_offsets(
    ocr_values: np.ndarray,
    ocr_success: np.ndarray,
    j: int,
    j_end: int,
    e_resolution: float = 0.1
) -> Tuple[float, float, int]:
    """ウィンドウ内で最適なオフセットの組み合わせを探索

    グリッドサーチにより、制約条件を満たすフレーム数が最大となる
    (e_j, e_{j_end})の組み合わせを求める。

    Args:
        ocr_values: OCR読み取り距離（mm）
        ocr_success: OCR成功フラグ
        j: ウィンドウ開始フレームインデックス
        j_end: ウィンドウ終了フレームインデックス
        e_resolution: オフセット解像度（mm）

    Returns:
        (best_e_j, best_e_j_end, max_consistent_count)
        - best_e_j: フレームjの最適オフセット（mm）
        - best_e_j_end: フレームj_endの最適オフセット（mm）
        - max_consistent_count: 制約を満たすフレーム数

    Example:
        >>> ocr_values = np.array([700, 700, 710, 710, 720, 720, 730, 730, 730, 730, 740])
        >>> ocr_success = np.ones(11, dtype=bool)
        >>> e_j, e_j_end, count = optimize_offsets(ocr_values, ocr_success, 0, 10)
        >>> # e_j ≈ 2.5, e_j_end ≈ 2.5, count = 11 (全フレーム整合)
    """
    # オフセット候補: -5mm ≤ e < 5mm を0.1mm刻みで探索
    e_candidates = np.arange(-5.0, 5.0, e_resolution)

    best_e_j = 0.0
    best_e_j_end = 0.0
    max_consistent_count = 0

    logger.info(
        f"    Optimizing offsets for window [{j}, {j_end}]: "
        f"{len(e_candidates)}x{len(e_candidates)}={len(e_candidates)**2} combinations"
    )

    # グリッドサーチ
    for e_j in e_candidates:
        for e_j_end in e_candidates:
            count = count_consistent_frames(
                ocr_values, ocr_success, j, j_end, e_j, e_j_end
            )

            if count > max_consistent_count:
                max_consistent_count = count
                best_e_j = e_j
                best_e_j_end = e_j_end

    logger.debug(
        f"Optimized offsets for window [{j}, {j_end}]: "
        f"e_j={best_e_j:.1f}mm, e_j_end={best_e_j_end:.1f}mm, "
        f"consistent_count={max_consistent_count}/{j_end - j + 1}"
    )

    return best_e_j, best_e_j_end, max_consistent_count


def average_overlapping_estimates(
    estimations: List[Tuple[int, int, np.ndarray]],
    n_frames: int,
    estimation_counts: np.ndarray
) -> np.ndarray:
    """オーバーラップする推定値を平均化

    Args:
        estimations: [(j, j_end, window_estimates), ...]のリスト
        n_frames: 総フレーム数
        estimation_counts: 各フレームの推定回数、shape (n_frames,)

    Returns:
        全フレームの最終推定距離、shape (n_frames,)
    """
    final_estimates = np.zeros(n_frames, dtype=float)

    # 各ウィンドウの推定値を累積
    for j, j_end, window_estimates in estimations:
        # 範囲チェック: j_endがn_framesを超えないようにする
        if j_end >= n_frames:
            logger.warning(
                f"Window end index out of bounds: j_end={j_end}, n_frames={n_frames}. "
                f"Clipping to {n_frames-1}"
            )
            j_end = n_frames - 1

        # window_estimatesのサイズを確認
        expected_size = j_end - j + 1
        actual_size = len(window_estimates)

        if actual_size == expected_size:
            # 通常の場合（最初のウィンドウなど）: [j, j_end] 全体
            for i in range(j, j_end + 1):
                final_estimates[i] += window_estimates[i - j]
        elif actual_size == expected_size - 1:
            # 連続性保持のため開始フレームが除外されている場合
            # window_estimates は [j+1, j_end] の範囲の値を持つ
            for idx, i in enumerate(range(j + 1, j_end + 1)):
                final_estimates[i] += window_estimates[idx]
        else:
            logger.error(
                f"Unexpected window_estimates size: expected={expected_size} or {expected_size-1}, "
                f"actual={actual_size}, j={j}, j_end={j_end}"
            )

    # 推定回数で割って平均化
    for i in range(n_frames):
        if estimation_counts[i] > 0:
            final_estimates[i] /= estimation_counts[i]

    # 推定されていないフレームを線形補間
    if np.any(estimation_counts == 0):
        # 推定されたフレームのインデックス
        estimated_indices = np.where(estimation_counts > 0)[0]
        if len(estimated_indices) >= 2:
            # 線形補間
            final_estimates = np.interp(
                np.arange(n_frames),
                estimated_indices,
                final_estimates[estimated_indices]
            )
        elif len(estimated_indices) == 1:
            # 1つしかない場合は全フレームにその値を使用
            final_estimates[:] = final_estimates[estimated_indices[0]]

    return final_estimates


def check_ocr_constraint_violations(
    window_estimates: np.ndarray,
    ocr_values: np.ndarray,
    ocr_success: np.ndarray,
    j: int,
    j_end: int
) -> int:
    """ウィンドウ内のOCR制約範囲逸脱フレーム数をカウント

    Args:
        window_estimates: ウィンドウ内の推定距離、shape (j_end - j + 1,)
        ocr_values: OCR読み取り距離（mm）、shape (N,)
        ocr_success: OCR成功フラグ、shape (N,)
        j: ウィンドウ開始フレームインデックス
        j_end: ウィンドウ終了フレームインデックス

    Returns:
        OCR制約範囲を逸脱したフレーム数

    Note:
        制約条件: ocr_values[i] <= z*[i] < ocr_values[i] + 10
    """
    violation_count = 0

    for i in range(j, j_end + 1):
        if not ocr_success[i]:
            # OCR失敗フレームはスキップ
            continue

        z_star = window_estimates[i - j]
        ocr_value = ocr_values[i]

        # 制約条件チェック: ocr_value <= z* < ocr_value + 10
        if z_star < ocr_value or z_star >= ocr_value + 10:
            violation_count += 1

    return violation_count


def optimize_window_with_adaptive_size(
    ocr_values: np.ndarray,
    ocr_success: np.ndarray,
    j: int,
    j_end: int,
    initial_window_size: int,
    min_window_size: int,
    e_resolution: float
) -> Tuple[int, int, float, float, np.ndarray]:
    """適応的ウィンドウサイズでOCR制約を満たす最大ウィンドウを探索

    初期ウィンドウサイズから開始し、OCR制約範囲を逸脱する場合は
    ウィンドウサイズを段階的に縮小してリトライする。

    Args:
        ocr_values: OCR読み取り距離（mm）
        ocr_success: OCR成功フラグ
        j: ウィンドウ開始フレームインデックス
        j_end: ウィンドウ終了フレームインデックス
        initial_window_size: 初期ウィンドウサイズ
        min_window_size: 最小ウィンドウサイズ
        e_resolution: オフセット解像度（mm）

    Returns:
        Tuple[int, int, float, float, np.ndarray]:
            - adjusted_j: 調整後のウィンドウ開始インデックス
            - adjusted_j_end: 調整後のウィンドウ終了インデックス
            - best_e_j: 最適オフセット（開始）
            - best_e_j_end: 最適オフセット（終了）
            - window_estimates: ウィンドウ内推定距離

    Note:
        ウィンドウサイズ候補: [initial, initial-2, initial-4, ..., min]
        例: [10, 8, 6, 4, 2]
        OCR範囲内に収まる最大のウィンドウサイズを採用
    """
    # ウィンドウサイズ候補を生成（降順）
    window_sizes = list(range(initial_window_size, min_window_size - 1, -2))
    if initial_window_size not in window_sizes:
        window_sizes.insert(0, initial_window_size)
    if min_window_size not in window_sizes:
        window_sizes.append(min_window_size)

    logger.info(f"Adaptive window: trying sizes {window_sizes} for window [{j}, {j_end}]")

    for trial_window_size in window_sizes:
        logger.info(f"  Trying window_size={trial_window_size}")
        # ウィンドウサイズに合わせてj_endを調整
        trial_j_end = min(j + trial_window_size - 1, j_end)

        # ウィンドウの終了がOCR成功フレームか確認
        if not ocr_success[trial_j_end]:
            # trial_j_end以前で最後のOCR成功フレームを探す
            success_indices = np.where(ocr_success[j:trial_j_end+1])[0]
            if len(success_indices) > 0:
                trial_j_end = j + success_indices[-1]
            else:
                continue

        # ウィンドウ内に少なくとも2つのOCR成功フレームがあるか確認
        window_success_count = np.sum(ocr_success[j:trial_j_end+1])
        if window_success_count < 2:
            continue

        # 最適なオフセットを探索
        e_j, e_j_end, _ = optimize_offsets(
            ocr_values, ocr_success, j, trial_j_end, e_resolution
        )

        # ウィンドウ内の推定距離を計算
        window_estimates = compute_window_estimates(
            ocr_values, j, trial_j_end, e_j, e_j_end
        )

        # OCR制約範囲逸脱をチェック
        violation_count = check_ocr_constraint_violations(
            window_estimates, ocr_values, ocr_success, j, trial_j_end
        )

        # 逸脱がなければ成功（最大のウィンドウサイズを採用）
        if violation_count == 0:
            if trial_window_size != initial_window_size:
                logger.debug(
                    f"Adaptive window adjustment: "
                    f"window_size {initial_window_size} → {trial_window_size} "
                    f"at frame [{j}, {trial_j_end}]"
                )
            return j, trial_j_end, e_j, e_j_end, window_estimates

    # すべてのウィンドウサイズで逸脱が発生する場合は、最小ウィンドウサイズを採用
    logger.warning(
        f"OCR constraint violations persist even with minimum window size "
        f"at frame [{j}, {j_end}]. Using minimum window size {min_window_size}."
    )

    # 最小ウィンドウサイズで再計算
    trial_j_end = min(j + min_window_size - 1, j_end)
    if not ocr_success[trial_j_end]:
        success_indices = np.where(ocr_success[j:trial_j_end+1])[0]
        if len(success_indices) > 0:
            trial_j_end = j + success_indices[-1]

    e_j, e_j_end, _ = optimize_offsets(
        ocr_values, ocr_success, j, trial_j_end, e_resolution
    )

    window_estimates = compute_window_estimates(
        ocr_values, j, trial_j_end, e_j, e_j_end
    )

    return j, trial_j_end, e_j, e_j_end, window_estimates


def estimate_positions_offset_moving_average(
    ocr_values: np.ndarray,
    ocr_success: np.ndarray,
    window_size: int = 10,
    overlap: int = 5,
    e_resolution: float = 0.1,
    enable_adaptive_window: bool = True,
    min_window_size: int = 2
) -> np.ndarray:
    """オフセット移動平均距離推定法（適応的ウィンドウサイズ対応）

    10mm刻みのOCR読み取り値から、0.1mm精度の距離推定を行う。
    ウィンドウ内で制約条件を最も多く満たすオフセットの組み合わせを探索し、
    オーバーラップ処理で安定化する。

    適応的ウィンドウサイズ調整機能:
    - 推定値がOCR制約範囲を逸脱した場合、ウィンドウサイズを段階的に縮小してリトライ
    - ウィンドウサイズ候補: [10, 8, 6, 4, 2] (min_window_sizeまで)
    - OCR範囲内に収まる最大のウィンドウサイズを採用（大きいほど安定）

    Args:
        ocr_values: OCR読み取り距離（mm）、shape (N,)
        ocr_success: OCR成功フラグ、shape (N,)
        window_size: 初期ウィンドウサイズ（デフォルト10フレーム）
        overlap: オーバーラップ（デフォルト5フレーム、ウィンドウの半分）
        e_resolution: オフセット解像度（デフォルト0.1mm）
        enable_adaptive_window: 適応的ウィンドウサイズ調整を有効化（デフォルトTrue）
        min_window_size: 最小ウィンドウサイズ（デフォルト2フレーム）

    Returns:
        全フレームの推定距離（mm）、shape (N,)

    Raises:
        ValueError: フレーム数がwindow_size未満の場合、またはOCR成功フレームが不足

    Example:
        >>> ocr_values = np.array([700, 700, 700, 710, 710, 710, 720, 720, 720, 720, 730])
        >>> ocr_success = np.ones(11, dtype=bool)
        >>> estimated = estimate_positions_offset_moving_average(ocr_values, ocr_success)
        >>> # estimated[0] ≈ 702.5, estimated[5] ≈ 713.2, ... (0.1mm精度)
    """
    n_frames = len(ocr_values)

    if n_frames < window_size:
        raise ValueError(
            f"Number of frames ({n_frames}) must be >= window_size ({window_size})"
        )

    # OCR成功フレームのインデックスを取得
    success_indices = np.where(ocr_success)[0]
    if len(success_indices) < 2:
        raise ValueError(
            f"Need at least 2 successful OCR readings, got {len(success_indices)}"
        )

    logger.info(
        f"Starting offset moving average estimation: "
        f"n_frames={n_frames}, window_size={window_size}, overlap={overlap}, "
        f"ocr_success_rate={len(success_indices)/n_frames*100:.1f}%"
    )

    # ウィンドウをスライドさせて最適化
    # ウィンドウの両端は必ずOCR成功フレームにする
    # 注: 適応的ウィンドウ調整時はstrideを動的に計算
    stride = window_size - overlap  # 初期値
    estimations = []
    estimation_counts = np.zeros(n_frames, dtype=int)

    j = 0
    window_count = 0
    while j < n_frames:
        window_count += 1
        logger.info(f"Processing window #{window_count}, j={j}")
        j_end = min(j + window_size - 1, n_frames - 1)  # n_frames-1を超えないようにクリップ

        # ウィンドウの開始と終了がOCR成功フレームか確認
        # 失敗している場合は最も近い成功フレームを探す
        if not ocr_success[j]:
            # j以降で最初のOCR成功フレームを探す
            next_success = success_indices[success_indices >= j]
            if len(next_success) > 0 and next_success[0] <= j_end:
                j = next_success[0]
            else:
                j += stride
                continue

        if not ocr_success[j_end]:
            # j_end以前で最後のOCR成功フレームを探す
            prev_success = success_indices[success_indices <= j_end]
            if len(prev_success) > 0 and prev_success[-1] >= j:
                j_end = prev_success[-1]
            else:
                j += stride
                continue

        # ウィンドウ内に少なくとも2つのOCR成功フレームがあるか確認
        window_success_count = np.sum(ocr_success[j:j_end+1])
        if window_success_count < 2:
            j += stride
            continue

        # 適応的ウィンドウサイズ調整（有効化されている場合）
        if enable_adaptive_window:
            adjusted_j, adjusted_j_end, e_j, e_j_end, window_estimates = \
                optimize_window_with_adaptive_size(
                    ocr_values, ocr_success, j, j_end,
                    window_size, min_window_size, e_resolution
                )
        else:
            # 従来の固定ウィンドウサイズ処理
            e_j, e_j_end, count = optimize_offsets(
                ocr_values, ocr_success, j, j_end, e_resolution
            )
            window_estimates = compute_window_estimates(
                ocr_values, j, j_end, e_j, e_j_end
            )
            adjusted_j = j
            adjusted_j_end = j_end

        # 推定値を保存
        # 連続性を保つため、最初のウィンドウ以外は開始フレームを除外
        if len(estimations) == 0:
            # 最初のウィンドウ: 全フレーム保存
            estimations.append((adjusted_j, adjusted_j_end, window_estimates))
            for i in range(adjusted_j, adjusted_j_end + 1):
                estimation_counts[i] += 1
        else:
            # 2番目以降のウィンドウ: 開始フレームは前のウィンドウの値を保持
            # 保存範囲は [adjusted_j+1, adjusted_j_end]
            # 区間が空でないかチェック
            if adjusted_j + 1 <= adjusted_j_end:
                estimations.append((adjusted_j + 1, adjusted_j_end, window_estimates[1:]))
                for i in range(adjusted_j + 1, adjusted_j_end + 1):
                    estimation_counts[i] += 1

        # 次のウィンドウの開始位置を計算
        # overlap+1フレームずつオーバーラップ
        if enable_adaptive_window:
            # 適応的ウィンドウ: 調整後の終了位置からoverlapフレーム戻る
            # ただし、調整後のウィンドウサイズが小さい場合は最小限のシフト
            actual_window_size = adjusted_j_end - adjusted_j + 1
            actual_overlap = min(overlap, actual_window_size - 1)
            j = adjusted_j_end - actual_overlap + 1
        else:
            # 固定ウィンドウ: 初期strideでスライド
            j += stride

        # ウィンドウ終了チェック
        if j + min_window_size > n_frames:
            break

    # 最後のウィンドウが含まれていない場合の処理
    # 最後のOCR成功フレームが推定されているかチェック
    if len(success_indices) > 0:
        last_success_idx = success_indices[-1]
        if estimation_counts[last_success_idx] == 0:
            # 最後の成功フレームを含むウィンドウを探す
            # 最後のOCR成功フレームから逆算してウィンドウ開始を決定
            j_end = last_success_idx
            j = max(0, j_end - window_size + 1)

            # jがOCR成功フレームになるよう調整
            if not ocr_success[j]:
                next_success = success_indices[success_indices >= j]
                if len(next_success) > 0 and next_success[0] < j_end:
                    j = next_success[0]

            window_success_count = np.sum(ocr_success[j:j_end+1])
            if window_success_count >= 2:
                # 適応的ウィンドウサイズ調整（有効化されている場合）
                if enable_adaptive_window:
                    adjusted_j, adjusted_j_end, e_j, e_j_end, window_estimates = \
                        optimize_window_with_adaptive_size(
                            ocr_values, ocr_success, j, j_end,
                            window_size, min_window_size, e_resolution
                        )
                else:
                    # 従来の固定ウィンドウサイズ処理
                    e_j, e_j_end, count = optimize_offsets(
                        ocr_values, ocr_success, j, j_end, e_resolution
                    )
                    window_estimates = compute_window_estimates(
                        ocr_values, j, j_end, e_j, e_j_end
                    )
                    adjusted_j = j
                    adjusted_j_end = j_end

                # 連続性を保つため、開始フレームは前のウィンドウの値を保持
                if len(estimations) > 0:
                    # 区間が空でないかチェック
                    if adjusted_j + 1 <= adjusted_j_end:
                        estimations.append((adjusted_j + 1, adjusted_j_end, window_estimates[1:]))
                        for i in range(adjusted_j + 1, adjusted_j_end + 1):
                            estimation_counts[i] += 1
                else:
                    estimations.append((adjusted_j, adjusted_j_end, window_estimates))
                    for i in range(adjusted_j, adjusted_j_end + 1):
                        estimation_counts[i] += 1

    # 推定されていないフレームがあるかチェック
    if np.any(estimation_counts == 0):
        logger.warning(
            f"Some frames were not estimated: "
            f"{np.sum(estimation_counts == 0)}/{n_frames} frames"
        )
        # 推定されていないフレームには、最も近い推定値を補間
        # （後処理として実装）

    # estimationsリストの内容をデバッグ出力
    logger.info(f"Estimations list: {len(estimations)} windows")
    for idx, (j, j_end, estimates) in enumerate(estimations):
        logger.info(
            f"  Window {idx}: [{j}, {j_end}], estimates_len={len(estimates)}, "
            f"expected={(j_end - j + 1)}"
        )

    # オーバーラップする推定値を平均化
    final_estimates = average_overlapping_estimates(
        estimations, n_frames, estimation_counts
    )

    logger.info(
        f"Offset moving average estimation completed: "
        f"n_windows={len(estimations)}, "
        f"avg_estimation_count={np.mean(estimation_counts[estimation_counts > 0]):.1f}"
    )

    logger.info(
        f"Final estimates shape: {final_estimates.shape}, "
        f"range: [{np.min(final_estimates):.1f}, {np.max(final_estimates):.1f}]mm"
    )

    return final_estimates
