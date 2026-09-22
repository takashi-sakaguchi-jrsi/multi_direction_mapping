"""
colormap_correction.py

展開画像補正モジュール

OCR距離推定値を基準として、展開画像の横方向（Z軸方向）を
PCHIPスプラインで非線形補正する。

レガシーコード: CameraCarSim_collection.py の correct_wall_image_spline() を
クラスベースで再実装。
"""

from typing import Tuple, Optional, List
from pathlib import Path
import logging

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


class ColormapCorrectionError(Exception):
    """展開画像補正処理のエラー"""
    pass


class ColormapCorrector:
    """展開画像補正クラス

    OCR距離推定値を基準として、展開画像の横方向（Z軸方向）を
    PCHIPスプラインで非線形補正する。

    Attributes:
        logger: ロガー
        ocr_interval_mm: 補正基準OCR距離間隔 L (mm)
        max_image_size: 最大画像サイズ（OpenCV制限）
    """

    def __init__(
        self,
        logger: logging.Logger,
        ocr_interval_mm: float = 100.0,
        max_image_size: int = 30000
    ):
        """初期化

        Args:
            logger: ロガー
            ocr_interval_mm: 補正基準OCR距離間隔 L (mm)
            max_image_size: 最大画像サイズ（OpenCV制限）
        """
        self.logger = logger
        self.ocr_interval_mm = ocr_interval_mm
        self.max_image_size = max_image_size

    def correct_colormap(
        self,
        image_path: Path,
        excel_path: Path,
        output_path: Path,
        pixels_per_mm: float
    ) -> bool:
        """展開画像を補正

        Args:
            image_path: 元展開画像パス（temporary_colormap_path）
            excel_path: Excelファイルパス（z, z_ocr列を含む）
            output_path: 補正済み画像パス（colormap_path）
            pixels_per_mm: ピクセル/mm変換係数

        Returns:
            補正成功時True

        Raises:
            ColormapCorrectionError: 補正処理エラー
        """
        try:
            self.logger.info("展開画像補正を開始します")
            self.logger.info(f"入力画像: {image_path}")
            self.logger.info(f"Excelファイル: {excel_path}")
            self.logger.info(f"出力画像: {output_path}")
            self.logger.info(f"変換係数: {pixels_per_mm:.3f} px/mm")
            self.logger.info(f"補正基準間隔: {self.ocr_interval_mm} mm")

            # 1. ファイル存在チェック
            if not image_path.exists():
                raise ColormapCorrectionError(f"画像ファイルが見つかりません: {image_path}")
            if not excel_path.exists():
                raise ColormapCorrectionError(f"Excelファイルが見つかりません: {excel_path}")

            # 2. 画像読み込み
            src_image = cv2.imread(str(image_path))
            if src_image is None:
                raise ColormapCorrectionError(f"画像ファイルを読み込めません: {image_path}")

            # BGR→RGB変換（OpenCVはBGR、matplotlibはRGBを使用）
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
            self.logger.info("BGR→RGB変換完了")

            h, w = src_image.shape[:2]
            self.logger.info(f"元画像サイズ: {w} x {h}")

            # 3. 画像サイズチェック（必要に応じてリサイズ）
            scale_factor = 1.0
            if w > self.max_image_size or h > self.max_image_size:
                scale_factor = min(self.max_image_size / w, self.max_image_size / h)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                src_image = cv2.resize(src_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w
                self.logger.info(f"画像をリサイズしました: スケール={scale_factor:.3f}, 新サイズ={w}x{h}")

            # pixels_per_mmをスケール補正
            pixels_per_mm_scaled = pixels_per_mm * scale_factor

            # 4. Excelから補正データを読み込み
            z_mm, z_ocr_mm = self._load_correction_data(excel_path)
            self.logger.info(f"補正データ読み込み完了: {len(z_mm)}フレーム")
            self.logger.info(f"z範囲: {z_mm.min():.1f} ～ {z_mm.max():.1f} mm")
            self.logger.info(f"z_ocr範囲: {z_ocr_mm.min():.1f} ～ {z_ocr_mm.max():.1f} mm")

            # 5. 補正基準フレームをフィルタリング
            z_ref, z_ocr_ref = self._filter_reference_frames(z_mm, z_ocr_mm)
            self.logger.info(f"補正基準フレーム数: {len(z_ref)}フレーム")

            # 6. 補正マッピングを作成
            forward_mapping, inverse_mapping = self._create_correction_mapping(
                z_ref, z_ocr_ref
            )
            self.logger.info("補正マッピング作成完了（PCHIPスプライン）")

            # 7. 画像に補正を適用
            corrected_image = self._apply_correction(
                src_image, inverse_mapping, pixels_per_mm_scaled, z_mm, z_ocr_mm
            )
            self.logger.info(f"補正後画像サイズ: {corrected_image.shape[1]} x {corrected_image.shape[0]}")

            # 8. 出力ディレクトリ作成
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 9. 補正済み画像を保存
            plt.imsave(str(output_path), corrected_image)
            self.logger.info(f"補正済み画像を保存しました: {output_path}")

            return True

        except Exception as e:
            self.logger.error(f"展開画像補正に失敗しました: {e}")
            raise ColormapCorrectionError(f"補正処理エラー: {e}") from e

    def _load_correction_data(
        self,
        excel_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Excelから補正データを読み込み

        Args:
            excel_path: Excelファイルパス

        Returns:
            (z_mm, z_ocr_mm): 推定距離とOCR距離の配列

        Raises:
            ColormapCorrectionError: データ読み込みエラー
        """
        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            raise ColormapCorrectionError(f"Excelファイルの読み込みに失敗しました: {e}")

        # 必要な列の確認（英語表記と日本語表記の両方に対応）
        z_col = None
        z_ocr_col = None

        for col in df.columns:
            if col.lower() == 'z' or col == 'Z (mm)':
                z_col = col
            if col.lower() == 'z_ocr' or col == 'Z_OCR (mm)':
                z_ocr_col = col

        if z_col is None:
            raise ColormapCorrectionError("Excelファイルに'z'または'Z (mm)'列がありません")
        if z_ocr_col is None:
            raise ColormapCorrectionError("Excelファイルに'z_ocr'または'Z_OCR (mm)'列がありません")

        # データ抽出（NaN除外）
        valid_mask = df[z_col].notna() & df[z_ocr_col].notna()
        z_mm = df.loc[valid_mask, z_col].astype(float).values
        z_ocr_mm = df.loc[valid_mask, z_ocr_col].astype(float).values

        if len(z_mm) < 2:
            raise ColormapCorrectionError(
                f"補正に必要なデータが不足しています（最低2点必要）: {len(z_mm)}点"
            )

        # 注: 0基準化済みデータでは原点(0, 0)追加は不要
        # 以前のレガシーコードでは追加していたが、0基準化後は不要
        # （追加すると補正マッピングが歪み、画像が過剰に引き伸ばされる）
        # z_mm = np.r_[0.0, z_mm]  # 削除
        # z_ocr_mm = np.r_[0.0, z_ocr_mm]  # 削除

        # 単調性確保のためソート
        sort_indices = np.argsort(z_mm)
        z_mm = z_mm[sort_indices]
        z_ocr_mm = z_ocr_mm[sort_indices]

        return z_mm, z_ocr_mm

    def _filter_reference_frames(
        self,
        z_mm: np.ndarray,
        z_ocr_mm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """補正基準フレームをフィルタリング

        開始フレームからOCR距離L(mm)間隔でフレームをピックアップ。

        Args:
            z_mm: 推定距離配列（全フレーム）
            z_ocr_mm: OCR距離配列（全フレーム）

        Returns:
            (z_ref, z_ocr_ref): フィルタリング後の配列
        """
        # z_ocr_mmをL(mm)間隔で間引く
        reference_indices = [0]  # 原点は必ず含める

        current_z_ocr = 0.0
        for i in range(1, len(z_ocr_mm)):
            if z_ocr_mm[i] >= current_z_ocr + self.ocr_interval_mm:
                reference_indices.append(i)
                current_z_ocr = z_ocr_mm[i]

        # 最後のフレームも含める
        if reference_indices[-1] != len(z_ocr_mm) - 1:
            reference_indices.append(len(z_ocr_mm) - 1)

        z_ref = z_mm[reference_indices]
        z_ocr_ref = z_ocr_mm[reference_indices]

        return z_ref, z_ocr_ref

    def _create_correction_mapping(
        self,
        z_mm: np.ndarray,
        z_ocr_mm: np.ndarray
    ) -> Tuple[PchipInterpolator, PchipInterpolator]:
        """補正マッピングを作成

        Args:
            z_mm: 推定距離配列
            z_ocr_mm: OCR距離配列

        Returns:
            (forward, inverse): 順方向・逆方向のPCHIPスプライン
                forward: z -> z_ocr
                inverse: z_ocr -> z
        """
        # 順方向: z -> z_ocr
        forward = PchipInterpolator(z_mm, z_ocr_mm, extrapolate=True)

        # 逆方向: z_ocr -> z
        # z_ocr_mmでソートして単調性を確保
        za_sort_indices = np.argsort(z_ocr_mm)
        z_ocr_mm_sorted = z_ocr_mm[za_sort_indices]
        z_mm_for_inverse = z_mm[za_sort_indices]

        inverse = PchipInterpolator(z_ocr_mm_sorted, z_mm_for_inverse, extrapolate=True)

        return forward, inverse

    def _apply_correction(
        self,
        src_image: np.ndarray,
        inverse_mapping: PchipInterpolator,
        pixels_per_mm: float,
        z_mm: np.ndarray,
        z_ocr_mm: np.ndarray
    ) -> np.ndarray:
        """画像に補正を適用

        Args:
            src_image: 元展開画像
            inverse_mapping: 逆方向マッピング（z_ocr -> z）
            pixels_per_mm: ピクセル/mm変換係数
            z_mm: 推定距離配列（全フレーム）
            z_ocr_mm: OCR距離配列（全フレーム）

        Returns:
            補正済み画像

        Raises:
            ColormapCorrectionError: 補正適用エラー
        """
        h, w = src_image.shape[:2]

        # Z範囲の取得
        # Z座標の最大値を取得
        z_max = np.max(z_mm)          # 推定値の最大
        za_max = np.max(z_ocr_mm)     # OCR値の最大

        self.logger.info(f"Z範囲: 0 ～ {z_max:.2f} mm")
        self.logger.info(f"Z_OCR範囲: 0 ～ {za_max:.2f} mm")

        # 補正後の幅を計算
        # 補正前の幅: w
        # 補正前の最終フレームより右側: w - z_max × pixels_per_mm
        # 補正後の幅: za_max × pixels_per_mm + (w - z_max × pixels_per_mm)
        #          = w + (za_max - z_max) × pixels_per_mm
        new_w = int(w + (za_max - z_max) * pixels_per_mm + 0.5)
        self.logger.info(f"画像幅: {w} px → {new_w} px (差分: {za_max - z_max:.2f} mm × {pixels_per_mm:.3f} px/mm = {(za_max - z_max) * pixels_per_mm:.1f} px)")

        # OpenCVの制限チェック (SHRT_MAX = 32767)
        SHRT_MAX = 32767
        if new_w >= SHRT_MAX or h >= SHRT_MAX or w >= SHRT_MAX:
            raise ColormapCorrectionError(
                f"画像サイズがOpenCVの制限を超えています。new_w={new_w}, h={h}, w={w} (制限: {SHRT_MAX})"
            )

        # 出力画像の各ピクセルに対応するmm座標を計算
        # ピクセル密度は一定（pixels_per_mm）
        dst_x = np.arange(new_w, dtype=np.float32)  # 出力 x座標 [0, new_w)
        mm_dst = dst_x / pixels_per_mm  # ピクセル位置をmm座標に変換

        # BUG-009修正: 補正範囲と外挿領域を分離
        # 補正範囲 (mm_dst <= za_max): inverse_mapping でPCHIPスプライン補正
        # 外挿領域 (mm_dst > za_max): 元画像から直接コピー（鏡像回避）
        mask_extrapolation = mm_dst > za_max

        # 補正範囲: inverse_mapping を使用
        mm_src_corrected = inverse_mapping(mm_dst)

        # 外挿領域: 元画像の対応位置を直接計算
        # mm_dst が za_max を超える部分は、z_max + (mm_dst - za_max) の位置からコピー
        # 例: mm_dst = 741mm, za_max = 740mm, z_max = 666mm
        #     → mm_src = 666 + (741 - 740) = 667mm
        mm_src_direct = z_max + (mm_dst - za_max)

        # 結合
        mm_src = np.where(mask_extrapolation, mm_src_direct, mm_src_corrected)

        self.logger.info(f"補正範囲: {(~mask_extrapolation).sum()}px, 外挿領域: {mask_extrapolation.sum()}px")

        # NaNチェック
        nan_count = np.isnan(mm_src).sum()
        if nan_count > 0:
            self.logger.warning(f"mm_srcに{nan_count}個のNaN値があります（境界値でクリップします）")
            mm_src = np.where(np.isnan(mm_src), 0, mm_src)

        # 元画像のピクセル座標に変換
        map_x = (mm_src * pixels_per_mm).astype(np.float32)

        # 元画像の範囲を超える場合はクリップ
        map_x = np.clip(map_x, 0, w - 1)

        # map_xを2D配列に拡張（H×W）
        map_x = map_x.reshape(1, -1)  # 1×W
        map_x = np.repeat(map_x, h, axis=0)  # H×W

        # map_y（縦方向は変化なし）
        map_y = np.repeat(
            np.arange(h, dtype=np.float32).reshape(-1, 1), new_w, axis=1
        )

        # remap で一括補正
        corrected = cv2.remap(
            src_image, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return corrected
