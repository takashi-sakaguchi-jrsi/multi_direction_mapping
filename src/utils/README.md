# フレーム読み込みユーティリティ（src/utils）

メモリ効率的なビデオフレーム処理のための共通ユーティリティモジュールです。

## 概要

本モジュールは、管内カメラカーシミュレーション用壁面画像生成処理において、メモリ効率的なフレーム処理を実現するための共通ユーティリティクラスを提供します。

### 設計思想

- **メモリ効率最優先**: 本番プログラム（main_twopass.py）の設計思想に準拠
- **ストリーミング処理**: メモリ使用量を最小化（最大2フレーム程度）
- **安全性**: コンテキストマネージャで確実なリソース解放
- **統一性**: エラーハンドリングとログ出力の統一

## 主要クラス

### 1. LazyFrameLoader

フレームパスのみを保持し、必要時に読み込むストリーミング型フレームローダー。

**特徴:**
- メモリに最大1フレームのみ保持
- コンテキストマネージャで安全なリソース管理
- イテレータパターンでストリーミング処理
- `__getitem__`でランダムアクセス対応
- `__len__`でフレーム数取得対応
- `load_selected_frames()`で指定インデックスのフレームのみ読み込み

**使用例:**

```python
from pathlib import Path
from src.utils import LazyFrameLoader

video_path = Path("data/input/sample.mp4")

# 基本的なストリーミング処理
with LazyFrameLoader(video_path, start_frame=0, end_frame=100) as loader:
    for frame_idx, frame in loader:
        # メモリに最大1フレームのみ保持
        process_frame(frame)

# ランダムアクセス
with LazyFrameLoader(video_path) as loader:
    frame_50 = loader[50]
    frame_100 = loader[100]
    
    # フレーム数取得
    total_frames = len(loader)

# 選択的フレーム読み込み（品質フィルタリング後等）
with LazyFrameLoader(video_path) as loader:
    selected_indices = [10, 20, 30, 40, 50]
    frames, indices = loader.load_selected_frames(selected_indices)
```

### 2. BatchFrameProcessor

バッチ処理向けフレームプロセッサ。

**特徴:**
- メモリ効率を保ちながらバッチ処理
- 処理関数をカスタマイズ可能
- 進捗ログ出力

**使用例:**

```python
from pathlib import Path
from src.utils import BatchFrameProcessor

video_path = Path("data/input/sample.mp4")

# バッチ処理関数の定義
def evaluate_batch(batch):
    """バッチ内の全フレームを評価"""
    results = []
    for frame_idx, frame in batch:
        score = evaluate_quality(frame)
        results.append({'frame_idx': frame_idx, 'score': score})
    return results

# バッチ処理実行
processor = BatchFrameProcessor(evaluate_batch, batch_size=32)
results = processor.process_video(video_path, start_frame=0, end_frame=500)

# 結果分析
scores = [r['score'] for r in results]
print(f"Mean quality: {np.mean(scores):.2f}")
```

## 実装パターン

### パターン1: ストリーミング処理（メモリ最小化）

本番処理（カラーマップ生成）で使用するパターン。

```python
from pathlib import Path
from src.utils import LazyFrameLoader

video_path = Path("data/input/sample.mp4")

# メモリ: 最大1フレームのみ
with LazyFrameLoader(video_path) as loader:
    for frame_idx, frame in loader:
        # フレーム処理
        process_frame(frame)
```

**メモリ使用量:** 約5-10 MB（1フレーム分のみ）

### パターン2: バッチ処理（品質評価等）

キャリブレーション等の複数パス処理で使用するパターン。

```python
from pathlib import Path
from src.utils import BatchFrameProcessor

video_path = Path("data/input/sample.mp4")

def evaluate_batch(batch):
    return [evaluate_quality(idx, frame) for idx, frame in batch]

# メモリ: batch_size フレーム分
processor = BatchFrameProcessor(evaluate_batch, batch_size=32)
results = processor.process_video(video_path)
```

**メモリ使用量:** 約160-320 MB（32フレーム × 5-10 MB）

### パターン3: 選択的読み込み（品質フィルタリング後）

キャリブレーション最適化で使用するパターン。

```python
from pathlib import Path
from src.utils import LazyFrameLoader

video_path = Path("data/input/sample.mp4")

# Phase 1: 全フレームの品質評価（ストリーミング）
quality_scores = []
with LazyFrameLoader(video_path) as loader:
    for frame_idx, frame in loader:
        score = evaluate_quality(frame)
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

# Phase 4: キャリブレーション実行
calibrate_frames(good_frames)
```

**メモリ使用量:** 約250-500 MB（50-70フレーム × 5-10 MB）

## エラーハンドリング

### カスタム例外

- `FrameReaderError`: フレーム読み込み関連のベース例外
- `VideoFileError`: ビデオファイル関連のエラー
- `FrameReadError`: フレーム読み込みエラー

### エラーハンドリング例

```python
from pathlib import Path
from src.utils import LazyFrameLoader, VideoFileError, FrameReadError

try:
    with LazyFrameLoader(video_path) as loader:
        for frame_idx, frame in loader:
            process_frame(frame)

except VideoFileError as e:
    logger.error(f"ビデオファイルを開けませんでした: {e}")

except FrameReadError as e:
    logger.error(f"フレーム読み込みに失敗しました: {e}")

except Exception as e:
    logger.error(f"予期しないエラーが発生しました: {e}")
```

## パフォーマンス比較

| 処理種別 | 推奨パターン | メモリ使用量 | 処理速度 |
|---------|-----------|-----------|--------|
| **本番処理（カラーマップ生成）** | ストリーミング（LazyFrameLoader） | 5-10 MB | ⭐⭐⭐⭐⭐ |
| **キャリブレーション** | バッチ処理（BatchFrameProcessor） | 160-320 MB | ⭐⭐⭐⭐ |
| **キャリブレーション最適化** | 品質フィルタリング → 選択読み込み | 250-500 MB | ⭐⭐⭐ |
| **全フレーム事前読み込み（非推奨）** | - | 5-10 GB | ⭐⭐ |

## テスト

テストコードは `tests/test_frame_reader.py` に配置されています。

### テスト実行

```bash
# 全テスト実行
pytest tests/test_frame_reader.py -v

# 特定のテストクラスのみ実行
pytest tests/test_frame_reader.py::TestLazyFrameLoader -v

# パフォーマンステストを除外
pytest tests/test_frame_reader.py -v -m "not slow"
```

### テスト項目

- LazyFrameLoaderの基本動作
- BatchFrameProcessorの基本動作
- エラーハンドリング
- メモリ効率性の確認
- 統合ワークフロー

## 使用上の注意

### コンテキストマネージャの使用必須

LazyFrameLoaderは**必ずコンテキストマネージャ（`with`文）**を使用してください。

```python
# ✅ 正しい使い方
with LazyFrameLoader(video_path) as loader:
    for frame_idx, frame in loader:
        process_frame(frame)

# ❌ 誤った使い方（リソースリークの危険）
loader = LazyFrameLoader(video_path)
for frame_idx, frame in loader:  # RuntimeError
    process_frame(frame)
```

### フレーム番号の扱い

- `start_frame`, `end_frame`: ビデオ内の**絶対位置**
- イテレーション時のフレーム番号: **絶対位置**
- `__getitem__`のインデックス: **絶対位置**

```python
# 例: フレーム10-50を処理
with LazyFrameLoader(video_path, start_frame=10, end_frame=50) as loader:
    for frame_idx, frame in loader:
        # frame_idx は 10, 11, 12, ..., 49
        pass
```

### メモリ使用量の見積もり

- 1フレーム（1920×1080, BGR）: 約6 MB
- バッチサイズ32: 約192 MB
- 全フレーム事前読み込み（1000フレーム）: 約6 GB

## 関連ドキュメント

- [MEMORY_PATTERNS_IMPLEMENTATION.md](/home/jrsi/work/CameraCarSim/MEMORY_PATTERNS_IMPLEMENTATION.md): メモリ効率実装パターン詳細ガイド
- [src/main_twopass.py](/home/jrsi/work/CameraCarSim/src/main_twopass.py): 本番プログラム（設計思想の参考）
- [tests/test_frame_reader.py](/home/jrsi/work/CameraCarSim/tests/test_frame_reader.py): テストコード

## 今後の拡張案

- [ ] メモリモニタリング機能の統合
- [ ] フレームキャッシュ機能の追加
- [ ] 並列処理対応（マルチプロセス/マルチスレッド）
- [ ] 進捗バー表示機能の統合
- [ ] ビデオエンコード対応の拡張

## 問い合わせ

不明点や改善提案がある場合は、プロジェクト管理者にお問い合わせください。
