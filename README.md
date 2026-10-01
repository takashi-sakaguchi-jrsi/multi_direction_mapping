# 2方向動画合成カラーマップ検証

仕様書 `doc/2方向動画合成カラーマップ生成_検証仕様書_v0_4_rev3.docx` に対する側視の採用事項は `doc/2方向動画合成カラーマップ生成_実装ノート.md`。  
正面/側視を同一 EXE で切り替える統合は `doc/正面側視統合_改修ノート.md`。

## 製品版

エントリは `src/main_twopass.py`（EXE 名 `main_twopass.exe`）。`--pipeline front|sideview`（未指定時は設定の `pipeline`、既定 sideview）。

```bash
python src/main_twopass.py --help

# 正面（動画 1 本）
python src/main_twopass.py --pipeline front --input data/input/videos/phi250_frontview.mp4 --start 1 --end 500

# 正面（フォルダ + 名前 → {name}.{ext}）
python src/main_twopass.py --pipeline front --input-dir data/input/videos --name phi250_frontview

# 側視（{name}_U/_R/_L。L が無ければ 2 方向）
python src/main_twopass.py --pipeline sideview --input-dir data/input/videos --name phi250_baseline_calib --config data/config/two_direction_config.json
```

`--input` は正面専用。`--input-dir` と `--name` は正面・側視とも使う。`--start/--end` は 1 始まり（含む）。

Windows 向け一式は `build\build_installer.bat` が `dist\main_twopass_<VERSION>\` に出力します（`python` または `py -3` が PATH にあること。PyInstaller は `python -m PyInstaller` で呼び出します）。進捗ウィンドウは同じフォルダへ `ProgressViewer.exe` として入ります。

仮想検証（既知 z）は従来どおり:

```bash
python src/main_two_direction_validation.py --config data/config/two_direction_config.json
pytest tests/ -q
```

## 実行

CLI は `src/main_twopass.py` と同じ 4 層設定です（dataclass 初期値 < default_config.json < `--config` < 個別引数）。

## 入力動画

配置方法は `data/input/videos/README.md` を参照。

実動画の前に、既存展開図から魚眼の 2 方向テスト動画を作れます（距離 overlay / Phase1 なし）。
再現テストの正本は `original_colormap/phi250tenkaizu.png`（管路直径 250mm）。

```bash
python src/generate_two_direction_test_videos.py --fov 181 --runs U,R --frames 150 --z-start-mm 10 --z-step-mm 4.5 --jitter
python src/generate_two_direction_test_videos.py --calibration data/calibration/fisheye_recalibrated_20260217.json --runs U,R,L --frames 150 --z-start-mm 10 --z-step-mm 4.5 --jitter
```

プロジェクトルートで実行してください。`python src/...` でも動くようにしてあります。

距離が長い展開図でも、既定は 150 フレーム（約 100～300 推奨）だけ生成します。`--z-start-mm` で開始距離を変えると、別の特徴区間で試せます。
