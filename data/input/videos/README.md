# 2方向検証用動画の配置

撮影後、同一セッションのファイルをこのディレクトリへ置いてください。

```
data/input/videos/
  <name>_U.mp4   # 真上   物理 roll = 0°     内部 0°
  <name>_R.mp4   # 右下   物理 roll = 120°    内部 +120°
  <name>_L.mp4   # 左下   物理 roll = 240°    内部 -120°
```

例: `vu250_side_U.mp4`, `vu250_side_R.mp4`

PoC は3本のうち2本を `data/config/two_direction_config.json` の `run_A` / `run_B` で指定します。
`video_path` の suffix `_U` / `_R` / `_L` からも方向を補完できますが、config の `physical_roll_deg` が優先です。

## 展開図からの生成（Phase1省略）

CameraCarDemo 相当の円筒サンプリングで、カメラモデルだけ等距離魚眼（既定 FOV=181°）に差し替えたテスト動画。
正本展開図は `original_colormap/phi250tenkaizu.png`（φ250mm、半径 125mm）。

```bash
python src/generate_two_direction_test_videos.py --fov 181 --runs U,R --frames 150 --z-start-mm 10 --z-step-mm 4.5 --jitter
```

`--jitter` はフレームごとの dz / dpitch / dyaw に、十数フレーム周期の緩やかな中程度振動と 2〜3 フレームの小さい振動を合成します。OCR 列 `ocr_z_mm` は真の累積 z に 10mm 遅れモデルを掛けたものです。

距離 overlay は焼き込みません。`--reconstruct` は既知の z / 姿勢で、生成した距離区間付近の展開図へ戻す確認用です。OCR や特徴点姿勢推定は使いません。
