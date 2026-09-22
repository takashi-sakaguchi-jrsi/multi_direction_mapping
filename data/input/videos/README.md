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
python src/generate_two_direction_test_videos.py --fov 181 --runs U,R --frames 150 --z-start-mm 0 --z-step-mm 10 --reconstruct
```

全長は使わず、開始距離から 100～300 フレーム程度だけ出します。`--z-start-mm` を変えると特徴の違う区間で確認できます。

出力例: `phi250_fisheye_side_U.mp4`, `phi250_fisheye_side_R.mp4` と `*_metadata.json`。
距離 overlay は焼き込みません。`--reconstruct` は既知の z / roll / pitch で、生成した距離区間付近の展開図へ戻す確認用です。OCR や特徴点姿勢推定は使いません。
