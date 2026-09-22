# 2方向動画合成カラーマップ検証

## 実行

```bash
python src/main_two_direction_validation.py --config data/config/two_direction_config.json
pytest tests/ -q
```

CLI は `src/main_twopass.py` と同じ4層設定です（dataclass 初期値 < default_config.json < `--config` < 個別引数）。

## 入力動画

配置方法は `data/input/videos/README.md` を参照。

実動画の前に、既存展開図から魚眼の2方向テスト動画を作れます（距離 overlay / Phase1 なし）。
再現テストの正本は `original_colormap/phi250tenkaizu.png`（管路直径 250mm）。

```bash
python src/generate_two_direction_test_videos.py --fov 181 --runs U,R --frames 150 --z-start-mm 0 --reconstruct
```

プロジェクトルートで実行してください。`python src/...` でも動くようにしてあります。

距離が長い展開図でも、既定は 150 フレーム（約 100～300 推奨）だけ生成します。`--z-start-mm` で開始距離を変えると、別の特徴区間で試せます。
# multi_direction_mapping
