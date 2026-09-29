3方向合成 progress.json ビューア

元プログラム: /home/tsakaguchi/progress_viewer/progress_viewer
（全体バー + 現在ステップバー + ステップ一覧 + details。500ms ポーリング）

3方向合成向けの変更:
  - 既定パスは製品配置の progress/progress.json
  - ステップ名は Python が出す name_ja をそのまま表示
      初期化
      フレーム解析（1/N）…（N/N）  N=方向数
      部分図生成（1/N）…（N/N）
      接合・補正
  - 経過時間・残り時間・終了予定時刻を表示（progress.json の estimated_end_time）
  - details の run_id / phase / current_frame / total_frames / skip_reason を一覧と下部に表示
  - 状態を日本語表示（待機 / 実行中 / 完了 / エラー / スキップ）
  - Windows で書き込み中でも読めるよう FileShare.ReadWrite

ビルド:
  dotnet build progress_viewer/ProgressViewer.csproj

実行:
  ProgressViewer.exe [progress.json のパス]
  省略時は progress/progress.json を探し、無ければファイル選択ダイアログを出します。
