管内カメラカーシミュレータ 3方向合成カラーマップ
Version: {VERSION}

フォルダ:
  data\calibration   レンズ校正 JSON
  data\config        default_config.json
  data\input         U.mp4 / R.mp4 / L.mp4
  data\output        最終カラーマップ
  debug              デバッグ画像
  Log\process.log    ローテーション付きログ
  progress\progress.json
  reports\report_yyyymmdd_hhmmss.xlsx
  ProgressViewer.exe 進捗ウィンドウ（経過・残り・終了予定）

使い方:
  main_twopass.exe --help
  main_twopass.exe --config data\config\default_config.json
  main_twopass.exe --input data\input --output data\output --start 1 --end 500
  main_twopass.exe --pi 250 --ppm 2.55 --debug
  ProgressViewer.exe progress\progress.json

要件: Windows 11 64bit, Tesseract-OCR 4.0+
