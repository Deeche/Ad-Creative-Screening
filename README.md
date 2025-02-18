# Ad Creative Screening

広告クリエイティブの自動審査システム

## 概要

このプロジェクトは、画像広告の自動審査を行うWebアプリケーションです。アップロードされた広告画像を分析し、コンプライアンスや品質の観点から審査を行い、結果をレポートとして提供します。

## 主な機能

- 複数画像の一括アップロードと分析
- 広告コンテンツの自動審査
- 審査結果のJSON形式での保存
- 審査結果のCSVダウンロード機能
- 審査結果の集計レポート

## 必要要件

- Python 3.8以上
- GPU（推奨）またはCPU
- 必要なPythonパッケージ（requirements.txtに記載）

## セットアップ手順

1. リポジトリのクローン
```bash
git clone [repository-url]
cd Ad-Creative-Screening
```

2. 仮想環境の作成と有効化
```bash
python -m venv venv
source venv/bin/activate  # Linuxの場合
venv\Scripts\activate     # Windowsの場合
```

3. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

4. アプリケーションの起動
```bash
python app.py
```

## 使用方法

1. ブラウザで `http://localhost:51473` にアクセス
2. 「ファイルを選択」ボタンから審査したい広告画像を選択
3. 「アップロード」ボタンをクリックして分析を開始
4. 審査結果が表示されるのを待つ
5. 必要に応じてCSVレポートをダウンロード

## 審査結果について

審査結果は以下の3段階で判定されます：
- 承認：問題なく使用可能
- 保留：確認が必要
- 却下：使用不可

各判定には以下の情報が含まれます：
- 判定結果
- 判定理由
- リスクスコア
- 違反項目（該当する場合）

## ディレクトリ構造

```
Ad-Creative-Screening/
├── app.py              # メインアプリケーション
├── backend/            # バックエンド処理
├── frontend/           # フロントエンド
├── static/            # 静的ファイル
│   ├── uploads/      # アップロードされた画像
│   └── results/      # 分析結果
├── templates/         # HTMLテンプレート
└── requirements.txt   # 依存パッケージ
```

## 注意事項

- アップロード可能なファイル形式: PNG, JPG, JPEG, GIF
- 最大ファイルサイズ: 16MB
- 結果は自動保存され、JSONファイルとして保存されます
- GPUが利用可能な場合は自動的に使用されます