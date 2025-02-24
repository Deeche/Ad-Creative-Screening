# Ad Creative Screening

広告クリエイティブの自動審査システム

## 概要

このプロジェクトは、広告クリエイティブの自動審査を行うWebアプリケーションです。アップロードされた広告（テキストおよび画像）を分析し、コンプライアンスや品質の観点から審査を行い、結果をレポートとして提供します。

## 主な機能

- 複数広告の一括審査
- RAG（Retrieval-Augmented Generation）を使用した広告ガイドライン違反の検出
- 広告コンテンツの自動審査
- 審査結果のJSON形式での保存
- 審査結果のCSVダウンロード機能
- 審査結果の集計レポート

## 開発環境

### 必要要件

- Docker Desktop
- Python 3.8以上（ローカル開発用）
- Git

### Docker環境のセットアップ

1. リポジトリのクローン
```bash
git clone [repository-url]
cd Ad-Creative-Screening
```

2. Dockerイメージのビルドと起動
```bash
docker-compose up --build
```

これにより以下の環境が構築されます：
- Python 3.9環境
- 必要なPythonパッケージ（requirements.txtに記載）
- ChromaDBのベクトルデータベース
- テスト実行環境

### ローカル開発環境のセットアップ

1. 仮想環境の作成と有効化
```bash
python -m venv venv
source venv/bin/activate  # Linuxの場合
venv\Scripts\activate     # Windowsの場合
```

2. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

## Docker環境でできること

1. **テストの実行**
```bash
# すべてのテストを実行
docker-compose up

# 特定のテストを実行
docker-compose run app python -m unittest backend/test_rag_system.py -k test_guideline_search
```

2. **開発環境での作業**
```bash
# コンテナ内でシェルを起動
docker-compose run app bash
```

3. **ベクトルデータベースの永続化**
- `data/chroma_db`ディレクトリにベクトルDBのデータが保存されます
- Dockerボリュームによってデータは永続化されます

## アプリケーションの実行

### Docker環境での実行
```bash
# アプリケーションの起動
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### ローカル環境での実行
```bash
python app.py
```

## 使用方法

1. ブラウザで `http://localhost:51473` にアクセス
2. 広告コンテンツ（テキストまたは画像）を入力
3. 「審査開始」ボタンをクリックして分析を開始
4. 審査結果が表示されるのを待つ
5. 必要に応じてCSVレポートをダウンロード

## 審査結果について

審査結果は以下の3段階で判定されます：
- 承認：問題なく使用可能
- 要確認：確認が必要
- 却下：使用不可

各判定には以下の情報が含まれます：
- 判定結果
- 判定理由
- リスクスコア
- 違反項目（該当する場合）
- 関連するガイドライン

## ディレクトリ構造

```
Ad-Creative-Screening/
├── app.py              # メインアプリケーション
├── backend/            # バックエンド処理
│   ├── ad_reviewer.py  # 広告審査ロジック
│   ├── rag_system.py   # RAGシステム
│   └── image_analyzer.py # 画像分析
├── static/            # 静的ファイル
│   ├── uploads/      # アップロードされた画像
│   └── results/      # 分析結果
├── templates/         # HTMLテンプレート
├── data/             # データ保存ディレクトリ
│   └── chroma_db/    # ベクトルDB
├── Dockerfile        # Dockerイメージ定義
├── docker-compose.yml # Docker構成
└── requirements.txt   # 依存パッケージ
```

## 開発ガイドライン

1. 新機能の開発は必ずfeatureブランチを作成して行う
```bash
git checkout -b feature/機能名
```

2. コードの変更前にテストを作成
3. 変更後、すべてのテストが通ることを確認
4. プルリクエストを作成してレビューを依頼

## 注意事項

- アップロード可能なファイル形式: PNG, JPG, JPEG, GIF
- 最大ファイルサイズ: 16MB
- 結果は自動保存され、JSONファイルとして保存されます
- ベクトルDBのデータは`data/chroma_db`ディレクトリに保存されます