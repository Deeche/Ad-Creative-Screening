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

## システムアーキテクチャ

### 1. RAGシステム（backend/rag_system.py）

RAG（Retrieval-Augmented Generation）システムは、以下のコンポーネントで構成されています：

#### ベクトルデータベース
- ChromaDBを使用して、ガイドラインと事例のベクトル化されたデータを保存
- Sentence Transformersによる埋め込み生成（all-MiniLM-L6-v2モデル使用）
- ガイドラインと事例は`data/documents`ディレクトリに保存され、自動的にベクトル化

#### 類似度検索
- 入力された広告コンテンツに対して、関連するガイドラインを検索
- コサイン類似度に基づく検索（スコア範囲: 0-1）
- 上位k件（デフォルト3件）の関連ガイドラインを返却

#### LLM分析
- Google Cloud VertexAI（Gemini Pro）を使用
- プロンプトエンジニアリングによる構造化された出力
- JSON形式での分析結果の生成

### 2. ワークフローエンジニアリング（backend/workflow/ad_review_workflow.py）

LangGraphを使用した広告審査ワークフローは、以下のステップで構成されています：

#### StateGraph
- 状態管理を行うグラフベースのワークフロー
- 各ノードは非同期で実行される独立した処理を表現
- エッジは処理の流れを定義

#### ワークフローステップ
1. **初期スクリーニング**（initial_screening）
   - 広告コンテンツの基本的なチェック
   - 禁止キーワードや不適切な表現の検出

   詳細な処理フロー：
   ```python
   # 初期スクリーニングのプロンプト
   prompt = f"""
   あなたは広告審査の専門家です。以下の広告コンテンツを初期スクリーニングしてください。

   【広告コンテンツ】
   {ad_content}

   以下の観点から分析し、JSON形式で回答してください：

   1. 明らかな禁止表現
      - 暴力的な表現
      - 差別的な表現
      - 不適切な性的表現
      - 過度に刺激的な表現
      - 恐怖を煽る表現

   2. 法令違反の可能性
      - 景品表示法違反
      - 医薬品医療機器等法違反
      - 特定商取引法違反
      - その他の法令違反

   3. 不適切な表現・主張
      - 根拠のない最上級表現
      - 誇大な表現
      - 断定的な表現
      - 比較表現
      - 安全性の保証

   回答形式：
   {
       "has_issues": true/false,          # 問題の有無
       "risk_level": "high/medium/low",   # リスクレベル
       "detected_issues": [               # 検出された問題
           {
               "type": "表現タイプ",
               "description": "具体的な問題",
               "severity": 0-100
           }
       ],
       "immediate_rejection": true/false,  # 即時却下が必要か
       "rejection_reason": "即時却下の理由"
   }
   """
   ```

   処理ステップ：
   1. **入力検証**
      - 広告コンテンツの長さチェック
      - 基本的な文字列バリデーション
      - 空白や特殊文字の正規化

   2. **LLMによる分析**
      - Gemini Proによる広告コンテンツの分析
      - 構造化されたJSON形式での結果取得
      - エラーハンドリングとリトライ処理

   3. **結果の評価**
      - 即時却下が必要な重大な違反の確認
      - リスクレベルの判定
      - 検出された問題の重大度評価

   4. **状態の更新**
      ```python
      state.update({
          "screening_result": {
              "has_issues": result["has_issues"],
              "risk_level": result["risk_level"],
              "detected_issues": result["detected_issues"],
              "immediate_rejection": result["immediate_rejection"],
              "rejection_reason": result.get("rejection_reason", "")
          }
      })
      ```

   判定基準：
   - **即時却下**（以下のいずれかに該当）
     - 暴力的な表現が含まれる
     - 差別的な表現が含まれる
     - 明らかな法令違反がある
     - 不適切な性的表現が含まれる

   - **リスクレベル判定**
     - High: 重大な問題が1つ以上、または中程度の問題が3つ以上
     - Medium: 中程度の問題が1-2つ、または軽度の問題が3つ以上
     - Low: 軽度の問題のみ、または問題なし

   エラーハンドリング：
   - LLM呼び出しの失敗時は最大3回リトライ
   - JSON解析エラー時はフォールバック処理
   - タイムアウト時は部分的な結果を返却

2. **ガイドラインチェック**（check_guidelines）
   - RAGシステムによる関連ガイドラインの検索
   - ガイドライン違反の可能性を評価

   詳細な処理フロー：
   ```python
   # ガイドラインチェックのプロンプト
   prompt = f"""
   あなたは広告審査の専門家です。以下の広告コンテンツと関連するガイドラインに基づいて、
   違反の有無を詳細に分析してください。

   【広告コンテンツ】
   {ad_content}

   【関連ガイドライン】
   {guidelines_text}

   【初期スクリーニング結果】
   {screening_result}

   以下の観点から分析し、JSON形式で回答してください：

   1. ガイドライン違反の特定
      - 各ガイドラインに対する違反の有無
      - 違反の具体的な内容
      - 該当する条項や規定

   2. 違反の文脈
      - 広告全体における位置づけ
      - 消費者への影響度
      - 誤解を招く可能性

   3. 修正の必要性
      - 必須の修正項目
      - 推奨される修正項目
      - 代替表現の可能性

   回答形式：
   {
       "violations": [
           {
               "guideline_id": "違反したガイドラインの識別子",
               "content": "違反したガイドラインの内容",
               "violation_details": "具体的な違反内容",
               "context": "違反の文脈説明",
               "impact_level": "high/medium/low",
               "required_fixes": ["必須の修正項目"],
               "recommended_fixes": ["推奨される修正項目"]
           }
       ],
       "total_violations": 0,  # 違反の総数
       "max_impact_level": "high/medium/low",  # 最も深刻な違反のレベル
       "requires_immediate_action": true/false  # 即時対応の必要性
   }
   """
   ```

   処理ステップ：
   1. **関連ガイドラインの検索**
      ```python
      # ChromaDBを使用した類似度検索
      results = self.vector_store.similarity_search_with_relevance_scores(
          ad_content,
          k=5  # 上位5件のガイドラインを取得
      )
      
      # スコアの正規化と閾値によるフィルタリング
      relevant_guidelines = [
          doc for doc, score in results
          if score > 0.7  # 類似度スコアが0.7以上のものを採用
      ]
      ```

   2. **コンテキストの構築**
      - 初期スクリーニング結果の参照
      - 関連ガイドラインの統合
      - 過去の判定事例の参照（該当する場合）

   3. **LLMによる詳細分析**
      - Gemini Proによるガイドライン違反の分析
      - 構造化されたJSON形式での結果取得
      - 違反の重大度評価

   4. **結果の検証と統合**
      ```python
      # 結果の検証
      if not result.get("violations"):
          # 違反が検出されない場合のフォールバック処理
          return self._generate_fallback_violations(relevant_guidelines)
      
      # 状態の更新
      state.update({
          "guideline_violations": result["violations"],
          "total_violations": result["total_violations"],
          "max_impact_level": result["max_impact_level"],
          "requires_immediate_action": result["requires_immediate_action"]
      })
      ```

   判定基準：
   - **違反の重大度**
     - High: ガイドラインの重要な規定に違反
     - Medium: 軽微な規定違反または解釈の余地がある違反
     - Low: 推奨事項への非準拠

   - **即時対応の必要性判断**
     - 法令違反の可能性がある場合
     - 消費者への重大な影響が予想される場合
     - 複数の重大な違反が検出された場合

   エラーハンドリング：
   - ガイドライン検索失敗時の代替処理
   - LLM応答のJSON解析エラー時の処理
   - 不完全な応答に対するフォールバック処理

   最適化：
   - ガイドラインのキャッシング
   - 類似度計算の並列処理
   - 検索結果のメモ化

3. **重大度分析**（analyze_severity）
   - 違反内容の重大度を評価し、リスクレベルを判定します。

   詳細な処理フロー：
   ```python
   # 重大度分析のプロンプト
   prompt = f"""
   あなたは広告審査の専門家です。以下の広告コンテンツとガイドライン違反について、
   その重大度を詳細に分析してください。

   【広告コンテンツ】
   {ad_content}

   【検出された違反】
   {violations}

   【初期スクリーニング結果】
   {screening_result}

   以下の項目について分析し、JSON形式で回答してください：

   1. 各違反の重大度（1-10のスコア）
   2. 総合リスクレベル（High/Medium/Low）
   3. 即時対応の必要性
   4. 具体的な影響範囲
   5. 改善の優先度

   回答は以下のJSON形式で提供してください：
   {
       "violations_severity": [
           {
               "violation": "違反内容",
               "severity_score": 1-10,
               "impact": "影響範囲の説明",
               "priority": "High/Medium/Low"
           }
       ],
       "overall_risk_level": "High/Medium/Low",
       "requires_immediate_action": true/false,
       "risk_factors": ["リスク要因1", "リスク要因2"],
       "severity_reasons": ["重大度判定の理由1", "理由2"]
   }
   """
   ```

   判定基準：
   1. **重大度スコア（1-10）**
      - 10: 法令違反、即時対応必須
      - 7-9: 重大な違反、早急な対応が必要
      - 4-6: 中程度の違反、改善が必要
      - 1-3: 軽微な違反、改善が望ましい

   2. **リスクレベル判定**
      - High: 以下のいずれかに該当
        - 単一の違反で重大度8以上
        - 複数の違反で重大度6以上の合計が3つ以上
        - 法令違反の可能性がある
      - Medium: 以下のいずれかに該当
        - 重大度4-7の違反が2つ以上
        - 複数の軽微な違反の累積
      - Low: 以下の場合
        - すべての違反が重大度3以下
        - 技術的な改善事項のみ

   3. **即時対応の判定基準**
      - 法令違反の可能性
      - 重大な誤認を招く可能性
      - ブランドイメージへの重大な影響
      - 社会的影響の大きい問題

   エラーハンドリング：
   - LLM応答のタイムアウト時は部分的な結果を返却
   - 解析不能な応答の場合は保守的な判定を実施
   - 重大度スコアの正規化による一貫性の確保

4. **改善提案生成**（generate_improvements）
   - 検出された違反に対する具体的な改善案を生成します。

   詳細な処理フロー：
   ```python
   # 改善提案生成のプロンプト
   prompt = f"""
   あなたは広告審査の専門家です。以下の広告コンテンツと検出された違反について、
   具体的な改善提案を生成してください。

   【広告コンテンツ】
   {ad_content}

   【検出された違反】
   {violations}

   【重大度分析結果】
   {severity_analysis}

   以下の項目について分析し、JSON形式で回答してください：

   1. 各違反に対する具体的な改善案
   2. 代替表現の提案
   3. 優先順位付けされた修正手順
   4. 改善後の効果予測
   5. 追加の推奨事項

   回答は以下のJSON形式で提供してください：
   {
       "improvements": [
           {
               "violation": "違反内容",
               "current_expression": "現在の表現",
               "suggested_fixes": [
                   {
                       "fix": "改善案",
                       "reason": "改善理由",
                       "expected_impact": "期待される効果"
                   }
               ],
               "alternative_expressions": ["代替表現1", "代替表現2"],
               "priority": "High/Medium/Low"
           }
       ],
       "general_recommendations": ["全般的な改善提案1", "改善提案2"],
       "implementation_steps": ["手順1", "手順2"],
       "expected_outcomes": {
           "compliance_improvement": "コンプライアンス改善効果",
           "risk_reduction": "リスク低減効果",
           "message_effectiveness": "メッセージ効果の維持・向上"
       }
   }
   """
   ```

   改善提案の生成基準：
   1. **表現の改善**
      - 誇大表現の適切な表現への置き換え
      - 具体的な数値やデータの追加
      - 条件や制限事項の明確化
      - 客観的な表現への修正

   2. **構造的な改善**
      - 情報の優先順位付け
      - 重要な情報の強調
      - レイアウトや構成の改善
      - 文脈の明確化

   3. **コンプライアンス対応**
      - 法令遵守の観点からの修正
      - 業界ガイドラインへの準拠
      - 必要な免責事項の追加
      - 表示要件の充足

   4. **代替表現の提案基準**
      - オリジナルのメッセージ性の維持
      - ターゲット層への適切性
      - 実現可能性の考慮
      - ブランドトーンの一貫性

   優先順位付けの基準：
   - **High Priority**
     - 法令違反の修正
     - 重大な誤解を招く表現の修正
     - 即時対応が必要な問題の修正

   - **Medium Priority**
     - 誤解を招く可能性のある表現の改善
     - 表示要件の充足
     - 明確性の向上

   - **Low Priority**
     - 表現の最適化
     - 追加情報の提供
     - 視認性の向上

   実装手順の生成：
   1. 重大度に基づく修正順序の決定
   2. 依存関係を考慮した手順の組み立て
   3. 実現可能性の評価
   4. 段階的な改善プロセスの提案

   効果予測の評価基準：
   - コンプライアンス改善度
   - リスク低減効果
   - メッセージの効果維持
   - 実装の容易さ

   エラーハンドリング：
   - LLM応答タイムアウト時の部分的提案生成
   - 不完全な応答時の基本改善案の提供
   - 提案生成失敗時のフォールバック処理

5. **最終判定**（make_decision）
   - 収集した情報に基づいて最終的な判断を行います。

   詳細な処理フロー：
   ```python
   # 最終判定のプロンプト
   prompt = f"""
   あなたは広告審査の最終判定者です。以下の情報に基づいて、広告の最終判定を行ってください。

   【広告コンテンツ】
   {ad_content}

   【初期スクリーニング結果】
   {screening_result}

   【ガイドライン違反】
   {violations}

   【重大度分析】
   {severity_analysis}

   【改善提案】
   {improvements}

   以下の項目について分析し、JSON形式で回答してください：

   1. 最終判定（承認/要確認/却下）
   2. 判定理由の詳細説明
   3. 信頼度スコア（0-100）
   4. 追加の注意事項や条件
   5. フォローアップ推奨事項

   回答は以下のJSON形式で提供してください：
   {
       "decision": "承認/要確認/却下",
       "confidence_score": 0-100,
       "primary_reason": "主たる判定理由",
       "detailed_reasons": [
           {
               "reason": "理由の説明",
               "weight": "High/Medium/Low",
               "evidence": "判断の根拠"
           }
       ],
       "conditions": ["条件1", "条件2"],  # 承認/要確認の場合の条件
       "follow_up_actions": ["アクション1", "アクション2"],
       "risk_assessment": {
           "legal_risk": "High/Medium/Low",
           "brand_risk": "High/Medium/Low",
           "operational_risk": "High/Medium/Low"
       }
   }
   """
   ```

   判定基準：
   1. **承認**（以下のすべてを満たす）
      - 法令違反がない
      - 重大な違反がない（重大度7以上の違反なし）
      - 軽微な違反が2件以下
      - 改善提案が実装済みまたは不要
      - 信頼度スコアが80以上

   2. **要確認**（以下のいずれかに該当）
      - 中程度の違反（重大度4-6）が存在
      - 複数の軽微な違反が累積
      - 改善提案の実装が必要
      - 判断に迷う要素がある
      - 信頼度スコアが50-79

   3. **却下**（以下のいずれかに該当）
      - 法令違反がある
      - 重大な違反（重大度7以上）がある
      - 改善が困難な問題がある
      - 複数の中程度の違反が存在
      - 信頼度スコアが50未満

   信頼度スコアの算出基準：
   1. **基本スコア（0-50ポイント）**
      - 判定の一貫性: 20ポイント
      - 証拠の質: 15ポイント
      - データの完全性: 15ポイント

   2. **補正要素（±30ポイント）**
      - 過去の類似判定との整合性: ±10
      - ガイドラインの明確性: ±10
      - コンテキストの理解度: ±10

   3. **リスク評価（±20ポイント）**
      - 法的リスク: ±10
      - ブランドリスク: ±5
      - オペレーショナルリスク: ±5

   実装フロー：
   1. **データの集約と検証**
      ```python
      def aggregate_data(self, state: State) -> Dict:
          return {
              "screening": state.screening_result,
              "violations": state.guideline_violations,
              "severity": state.severity_analysis,
              "improvements": state.improvements,
              "context": state.context
          }
      ```

   2. **判定ロジックの適用**
      ```python
      def apply_decision_rules(self, aggregated_data: Dict) -> str:
          if self._has_critical_violations(aggregated_data):
              return "却下"
          elif self._needs_confirmation(aggregated_data):
              return "要確認"
          return "承認"
      ```

   3. **信頼度スコアの計算**
      ```python
      def calculate_confidence(self, data: Dict, decision: str) -> float:
          base_score = self._calculate_base_score(data)
          adjustments = self._apply_adjustments(data, decision)
          risk_score = self._evaluate_risks(data)
          return min(100, max(0, base_score + adjustments + risk_score))
      ```

   エラーハンドリング：
   - 判定に必要なデータが不足している場合の処理
   - 矛盾する判定結果の検出と解決
   - 信頼度スコアが低い場合の警告
   - システムエラー時のフォールバック判定

   最適化とパフォーマンス：
   - 判定ルールのキャッシング
   - 並列処理による高速化
   - メモ化による計算の効率化
   - バッチ処理のサポート

### 3. 広告レビューシステム（backend/ad_reviewer.py）

#### 主な機能
- ワークフローの実行管理
- 非同期処理のハンドリング
- エラー処理とフォールバック
- 結果のフォーマット

#### エラーハンドリング
- 空の入力チェック
- 非同期実行時のエラー処理
- タイムアウト処理
- 結果の検証

### 4. ドキュメント管理（backend/document_manager.py）

#### 機能
- ガイドラインと事例の管理
- ファイルシステムベースのストレージ
- カテゴリ分類（ガイドライン/違反事例/適切な事例）
- CRUD操作のサポート

#### ディレクトリ構造
```
data/documents/
├── guidelines/      # ガイドライン文書
├── examples/
│   ├── violations/  # 違反事例
│   └── compliant/   # 適切な事例
```

## システムフロー

1. **入力受付**
   - Webインターフェースで広告テキストを受付
   - バリデーションチェック

2. **RAG処理**
   - 広告テキストのベクトル化
   - 関連ガイドラインの検索
   - 類似度スコアの計算

3. **ワークフロー実行**
   - 各ステップの非同期実行
   - 状態の更新と管理
   - エラーハンドリング

4. **結果生成**
   - 判定結果の集約
   - レスポンスの構造化
   - フロントエンドへの返却

5. **結果表示**
   - 判定結果の表示
   - 違反内容の詳細表示
   - 改善提案の提示

### 5. バナー広告審査システム（backend/image_analyzer.py）

バナー広告の審査は、テキストと画像の両方の要素を分析し、総合的な判断を行います。

#### テキスト抽出と分析
1. **OCR処理**
   - EasyOCRを使用して画像からテキストを抽出
   - 日本語と英語の両方に対応
   - 抽出されたテキストは既存の広告審査ワークフローで処理

2. **画像分析**
   - Gemini Pro Vision APIを使用した画像内容の分析
   - 以下の観点から評価：
     - 全体的な印象
     - 不適切な要素の有無
     - ブランドガイドラインとの整合性
     - 視覚的な効果とインパクト
     - 配色やレイアウトの適切性
     - 画像の品質と解像度

#### 画像分析の評価基準
1. **不適切な要素のチェック**
   - 暴力的な表現
   - 性的な表現
   - 差別的な表現
   - 過度に刺激的な表現

2. **視覚要素の評価**
   - 配色の適切性
   - レイアウトの効果
   - 画質の基準適合性

3. **コンプライアンス評価**
   - ブランドガイドラインへの準拠
   - 法的規制への適合
   - 業界標準の遵守

#### 総合判定プロセス
1. **テキスト審査**
   - 抽出されたテキストに対する通常の広告審査
   - ガイドライン違反のチェック
   - リスクスコアの算出

2. **画像審査**
   - 視覚的要素の分析
   - 不適切なコンテンツの検出
   - コンプライアンス評価

3. **統合評価**
   - テキストと画像の分析結果を統合
   - 総合リスクスコアの算出
   - 最終判定（承認/要確認/却下）の決定

#### 判定基準
1. **却下**（以下のいずれかに該当）
   - リスクスコア70以上
   - 重大な不適切要素の検出
   - 深刻なコンプライアンス違反

2. **要確認**（以下のいずれかに該当）
   - リスクスコア40-69
   - 中程度の不適切要素
   - 軽微なコンプライアンス違反

3. **承認**
   - リスクスコア40未満
   - 不適切要素なし
   - コンプライアンス違反なし

#### エラーハンドリング
1. **テキスト抽出エラー**
   - OCR処理の失敗
   - テキスト認識精度の低下
   - 文字化けや不明な文字

2. **画像分析エラー**
   - 画像読み込みエラー
   - API通信エラー
   - 解析不能な画像形式

3. **フォールバック処理**
   - エラー時の代替分析結果
   - 保守的な判定の適用
   - エラーログの記録

#### パフォーマンス最適化
1. **画像処理**
   - 画像サイズの最適化
   - キャッシング
   - 並列処理

2. **API呼び出し**
   - レート制限の管理
   - リトライ処理
   - 結果のキャッシング