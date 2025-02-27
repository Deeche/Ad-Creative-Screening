from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import numpy as np
from backend.document_manager import DocumentManager

class AdGuidelineRAG:
    def __init__(self):
        load_dotenv()
        
        # サービスアカウントの認証情報を読み込み
        credentials = service_account.Credentials.from_service_account_file(
            'credentials/ad-creative-screening-67985e10c506.json',
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Gemini Proの初期化
        vertexai.init(
            project=os.getenv("PROJECT_ID"),
            location=os.getenv("LOCATION"),
            credentials=credentials
        )
        self.model = GenerativeModel("gemini-1.5-flash-002")
        
        # ベクトル検索の初期化
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self._initialize_vector_store()

        # 日本語に特化したモデルを使用
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.document_manager = DocumentManager()
        self.guidelines = []
        self.guideline_embeddings = None
        self._initialize_guidelines()

    def _initialize_vector_store(self):
        # ガイドラインのサンプルデータ
        guidelines = []
        guidelines_dir = "data/documents/guidelines"
        examples_dir = "data/documents/examples"
        
        # ガイドラインの読み込み
        for root, _, files in os.walk(guidelines_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        guidelines.append(f.read())
        
        # 事例の読み込み
        for root, _, files in os.walk(examples_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        guidelines.append(f.read())
        
        # ドキュメントの作成
        docs = [Document(page_content=text) for text in guidelines]
        texts = self.text_splitter.split_documents(docs)
        
        # ベクトルストアの作成
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./data/chroma_db"
        )

    def _initialize_guidelines(self):
        """ガイドラインの初期化とembeddingの計算"""
        # ガイドラインの読み込み
        for doc in self.document_manager.get_document_list():
            if doc['category'] == 'guidelines':
                content = self.document_manager.get_document_content(doc['path'])
                if content:
                    # ガイドラインを行単位で分割
                    lines = [line.strip() for line in content.split('\n')
                            if line.strip() and not line.startswith('#')]
                    self.guidelines.extend(lines)

        # ガイドラインのembeddingを計算
        if self.guidelines:
            self.guideline_embeddings = self.model.encode(
                self.guidelines,
                convert_to_tensor=True,
                show_progress_bar=False
            )

    def search_relevant_guidelines(self, query: str, k: int = 3) -> List[Dict]:
        """関連するガイドラインを検索"""
        if not self.guidelines or self.guideline_embeddings is None:
            return []

        # クエリのembeddingを計算
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # コサイン類似度を計算
        cos_scores = util.cos_sim(query_embedding, self.guideline_embeddings)[0]
        
        # 上位k件の結果を取得
        top_results = []
        top_k_idx = np.argsort(cos_scores.numpy())[-k:][::-1]
        
        for idx in top_k_idx:
            score = cos_scores[idx].item()
            if score > 0.3:  # スコアが一定以上の場合のみ
                top_results.append({
                    'content': self.guidelines[idx],
                    'score': score
                })

        return top_results

    def _generate_fallback_response(self, relevant_guidelines: List[Dict]) -> Dict:
        """フォールバックレスポンスの生成"""
        if not relevant_guidelines:
            return {
                "judgement": "要確認",
                "reason": "ガイドラインとの照合ができませんでした",
                "violations": [],
                "improvements": ["広告内容の詳細な確認を推奨します"],
                "relevant_guidelines": []
            }

        return {
            "judgement": "要確認",
            "reason": "ガイドラインの確認が必要です",
            "violations": [],
            "improvements": ["関連するガイドラインに基づいて内容を確認してください"],
            "relevant_guidelines": relevant_guidelines
        }

    def analyze_with_llm(self, ad_content: str, relevant_guidelines: List[Dict]) -> Dict:
        """Gemini Proを使用して広告の詳細な分析を行う"""
        try:
            # プロンプトの作成
            prompt = f"""
あなたは広告審査の専門家です。以下の広告コンテンツを、関連するガイドラインに基づいて分析してください。

【広告コンテンツ】
{ad_content}

【関連するガイドライン】
{[g["content"] for g in relevant_guidelines]}

以下の項目について分析し、JSON形式で回答してください：

1. ガイドライン違反の有無
2. 違反の重大度（0-100のスコア）
3. 具体的な違反内容
4. 改善提案
5. 判定結果（承認/要確認/却下）

回答は以下のJSON形式で提供してください：
{{
    "has_violations": true/false,
    "violation_score": 0-100,
    "violations": ["違反1", "違反2", ...],
    "improvements": ["改善案1", "改善案2", ...],
    "judgement": "承認/要確認/却下",
    "reason": "判定理由"
}}

注意：必ず上記のJSON形式で回答してください。
"""
            # Gemini Proによる分析
            response = self.model.generate_content(prompt)
            
            # レスポンスの検証と整形
            try:
                # 文字列をPythonオブジェクトに変換
                result = eval(response.text)
                
                # 必須フィールドの存在確認
                required_fields = ["has_violations", "violation_score", "violations", 
                                "improvements", "judgement", "reason"]
                for field in required_fields:
                    if field not in result:
                        print(f"Missing field '{field}' in response: {response.text}")
                        return self._generate_fallback_response(relevant_guidelines)
                
                # judgementの値を検証
                if result["judgement"] not in ["承認", "要確認", "却下"]:
                    print(f"Invalid judgement value: {result['judgement']}")
                    return self._generate_fallback_response(relevant_guidelines)
                
                # violation_scoreの範囲を検証
                if not (0 <= result["violation_score"] <= 100):
                    result["violation_score"] = max(0, min(100, result["violation_score"]))
                
                return result
                
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                print(f"Raw response: {response.text}")
                return self._generate_fallback_response(relevant_guidelines)
                
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return self._generate_fallback_response(relevant_guidelines)

    def analyze_ad_content(self, ad_content: str) -> Dict:
        """広告コンテンツの分析"""
        try:
            # 関連するガイドラインを検索
            relevant_guidelines = self.search_relevant_guidelines(ad_content)
            
            # 違反の重大度を判定
            severity = "Low"
            violations = []
            
            for guideline in relevant_guidelines:
                if guideline['score'] > 0.7:  # 高い類似度の場合
                    severity = "High"
                    violations.append({
                        "guideline": guideline['content'],
                        "score": guideline['score']
                    })
                elif guideline['score'] > 0.5:  # 中程度の類似度の場合
                    if severity != "High":
                        severity = "Medium"
                    violations.append({
                        "guideline": guideline['content'],
                        "score": guideline['score']
                    })

            # 判定結果の生成
            if severity == "High":
                judgement = "却下"
                reason = "重大なガイドライン違反の可能性があります"
            elif severity == "Medium":
                judgement = "要確認"
                reason = "ガイドラインとの整合性の確認が必要です"
            else:
                judgement = "承認"
                reason = "特に問題は検出されませんでした"

            return {
                "judgement": judgement,
                "reason": reason,
                "violations": violations,
                "improvements": [
                    "検出された違反に基づいて広告内容を修正してください" if violations else
                    "現状の内容で問題ありません"
                ],
                "relevant_guidelines": relevant_guidelines
            }

        except Exception as e:
            print(f"Error in analyze_ad_content: {e}")
            return self._generate_fallback_response([]) 