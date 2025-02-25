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

    def search_relevant_guidelines(self, query: str, k: int = 3) -> List[Dict]:
        """
        広告コンテンツに関連するガイドラインを検索します。
        
        Args:
            query: 検索クエリ（広告コンテンツの説明など）
            k: 返す結果の数
            
        Returns:
            関連するガイドラインのリスト
        """
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized")
            
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return [
            {
                "content": doc.page_content,
                "score": score
            }
            for doc, score in results
        ]

    def _generate_fallback_response(self, relevant_guidelines: List[Dict]) -> Dict:
        """LLM分析が失敗した場合のフォールバックレスポンスを生成"""
        violations = [
            f"ガイドライン違反の可能性: {guideline['content']}"
            for guideline in relevant_guidelines
            if guideline["score"] > 0.8
        ]
        
        violation_score = len(violations) * 20
        if violation_score > 60:
            judgement = "却下"
            reason = "重大なガイドライン違反が検出されました"
        elif violation_score > 30:
            judgement = "要確認"
            reason = "潜在的なガイドライン違反が検出されました"
        else:
            judgement = "承認"
            reason = "重大な違反は検出されませんでした"

        return {
            "has_violations": len(violations) > 0,
            "violation_score": min(100, violation_score),
            "violations": violations,
            "improvements": [
                "具体的な数値や根拠を示してください",
                "誇大な表現を避けてください",
                "条件や制限事項を明確に記載してください"
            ],
            "judgement": judgement,
            "reason": reason
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
        """
        広告コンテンツを分析し、関連するガイドラインと違反の可能性を評価します。
        
        Args:
            ad_content: 分析する広告コンテンツ
            
        Returns:
            分析結果を含む辞書
        """
        try:
            # 関連ガイドラインの検索
            relevant_guidelines = self.search_relevant_guidelines(ad_content)
            
            # LLMによる詳細分析
            llm_analysis = self.analyze_with_llm(ad_content, relevant_guidelines)
            
            return {
                "ad_content": ad_content,
                "relevant_guidelines": relevant_guidelines,
                "analysis": llm_analysis,
                "potential_violations": llm_analysis["violations"],
                "judgement": llm_analysis["judgement"],
                "reason": llm_analysis["reason"],
                "risk_score": llm_analysis["violation_score"],
                "improvements": llm_analysis["improvements"]
            }
            
        except Exception as e:
            print(f"Error in analyze_ad_content: {e}")
            fallback = self._generate_fallback_response(
                self.search_relevant_guidelines(ad_content)
            )
            return {
                "ad_content": ad_content,
                "relevant_guidelines": relevant_guidelines,
                "analysis": fallback,
                "potential_violations": fallback["violations"],
                "judgement": fallback["judgement"],
                "reason": fallback["reason"],
                "risk_score": fallback["violation_score"],
                "improvements": fallback["improvements"]
            } 