from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

class AdGuidelineRAG:
    def __init__(self):
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
        guidelines = [
            "広告には暴力的な表現を含めることはできません。",
            "医薬品の効能・効果の表現には制限があります。",
            "個人情報の取り扱いには十分な注意が必要です。",
            "景品表示法に違反する誇大な表現は禁止されています。",
            "著作権を侵害するコンテンツは使用できません。",
        ]
        
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

    def analyze_ad_content(self, ad_content: str) -> Dict:
        """
        広告コンテンツを分析し、関連するガイドラインと違反の可能性を評価します。
        
        Args:
            ad_content: 分析する広告コンテンツ
            
        Returns:
            分析結果を含む辞書
        """
        relevant_guidelines = self.search_relevant_guidelines(ad_content)
        
        # 分析結果を返す
        return {
            "ad_content": ad_content,
            "relevant_guidelines": relevant_guidelines,
            "potential_violations": [
                guideline for guideline in relevant_guidelines
                if guideline["score"] > 0.8  # スコアが高いものを潜在的な違反として扱う
            ]
        } 