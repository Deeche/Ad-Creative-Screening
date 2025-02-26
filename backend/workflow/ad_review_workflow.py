from typing import Dict, List, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import VertexAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
import json
import os
import logging
from google.oauth2 import service_account
from google.cloud import aiplatform

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdReviewState(TypedDict):
    """広告審査の状態を表すクラス"""
    ad_content: Annotated[str, "広告コンテンツ"]
    screening_result: Annotated[Dict, "スクリーニング結果"]
    guideline_violations: Annotated[List[Dict], "ガイドライン違反"]
    severity_analysis: Annotated[Dict, "重大度分析"]
    improvements: Annotated[List[str], "改善提案"]
    final_decision: Annotated[Dict, "最終判定"]
    confidence: Annotated[float, "信頼度"]
    context: Annotated[Dict, "コンテキスト"]

class AdReviewWorkflow:
    def __init__(self):
        # Google Cloud認証の初期化
        try:
            logger.info("Google Cloud認証を開始します")
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'credentials/ad-creative-screening-67985e10c506.json')
            logger.info(f"認証ファイルのパス: {credentials_path}")
            
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"認証ファイルが見つかりません: {credentials_path}")
            
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Vertex AI の初期化
            project_id = os.getenv('PROJECT_ID', 'ad-creative-screening')
            location = os.getenv('LOCATION', 'asia-northeast1')
            
            logger.info(f"Vertex AIを初期化: project_id={project_id}, location={location}")
            
            aiplatform.init(
                project=project_id,
                location=location,
                credentials=credentials
            )
            
            # LLMの初期化
            self.llm = VertexAI(
                model_name="gemini-1.5-flash-002",
                project=project_id,
                location=location,
                credentials=credentials
            )
            logger.info("Google Cloud認証が完了しました")
            
        except Exception as e:
            logger.error(f"Google Cloud認証エラー: {str(e)}")
            raise RuntimeError(f"Google Cloud認証に失敗しました: {str(e)}")
        
        try:
            # エンベッディングモデルの初期化
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # ベクトルストアの初期化
            self.vector_store = Chroma(
                persist_directory="./data/chroma_db",
                embedding_function=self.embeddings
            )
            logger.info("モデルの初期化が完了しました")
        except Exception as e:
            logger.error(f"モデル初期化エラー: {str(e)}")
            raise RuntimeError(f"モデルの初期化に失敗しました: {str(e)}")
        
        # ワークフローの構築
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """ワークフローグラフの構築"""
        # ステートの型を定義
        class State(TypedDict):
            ad_content: str
            screening_result: Dict
            guideline_violations: List[Dict]
            severity_analysis: Dict
            improvements: List[str]
            final_decision: Dict
            confidence: float
            context: Dict

        # ワークフローの構築
        workflow = StateGraph(state_schema=State)

        # ノードの追加
        workflow.add_node("screening", self.initial_screening)
        workflow.add_node("check_guidelines", self.check_guidelines)
        workflow.add_node("analyze_severity", self.analyze_severity)
        workflow.add_node("generate_improvements", self.generate_improvements)
        workflow.add_node("make_decision", self.make_decision)

        # エッジの定義
        workflow.add_edge("screening", "check_guidelines")
        workflow.add_edge("check_guidelines", "analyze_severity")
        workflow.add_edge("analyze_severity", "generate_improvements")
        workflow.add_edge("generate_improvements", "make_decision")
        workflow.add_edge("make_decision", END)

        # エントリーポイントの設定
        workflow.set_entry_point("screening")
        
        return workflow.compile()

    async def initial_screening(self, state: AdReviewState) -> AdReviewState:
        """初期スクリーニング"""
        try:
            prompt = f"""
            以下の広告コンテンツの初期スクリーニングを行ってください。
            結果は必ずJSON形式で返してください。

            広告コンテンツ：
            {state['ad_content']}

            以下の項目を評価してJSON形式で回答してください：
            {{
                "obvious_violations": true/false,
                "warning_keywords": ["キーワード1", "キーワード2"],
                "category": "広告カテゴリ"
            }}
            """

            logger.info("初期スクリーニングを開始")
            response = await self.llm.ainvoke(prompt)
            logger.info(f"LLMの応答: {response}")

            try:
                if hasattr(response, 'content'):
                    logger.info("Message型の応答を処理")
                    content = response.content
                else:
                    logger.info("文字列型の応答を処理")
                    content = str(response)

                # 応答から余分な空白や改行を削除
                content = content.strip()
                
                # JSON部分を抽出（応答に余分なテキストが含まれている可能性がある）
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    logger.info(f"抽出したJSON文字列: {json_str}")
                    state["screening_result"] = json.loads(json_str)
                else:
                    raise ValueError("JSON形式の応答が見つかりません")
            except Exception as json_error:
                logger.error(f"JSON解析エラー: {json_error}")
                raise

            return state
        except Exception as e:
            print(f"初期スクリーニングエラー: {e}")
            state["screening_result"] = {
                "error": str(e),
                "obvious_violations": True,
                "warning_keywords": [],
                "category": "エラー"
            }
            return state

    def should_escalate(self, state: AdReviewState) -> str:
        """エスカレーションの判断"""
        screening = state["screening_result"]  # キー名を変更
        if screening.get("obvious_violations", False):
            return "escalate"
        return "continue"

    async def check_guidelines(self, state: AdReviewState) -> AdReviewState:
        """ガイドライン違反のチェック"""
        try:
            # 関連ガイドラインの検索
            results = self.vector_store.similarity_search_with_relevance_scores(
                state["ad_content"],
                k=5
            )

            # スコアを0-1の範囲に正規化
            normalized_results = [
                (doc, abs(score) if score < 0 else score)
                for doc, score in results
            ]

            prompt = f"""
            以下の広告コンテンツを、関連するガイドラインに基づいて分析してください。
            結果は必ずJSON形式で返してください。

            【広告コンテンツ】
            {state['ad_content']}

            【関連ガイドライン】
            {[doc.page_content for doc, _ in normalized_results]}

            以下の形式でJSON形式で回答してください：
            [
                {{
                    "violation": "違反内容の説明",
                    "guideline": "関連するガイドライン",
                    "severity": "高/中/低"
                }}
            ]
            """

            logger.info("ガイドライン違反チェックを開始")
            response = await self.llm.ainvoke(prompt)
            logger.info(f"LLMの応答: {response}")

            try:
                if hasattr(response, 'content'):
                    logger.info("Message型の応答を処理")
                    content = response.content
                else:
                    logger.info("文字列型の応答を処理")
                    content = str(response)

                # 応答から余分な空白や改行を削除
                content = content.strip()
                
                # JSON部分を抽出
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    logger.info(f"抽出したJSON文字列: {json_str}")
                    state["guideline_violations"] = json.loads(json_str)
                else:
                    raise ValueError("JSON形式の応答が見つかりません")
            except Exception as json_error:
                logger.error(f"JSON解析エラー: {json_error}")
                raise

            return state
        except Exception as e:
            logger.error(f"ガイドライン違反チェックエラー: {str(e)}")
            state["guideline_violations"] = []
            return state

    async def analyze_severity(self, state: AdReviewState) -> AdReviewState:
        """違反の重大度分析"""
        violations = state["guideline_violations"]
        prompt = f"""
        以下の違反項目の重大度を分析してください。
        結果は必ずJSON形式で返してください。

        違反項目：
        {json.dumps(violations, ensure_ascii=False)}

        以下の形式でJSON形式で回答してください：
        {{
            "violations": [
                {{
                    "violation": "違反内容",
                    "severity_score": 0-100,
                    "impact": "影響範囲の説明",
                    "risk": "リスク評価"
                }}
            ]
        }}
        """

        logger.info("重大度分析を開始")
        response = await self.llm.ainvoke(prompt)
        logger.info(f"LLMの応答: {response}")

        try:
            if hasattr(response, 'content'):
                logger.info("Message型の応答を処理")
                content = response.content
            else:
                logger.info("文字列型の応答を処理")
                content = str(response)

            # 応答から余分な空白や改行を削除
            content = content.strip()
            
            # JSON部分を抽出
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                logger.info(f"抽出したJSON文字列: {json_str}")
                state["severity_analysis"] = json.loads(json_str)
            else:
                raise ValueError("JSON形式の応答が見つかりません")
        except Exception as json_error:
            logger.error(f"JSON解析エラー: {json_error}")
            raise

        return state

    async def generate_improvements(self, state: AdReviewState) -> AdReviewState:
        """改善提案の生成"""
        violations = state["guideline_violations"]
        severity = state["severity_analysis"]

        prompt = f"""
        以下の違反項目と重大度分析に基づいて、具体的な改善提案を生成してください。
        結果は必ずJSON形式で返してください。

        【違反項目】
        {json.dumps(violations, ensure_ascii=False)}

        【重大度分析】
        {json.dumps(severity, ensure_ascii=False)}

        以下の形式でJSON形式で回答してください：
        {{
            "improvements": [
                "改善提案1",
                "改善提案2"
            ]
        }}
        """

        logger.info("改善提案の生成を開始")
        response = await self.llm.ainvoke(prompt)
        logger.info(f"LLMの応答: {response}")

        try:
            if hasattr(response, 'content'):
                logger.info("Message型の応答を処理")
                content = response.content
            else:
                logger.info("文字列型の応答を処理")
                content = str(response)

            # 応答から余分な空白や改行を削除
            content = content.strip()
            
            # JSON部分を抽出
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                logger.info(f"抽出したJSON文字列: {json_str}")
                result = json.loads(json_str)
                state["improvements"] = result.get("improvements", [])
            else:
                raise ValueError("JSON形式の応答が見つかりません")
        except Exception as json_error:
            logger.error(f"JSON解析エラー: {json_error}")
            raise

        return state

    async def make_decision(self, state: AdReviewState) -> AdReviewState:
        """
        最終的な判定を行う
        """
        try:
            logger.info("最終判定を開始")
            
            # 判定用のプロンプトを構築
            prompt = f"""
            以下の広告審査結果に基づいて、最終的な判定を行ってください。
            結果は必ずJSON形式で返してください。

            【審査結果】
            初期スクリーニング: {state['screening_result']}
            ガイドライン違反: {state['guideline_violations']}
            重大度分析: {state['severity_analysis']}
            改善提案: {state['improvements']}

            以下の形式でJSON形式で回答してください：
            {{
                "judgement": "承認/要確認/却下",
                "reason": "判定理由の説明",
                "confidence": 0-100の数値
            }}
            """

            # 非同期でLLMを呼び出し
            response = await self.llm.ainvoke(prompt)
            logger.info(f"LLMの応答: {response}")

            # レスポンスの処理
            try:
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)

                # JSON部分を抽出
                content = content.strip()
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    decision = json.loads(json_str)
                else:
                    raise ValueError("JSON形式の応答が見つかりません")

                # 結果を状態に追加
                state['final_decision'] = decision
                state['confidence'] = float(decision.get('confidence', 50))

            except json.JSONDecodeError as e:
                logger.error(f"JSON解析エラー: {e}")
                state['final_decision'] = {
                    "judgement": "判定不可",
                    "reason": "応答の解析に失敗しました",
                    "confidence": 0
                }
            except Exception as e:
                logger.error(f"予期せぬエラー: {e}")
                state['final_decision'] = {
                    "judgement": "判定不可",
                    "reason": f"エラーが発生しました: {str(e)}",
                    "confidence": 0
                }

        except Exception as e:
            logger.error(f"make_decision処理中のエラー: {e}")
            state['final_decision'] = {
                "judgement": "判定不可",
                "reason": f"判定処理中にエラーが発生しました: {str(e)}",
                "confidence": 0
            }

        return state

    async def review_ad(self, ad_content: str) -> Dict:
        """広告の審査を実行"""
        try:
            logger.info(f"広告審査を開始: {ad_content[:100]}...")
            
            # 初期状態の作成
            initial_state = {
                "ad_content": ad_content,
                "screening_result": {},
                "guideline_violations": [],
                "severity_analysis": {},
                "improvements": [],
                "final_decision": {},
                "confidence": 0.0,
                "context": {}
            }

            # 関連ガイドラインを検索
            relevant_guidelines = []
            try:
                logger.info("関連ガイドラインの検索を開始")
                results = self.vector_store.similarity_search_with_relevance_scores(
                    ad_content,
                    k=5
                )
                normalized_results = [
                    {"content": doc.page_content, "score": abs(score) if score < 0 else score}
                    for doc, score in results
                ]
                relevant_guidelines = normalized_results
                logger.info(f"関連ガイドライン検索結果: {len(relevant_guidelines)}件")
                
                # コンテキストに関連ガイドラインを追加
                initial_state["context"]["relevant_guidelines"] = relevant_guidelines
                
            except Exception as e:
                logger.error(f"ガイドライン検索エラー: {str(e)}")
                relevant_guidelines = []

            # ワークフローの実行
            logger.info("ワークフローの実行を開始")
            final_state = await self.workflow.ainvoke(initial_state)
            logger.info("広告審査が完了しました")
            
            return {
                "judgement": final_state["final_decision"].get("judgement", "判定不可"),
                "reason": final_state["final_decision"].get("reason", "判定理由を取得できませんでした"),
                "confidence": final_state["final_decision"].get("confidence", 0),
                "violations": final_state["guideline_violations"],
                "severity": final_state["severity_analysis"],
                "improvements": final_state["improvements"],
                "relevant_guidelines": relevant_guidelines
            }
        except Exception as e:
            logger.error(f"広告審査エラー: {str(e)}")
            return {
                "error": str(e),
                "judgement": "判定不可",
                "reason": f"審査処理中にエラーが発生しました: {str(e)}",
                "confidence": 0,
                "violations": [],
                "improvements": [],
                "relevant_guidelines": []
            } 