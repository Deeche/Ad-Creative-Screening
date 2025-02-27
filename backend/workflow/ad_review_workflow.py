from typing import Dict, List, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel
import json
import os
import logging
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv()

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
        try:
            logger.info("Gemini APIの初期化を開始します")
            
            # 環境変数の確認
            api_key = os.getenv('GOOGLE_API_KEY')
            project_id = os.getenv('PROJECT_ID')
            location = os.getenv('LOCATION')
            
            logger.info(f"環境変数: PROJECT_ID={project_id}, LOCATION={location}")
            logger.info(f"GOOGLE_API_KEY設定状況: {'設定済み' if api_key else '未設定'}")
            
            if not api_key:
                raise ValueError("GOOGLE_API_KEY環境変数が設定されていません")
            
            # LLMの初期化 - モデル名を更新
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-002",
                convert_system_message_to_human=True,
                temperature=0.3,
                google_api_key=api_key
            )
            logger.info("Gemini APIの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"Gemini API初期化エラー: {str(e)}")
            raise RuntimeError(f"Gemini APIの初期化に失敗しました: {str(e)}")

        # ベクトル検索の初期化
        try:
            # 新しい初期化方法を使用
            self.embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-multilingual-mpnet-base-v2", 
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("埋め込みモデルの初期化が完了しました")
        except Exception as e:
            logger.error(f"埋め込みモデルの初期化エラー: {str(e)}")
            # 代替方法を試みる
            from langchain_community.embeddings import HuggingFaceEmbeddings as CommunityHuggingFaceEmbeddings
            logger.info("代替埋め込みモデルを試行します")
            self.embeddings = CommunityHuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
            logger.info("代替埋め込みモデルの初期化が完了しました")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None
        self._initialize_vector_store()
        
        # ワークフローの構築
        self.workflow = self._build_workflow()

    def _initialize_vector_store(self):
        """ベクトルストアの初期化"""
        try:
            logger.info("ベクトルストアの初期化を開始")
            guidelines = self._load_guidelines()
            
            # ガイドラインをドキュメントに変換
            docs = []
            for guideline in guidelines:
                # ガイドラインを段落に分割
                paragraphs = guideline["content"].split("\n\n")
                for paragraph in paragraphs:
                    if paragraph.strip():
                        docs.append(Document(
                            page_content=paragraph.strip(),
                            metadata={"category": guideline["category"]}
                        ))
            
            # テキストの分割
            texts = self.text_splitter.split_documents(docs)
            
            # ベクトルストアの作成
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="chroma_db"
            )
            logger.info("ベクトルストアの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"ベクトルストア初期化エラー: {str(e)}")
            raise

    def _load_guidelines(self) -> List[Dict]:
        """ガイドラインをファイルから読み込む"""
        guidelines = []
        guideline_dir = "data/documents/guidelines"
        try:
            for filename in os.listdir(guideline_dir):
                if filename.endswith(".txt"):
                    with open(os.path.join(guideline_dir, filename), "r", encoding="utf-8") as f:
                        content = f.read()
                        guidelines.append({
                            "content": content,
                            "category": filename.replace(".txt", "")
                        })
            return guidelines
        except Exception as e:
            logger.error(f"ガイドライン読み込みエラー: {str(e)}")
            return []

    def _find_relevant_guidelines(self, ad_content: str, k: int = 5) -> List[Dict]:
        """関連するガイドラインを検索"""
        try:
            if not self.vector_store:
                logger.error("ベクトルストアが初期化されていません")
                return []
            
            # ベクトル検索の実行
            results = self.vector_store.similarity_search_with_relevance_scores(
                ad_content,
                k=k
            )
            
            # 結果の整形
            relevant_guidelines = []
            for doc, score in results:
                if score > 0.3:  # スコアが一定以上の場合のみ
                    relevant_guidelines.append({
                        "content": doc.page_content,
                        "category": doc.metadata.get("category", "unknown"),
                        "score": score
                    })
            
            return sorted(relevant_guidelines, key=lambda x: x["score"], reverse=True)
            
        except Exception as e:
            logger.error(f"ガイドライン検索エラー: {str(e)}")
            return []

    async def check_guidelines(self, state: AdReviewState) -> AdReviewState:
        """ガイドライン違反のチェック"""
        try:
            # 関連ガイドラインの検索
            relevant_guidelines = self._find_relevant_guidelines(state["ad_content"])

            prompt = f"""
            以下の広告コンテンツを、関連するガイドラインに基づいて分析してください。
            結果は必ずJSON形式で返してください。

            【広告コンテンツ】
            {state['ad_content']}

            【関連ガイドライン】
            {[g["content"] for g in relevant_guidelines]}

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
            # より安全な非同期呼び出し
            try:
                response = await self.llm.ainvoke(prompt)
                logger.info(f"LLMの応答: {response}")
            except RuntimeError as e:
                if "attached to a different loop" in str(e):
                    logger.warning("ループエラーが発生しました。同期的に実行を試みます。")
                    response = self.llm.invoke(prompt)
                    logger.info(f"同期的なLLMの応答: {response}")
                else:
                    raise

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

    def review_ad(self, ad_content: str) -> Dict:
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
            relevant_guidelines = self._find_relevant_guidelines(ad_content)
            initial_state["context"]["relevant_guidelines"] = relevant_guidelines

            # 非同期からの変更: ワークフローを同期的に呼び出す
            logger.info("ワークフローの実行を開始")
            try:
                # 非同期の代わりに同期呼び出しを使用
                # langgraphの同期APIを使用
                # self.workflowはcompileされたグラフなので.invokeを使う
                final_state = self.workflow.invoke(initial_state)
                logger.info("広告審査が完了しました")
            except Exception as e:
                logger.error(f"ワークフロー実行エラー: {str(e)}")
                raise
            
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

        # ノードの追加 - 非同期関数から同期関数に変更
        workflow.add_node("screening", self.sync_initial_screening)
        workflow.add_node("check_guidelines", self.sync_check_guidelines)
        workflow.add_node("analyze_severity", self.sync_analyze_severity)
        workflow.add_node("generate_improvements", self.sync_generate_improvements)
        workflow.add_node("make_decision", self.sync_make_decision)

        # エッジの定義
        workflow.add_edge("screening", "check_guidelines")
        workflow.add_edge("check_guidelines", "analyze_severity")
        workflow.add_edge("analyze_severity", "generate_improvements")
        workflow.add_edge("generate_improvements", "make_decision")
        workflow.add_edge("make_decision", END)

        # エントリーポイントの設定
        workflow.set_entry_point("screening")
        
        return workflow.compile()

    # 同期バージョンのメソッドを作成
    def sync_initial_screening(self, state: AdReviewState) -> AdReviewState:
        """初期スクリーニング（同期版）"""
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
            # 同期的に実行
            response = self.llm.invoke(prompt)
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

    def sync_check_guidelines(self, state: AdReviewState) -> AdReviewState:
        """ガイドライン違反のチェック（同期版）"""
        try:
            # 関連ガイドラインの検索
            relevant_guidelines = self._find_relevant_guidelines(state["ad_content"])

            prompt = f"""
            以下の広告コンテンツを、関連するガイドラインに基づいて分析してください。
            結果は必ずJSON形式で返してください。

            【広告コンテンツ】
            {state['ad_content']}

            【関連ガイドライン】
            {[g["content"] for g in relevant_guidelines]}

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
            # 同期的に実行
            response = self.llm.invoke(prompt)
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

    def sync_analyze_severity(self, state: AdReviewState) -> AdReviewState:
        """違反の重大度分析（同期版）"""
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
        # 同期的に実行
        response = self.llm.invoke(prompt)
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

    def sync_generate_improvements(self, state: AdReviewState) -> AdReviewState:
        """改善提案の生成（同期版）"""
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
        # 同期的に実行
        response = self.llm.invoke(prompt)
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

    def sync_make_decision(self, state: AdReviewState) -> AdReviewState:
        """最終的な判定を行う（同期版）"""
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

            # 同期的に実行
            response = self.llm.invoke(prompt)
            logger.info(f"LLMの応答: {response}")

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