import json
import asyncio
from .workflow.ad_review_workflow import AdReviewWorkflow

class AdReviewer:
    def __init__(self):
        self.workflow = AdReviewWorkflow()
    
    def review_ad(self, analysis_result):
        """
        広告の審査を実行
        
        Args:
            analysis_result: 広告分析の結果（テキスト、画像分析結果など）
            
        Returns:
            審査結果を含む辞書
        """
        try:
            # テキストの取得
            text = analysis_result.get('text', '')
            if not text:
                return {
                    "error": "広告テキストが空です",
                    "judgement": "判定不可",
                    "reason": "広告テキストが提供されていません",
                    "risk_score": 0,
                    "violations": [],
                    "improvements": [],
                    "relevant_guidelines": []
                }
            
            # 非同期関数を同期的に実行
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            review_result = loop.run_until_complete(
                self.workflow.review_ad(text)
            )
            
            if review_result is None:
                return {
                    "error": "審査処理中にエラーが発生しました",
                    "judgement": "判定不可",
                    "reason": "システムエラーが発生しました",
                    "risk_score": 0,
                    "violations": [],
                    "improvements": [],
                    "relevant_guidelines": []
                }
            
            return {
                "judgement": review_result["judgement"],
                "reason": review_result["reason"],
                "risk_score": review_result.get("confidence", 50),
                "violations": review_result.get("violations", []),
                "improvements": review_result.get("improvements", []),
                "relevant_guidelines": review_result.get("relevant_guidelines", [])
            }
            
        except Exception as e:
            print(f"Error in review_ad: {e}")
            return {
                "error": str(e),
                "judgement": "判定不可",
                "reason": f"エラーが発生しました: {str(e)}",
                "risk_score": 0,
                "violations": [],
                "improvements": [],
                "relevant_guidelines": []
            }