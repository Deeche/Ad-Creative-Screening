import json
import asyncio
import nest_asyncio
import threading
from typing import Dict, Optional, Union
from .workflow.ad_review_workflow import AdReviewWorkflow
from .image_analyzer import ImageAnalyzer

# asyncioループの入れ子を許可（Jupyterなど他のasyncioアプリケーションと共存できるようにする）
nest_asyncio.apply()

class AdReviewer:
    def __init__(self):
        self.workflow = AdReviewWorkflow()
        self.image_analyzer = ImageAnalyzer()
    
    def review_ad(self, analysis_result: Dict) -> Dict:
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
            
            # AdReviewWorkflowクラスのreview_adメソッドが非同期だが
            # 内部で同期呼び出しを行うように修正されているので、
            # 同期的に呼び出すだけでよい
            try:
                # async/awaitを使わずに直接呼び出し
                # コルーチンが返されても自動的に実行される
                review_result = self.workflow.review_ad(text)
            except Exception as e:
                print(f"Workflow execution error: {e}")
                raise
            
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

    def review_banner_ad(self, image_path: str) -> Dict:
        """
        バナー広告の審査を実行
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            審査結果を含む辞書
        """
        try:
            # 画像分析の実行
            banner_analysis = self.image_analyzer.analyze_banner_ad(image_path)
            
            if not banner_analysis["success"]:
                return {
                    "error": "画像分析中にエラーが発生しました",
                    "judgement": "判定不可",
                    "reason": banner_analysis.get("error", "不明なエラー"),
                    "risk_score": 0,
                    "violations": [],
                    "improvements": [],
                    "relevant_guidelines": [],
                    "image_analysis": banner_analysis["image_analysis"]
                }
            
            # テキストコンテンツの審査
            text_review = None
            if banner_analysis["text_content"]:
                text_review = self.review_ad({"text": banner_analysis["text_content"]})
            
            # 画像分析結果の取得
            image_analysis = banner_analysis["image_analysis"]
            
            # 総合判定
            final_judgement = self._make_final_banner_judgement(text_review, image_analysis)
            
            return {
                "judgement": final_judgement["judgement"],
                "reason": final_judgement["reason"],
                "risk_score": final_judgement["risk_score"],
                "violations": final_judgement["violations"],
                "improvements": final_judgement["improvements"],
                "text_review": text_review,
                "image_analysis": image_analysis,
                "relevant_guidelines": text_review.get("relevant_guidelines", []) if text_review else []
            }
            
        except Exception as e:
            print(f"Error in review_banner_ad: {e}")
            return {
                "error": str(e),
                "judgement": "判定不可",
                "reason": f"エラーが発生しました: {str(e)}",
                "risk_score": 0,
                "violations": [],
                "improvements": [],
                "relevant_guidelines": [],
                "image_analysis": None,
                "text_review": None
            }

    def _make_final_banner_judgement(self, text_review: Optional[Dict], image_analysis: Dict) -> Dict:
        """
        テキストと画像の分析結果から最終判定を行う
        
        Args:
            text_review: テキスト審査結果
            image_analysis: 画像分析結果
            
        Returns:
            最終判定結果
        """
        violations = []
        improvements = []
        risk_score = 0
        reasons = []
        
        # 画像分析結果の評価
        if image_analysis["inappropriate_elements"]["has_inappropriate"]:
            violations.extend(image_analysis["inappropriate_elements"]["details"])
            risk_score = max(risk_score, 
                {"High": 100, "Medium": 70, "Low": 30}[image_analysis["inappropriate_elements"]["severity"]])
            reasons.append("不適切な視覚表現が検出されました")
        
        if not image_analysis["compliance"]["is_compliant"]:
            violations.extend(image_analysis["compliance"]["violations"])
            risk_score = max(risk_score, 
                {"High": 100, "Medium": 70, "Low": 30}[image_analysis["compliance"]["risk_level"]])
            reasons.append("コンプライアンス違反が検出されました")
        
        improvements.extend(image_analysis["recommendations"])
        
        # テキスト審査結果の評価
        if text_review and not text_review.get("error"):
            violations.extend(text_review.get("violations", []))
            improvements.extend(text_review.get("improvements", []))
            risk_score = max(risk_score, text_review.get("risk_score", 0))
            if text_review.get("reason"):
                reasons.append(text_review["reason"])
        
        # 最終判定
        if risk_score >= 70:
            judgement = "却下"
        elif risk_score >= 40:
            judgement = "要確認"
        else:
            judgement = "承認"
        
        return {
            "judgement": judgement,
            "reason": " / ".join(reasons) if reasons else "特に問題は検出されませんでした",
            "risk_score": risk_score,
            "violations": list(set(violations)),  # 重複を除去
            "improvements": list(set(improvements))  # 重複を除去
        }