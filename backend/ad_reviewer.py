import json
from .rag_system import AdGuidelineRAG

class AdReviewer:
    def __init__(self):
        self.rag_system = AdGuidelineRAG()
    
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
            
            # RAGシステムを使用してガイドライン違反をチェック
            rag_analysis = self.rag_system.analyze_ad_content(text)
            potential_violations = rag_analysis["potential_violations"]
            
            # 違反の重大度に基づいてリスクスコアを計算
            violation_score = len(potential_violations) * 20
            base_score = analysis_result.get('risk_score', 50)
            final_score = base_score + violation_score
            final_score = min(100, max(0, final_score))
            
            # 判定とその理由を決定
            if final_score < 30:
                judgement = "承認"
                reason = "重大な違反は検出されませんでした"
            elif final_score < 70:
                judgement = "要確認"
                reason = "潜在的な違反が検出されました"
            else:
                judgement = "却下"
                reason = "重大な違反が検出されました"
            
            # 審査結果の作成
            result = {
                "judgement": judgement,
                "reason": reason,
                "risk_score": final_score,
                "violations": [v["content"] for v in potential_violations],
                "relevant_guidelines": rag_analysis["relevant_guidelines"]
            }
            
            return result
            
        except Exception as e:
            print(f"Error in review_ad: {e}")
            return None