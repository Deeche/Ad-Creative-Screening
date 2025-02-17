import json

class AdReviewer:
    def __init__(self):
        pass
    
    def review_ad(self, analysis_result):
        """広告の審査を実行（ダミー実装）"""
        try:
            # テキストに基づく簡単なキーワードチェック
            text = analysis_result.get('text', '').lower()
            risk_words = ['危険', '無料', '簡単', '儲かる', '即金', '必ず']
            violations = [word for word in risk_words if word in text]
            
            # リスクスコアに基づく判定
            base_score = analysis_result.get('risk_score', 50)
            final_score = base_score + (len(violations) * 10)
            final_score = min(100, max(0, final_score))
            
            if final_score < 30:
                judgement = "承認"
                reason = "リスクは低いと判断されました"
            elif final_score < 70:
                judgement = "保留"
                reason = "要確認項目が見つかりました"
            else:
                judgement = "却下"
                reason = "リスクが高いと判断されました"
            
            result = {
                "judgement": judgement,
                "reason": reason,
                "risk_score": final_score,
                "violations": violations
            }
            return result
        except Exception as e:
            print(f"Error in review_ad: {e}")
            return None