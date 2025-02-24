import unittest
from backend.rag_system import AdGuidelineRAG
from backend.ad_reviewer import AdReviewer

class TestAdGuidelineRAG(unittest.TestCase):
    def setUp(self):
        self.rag = AdGuidelineRAG()
        self.reviewer = AdReviewer()

    def test_guideline_search(self):
        """ガイドライン検索の精度をテストします"""
        # テストケース
        test_cases = [
            {
                "query": "この広告では暴力的なシーンを使用しています",
                "expected_guideline": "広告には暴力的な表現を含めることはできません。",
                "should_violate": True
            },
            {
                "query": "この商品は100%効果があることを保証します",
                "expected_guideline": "景品表示法に違反する誇大な表現は禁止されています。",
                "should_violate": True
            },
            {
                "query": "この健康食品で病気が治ります",
                "expected_guideline": "医薬品の効能・効果の表現には制限があります。",
                "should_violate": True
            },
            {
                "query": "商品の特徴を説明する一般的な広告です",
                "should_violate": False
            }
        ]

        for test_case in test_cases:
            query = test_case["query"]
            results = self.rag.search_relevant_guidelines(query)
            
            if test_case.get("should_violate", False):
                # 期待されるガイドラインが上位の結果に含まれているか確認
                expected_guideline = test_case["expected_guideline"]
                found = any(result["content"] == expected_guideline for result in results)
                self.assertTrue(found, f"期待されるガイドライン '{expected_guideline}' が検出されませんでした")
            else:
                # 違反がないケースでは、関連スコアが低いことを確認
                for result in results:
                    self.assertLess(result["score"], 0.8, "違反がないはずのケースで高スコアが検出されました")

    def test_ad_review_system(self):
        """広告審査システム全体の機能をテストします"""
        test_cases = [
            {
                "description": "暴力的な表現を含む広告",
                "input": {
                    "text": "激しい戦闘シーンで商品の強さを表現します",
                    "risk_score": 50
                },
                "expected_judgement": "却下"
            },
            {
                "description": "医薬品の誇大広告",
                "input": {
                    "text": "この薬を飲めば、どんな病気でも完治します",
                    "risk_score": 50
                },
                "expected_judgement": "却下"
            },
            {
                "description": "適切な広告",
                "input": {
                    "text": "新商品は快適な使用感と丈夫な作りが特徴です",
                    "risk_score": 30
                },
                "expected_judgement": "承認"
            }
        ]

        for test_case in test_cases:
            result = self.reviewer.review_ad(test_case["input"])
            self.assertEqual(
                result["judgement"],
                test_case["expected_judgement"],
                f"{test_case['description']}の判定が期待と異なります"
            )

    def test_performance(self):
        """システムのパフォーマンスをテストします"""
        import time
        
        # 検索性能の測定
        query = "商品の説明文です"
        start_time = time.time()
        self.rag.search_relevant_guidelines(query)
        search_time = time.time() - start_time
        
        # 検索は1秒以内に完了すべき
        self.assertLess(search_time, 1.0, "ガイドライン検索に時間がかかりすぎています")
        
        # 審査システム全体の性能測定
        start_time = time.time()
        self.reviewer.review_ad({"text": query, "risk_score": 50})
        review_time = time.time() - start_time
        
        # 審査全体は2秒以内に完了すべき
        self.assertLess(review_time, 2.0, "広告審査に時間がかかりすぎています")

if __name__ == '__main__':
    unittest.main() 