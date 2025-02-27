import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
from dotenv import load_dotenv
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        load_dotenv()
        
        try:
            logger.info("画像分析モジュールの初期化を開始")
            
            # 環境変数の確認
            project_id = os.getenv('PROJECT_ID')
            location = os.getenv('LOCATION')
            
            logger.info(f"環境変数: PROJECT_ID={project_id}, LOCATION={location}")
            
            if not project_id or not location:
                raise ValueError("PROJECT_IDまたはLOCATION環境変数が設定されていません")
            
            # サービスアカウントの認証情報を読み込み
            credentials = service_account.Credentials.from_service_account_file(
                'credentials/ad-creative-screening-67985e10c506.json',
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Gemini Proの初期化
            vertexai.init(
                project=project_id,
                location=location,
                credentials=credentials
            )
            self.model = GenerativeModel("gemini-pro-vision")
            logger.info("画像分析モジュールの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"画像分析モジュール初期化エラー: {str(e)}")
            raise RuntimeError(f"画像分析モジュールの初期化に失敗しました: {str(e)}")

    def extract_text(self, image_path: str) -> str:
        """画像からテキストを抽出"""
        try:
            logger.info(f"画像からテキストの抽出を開始: {image_path}")
            
            # 画像を読み込み
            image = Image.open(image_path)
            logger.info(f"画像情報: サイズ={image.size}, フォーマット={image.format}")
            
            # 画像をバイト列に変換
            import io
            img_byte_arr = io.BytesIO()
            
            # 画像のフォーマットを確実に指定してバイト列に変換
            if image.format:
                image.save(img_byte_arr, format=image.format)
                mime_type = f"image/{image.format.lower()}"
            else:
                image.save(img_byte_arr, format='JPEG')
                mime_type = "image/jpeg"
                
            img_byte_arr = img_byte_arr.getvalue()
            logger.info(f"画像をバイト列に変換: サイズ={len(img_byte_arr)}バイト, MIMEタイプ={mime_type}")
            
            # Gemini Proによるテキスト抽出
            prompt = """
            この広告画像から全てのテキストを抽出してください。
            以下の点に注意して抽出を行ってください：
            
            1. 画像内の全てのテキストを抽出（メインテキスト、サブテキスト、注釈など）
            2. テキストの配置や大きさに関係なく、全ての文字を抽出
            3. 日本語と英語の両方に対応
            4. 抽出したテキストは原文のまま出力
            5. 改行やスペースなども可能な限り保持
            
            テキストのみを出力し、説明や解説は不要です。
            """
            
            logger.info("Gemini APIにリクエストを送信")
            # 正しいフォーマットでのコンテンツ送信
            content_parts = [
                prompt,
                Part.from_data(mime_type=mime_type, data=img_byte_arr)
            ]
            response = self.model.generate_content(content_parts)
            logger.info("テキスト抽出が完了しました")
            
            return response.text.strip()
            
        except FileNotFoundError as e:
            logger.error(f"画像ファイルが見つかりません: {e}")
            return ""
        except IOError as e:
            logger.error(f"画像ファイルの読み込みエラー: {e}")
            return ""
        except Exception as e:
            logger.error(f"テキスト抽出中のエラー: {str(e)}")
            return ""

    def analyze_image_content(self, image_path: str) -> Dict:
        """画像の内容を分析"""
        try:
            logger.info(f"画像の内容分析を開始: {image_path}")
            
            # 画像を読み込み
            image = Image.open(image_path)
            logger.info(f"画像情報: サイズ={image.size}, フォーマット={image.format}")
            
            # 画像をバイト列に変換
            import io
            img_byte_arr = io.BytesIO()
            
            # 画像のフォーマットを確実に指定してバイト列に変換
            if image.format:
                image.save(img_byte_arr, format=image.format)
                mime_type = f"image/{image.format.lower()}"
            else:
                image.save(img_byte_arr, format='JPEG')
                mime_type = "image/jpeg"
                
            img_byte_arr = img_byte_arr.getvalue()
            logger.info(f"画像をバイト列に変換: サイズ={len(img_byte_arr)}バイト, MIMEタイプ={mime_type}")
            
            # Gemini Proによる画像分析
            prompt = """
            あなたは広告画像の審査専門家です。この広告画像を分析し、以下の観点から評価してください：

            1. 画像の全体的な印象
            2. 不適切な要素の有無
               - 暴力的な表現
               - 性的な表現
               - 差別的な表現
               - 過度に刺激的な表現
            3. ブランドガイドラインとの整合性
            4. 視覚的な効果とインパクト
            5. 配色やレイアウトの適切性
            6. 画像の品質と解像度

            以下のJSON形式で回答してください：
            {
                "overall_impression": "全体的な印象",
                "inappropriate_elements": {
                    "has_inappropriate": true/false,
                    "details": ["不適切な要素の詳細"],
                    "severity": "High/Medium/Low"
                },
                "visual_elements": {
                    "color_scheme": "配色の評価",
                    "layout": "レイアウトの評価",
                    "quality": "画質の評価"
                },
                "compliance": {
                    "is_compliant": true/false,
                    "violations": ["違反内容"],
                    "risk_level": "High/Medium/Low"
                },
                "recommendations": ["改善提案"]
            }
            """
            
            logger.info("Gemini APIに画像分析リクエストを送信")
            # 正しいフォーマットでのコンテンツ送信
            content_parts = [
                prompt,
                Part.from_data(mime_type=mime_type, data=img_byte_arr)
            ]
            response = self.model.generate_content(content_parts)
            logger.info("画像分析が完了しました")
            
            try:
                # 応答の解析を改善
                text_response = response.text.strip()
                logger.info(f"応答テキスト（一部）: {text_response[:100]}...")
                
                # JSON解析を試みる
                import json
                import re
                
                # JSON部分を抽出する正規表現
                json_pattern = r'\{.*\}'
                match = re.search(json_pattern, text_response, re.DOTALL)
                
                if match:
                    json_str = match.group(0)
                    logger.info(f"抽出したJSON文字列（一部）: {json_str[:100]}...")
                    result = json.loads(json_str)
                    return result
                else:
                    # evalを使用した解析を試みる
                    result = eval(text_response)
                    return result
                    
            except Exception as e:
                logger.error(f"画像分析応答の解析エラー: {e}")
                logger.error(f"応答テキスト: {response.text}")
                return self._generate_fallback_analysis()
                
        except FileNotFoundError as e:
            logger.error(f"画像ファイルが見つかりません: {e}")
            return self._generate_fallback_analysis()
        except IOError as e:
            logger.error(f"画像ファイルの読み込みエラー: {e}")
            return self._generate_fallback_analysis()
        except Exception as e:
            logger.error(f"画像分析中のエラー: {str(e)}")
            return self._generate_fallback_analysis()

    def _generate_fallback_analysis(self) -> Dict:
        """エラー時のフォールバック分析結果を生成"""
        return {
            "overall_impression": "分析不能",
            "inappropriate_elements": {
                "has_inappropriate": False,
                "details": [],
                "severity": "Low"
            },
            "visual_elements": {
                "color_scheme": "評価不能",
                "layout": "評価不能",
                "quality": "評価不能"
            },
            "compliance": {
                "is_compliant": True,
                "violations": [],
                "risk_level": "Low"
            },
            "recommendations": ["画像の再分析を推奨"]
        }

    def analyze_banner_ad(self, image_path: str) -> Dict:
        """バナー広告の総合分析を実行"""
        try:
            logger.info(f"バナー広告の総合分析を開始: {image_path}")
            
            # ファイルの存在確認
            if not os.path.exists(image_path):
                logger.error(f"画像ファイルが存在しません: {image_path}")
                return {
                    "text_content": "",
                    "image_analysis": self._generate_fallback_analysis(),
                    "success": False,
                    "error": f"画像ファイルが存在しません: {image_path}"
                }
            
            # テキスト抽出
            logger.info("テキスト抽出処理を開始")
            extracted_text = self.extract_text(image_path)
            logger.info(f"抽出されたテキスト（一部）: {extracted_text[:50]}...")
            
            # 画像分析
            logger.info("画像内容分析処理を開始")
            image_analysis = self.analyze_image_content(image_path)
            logger.info("画像分析が完了しました")
            
            logger.info("バナー広告の総合分析が完了しました")
            return {
                "text_content": extracted_text,
                "image_analysis": image_analysis,
                "success": True
            }
            
        except FileNotFoundError as e:
            logger.error(f"画像ファイルが見つかりません: {e}")
            return {
                "text_content": "",
                "image_analysis": self._generate_fallback_analysis(),
                "success": False,
                "error": str(e)
            }
        except IOError as e:
            logger.error(f"画像ファイル読み込みエラー: {e}")
            return {
                "text_content": "",
                "image_analysis": self._generate_fallback_analysis(),
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"バナー広告分析中のエラー: {str(e)}")
            return {
                "text_content": "",
                "image_analysis": self._generate_fallback_analysis(),
                "success": False,
                "error": str(e)
            }

    def analyze_batch(self, image_paths: List[str]) -> List[Dict]:
        """複数の画像を一括分析"""
        logger.info(f"一括画像分析を開始: {len(image_paths)}件の画像")
        results = []
        
        for image_path in image_paths:
            try:
                # 画像分析の実行
                result = self.analyze_banner_ad(image_path)
                results.append(result)
                logger.info(f"画像の分析が完了: {image_path}")
            except Exception as e:
                logger.error(f"画像分析中のエラー: {str(e)}")
                results.append({
                    "text_content": "",
                    "image_analysis": self._generate_fallback_analysis(),
                    "success": False,
                    "error": str(e)
                })
                
        logger.info(f"一括画像分析が完了: {len(results)}件の結果")
        return results