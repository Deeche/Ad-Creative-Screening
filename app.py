from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
from backend.image_analyzer import ImageAnalyzer
from backend.ad_reviewer import AdReviewer
from backend.document_manager import DocumentManager
import json
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

image_analyzer = ImageAnalyzer()
ad_reviewer = AdReviewer()
document_manager = DocumentManager()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/documents')
def documents():
    return render_template('documents.html')

def save_file(file, upload_dir):
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            return {
                'success': True,
                'filename': filename,
                'path': file_path
            }
    except Exception as e:
        logger.error(f'Error saving file {file.filename}: {str(e)}')
        return {
            'success': False,
            'filename': file.filename,
            'error': str(e)
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'files[]' not in request.files:
            logger.error('No files part in request')
            return jsonify({'error': 'No files part'}), 400
        
        files = request.files.getlist('files[]')
        logger.info(f'Received {len(files)} files for upload')
        
        uploaded_files = []
        file_paths = []
        
        for file in files:
            result = save_file(file, app.config['UPLOAD_FOLDER'])
            if result['success']:
                uploaded_files.append(result)
                file_paths.append(result['path'])
                logger.info(f'Successfully saved file: {result["filename"]}')
            else:
                logger.error(f'Failed to save file: {result.get("error", "Unknown error")}')
        
        if not uploaded_files:
            logger.error('No valid files were uploaded')
            return jsonify({'error': 'No valid files were uploaded'}), 400
        
        logger.info(f'Starting analysis for {len(file_paths)} files')
        
        try:
            analysis_results = image_analyzer.analyze_batch(file_paths)
            logger.info('Image analysis completed')
        except Exception as e:
            logger.error(f'Error during image analysis: {str(e)}')
            return jsonify({'error': f'Error during image analysis: {str(e)}'}), 500
        
        review_results = []
        for upload, analysis in zip(uploaded_files, analysis_results):
            try:
                if analysis:
                    review_result = ad_reviewer.review_ad(analysis)
                    if review_result:
                        result = {
                            'filename': upload['filename'],
                            'path': upload['path'],
                            'analysis': analysis,
                            'review': review_result
                        }
                        review_results.append(result)
                        logger.info(f'Review completed for {upload["filename"]}')
            except Exception as e:
                logger.error(f'Error reviewing file {upload["filename"]}: {str(e)}')
                return jsonify({'error': f'Error reviewing file {upload["filename"]}: {str(e)}'}), 500
        
        if review_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(RESULTS_FOLDER, f'results_{timestamp}.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(review_results, f, ensure_ascii=False, indent=2)
            
            return jsonify({
                'message': 'Files uploaded and analyzed successfully',
                'files': review_results,
                'results_file': results_file
            })
        
        return jsonify({'error': 'No results were generated'}), 500
        
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/results')
def get_results():
    results_files = []
    for file in os.listdir(RESULTS_FOLDER):
        if file.endswith('.json'):
            file_path = os.path.join(RESULTS_FOLDER, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            results_files.append({
                'file': file,
                'results': results
            })
    return jsonify({'results': results_files})

@app.route('/download_csv/<timestamp>')
def download_csv(timestamp):
    json_file = os.path.join(RESULTS_FOLDER, f'results_{timestamp}.json')
    if not os.path.exists(json_file):
        logger.error(f'Results file not found: {json_file}')
        return jsonify({'error': 'Results file not found', 'path': json_file}), 404
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        data = []
        for result in results:
            data.append({
                'image_name': result['filename'],
                'judgement': result['review']['judgement'],
                'reason': result['review']['reason'],
                'risk_score': result['review']['risk_score'],
                'violations': ', '.join(result['review']['violations']) if result['review']['violations'] else '-'
            })
        
        df = pd.DataFrame(data)
        csv_file = os.path.join(RESULTS_FOLDER, f'results_{timestamp}.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        summary = df['judgement'].value_counts()
        with open(csv_file, 'a', encoding='utf-8-sig') as f:
            f.write('\n\n審査結果集計\n')
            f.write(f'承認: {summary.get("承認", 0)}件\n')
            f.write(f'保留: {summary.get("保留", 0)}件\n')
            f.write(f'却下: {summary.get("却下", 0)}件\n')
            f.write(f'合計: {len(df)}件\n')
        
        return send_file(
            csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'ad_review_results_{timestamp}.csv'
        )
    except Exception as e:
        logger.error(f'Error in download_csv: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/review', methods=['POST'])
def review_ad():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '広告テキストが提供されていません'}), 400

        # 広告の審査を実行
        result = ad_reviewer.review_ad({
            'text': data['text'],
            'risk_score': data.get('risk_score', 50)
        })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/guidelines', methods=['GET'])
def get_guidelines():
    try:
        # 現在のガイドラインを取得
        guidelines = [
            "広告には暴力的な表現を含めることはできません。",
            "医薬品の効能・効果の表現には制限があります。",
            "個人情報の取り扱いには十分な注意が必要です。",
            "景品表示法に違反する誇大な表現は禁止されています。",
            "著作権を侵害するコンテンツは使用できません。"
        ]
        return jsonify(guidelines)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """ドキュメント一覧を取得"""
    try:
        documents = document_manager.get_document_list()
        return jsonify(documents)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<path:doc_path>', methods=['GET'])
def get_document(doc_path):
    """特定のドキュメントの内容を取得"""
    try:
        content = document_manager.get_document_content(doc_path)
        if content is None:
            return jsonify({'error': 'Document not found'}), 404
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<path:doc_path>', methods=['PUT'])
def update_document(doc_path):
    """ドキュメントを更新"""
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'Content is required'}), 400
        
        if document_manager.save_document(doc_path, data['content']):
            return jsonify({'message': 'Document updated successfully'})
        return jsonify({'error': 'Failed to update document'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['POST'])
def create_document():
    """新規ドキュメントを作成"""
    try:
        data = request.get_json()
        if not data or 'name' not in data or 'category' not in data:
            return jsonify({'error': 'Name and category are required'}), 400
        
        doc_path = document_manager.create_document(
            data['category'],
            data.get('subcategory'),
            data['name'],
            data.get('content', '')
        )
        
        if doc_path:
            return jsonify({
                'message': 'Document created successfully',
                'path': doc_path
            })
        return jsonify({'error': 'Failed to create document'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<path:doc_path>', methods=['DELETE'])
def delete_document(doc_path):
    """ドキュメントを削除"""
    try:
        if document_manager.delete_document(doc_path):
            return jsonify({'message': 'Document deleted successfully'})
        return jsonify({'error': 'Failed to delete document'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_banner', methods=['POST'])
def analyze_banner():
    """バナー広告の分析を実行"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '画像ファイルが提供されていません'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'ファイルが選択されていません'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': '許可されていないファイル形式です'}), 400
            
        # ファイルの保存
        result = save_file(file, app.config['UPLOAD_FOLDER'])
        if not result['success']:
            return jsonify({'error': 'ファイルの保存に失敗しました'}), 500
            
        # バナー広告の分析
        analysis_result = image_analyzer.analyze_banner_ad(result['path'])
        if not analysis_result['success']:
            return jsonify({'error': analysis_result.get('error', '分析に失敗しました')}), 500
            
        # テキストの審査（テキストが抽出された場合）
        if analysis_result['text_content']:
            text_review = ad_reviewer.review_ad({
                'text': analysis_result['text_content'],
                'risk_score': 50
            })
        else:
            text_review = None
            
        return jsonify({
            'filename': result['filename'],
            'path': result['path'],
            'text_content': analysis_result['text_content'],
            'image_analysis': analysis_result['image_analysis'],
            'text_review': text_review
        })
            
    except Exception as e:
        logger.error(f'Error in analyze_banner: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 必要なディレクトリの作成
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('data/chroma_db', exist_ok=True)
    
    app.run(host='0.0.0.0', port=51473, debug=False)
