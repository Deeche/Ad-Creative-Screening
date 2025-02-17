from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from backend.image_analyzer import ImageAnalyzer
from backend.ad_reviewer import AdReviewer
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 画像分析とレビューのインスタンスを作成
image_analyzer = ImageAnalyzer()
ad_reviewer = AdReviewer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files[]')
    uploaded_files = []
    review_results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 画像分析を実行
            analysis_result = image_analyzer.analyze_image(file_path)
            
            # 広告審査を実行
            review_result = ad_reviewer.review_ad(analysis_result)
            
            if review_result:
                result = {
                    'filename': filename,
                    'path': file_path,
                    'analysis': analysis_result,
                    'review': review_result
                }
                review_results.append(result)
                uploaded_files.append(result)
    
    if uploaded_files:
        # 結果をJSONファイルとして保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        results_file = os.path.join(RESULTS_FOLDER, f'results_{timestamp}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(review_results, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'message': 'Files uploaded and analyzed successfully',
            'files': uploaded_files,
            'results_file': results_file
        })
    
    return jsonify({'error': 'No valid files uploaded'}), 400

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
        return jsonify({'error': 'Results file not found'}), 404
    
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 結果をDataFrameに変換
    data = []
    for result in results:
        data.append({
            'image_name': result['filename'],
            'judgement': result['review']['judgement'],
            'reason': result['review']['reason'],
            'risk_score': result['review']['risk_score']
        })
    
    df = pd.DataFrame(data)
    
    # CSVファイルを作成
    csv_file = os.path.join(RESULTS_FOLDER, f'results_{timestamp}.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    return send_file(
        csv_file,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'ad_review_results_{timestamp}.csv'
    )

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=51473, debug=True)