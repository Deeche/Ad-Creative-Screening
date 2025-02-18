from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
from backend.image_analyzer import ImageAnalyzer
from backend.ad_reviewer import AdReviewer
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    import torch
    if torch.cuda.is_available():
        logger.info(f'GPU available: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB')
    else:
        logger.warning('No GPU available, using CPU')
    
    app.run(host='0.0.0.0', port=51473, debug=True)
