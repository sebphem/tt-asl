from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort, jsonify
import os
from werkzeug.utils import secure_filename
from draw_hands import draw_hand_landmarks_and_connections_return_image_path
from pathlib import Path
from backend.asl_prediction_buda import get_prediction_given_tensor, setup_image
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('uploaded_file', filename=filename))
    else:
        return "File type not allowed", 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        abort(404)
    return render_template('uploaded.html', filename=filename)

@app.route('/uploads/<filename>/serve')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/run_ml', methods=['POST'])
def run_ml():
    filename = request.form.get('filename')
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404
    
    # Replace with actual ML processing code
    processed_image_path = 'temp_proc.png'  # Path to the processed image
    label = 'A'  # Placeholder label
    
    return jsonify(your_ml_function(file_path=filename))


def your_ml_function(file_path):
    # Replace with actual ML processing code

    write_back_folder = Path(os.path.abspath('.')) / "image_queue"
    processed_image_path = draw_hand_landmarks_and_connections_return_image_path(file_path,write_back_folder)

    tmp_img = Image.open(processed_image_path)
    tmp_tensor = setup_image(tmp_img)
    label = get_prediction_given_tensor(tmp_tensor)

    return {"image_path": url_for('static', filename=processed_image_path), "label": label}

@app.route('/delete_file', methods=['POST'])
def delete_file():
    filename = request.form.get('filename')
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        os.remove(file_path)
        return jsonify({"success": "File deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=8081)
