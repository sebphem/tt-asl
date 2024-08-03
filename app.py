from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort, jsonify
import os
from werkzeug.utils import secure_filename

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
    
    return jsonify({"image_path": url_for('static', filename=processed_image_path), "label": label})


def your_ml_function(file_path):
    # Replace with actual ML processing code
    return f"Processed image located at: {file_path}"

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
