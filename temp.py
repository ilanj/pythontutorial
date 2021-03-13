from flask import Flask, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/resume', methods=['POST'])
def resume():
    print('request.data', request.data)
    print('request.json', request.json)
    print('request.form', request.form)
    print('request.files', request.files)
    print('request.headers', request.headers)
    file = request.files['file']
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    return 'Successfully received'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555)