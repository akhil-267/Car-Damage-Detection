import argparse
import io
import os
import base64
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image

# Initialize the YOLO model only once
model = YOLO('best.pt')

# Configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'output.mp4')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
terminate_flag = False  # Global flag for webcam termination


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate():
    global terminate_flag
    terminate_flag = False
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened() and not terminate_flag:
            success, frame = cap.read()
            if not success:
                break

            try:
                results = model(frame)
                annotated_frame = results[0].plot()
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.route('/')
@app.route('/first')
def first():
    return render_template("first.html")


@app.route('/login')
def login():
    return render_template("login.html")






@app.route('/image')
def image():
    return render_template("image.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('image'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Invalid file type')
        return redirect(url_for('image'))

    try:
        upl_img = Image.open(file)
        result = model.predict(source=upl_img)[0]
        res_img = Image.fromarray(result.plot())
        image_byte_stream = io.BytesIO()
        res_img.save(image_byte_stream, format='PNG')
        image_byte_stream.seek(0)
        image_base64 = base64.b64encode(image_byte_stream.read()).decode('utf-8')
        return render_template('image.html', detection_results=image_base64)
    except Exception as e:
        flash(f'Prediction error: {e}')
        return redirect(url_for('image'))


@app.route("/video")
def video():
    return render_template('video.html')


@app.route("/predict_img", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            if not allowed_file(f.filename):
                flash("Unsupported file type")
                return redirect(url_for('video'))

            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)

            if filename.rsplit('.', 1)[1].lower() == 'mp4':
                cap = cv2.VideoCapture(filepath)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (frame_width, frame_height))

                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        results = model(frame)
                        res_frame = results[0].plot()
                        out.write(res_frame)
                except Exception as e:
                    flash(f"Video processing error: {e}")
                finally:
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()

                flash("Video processed successfully.")
                return redirect(url_for('video'))

    return render_template('video.html')


@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop', methods=['POST'])
def stop():
    global terminate_flag
    terminate_flag = True
    flash("Webcam stopped.")
    return redirect(url_for('first'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv8 models")
    parser.add_argument("--port", default=5000, type=int, help="Port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
