
import argparse
import io
import cv2
from re import DEBUG, sub
from flask import Flask, flash,render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
from ultralytics import YOLO
from PIL import Image
import base64


model = YOLO(model = 'best.pt')

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
PORT_NUMBER = 5000

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


def generate():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)

            # Yield the JPEG data to Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            key = cv2.waitKey(50) & 0xFF
            if key == 27 or key == ord("q"):  # Terminate the loop on 'q' key
                break

    # Release the webcam and redirect to another page
    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))  # Redirect to another page
         
@app.route('/')
@app.route('/first')
def first():
    return render_template("first.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")
@app.route('/performance')
def performance():
    return render_template("performance.html")
@app.route('/image')
def image():
    return render_template("image.html")


@app.route('/predict', methods=['POST'])
def predict():
     # Check if the file input is empty
    if 'file' not in request.files:
        return redirect(url_for('image'))

    file = request.files['file']
    print(file)

    # Check if the filename is empty
    if file.filename == '':
        return redirect(url_for('image'))

    # Check if the uploaded file is an MP4 file
    if file.filename.endswith('.mp4'):
        # If it's an MP4 file, redirect to another page
        return redirect(url_for('image'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upl_img = Image.open(file)
        extension = upl_img.format.lower()

        result = model.predict(source=upl_img)[0]
        res_img = Image.fromarray(result.plot())
        image_byte_stream = io.BytesIO()
        res_img.save(image_byte_stream, format='PNG')  # You can use a different format if desired, such as 'JPEG'
        image_byte_stream.seek(0)
        image_base64 = base64.b64encode(image_byte_stream.read()).decode('utf-8')

        return render_template('image.html', detection_results = image_base64)
@app.route("/video")
def video():
    return render_template('video.html')

    

@app.route("/predict_img", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)
            
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 50.0, (frame_width, frame_height))

                model = YOLO('best.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, save=True)
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)

                    out.write(res_plotted)

                    if (cv2.waitKey(25) & 0xFF) == ord('q'):
                        cv2.destroyAllWindows()
                        break

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
    return render_template('first.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    #model = torch.hub.load('.', 'custom','best.pt', source='local')
    model = YOLO('best.pt')
    app.run(host="0.0.0.0", port=args.port) 
