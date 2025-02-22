from flask import Flask, request, Response
import pyttsx3
import cv2
from flask_cors import CORS
import numpy as np
import ImageStuff

app = Flask(__name__)
CORS(app)

# Variable to store top two colors
current_objects = []


@app.route('/process-image', methods=['POST'])
def process_image():
    global current_objects

    # Get the image file from the request
    file = request.files.get('image')
    if not file:
        return {"error": "No image provided"}, 400

    # Convert the uploaded image to a NumPy array and decode it using OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # OpenCV loads images in BGR format

    # Process the image using YOLO
    current_objects = ImageStuff.read(image)

    return {"message": "true"}


@app.route('/objects-now', methods=['GET'])
def objects_now():
    global current_objects
    if current_objects:
        text = "Detected objects: " + ", ".join(current_objects)
    else:
        text = "No objects detected."

    engine = pyttsx3.init()
    engine.save_to_file(text, "output.mp3")
    engine.runAndWait()

    # Read the saved file and stream it
    with open("output.mp3", "rb") as f:
        audio_data = f.read()

    return Response(audio_data, mimetype="audio/mpeg")


if __name__ == '__main__':
    app.run(debug=True)
