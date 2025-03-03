from flask import Flask, request, render_template, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename, mimetype='text/css')



app = Flask(__name__)

# Load trained model
MODEL_PATH = "../cifar10_model.h5"  # Ensure the correct path
model = load_model(MODEL_PATH)

# Ensure static uploads folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    image_url = None
    predicted_class = None

    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Process image
        img = image.load_img(file_path, target_size=(32, 32))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)

        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        predicted_class = class_names[class_index]

        # Pass correct image URL
        image_url = f"/{file_path.replace('\\', '/')}"

    return render_template("index.html", image_url=image_url, predicted_class=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
