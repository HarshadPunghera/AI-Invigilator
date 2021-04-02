from flask import Flask,render_template,request,flash,make_response
import json
import cv2
import datetime
from datetime import datetime, timedelta
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from test import live_capturing,ml_predict



with open("./configs.json") as f:  # opening the config file.
    config_data = json.load(f)

VIDEO_FOLDER = config_data["IMAGE_FOLDER"]  # getting the upload folder path.
PROCESSED_FRAMES_FOLDER = config_data["PROCESSED_FOLDER"]  # getting the processed folder path.
TARGET_INPUT_SIZE = (
    config_data["TARGET_INPUT_WIDTH"],
    config_data["TARGET_INPUT_HEIGHT"],
)  # allowed input image size.

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = VIDEO_FOLDER  # setting the upload folder path.
app.config["PROCESSED_FOLDER"] = PROCESSED_FRAMES_FOLDER   # setting the processed folder.
# app.config["UPLOAD_EXTENSIONS"] = [".mvp4"]  # allowed video  extensions.

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/result',methods=["POST"])
def detect_objects():
    """Detects cheating instances in the video frame ."""
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return "no File Uploaded"
        f = request.files["file"]                                                    #get the input video from the server.
        # client_name = str(request.form.get("client"))                         
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], f.filename))                #save the input video in the static folder.
        # full_filename = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        in_path = app.config["UPLOAD_FOLDER"]
        out_path = app.config["PROCESSED_FOLDER"]
    
        saved_model = pickle.load(open(config_data["MODEL_FILE_PATH"]),'rb')
    
        live_capture_object = live_capturing()
        live_capture_object.convert_vid_to_images(in_path,out_path)
        ml_predict(out_path,saved_model)
    
    
    
if __name__ == "__main__":
    app.run()