#!/usr/bin/env python
# coding: utf-8

# # Front end with ML Predict Function
import cv2
import time
import datetime
from datetime import datetime, timedelta
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import logging as log
#from log import logger
import pickle
from keras.models import load_model

in_path = "static/uploaded"
out_path = "static/processed"

exam_start_time = datetime(2020, 12, 8, 11, 43, 0)
exam_end_time = datetime(2020, 12, 8, 11, 45, 13)


class live_capturing:
    """ Class for handling cheating detection in video files."""

    log.info("Initializing Live capturing.............")
    
    

    def live_video_capturing(in_path,exam_start_time,exam_end_time):
        """Captures live using webcam from start_time to end_time."""
        
        log.info("Entering function live_video_capturing within class live_capturing")
        ran_time = [5, 10, 15, 25, 30, 50, 60]

        while datetime.now() > exam_start_time and datetime.now() < exam_end_time:
            log.info("Entering exam time")
            ran_gap = random.choice(ran_time)
            time.sleep(ran_gap)
            file_name = in_path + 'stu_id' + str(time.time()) + '.mp4'
            log.info(file_name)

            v1 = cv2.VideoCapture(0)

            if (v1.isOpened() == False):
                log.info("Camera is already running")

            shoot = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MP4V'),
                                    20, (640, 480))

            record_end_time = datetime.now() + timedelta(seconds=60)
            while datetime.now() < record_end_time:
                ret, frame = v1.read()
                if ret == True:
                    shoot.write(frame)
                else:
                    break

            v1.release()
            shoot.release()
            cv2.destroyWindow
            log.info("Closing Window")

        cv2.destroyAllWindows()
        log.info('Exam Over')
        

    def convert_vid_to_images(in_path,out_path):
        """ Convert video to frames and stores them in the out_path."""

        print("Initializing convert_vid_to_images.........")
        fls = os.listdir(in_path)
        print(fls)
        if len(fls) > 0:
            for file in fls:
                print(file)
                vid1 = cv2.VideoCapture(in_path +"/"+ file)
                print("captured video from inpath")
                print(vid1)
                print(vid1.get(cv2.CAP_PROP_POS_MSEC))
                frame_count = int(vid1.get(cv2.CAP_PROP_FRAME_COUNT))
                print(frame_count)
                #filename = str(time.time()) + '.png'
                for i in range(frame_count):
                    _, img7 = vid1.read()
                    cv2.imwrite(out_path +"/"+ str(time.time()) + '.png', img7)
                    print("saving frame to outpath")

        log.info("Video to Frames conversion succesfully over.............")



def ml_predict(out_path,saved_model):
    """Calls the saved model on the processed frames and return cheating instances."""
    
    saved_model = load_model("models/vgg/final_model1.h5")

    fls = os.listdir(out_path)
    if len(fls) > 0:
        final_result=[]
        for file in fls:
            # load the image
            img = os.path.join(out_path, file)
            img12 = cv2.imread(img, cv2.IMREAD_COLOR)

            # resizing the image
            img12 = cv2.resize(img12, (244, 244))
            plt.figure()
            plt.imshow(img12) 
            plt.show()
            
            img11 = np.array(img12)
            img11 = np.expand_dims(img11, axis=0)
            
            
            test_result = np.round(saved_model.model.predict(img11))
            print(test_result)

            if test_result[0][0] == 1:
                print("Non Cheating Instance")
            else:
                final_result.append(test_result)
                print("Cheating Instance")
                


    print("Prediction done........ ")


def main():
    """Main logic for detecting cheating in video."""
    
    in_path = "static/uploaded"
    out_path = "static/"
    print("yes")
    saved_model = load_model("models/vgg/final_model1.h5")
    print("yes1")
    #saved_model = pickle.dump(open("/mnt/c/Users/Asus/Desktop/capstone/models/vgg/final_model1.h5",'rb'))
    #live_capturing.live_video_capturing(in_path,exam_start_time,exam_end_time)
    #live_capturing()
    #live_capturing.convert_vid_to_images(in_path,out_path)
    print("yes2")
    ml_predict(out_path,saved_model)
    print("yes3")


if __name__ == "__main__":
    main()

