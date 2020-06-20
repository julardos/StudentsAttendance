# import the necessary packages
from imutils.video import VideoStream
from train import TrainData
import numpy as np
import imutils
import uuid
import time
import cv2

class Detect :
    def __init__(self, datasets, temp_directory, deploy_path, caffe_path):
        self.train_class = TrainData(datasets, temp_directory, deploy_path, caffe_path)

    def video_rec(self):
        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        (startX, startY, endX, endY) = (0, 0, 0, 0)
        set_timer = 5
        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=800)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < 0.5:
                    continue
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # draw the bounding box of the face along with the associated
                # probability
                input = frame[startY:endY, startX:endX]
                text = self.train_class.predict(input)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 0, 0), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed or time end, break from the loop
            set_timer -= 1
            if set_timer == 0:
                input = frame[startY:endY, startX:endX]
                cv2.imwrite('temp/ceptured/' + str(uuid.uuid4()) + '.jpeg', input)
                print("Prediksi NRP :", self.train_class.predict(input))
                set_timer = 5

            if key == ord("q"):
                cv2.destroyAllWindows()
                break

        input = frame[startY:endY, startX:endX]
        cv2.imshow("Captured Photo", input)
        key2 = cv2.waitKey(1) & 0xFF
        print("Ceptured", end=" ")
        self.train_class.predict(input)

        if key2 == ord("q"):
            # do a bit of cleanup
            cv2.destroyAllWindows()
            vs.stop()
