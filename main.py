import cv2
from train import TrainData
from normalize import FaceDetector
from detect import Detect
import sys

def main(argv):
    # Prepare datasets
    base_dir = 'datasets'
    # Create directory to place clean datasets
    clean_dir = 'new_datasets'
    # For Face Detection we use caffemodel
    caffe_path = "res10_300x300_ssd_iter_140000.caffemodel"
    deploy_path = "deploy.prototxt.txt"
    # Input File
    test_input = ""

    # make an object TrainData
    model = TrainData(base_dir, clean_dir, deploy_path, caffe_path)
    for action in argv:
        if action == "train":
            model.train_data_flow()
        elif action == "predict":
            face = FaceDetector(deploy_path, caffe_path)
            img = cv2.imread("./datasets/2103181003/2103181003 (2).jpeg")
            image = face.normalize_faces(img)
            print(model.predict(image[0]))
        else :
            Detect(base_dir, clean_dir, deploy_path, caffe_path).video_rec()

if __name__ == '__main__':
    main(sys.argv[1:])