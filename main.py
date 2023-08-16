import os
import cv2
from numpy import asarray
from PIL import Image
from ultralytics import YOLO

# library that controls your mouses position
import pyautogui


def delete_directory(directory_path):
    file_list = os.listdir(directory_path)
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_directory(file_path)

    os.rmdir(directory_path)
    print(f"Directory '{directory_path}' and its contents successfully deleted.")


def XandYofBox(label_path):
    fileLines = open(label_path, "r")
    lines = fileLines.readlines()

    for classes in lines:
        cords = classes.split()
        xc, yc = float(cords[1]), float(cords[2])
        return xc, yc
        


model = YOLO("C:/OpenCV/ValorantDetection/trainedModel/weights/best.pt")

video = cv2.VideoCapture("C:/Users/Austi/Desktop/Youtube/OnlyEnemyAgentVid.mp4")

while True:
    ret, frame = video.read()
    result = model.predict(frame, save=True, save_txt=True, save_conf=True, max_det=5, conf=0.7)
    predict_img_path = "C:/OpenCV/ValorantDetection/runs/detect/predict/image0.jpg"

    # Move mouse to the center of "Body"
    new_img = cv2.imread(predict_img_path)
    img_height, img_width = new_img.shape[0], new_img.shape[1]
    # xc, yc = XandYofBox("C:/OpenCV/ValorantDetection/runs/detect/predict/labels/image0.txt")
    # pyautogui.moveTo(xc*img_width, yc*img_height)

    # turn Yolo predicted img to RGB img and display
    img = Image.open(predict_img_path)
    numpydata = asarray(img)
    final_img = cv2.cvtColor(numpydata, cv2.COLOR_BGR2RGB)
    cv2.imshow("VALORANT - Agent Detector", final_img)

    if cv2.waitKey(30) == 27:
        delete_directory("C:/OpenCV/ValorantDetection/runs/detect/predict")
        break

video.release()
cv2.destroyAllWindows()