import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import warnings
import ffmpeg
warnings.filterwarnings("ignore")

model = load_model('mnist_model.h5')

def get_and_process_user_input():
    print("Welcome to handwritten digit recogonition application!")
    while(1):
        val = input("Enter 1 for web camera or 2 for video file:  ")
        if val == '1':
            cap = cv2.VideoCapture(0)
            rotateCode = None
            CONTOUR_AREA_THRESHOLD = 225
            break
        if val == '2':
            cap = cv2.VideoCapture("original_video_6885.mp4")
            rotateCode = check_rotation("original_video_6885.mp4")
            CONTOUR_AREA_THRESHOLD = 200
            break
        else:
            print('ERROR: Invalid Input!')
    return cap, rotateCode, CONTOUR_AREA_THRESHOLD


def main():
    cap, rotateCode, CONTOUR_AREA_THRESHOLD = get_and_process_user_input()

    while (cap.isOpened()):
        ret, img = cap.read()
        if rotateCode is not None:
            img = correct_rotation(img, rotateCode)
        img, contours, thresh = get_image_contours_and_threshold(img)

        for every_contour in contours:
            if cv2.contourArea(every_contour) > CONTOUR_AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(every_contour)
                roi_Image = thresh[y:y + h, x:x + w]
                roi_digit = cv2.resize(roi_Image, (28, 28), interpolation=cv2.INTER_AREA)
                roi_digit = cv2.dilate(roi_digit, (3, 3))
                resized_image = np.reshape(roi_digit, [1, 28, 28, 1])
                result = model.predict(resized_image)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
                cv2.putText(img, str(int(np.argmax(result, axis=1))), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        cv2.imshow("Frame", img)
        k = cv2.waitKey(10)
        if k == 30:
            break

def get_image_contours_and_threshold(img):
    img = cv2.resize(img, (256, 256))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.Canny(imgray, 100, 200)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresh1 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 5) # all contours
    return img, contours, thresh

     
def check_rotation(path_video_file):
    meta_dict = ffmpeg.probe(path_video_file)
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    return rotateCode


def correct_rotation(frame, rotateCode):  
    return cv2.rotate(frame, rotateCode) 


main()
