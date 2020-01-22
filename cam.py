import cv2
import os
import numpy as np



def record(motionName, save_on_disk = True):
    if save_on_disk == True:
        motionIndex = 0
        rawPath='/home/feras/ProjectEye/Raw/'+ motionName
        files = os.listdir(rawPath)
        for files_in_motion in files:
            number = int((files_in_motion.split(".")[0]).split("_")[-1])
            motionIndex = max(motionIndex, number + 1)

        print ("Recording {}_{}".format(motionName, motionIndex))
        DestinationPathLeft = rawPath + "/" + "EyeLeft_{}_{}".format(motionName, motionIndex)
        DestinationPathRight = rawPath + "/" + "EyeRight_{}_{}".format(motionName, motionIndex)


    cap =cv2.VideoCapture(0)
    step = 0
    URL = "http://10.1.226.233:8080/shot.jpg"

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out_L = cv2.VideoWriter('output_L.avi', fourcc, 20.0, (60, 60))
    out_R = cv2.VideoWriter('output_R.avi', fourcc, 20.0, (60, 60))

    while True:
        eye_left_gray_resize = np.array([])
        eye_right_gray_resize = np.array([])
        while not eye_left_gray_resize.any()  or not eye_right_gray_resize.any():
            ret, img_color = cap.read()
#            imgResp = requests.get(URL)
#            imgNp = np.array(bytearray(imgResp.content), dtype=np.uint8)
#            img_color = cv2.imdecode(imgNp, -1)
            eye_left_gray_resize, eye_right_gray_resize, eye_right_color, eye_left_color = detect_eyes(img_color)
        out_R.write(eye_right_color)
        out_L.write(eye_left_color)
        eye_left_gray_resize = eye_left_gray_resize.astype(np.int16)
        eye_right_gray_resize = eye_right_gray_resize.astype(np.int16)
        if step == 0 :
            eye_left_frame = eye_left_gray_resize
            eye_right_frame = eye_right_gray_resize
        elif step == 1:
            eye_left_video_diff = eye_left_gray_resize - eye_left_frame_last
            eye_right_video_diff = eye_right_gray_resize - eye_right_frame_last
        else :
            eye_left_frame_diff = eye_left_gray_resize - eye_left_frame_last
            eye_left_video_diff= np.dstack((eye_left_video_diff,eye_left_frame_diff))
            eye_right_frame_diff = eye_right_gray_resize - eye_right_frame_last
            eye_right_video_diff = np.dstack((eye_right_video_diff,eye_right_frame_diff))


        eye_left_frame_last = eye_left_frame
        eye_right_frame_last = eye_right_frame
        step += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if step== 21:
            break
    cap.release()
    out_R.release()
    out_L.release()
    cv2.destroyAllWindows()

    eye_left_video_diff = np.transpose(eye_left_video_diff, (2, 0, 1)).reshape(step-1,24,24,1)
    eye_right_video_diff = np.transpose(eye_right_video_diff, (2, 0, 1)).reshape(step-1,24,24,1)
    np.save(DestinationPathLeft,eye_left_video_diff)
    np.save(DestinationPathRight, eye_right_video_diff)


def detect_eyes(img_color):
    face_cascade = cv2.CascadeClassifier('/home/feras/ProjectEye/haarcascade_frontalface_default.xml')
#    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/home/feras/ProjectEye/haarcascade_eye.xml')
#    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    face_coordinate = face_cascade.detectMultiScale(img_color, 1.3, 5)
    for (x, y, w, h) in face_coordinate:
        face_gray = img_gray[y:y + h, x:x + w]
        face_color = img_color[y:y + h, x:x + w]
        eye_coordinate = eye_cascade.detectMultiScale(face_color)
        if (len(eye_coordinate)) == 2:
            eye_left_coordinate = eye_coordinate[0]
            eye_right_coordinate = eye_coordinate[1]
            eye_left_coordinate = eye_left_coordinate.reshape((1, -1))
            eye_right_coordinate = eye_right_coordinate.reshape((1, -1))
            if eye_left_coordinate[0][0] < eye_right_coordinate[0][0]:
                temp_left = eye_left_coordinate
                temp_right = eye_right_coordinate
                eye_left_coordinate = temp_right
                eye_right_coordinate = temp_left
            for (ex, ey, ew, eh) in eye_left_coordinate:
                eye_left_gray = face_gray[ey:ey + eh, ex:ex + ew]
                eye_left_color = face_color[ey:ey + eh, ex:ex + ew]
                eye_left_color_resize = cv2.resize(eye_left_color, (60, 60), interpolation=cv2.INTER_CUBIC)
                cv2.imshow('Left Eye', eye_left_color)
                eye_left_gray_resize = cv2.resize(eye_left_gray, (24, 24), interpolation=cv2.INTER_CUBIC)

            for (ex, ey, ew, eh) in eye_right_coordinate:
                eye_right_gray = face_gray[ey:ey + eh, ex:ex + ew]
                eye_right_color = face_color[ey:ey + eh, ex:ex + ew]
                eye_right_color_resize = cv2.resize(eye_right_color, (60, 60), interpolation=cv2.INTER_CUBIC)
                cv2.imshow('Right Eye', eye_right_color)
                eye_right_gray_resize = cv2.resize(eye_right_gray, (24, 24), interpolation=cv2.INTER_CUBIC)

            return eye_left_gray_resize, eye_right_gray_resize, eye_right_color_resize, eye_left_color_resize
    return np.array([]), np.array([]), np.array([]), np.array([])


#record( 'L' , True)