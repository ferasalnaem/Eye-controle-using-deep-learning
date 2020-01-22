from model import eye_model
from cam import detect_eyes
from dataset import Eyes
import numpy as np
import cv2



max_length = 20
image_height = 24
image_width = 24
checkpoint_path = "/home/feras/ProjectEye/training_1/cp.ckpt"
checkpoint_path2 = "/home/feras/ProjectEye/training_2/cp.ckpt"

model = eye_model().get_model2(max_length, image_height, image_width)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(checkpoint_path2)

model2 = eye_model().get_model(max_length, image_height, image_width)
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.load_weights(checkpoint_path)

## to verify saved dataset
def saved_data_prediction():
    train_dataset_final, val_dataset_final = Eyes().get_eyes_data()

    for i, (eye_left, eye_right, y_label) in enumerate(val_dataset_final):
        eye_left = np.array(eye_left).astype(float)
        eye_right = np.array(eye_right).astype(float)
        y_label = np.array(y_label).astype(float)
        EyeLeft_dataset_train = eye_left.reshape(1, 20, 24, 24, 1)
        EyeRight_dataset_train = eye_right.reshape(1, 20, 24, 24, 1)
        feature_dataset_train = y_label.reshape(1, 4)
        s = model.predict([EyeLeft_dataset_train, EyeRight_dataset_train])
        d = model2.predict([EyeLeft_dataset_train, EyeRight_dataset_train])
        print ('Actual is '+ str(feature_dataset_train) + ' and without attention layer is ' + str(s) + ' and with attention layer is ' + str(d))

# Train to predict live videos

def live_predict():
    cap =cv2.VideoCapture(0)
    step = 0

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out_L = cv2.VideoWriter('output_L_predict.avi', fourcc, 20.0, (60, 60))
    out_R = cv2.VideoWriter('output_R_predict.avi', fourcc, 20.0, (60, 60))

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
    eye_left_diff = np.array(eye_left_video_diff).astype(float)
    eye_right_diff = np.array(eye_right_video_diff).astype(float)
    EyeLeft_dataset_train = eye_left_diff.reshape(1, 20, 24, 24, 1)
    EyeRight_dataset_train = eye_right_diff.reshape(1, 20, 24, 24, 1)
    s = model.predict([EyeLeft_dataset_train, EyeRight_dataset_train])
    d = model2.predict([EyeLeft_dataset_train, EyeRight_dataset_train])
    print ('Without attention layer is ' + str(s) + ' and with attention layer is ' + str(d))
    prediction_map = np.where(d == np.amax(d))[1]
    prediction_map = int(prediction_map)
    if prediction_map == 0:
        print ('The motion predicted is No eyes blinking')
    elif prediction_map == 1:
        print('The motion predicted is both eyes blinking')
    elif prediction_map == 2:
        print('The motion predicted is right eye is blinking')
    elif prediction_map == 3:
        print('The motion predicted is left eye is blinking')



#saved_data_prediction()
live_predict()