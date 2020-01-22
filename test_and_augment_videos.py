import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import cv2
import os

def rotatevid(input_filepath, output_filepath, degree):
    DestinationPathRight = output_filepath
    img_array = np.load(input_filepath)
    img_output = img_array.reshape(20, 24, 24)
    stepa = 0
    ch = 24
    cw = 24
    rotation_matrix = cv2.getRotationMatrix2D((cw/2,ch/2),degree,1)
    while stepa < 20:
        xx = cv2.warpAffine(img_output[stepa], rotation_matrix, (ch, cw))
        if stepa == 0:
            vid_output = xx
        elif stepa == 5:
            vid_output = np.dstack((vid_output,xx))
        else:
            vid_output = np.dstack((vid_output, xx))
        stepa += 1
    vid_output = np.transpose(vid_output, (2, 0, 1))
    np.save(DestinationPathRight, vid_output)
    return vid_output



def rotatevidfolder(rawpath2):
    for motname in ['N','B','R','L']:
        rawpath_motname = rawpath2 + motname + '/'
        files2 = os.listdir(rawpath_motname)
        inital_index = int(len(files2)/2)
        new_initial_index = inital_index
        for rotate_angel in [-5, -3, 3, 5]:
            for old_file_name in files2:
                new_index = int((old_file_name.split(".")[0]).split("_")[-1])
                new_index = new_index + new_initial_index
                which_eye = old_file_name.split(".")[0].split("_")[0]
                old_file_name = rawpath_motname + old_file_name
                new_file_name = rawpath_motname + which_eye + '_' + motname + '_' + str(new_index) + '.npy'
                rotatevid(old_file_name, new_file_name ,rotate_angel)
            new_initial_index = new_initial_index + inital_index


def updatefig(*args):
    global step
    step +=1
    while step < 20:
        im.set_array(img_output[step])
        return im



#rawpath2 = 'C:/Bassem/Study/DeepLearning/DLA2019/Lectures/Lecture_3/Homework/ProjectEye/Raw2/'
#rotatevidfolder(rawpath2)

img_array = np.load("C:/Bassem/Study/DeepLearning/DLA2019/Lectures/Lecture_3/Homework/ProjectEye/Raw3/B/EyeLeft_B_100.npy")
img_output = img_array.reshape(20,24,24)
fig = plt.figure()
step = 0
im = plt.imshow(img_output[0])
ani = animation.FuncAnimation(fig, updatefig,  interval=50, blit=False)
plt.show()