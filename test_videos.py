import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

img_array = np.load("/home/feras/ProjectEye/Raw/B/EyeLeft_B_1.npy")
img_output = img_array.reshape(20,24,24)

fig = plt.figure()
step = 0
im = plt.imshow(img_output[0])

def updatefig(*args):
    global step
    step +=1
    if step == 20:
        step = 0
    im.set_array(img_output[step])
    return im,


ani = animation.FuncAnimation(fig, updatefig,  interval=200, blit=False)
plt.show()
