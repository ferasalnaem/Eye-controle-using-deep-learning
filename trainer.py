import matplotlib.pyplot as plt
from dataset import Eyes
from model import eye_model
import numpy as np
import tensorflow as tf

class Trainer:
    def __init__(self):
        self.Eyes = Eyes()
        self.eye_model = eye_model()
        self.max_length = 20
        self.image_height = 24
        self.image_width = 24
        self.epochs = 7
        self.batch_size = 28


    def train(self, plot=False, predict=True):
        checkpoint_path = "/home/feras/ProjectEye/training_3/cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        train_dataset_final, val_dataset_final = self.Eyes.get_eyes_data()

        for i, (eye_left, eye_right, y_label) in enumerate(train_dataset_final):
            eye_left = np.array(eye_left)
            eye_right = np.array(eye_right)
            y_label = np.array(y_label)
            if i == 0:
                EyeLeft_dataset_train = eye_left.reshape(1,20,24,24,1)
                EyeRight_dataset_train = eye_right.reshape(1, 20, 24, 24, 1)
                feature_dataset_train = y_label.reshape (1,4)
            else :
                eye_left = eye_left.reshape(1,20,24,24,1)
                eye_right = eye_right.reshape(1, 20, 24, 24, 1)
                y_label = y_label.reshape (1,4)
                EyeLeft_dataset_train = np.concatenate((EyeLeft_dataset_train, eye_left))
                EyeRight_dataset_train = np.concatenate((EyeRight_dataset_train,eye_right))
                feature_dataset_train = np.concatenate((feature_dataset_train, y_label))

        for i, (eye_left, eye_right, y_label) in enumerate(val_dataset_final):
            eye_left = np.array(eye_left)
            eye_right = np.array(eye_right)
            y_label = np.array(y_label)
            if i == 0:
                EyeLeft_dataset_test = eye_left.reshape(1,20,24,24,1)
                EyeRight_dataset_test = eye_right.reshape(1, 20, 24, 24, 1)
                feature_dataset_test = y_label.reshape (1,4)
            else :
                eye_left = eye_left.reshape(1,20,24,24,1)
                eye_right = eye_right.reshape(1, 20, 24, 24, 1)
                y_label = y_label.reshape (1,4)
                EyeLeft_dataset_test = np.concatenate((EyeLeft_dataset_test, eye_left))
                EyeRight_dataset_test = np.concatenate((EyeRight_dataset_test,eye_right))
                feature_dataset_test = np.concatenate((feature_dataset_test, y_label))


        model = self.eye_model.get_model(self.max_length, self.image_height, self.image_width)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        STEPS_PER_EPOCH = int(feature_dataset_train.shape[0]*0.8)//40
        VALIDATION_STEPS = feature_dataset_test.shape[0]//self.batch_size//3
        model_history = model.fit([EyeLeft_dataset_train,EyeRight_dataset_train],feature_dataset_train, epochs=self.epochs,
#                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  steps_per_epoch=20,
#                                  validation_steps=VALIDATION_STEPS,
                                  validation_steps=30,
                                  batch_size=self.batch_size,
#                                  validation_split=0.2
                                  validation_data=([EyeLeft_dataset_test,EyeRight_dataset_test],feature_dataset_test),
                                  callbacks=[cp_callback]
                                  )

        if plot:
            loss = model_history.history['loss']
            val_loss = model_history.history['val_loss']

            epochs = range(self.epochs)

            plt.figure()
            plt.plot(epochs, loss, 'r', label='Training loss')
            plt.plot(epochs, val_loss, 'bo', label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.ylim([0, 1])
            plt.legend()
            plt.show()
            plt.show()

        if predict:
            for i, (eye_left, eye_right, y_label) in enumerate(val_dataset_final):
                eye_left = np.array(eye_left).astype(float)
                eye_right = np.array(eye_right).astype(float)
                y_label = np.array(y_label).astype(float)
                EyeLeft_dataset_train = eye_left.reshape(1, 20, 24, 24, 1)
                EyeRight_dataset_train = eye_right.reshape(1, 20, 24, 24, 1)
                feature_dataset_train = y_label.reshape(1, 4)
                s = model.predict([EyeLeft_dataset_train, EyeRight_dataset_train])
                print('Actual is ' + str(feature_dataset_train) + ' and after train is ' + str(s))

Trainer().train()