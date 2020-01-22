from tensorflow.keras.utils import plot_model
import tensorflow
from tensorflow.keras.layers import Dense, Conv2D, Dropout,Multiply, TimeDistributed, concatenate, Input, LSTM, Flatten,Bidirectional , Activation, Reshape, Lambda, RepeatVector
import tensorflow.keras.backend as K
from dataset import Eyes
import numpy as np




max_length = 20
image_height = 24
image_width = 24

def linear_transform(x):
    return np.array([x, x])

class eye_model:
    def get_model(self,max_length, image_height, image_width):
        eye_L_input = Input(shape=(max_length, image_height, image_width, 1))
        eye_L_Layers = TimeDistributed(Conv2D(16,3,padding='same', activation='linear',use_bias=True, kernel_initializer='glorot_normal'))(eye_L_input)
        eye_L_Layers= TimeDistributed(Dense(128, activation='elu', use_bias=True, kernel_initializer='glorot_normal'))(eye_L_Layers)
        eye_L_Layers = TimeDistributed(Dropout(0.5))(eye_L_Layers)
        eye_R_input = Input(shape=(max_length, image_height, image_width, 1))
        eye_R_Layers = TimeDistributed(Conv2D(16,3,padding='same', activation='linear',use_bias=True, kernel_initializer='glorot_normal'))(eye_R_input)
        eye_R_Layers= TimeDistributed(Dense(128, activation='elu', use_bias=True, kernel_initializer='glorot_normal'))(eye_R_Layers)
        eye_R_Layers = TimeDistributed(Dropout(0.5))(eye_R_Layers)
        concat_Layer = concatenate([eye_L_Layers, eye_R_Layers])
        attention_concat = TimeDistributed(Flatten())(concat_Layer)
        attention_concat  = TimeDistributed(Dense(1, activation='tanh'))(attention_concat)
        attention_concat = Reshape ((1,20))(attention_concat)
        attention_concat = Activation('softmax')(attention_concat)
        attention_concat = Reshape((20,1,1,1))(attention_concat)
        attention_concat = tensorflow.multiply( tensorflow.ones((20,3,2,4)), attention_concat)

        model = tensorflow.keras.Model(inputs=[eye_L_input, eye_R_input], outputs=[attention_concat])

        model.summary()
        return model

model = eye_model().get_model(max_length, image_height, image_width)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def saved_data_prediction():
    train_dataset_final, val_dataset_final = Eyes().get_eyes_data()

    for i, (eye_left, eye_right, y_label) in enumerate(val_dataset_final):
        eye_left = np.array(eye_left).astype(float)
        eye_right = np.array(eye_right).astype(float)
        y_label = np.array(y_label).astype(float)
        EyeLeft_dataset_train = eye_left.reshape(1, 20, 24, 24, 1)
        EyeRight_dataset_train = eye_right.reshape(1, 20, 24, 24, 1)
        feature_dataset_train = y_label.reshape(1, 4)
        if i == 5:
            s = model.predict([EyeLeft_dataset_train, EyeRight_dataset_train])
            print(s)
            print(s.shape)
#        print ('Actual is '+ str(feature_dataset_train)  + ' and after train is ' + str(s))

saved_data_prediction()