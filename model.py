import tensorflow
from tensorflow.keras.layers import Dense, Conv2D, Dropout, TimeDistributed, Multiply,  concatenate, Input, LSTM, Flatten, Activation, Reshape

nb_of_classes = 3
image_height = 24
image_width = 24
max_length = 20

class eye_model:
    def get_model(self,max_length, image_height, image_width):

        eye_L_input = Input(shape=(max_length, image_height, image_width, 1))
        eye_L_Layers = TimeDistributed(Conv2D(16,3,padding='same', activation='tanh',use_bias=True, kernel_initializer='glorot_normal'))(eye_L_input)
        eye_L_Layers= TimeDistributed(Dense(128, activation='tanh', use_bias=True, kernel_initializer='glorot_normal'))(eye_L_Layers)
        eye_L_Layers = TimeDistributed(Dropout(0.5))(eye_L_Layers)

        eye_R_input = Input(shape=(max_length, image_height, image_width, 1))
        eye_R_Layers = TimeDistributed(Conv2D(16,3,padding='same', activation='tanh',use_bias=True, kernel_initializer='glorot_normal'))(eye_R_input)
        eye_R_Layers= TimeDistributed(Dense(128, activation='tanh', use_bias=True, kernel_initializer='glorot_normal'))(eye_R_Layers)
        eye_R_Layers = TimeDistributed(Dropout(0.5))(eye_R_Layers)

        concat_Layer = concatenate([eye_L_Layers, eye_R_Layers])

        attention_concat = TimeDistributed(Flatten())(concat_Layer)
        attention_concat = TimeDistributed(Dense(128, activation='tanh'))(attention_concat)
        attention_concat = TimeDistributed(Dense(1, activation='tanh'))(attention_concat)
        attention_concat = Reshape((1, 20))(attention_concat)
        attention_concat = Activation('softmax')(attention_concat)
        attention_concat = Reshape((20, 1, 1, 1))(attention_concat)
        attention_concat = tensorflow.multiply(tensorflow.ones((max_length, image_height, (image_width), 256)),
                                               attention_concat)
        concat_Layer = Multiply()([concat_Layer, attention_concat])
        concat_Layer = TimeDistributed(Dense(128, activation='tanh', use_bias=True, kernel_initializer='glorot_normal'))(concat_Layer)
        concat_Layer = TimeDistributed(Flatten())(concat_Layer)
        concat_Layer = LSTM(128, dynamic = False)(concat_Layer)
        concat_Layer = Dropout(0.5)(concat_Layer)
        concat_Layer = Dense(64, activation = 'elu')(concat_Layer)
        concat_Layer = Dropout(0.5)(concat_Layer)
        concat_Layer = Dense(4, activation = 'softmax')(concat_Layer)

        model = tensorflow.keras.Model(inputs=[eye_L_input, eye_R_input], outputs=[concat_Layer])
        model.summary()
        return model

    def get_model2(self,max_length, image_height, image_width):

        eye_L_input = Input(shape=(max_length, image_height, image_width, 1))
        eye_L_Layers = TimeDistributed(Conv2D(16,3,padding='same', activation='linear',use_bias=True, kernel_initializer='glorot_normal'))(eye_L_input)
        eye_L_Layers= TimeDistributed(Dense(128, activation='tanh', use_bias=True, kernel_initializer='glorot_normal'))(eye_L_Layers)
        eye_L_Layers = TimeDistributed(Dropout(0.5))(eye_L_Layers)
        eye_R_input = Input(shape=(max_length, image_height, image_width, 1))
        eye_R_Layers = TimeDistributed(Conv2D(16,3,padding='same', activation='linear',use_bias=True, kernel_initializer='glorot_normal'))(eye_R_input)
        eye_R_Layers= TimeDistributed(Dense(128, activation='tanh', use_bias=True, kernel_initializer='glorot_normal'))(eye_R_Layers)
        eye_R_Layers = TimeDistributed(Dropout(0.5))(eye_R_Layers)
        concat_Layer = concatenate([eye_L_Layers, eye_R_Layers])
        concat_Layer = TimeDistributed(Dense(128, activation='tanh', use_bias=True, kernel_initializer='glorot_normal'))(concat_Layer)
        concat_Layer = TimeDistributed(Flatten())(concat_Layer)
        concat_Layer = LSTM(128, dynamic = False)(concat_Layer)
        concat_Layer = Dropout(0.5)(concat_Layer)
        concat_Layer = Dense(64, activation = 'tanh')(concat_Layer)
        concat_Layer = Dropout(0.5)(concat_Layer)
        concat_Layer = Dense(4, activation = 'softmax')(concat_Layer)
        model = tensorflow.keras.Model(inputs=[eye_L_input, eye_R_input], outputs=[concat_Layer])


        model.summary()
        return model

#model = eye_model().get_model(20,24,24)




