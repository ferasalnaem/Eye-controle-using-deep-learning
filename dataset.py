import os
import tensorflow as tf
import numpy as np

class Eyes:
    def get_eyes_data(self):
        rawPath='/home/feras/ProjectEye/Raw3/'
        motionName = ['N','B','R','L']
        for i in motionName:
            if i == 'N':
                y_label = np.array([1,0,0,0])
            elif i == 'B':
                y_label = np.array([0,1,0,0])
            elif i == 'R':
                y_label = np.array([0,0,1,0])
            elif i == 'L':
                y_label = np.array([0,0,0,1])
            destinationPath = rawPath + i + '/'
            files = os.listdir(destinationPath)
            motionIndex = 0
            for file_count in files:
                number = int((file_count.split(".")[0]).split("_")[-1])
                motionIndex = max(motionIndex, number + 1)
            for count in range(0,motionIndex):
                EyeLeft_file  = destinationPath + 'EyeLeft_'+ i + '_' + str(count) + '.npy'
                EyeRight_file = destinationPath + 'EyeRight_'+ i + '_' + str(count) + '.npy'
                EyeLeft_array = np.load(EyeLeft_file)
                EyeRight_array = np.load(EyeRight_file)
                if count == 0 and i == 'N':
                    EyeLeft_dataset = EyeLeft_array.reshape(1,20,24,24,1)
                    EyeRight_dataset = EyeRight_array.reshape(1, 20, 24, 24, 1)
                    feature_dataset = y_label.reshape (1,4)
                else :
                    EyeLeft_array = EyeLeft_array.reshape(1,20,24,24,1)
                    EyeRight_array = EyeRight_array.reshape(1, 20, 24, 24, 1)
                    y_label = y_label.reshape (1,4)
                    EyeLeft_dataset = np.concatenate((EyeLeft_dataset, EyeLeft_array))
                    EyeRight_dataset = np.concatenate((EyeRight_dataset,EyeRight_array))
                    feature_dataset = np.concatenate((feature_dataset, y_label))

        dataset_length = feature_dataset.shape[0]

        def our_generators():
            for i in range(dataset_length):
                eye_left = EyeLeft_dataset[i]
                eye_right = EyeRight_dataset[i]
                eye_label = feature_dataset[i]
                yield eye_left, eye_right, eye_label

        train_size = int(0.87 * dataset_length)

        full_eye_dataset = tf.data.Dataset.from_generator(our_generators, (tf.int16, tf.int16, tf.int16))
        full_eye_dataset = full_eye_dataset.shuffle(dataset_length, reshuffle_each_iteration=False)

        train_dataset = full_eye_dataset.take(train_size)
        val_dataset_final = full_eye_dataset.skip(train_size)
        train_dataset_final = train_dataset
#        train_dataset_final = train_dataset.repeat(3)


        return train_dataset_final, val_dataset_final


#x = Eyes().get_eyes_data()