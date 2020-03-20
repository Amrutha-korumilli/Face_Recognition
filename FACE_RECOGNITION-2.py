import cv2

import numpy as np

from os import listdir

from os.path import isfile,join

data_path='C:/Users/AMRUTHA/Desktop/image_proc/faces/'

onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]
print(onlyfiles)

training_data,labels=[],[]

for i,files in enumerate(onlyfiles):
           img_path=data_path +onlyfiles[i]
           imgs=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
           training_data.append(np.asarray(imgs,dtype='uint8'))
           labels.append(i)
labels=np.asarray(labels,dtype='int32')

model=cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(training_data),np.asarray(labels))

print("model training is complete")

