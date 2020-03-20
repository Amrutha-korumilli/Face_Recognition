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


face_classifier=cv2.CascadeClassifier('C:/Users/AMRUTHA/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    f=face_classifier.detectMultiScale(gray,1.3,3)
    if f is ():
        return img,[]
    
    for(x,y,w,h) in f:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi

cap=cv2.VideoCapture(0)  



while True:
    ret,frame=cap.read()

    image,face=face_detector(frame)
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)

        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))
            display_string=str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

        if confidence>75:
            cv2.putText(image,'unlocked',(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('face cropped',image)
        else:
            cv2.putText(image,'locked',(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('face cropped',image)
            
    except:
        cv2.putText(image,'face not found',(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)

        cv2.imshow('face cropped',image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()

