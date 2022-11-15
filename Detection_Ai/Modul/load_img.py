import glob
import numpy as np
import cv2
import pandas as pd
import os

def face_detection(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier('D:\Study\AI\Project\Detection_Ai\Modul\haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        new_img = gray[y:y+h, x:x+w]
        new_img = cv2.resize(src=new_img, dsize=(200, 200))
        return new_img[:,:]

def load_face(path):
    img =  cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_name_form_path(path):
    arr = path.split('\\')[-1]
    arr = arr.split('_')[-1]
    name = arr.split('.')[0]
    return name

class load_img:
    def __init__(self):
        pass
    W, H = 200, 200
    
    def load_im(self, path):
        X = []  # Lưu ma trận ảnh
        Y = []  # Lưu nhãn tên
        df = pd.DataFrame(columns=['Name', 'Quantity']) # Lưu tên và số lượng 
        # File chứa thư mục ảnh
        path = 'D:\\Study\\AI\\Project\\Detection_Ai\\image_training\\'
        # list_link chứa link các thư mục ảnh
        list_link = glob.glob(os.path.join(path, "*"))
        
        cnt_name = 0
        tmp = 'tmp'
        name =''
        for link in list_link:
            for filename in glob.glob( link + '\\' + '*.png'): 
                img = load_face(filename)
                img = cv2.resize(src=img, dsize=(200, 200))
                X.append(img.reshape(self.W * self.H))
                name = get_name_form_path(filename)
                Y.append(name)
                if not(name in np.array(df['Name'])):
                    df.loc[df.shape[0]] = [name, 1]
                else:
                    i = 0
                    while df['Name'][i] != name:
                        i+=1
                    a = df.iloc[i][1]
                    a += 1
                    df.loc[i] = [name, a]
        return np.array(X), np.array(Y), df

#ld = load_img()
#X, Y, df = ld.load_im('D:\\Study\\AI\\Project\\Detection_Ai\\image_training\\')
#print(df)