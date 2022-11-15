import cv2
import os

class saver:
    def __init__(self):
        pass
    def save(self,file_name, img, path):
        # Set vị trí lưu ảnh
        os.chdir(path)
        # Lưu ảnh
        cv2.imwrite(file_name, img)