import cv2
import os
import numpy as np
    
if __name__ == '__main__':
    name = os.listdir('./cell_images')
    np.random.shuffle(name)
    train = name[:int(0.80*len(name))]
    test = name[int(0.80*len(name)):]
    
    os.mkdir('./train')
    for i in range(0,len(train)):
        nm = train[i]
        img = cv2.imread('./cell_images/'+nm, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('./train/'+nm,img)
    
    os.mkdir('./test')
    for i in range(0,len(test)):
        nm = test[i]
        img = cv2.imread('./cell_images/'+nm, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('./test/'+nm,img)
