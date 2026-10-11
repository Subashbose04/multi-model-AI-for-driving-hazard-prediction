import cv2
import numpy as np

def test():
   
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('test_avc1.mp4', fourcc, 10.0, (200, 200))
    for i in range(10):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img.fill(i * 20)
        out.write(img)
    out.release()
    print("avc1 success")

test()
