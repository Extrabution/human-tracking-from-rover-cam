import cv2
import numpy as np
import os

images = os.listdir("images")
print(images)
img = []
for i in range(len(images)):
    img.append(cv2.imread(f"images/frame{str(i)}" + '.png'))

height, width, layers = img[1].shape

video = cv2.VideoWriter('video1.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 24, (width, height))

for j in range(len(img)):
    video.write(img[j])

cv2.destroyAllWindows()
video.release()