import glob
import cv2
from utils import text_detection
#IMAGE_PATH = 'D:\\pcb_project\\test_images\\s27_front.jpg'
IMAGE_PATH = 'D:\pcb_project\s27_front.jpg\ICs\IC0.png'

img = cv2.imread(IMAGE_PATH)

#(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

#img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
val = text_detection(img)
print(val)



