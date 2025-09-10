import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./image/image_8610.pgm", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title("PGM Image")
plt.axis("off")
plt.show()
