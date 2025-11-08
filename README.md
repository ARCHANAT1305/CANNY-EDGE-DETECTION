# CANNY-EDGE-DETECTION
## AIM: 
To implement the canny edge detection using python program.
## PROGRAM:
### NAME: ARCHANA T
### REGISTER NUMBER : 212223240013
```
from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt
uploaded = files.upload()
filename = next(iter(uploaded))
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found or cannot be opened.")
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(img)
sharpened = cv2.addWeighted(enhanced, 1.5, cv2.Laplacian(enhanced, cv2.CV_8U), -0.5, 0)
blurred = cv2.GaussianBlur(sharpened, (5,5), 0)
v = np.median(blurred)
lower = int(max(0, 0.4 * v))  
upper = int(min(255, 1.4 * v))
edges = cv2.Canny(blurred, lower, upper)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title('Detected Edges (Enhanced)')
plt.axis('off')

plt.show()

```
## OUTPUT:
<img width="415" height="516" alt="image" src="https://github.com/user-attachments/assets/df61a3ef-013f-47ae-8629-04e63b386c1a" />
<img width="561" height="781" alt="image" src="https://github.com/user-attachments/assets/a7d67c8a-bf68-4476-b5d6-2d65290358d6" />

## RESULT:
Thus the program to implement the canny edge detection was executed successfully.

