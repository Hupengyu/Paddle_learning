import cv2

img = cv2.imread(r'F:\python\work\cv_learn\clipboard.png',1)
cv2.imshow('input',img)
result = img.copy()
for j in range(3):
	result[:, :, j] = cv2.equalizeHist(img[:,:,j])
cv2.imshow('Result1', result)
cv2.waitKey(0)