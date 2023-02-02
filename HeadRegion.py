import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
import copy

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

def GetHeadDict(img):
	Height, Width = img.shape[:2]
	# Width : Height = 400 : ?
	img = cv.resize(img, (400, int(400 * Height / Width)))
	Height, Width = img.shape[:2]

	temp = copy.deepcopy(img)
	temp = tf.image.resize_with_pad(tf.expand_dims(temp, axis=0), 384,640)
	temp = tf.cast(temp, dtype=tf.int32)
	# Detection section
	results = movenet(temp)
	keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

	ShowImage = copy.deepcopy(img)
	HeadDict = {}
	HeadDict['originalimg'] = img.copy()
	HeadList = []
	for person in keypoints_with_scores:
		y, x, c = ShowImage.shape

		shaped = np.squeeze(np.multiply(person, [y,x,1]))
		RightArmY, RightArmX = shaped[5,:2]
		LeftArmY, LeftArmX = shaped[6,:2]

		WidthHeight = int(abs(RightArmX - LeftArmX))
		HeadCenter = (int((RightArmX + LeftArmX) / 2), int((RightArmY + LeftArmY) / 2 - WidthHeight / 2))
		x0,y0,x1,y1=int(HeadCenter[0] - WidthHeight/2),int(HeadCenter[1] - WidthHeight/2),int(HeadCenter[0] + WidthHeight/2),int(HeadCenter[1] + WidthHeight/2)
		HeadImage = img[y0:y1,x0:x1]
		HeadImage = cv.resize(HeadImage, (100, 100))

		HeadList.append({
			"head" : HeadImage,
			"headpos" : (x0, y0, x1, y1),
			"bodypos" : (int(x0 - WidthHeight/3),y0,int(x1 + WidthHeight/3),int(y1 + WidthHeight * 3))
		})
	HeadDict['headlist'] = HeadList
	return HeadDict