import numpy as np 
import cv2 
import math

def to_hsl(img):
	return cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

def isolate_yellow_hsl(img):
	# Lower value equivalent pure HSL is (30, 45, 15)
	low_threshold = np.array([15, 38, 115], dtype=np.uint8)
	# Higher value equivalent pure HSL is (75, 100, 80)
	high_threshold = np.array([35, 204, 255], dtype=np.uint8)  

	yellow_mask = cv2.inRange(img, low_threshold, high_threshold)

	return yellow_mask

def isolate_while_hsl(img):
	# Lower value equivalent pure HSL is (30, 45, 15)
	low_threshold = np.array([0, 200, 0], dtype=np.uint8)
	# Higher value equivalent pure HSL is (360, 100, 100)
	high_threshold = np.array([180, 255, 255], dtype=np.uint8)  

	white_mask = cv2.inRange(img, low_threshold, high_threshold)

	return white_mask

def combine_hsl(img, hsl_yellow, hsl_white):
	hsl_mark = cv2.bitwise_or(hsl_yellow,hsl_white)
	return cv2.bitwise_and(img, img, mask=hsl_mask)

def filter_img_hsl(img):
    '''
    This function will need to be amended if not using yellow mask
    '''
    hsl_img = to_hsl(img)
    hsl_yellow = isolate_yellow_hsl(hsl_img)
    hsl_white = isolate_white_hsl(hsl_img)
    return combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(grayscale_img, kernel_size=5):
    '''
    Adjust kernel_size to tune the amount of blurring required
    '''
    return cv2.GaussianBlur(grayscale_img, (kernel_size, kernel_size), 0)

def getVertices(img):
	"""This functions needs to be hard coded for 
	our resolution - might need tuning 
	"""
	img_shape = img.shape
	height = img_shape[0]
	width = img_shape[1]

	vert = None

	if (width, height) == (960, 540):
	    region_bottom_left = (130 ,img_shape[0] - 1)
	    region_top_left = (410, 330)
	    region_top_right = (650, 350)
	    region_bottom_right = (img_shape[1] - 30,img_shape[0] - 1)
	    vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
	else:
		region_bottom_left = (200 , 680)
	    region_top_left = (600, 450)
	    region_top_right = (750, 450)
	    region_bottom_right = (1100, 650)
	    vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

	return vert

def getROI(img):
	#define blank mask
	mask=np.zeros_like(img)

	#defining 3 channel or 1 channel color to fill mask
	if len(img.shape) > 2:
		channel_count = img.shape[2] # i.e. 3 or 4
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	vert = getVertices(img)

	#filling pixels inside the polygon defined by "vertices" with the fill color    
	cv2.fillPoly(mask, vert, ignore_mask_color)

	#return image only where mask nonzero
	return cv2.bitwise_and(img, mask)





