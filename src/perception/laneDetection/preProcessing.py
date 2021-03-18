import numpy as np
import cv2
import math
from scipy import stats
from collections import deque

def to_hsl(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

def isolate_yellow_hsl(img):
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([15, 38, 115], dtype=np.uint8)
    # Higher value equivalent pure HSL is (75, 100, 80)
    high_threshold = np.array([35, 204, 255], dtype=np.uint8)

    yellow_mask = cv2.inRange(img, low_threshold, high_threshold)

    return yellow_mask

def isolate_white_hsl(img):
    # Lower value equivalent pure HSL is (30, 45, 15)
    low_threshold = np.array([0, 200, 0], dtype=np.uint8)
    # Higher value equivalent pure HSL is (360, 100, 100)
    high_threshold = np.array([180, 255, 255], dtype=np.uint8)

    white_mask = cv2.inRange(img, low_threshold, high_threshold)

    return white_mask

def combine_hsl(img, hsl_yellow, hsl_white):
    hsl_mask = cv2.bitwise_or(hsl_yellow,hsl_white)
    return cv2.bitwise_and(img, img, mask=hsl_mask)

def filter_img_hsl(img):
    '''
    This function will need to be amended if not using yellow mask
    '''
    hsl_img = to_hsl(img)
    hsl_yellow = isolate_yellow_hsl(hsl_img)
    hsl_white = isolate_white_hsl(hsl_img)
    return combine_hsl(img, hsl_yellow, hsl_white)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(grayscale_img, kernel_size=5):
    '''
    Adjust kernel_size to tune the amount of blurring required
    '''
    return cv2.GaussianBlur(grayscale_img, (kernel_size, kernel_size), 0)

def getROI(img,mask_vertices):
    #define blank mask
    mask=np.zeros_like(img)

    #defining 3 channel or 1 channel color to fill mask
    if len(img.shape) > 2:
        channel_count = img.shape[2] # i.e. 3 or 4
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, mask_vertices, ignore_mask_color)

    #return image only where mask nonzero
    return cv2.bitwise_and(img, mask)

def hough_transform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def separate_lines(lines, img):
    img_shape = img.shape

    middle_x = img_shape[1] / 2

    left_lane_lines = []
    right_lane_lines = []
    horizontal_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1
            if dx == 0:
                #Discarding line since we can't gradient is undefined at this dx
                continue
            dy = y2 - y1

            # Similarly, if the y value remains constant as x increases, discard line
            # if dy == 0:
            #     continue

            slope = dy / dx

            # This is pure guess than anything...
            # but get rid of lines with a small slope as they are likely to be horizontal one
            epsilon = 0.1
            if abs(slope) <= epsilon:
                horizontal_lines.append([[x1, y1, x2, y2]])

            if slope < 0 and x1 < middle_x and x2 < middle_x:
                # Lane should also be within the left hand side of region of interest
                left_lane_lines.append([[x1, y1, x2, y2]])
            elif x1 >= middle_x and x2 >= middle_x:
                # Lane should also be within the right hand side of region of interest
                right_lane_lines.append([[x1, y1, x2, y2]])

    return left_lane_lines, right_lane_lines, horizontal_lines

def getLanesFormula(lines):
    xs = []
    ys = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)

    # a straight line is expressed as f(x) = Ax + b. Slope is the A, while intercept is the b
    return (slope, intercept)

def create_coefficients_list(length = 10):
    return deque(maxlen=length)

def mean_coefficients(coefficients_queue, axis=0):
    return [0, 0] if len(coefficients_queue) == 0 else np.mean(coefficients_queue, axis=axis)

def determine_line_coefficients(stored_coefficients, current_coefficients, MAXIMUM_SLOPE_DIFF=0.1, MAXIMUM_INTERCEPT_DIFF=50.0):

    if len(stored_coefficients)==0:
        stored_coefficients.append(current_coefficients)
        return current_coefficients
    mean = mean_coefficients(stored_coefficients)
    abs_slope_diff = abs(current_coefficients[0] - mean[0])
    abs_intercept_diff = abs(current_coefficients[1] - mean[1])

    if abs_slope_diff > MAXIMUM_SLOPE_DIFF or abs_intercept_diff > MAXIMUM_INTERCEPT_DIFF:
        # Identified big difference in slope
        # In this case use the mean
        return mean
    else:
        # Save our coefficients and returned a smoothened one
        stored_coefficients.append(current_coefficients)
        return mean_coefficients(stored_coefficients)

