import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import os
from moviepy.editor import VideoFileClip
from lanedetection import LaneDetection


class LaneDetectionChallenge(LaneDetection):
    
    def __init__(self):
        LaneDetection.__init__(self)
        self.vertices = np.array([[(100,720),(580,450), (732,450), (1240,720)]], dtype=np.int32)
        self.low_threshold = 50
        self.high_threshold = 150
        
        self.min_line_len = 60
        self.max_line_gap = 5
        self.threshold = 30
        self.img_frame_id = 1
        return
    def extrapolate_lines(self, lines, y_bottom):
        left_lines = []
        right_lines = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                k = (y2-y1)/float((x2-x1))
                if k < 0:
                    left_lines.append((x1,y1,x2,y2))
                if k > 0:
                    right_lines.append((x1,y1,x2,y2))
            
        y_top = 450
        left_line = self.extrapolate_one_lane(left_lines, y_bottom, y_top)  
        right_line = self.extrapolate_one_lane(right_lines, y_bottom, y_top)
        
        twolines = np.concatenate((left_line, right_line))[np.newaxis,:]
        return twolines
    def get_weighted_average_line_equation(self, lines):
        res = lines.copy()

        x1 = lines[:,0]
        y1 = lines[:,1]
        x2 = lines[:,2]
        y2 = lines[:,3]
        
        len = np.sqrt((y2-y1) **2 + (x2-x1)**2)
        weights = len/len.sum()
        k = (y2 - y1)/(x2 - x1).astype(np.float32)
        b = (y2 - k * x2).astype(np.float32)
        return (k*weights).sum(), (b*weights).sum()
#         return k.mean(), b.mean()
    def extend2_top_bottom_lines(self, lines, y_bottom, y_top):
        k, b = self.get_weighted_average_line_equation(lines)
        print 'k_mean {} b_mean {}'.format(k,b)
        
        x_bottom = int((y_bottom - b)/k)
        x_top = int((y_top - b)/k)
        res = np.array([x_bottom, y_bottom, x_top,y_top])[np.newaxis,:]
#         for line in lines:
#             extended_lines = self.extrapolate_one_line(line, y_bottom, y_top, k)
#             if not extended_lines is None:
#                 res = np.concatenate((res, extended_lines))
        return res
    def filter_outlier_lines(self, lines):
        x1 = lines[:,0]
        y1 = lines[:,1]
        x2 = lines[:,2]
        y2 = lines[:,3]
        #filter those line segments that that very different slope
        k = ((y2-y1)/(x2-x1).astype(np.float32))
       
        
        #filter those lines are near horiztal
        print "K value before filtering" , abs(k)
        lines = lines[abs(k) > 0.5]     
        print "lines after filtering" , lines
        return lines
    def format_coord(self, x, y):
        pt = self.hsv_img[y, x, :]
        return 'HSV value, x={:.0f}, y={:.0f}  [h={}, s={}, v={}]'.format(x, y, pt[0],pt[1],pt[2])
    def yellowtowhite(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV);
         
        lowerb = np.array([20, 100, 100], dtype=np.uint8)
        upperb = np.array([30, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lowerb, upperb)
        res_img = img.copy()
        res_img[mask == 255] = [255,255,255] 
        
        # for the purpose of showing hsv value
#         self.hsv_img = hsv_img 
#         _, ax = plt.subplots()
#         ax.format_coord = self.format_coord
#         ax.imshow(img)
        return res_img
    def visualize_roi(self, lines_img):
#         color = [0, 255, 0]
#         thickness= 5
#         cv2.polylines(lines_img, self.vertices, True, color=color, thickness= thickness)
#         plt.imshow(lines_img)
        return lines_img
    
#     def process_image(self, initial_img):
#         self.img_frame_id  +=1
#         if self.img_frame_id == 125:
#             self.save_image(initial_img, '../test_images/challenge_125.jpg')
#         plt.imshow(initial_img)    
#         return initial_img
    def test_hsv(self):
        img = self.load_image('../data/challenge_yellow.JPG')
#         plt.imshow(img)
        self.yellowtowhite(img)
        plt.show()
        return
    def run(self):
#         self.test_hsv()
#         self.test_on_one_image('../test_images/challenge_125.jpg')
#         self.test_on_one_image('../test_images/challenge_video_frame.jpg')
#         plt.show()
        self.test_on_videos('../challenge.mp4','../extra.mp4')

       
        
        return






if __name__ == "__main__":   
    obj= LaneDetectionChallenge()
    obj.run()