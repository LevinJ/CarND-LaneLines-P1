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
        self.low_threshold = 30
        self.high_threshold = 80
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
   
    def filter_outlier_lines(self, lines):
        x1 = lines[:,0]
        y1 = lines[:,1]
        x2 = lines[:,2]
        y2 = lines[:,3]
        #filter those line segments that that very different slope
        k = ((y2-y1)/(x2-x1).astype(np.float32))
       
        
        #filter those lines are near horiztal
        print abs(k)
        lines = lines[abs(k) > 0.5]     
        return lines

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
    def run(self):
#         self.test_on_one_image('../test_images/challenge_125.jpg')
#         self.test_on_one_image('../test_images/challenge_video_frame.jpg')
#         plt.show()
        self.test_on_videos('../challenge.mp4','../extra.mp4')

       
        
        return






if __name__ == "__main__":   
    obj= LaneDetectionChallenge()
    obj.run()