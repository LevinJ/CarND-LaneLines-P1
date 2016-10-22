import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import os
from moviepy.editor import VideoFileClip

class LaneDetection:
    
    def __init__(self):
        return
    def load_image(self, image_path):
        #reading in an image
        image = mpimg.imread(image_path)
        #printing out some stats and plotting
        print('This image is:', type(image), 'with dimesions:', image.shape)
        return image
    def save_image(self, img, img_path):
        mpimg.imsave(img_path, img)
        return
    def grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)
    def gaussian_noise(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=8):
        """
        NOTE: this is the function you might want to use as a starting point once you want to 
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).  
        
        Think about things like separating line segments by their 
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of 
        the lines and extrapolate to the top and bottom of the lane.
        
        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        lines = self.extrapolate_lines(lines, img.shape[0])
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.
            
        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros(list(img.shape) + [3], dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img
    def weighted_img(self, img, initial_img, a=0.8, b=1., lambda_param=0.0):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.
        
        `initial_img` should be the image before any processing.
        
        The result image is computed as follows:
        
        initial_img * a + img * b + lambda_param
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, a, img, b, lambda_param)
    def process_image(self, initial_img):
        #get gray images
        gray_img = self.grayscale(initial_img)
#         plt.imshow(gray_img, cmap='gray')
        #Guassian blur
        kernel_size = 5
        blur_img = self.gaussian_noise(gray_img, kernel_size)
#         plt.imshow(blur_img, cmap='gray')
        #canny edge detection
        low_threshold = 50
        high_threshold = 150
        canny_edges = self.canny(blur_img, low_threshold, high_threshold)
#         plt.imshow(canny_edges, cmap='gray')
        
        #crop region of interest
        vertices = np.array([[(10,539),(460,320), (495,320), (930,539)]], dtype=np.int32)
        roi_img = self.region_of_interest(canny_edges, vertices)
        plt.imshow(roi_img, cmap='gray')

        #hough line detection
        rho = 1
        theta = np.pi/180
        threshold = 2
        min_line_len = 15
        max_line_gap = 5
        lines_img = self.hough_lines(roi_img, rho, theta, threshold, min_line_len, max_line_gap)
        plt.imshow(lines_img, cmap='gray')

        #blendign the images
        a = 0.8
        b = 0.2
        lambda_param = 20
        final_img = self.weighted_img(lines_img, initial_img, a, b, lambda_param)
        
        return  final_img
    def process_image_file_path(self, img_file_path):
        #load the image
        initial_img = self.load_image(img_file_path)
#         plt.imshow(initial_img)
        final_img = self.process_image(initial_img)

        img_file_name = os.path.basename(img_file_path)
        new_img_file_path = os.path.dirname(img_file_path) + '/' + os.path.splitext(img_file_name)[0] + '_withlane' + os.path.splitext(img_file_name)[1]
        self.save_image(final_img, new_img_file_path)
       
        plt.imshow(final_img,cmap='gray' )
        plt.show()

        
        return final_img
    def extrapolate_lines(self, lines, y_bottom):
        left_lines = []
        right_lines = []
        y_top = 1000
        for line in lines:
            for x1,y1,x2,y2 in line:
                y_min = min(y1, y2)
                y_top = min(y_top, y_min)
                k = (y2-y1)/float((x2-x1))
                if k < 0:
                    left_lines.append((x1,y1,x2,y2))
                if k > 0:
                    right_lines.append((x1,y1,x2,y2))
            
        left_line = self.extrapolate_one_lane(left_lines, y_bottom, y_top)  
        right_line = self.extrapolate_one_lane(right_lines, y_bottom, y_top)
        
        twolines = np.concatenate((left_line, right_line))[np.newaxis,:]
        return twolines
    def extrapolate_one_lane(self,lines, y_bottom, y_top):
        res = self.extend2_top_bottom(lines, y_bottom, y_top)
        #         res = self.connect_top_bottom(lines, y_bottom)     
        return res
    def filter_outlier_lines(self, lines):
        x1 = lines[:,0]
        y1 = lines[:,1]
        x2 = lines[:,2]
        y2 = lines[:,3]
        
#         line_length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        # if the line segment is less than a certain amout, we ignore it.
        #This is to avoid extending some noise line segments
#         lines = lines[line_length > 15]
       
        
        #filter those line segments that that very different slope
        k = ((y2-y1)/(x2-x1).astype(np.float32))
        q75, q25,k_median = np.percentile(k, [75 ,25,50])
        iqr = q75 - q25
        #here outlieris 1.5 * irq away from the median
        lines = lines[abs((k-k_median)) < 0.2]
#         k_average = ((y2-y1)/(x2-x1).astype(np.float32)).mean()
#         lines = lines[(k - k_average) > 0.5]
        
      
        return lines
    def extend2_top_bottom(self,lines, y_bottom, y_top):
        #Each line extend to both bottom and top
        lines = np.array(lines)
        res = lines.copy()
        lines = self.filter_outlier_lines(lines)
#         if filtered_lines.size ==0:
#             return res
        
        
        
        #reinitalize 
        x1 = lines[:,0]
        y1 = lines[:,1]
        x2 = lines[:,2]
        y2 = lines[:,3]
        
        
        
#         y_top = min(y1.min(), y2.min())
        k_average = ((y2-y1)/(x2-x1).astype(np.float32)).mean()
        
        
        for line in lines:
            extended_lines = self.extrapolate_one_line(line, y_bottom, y_top, k_average)
            if not extended_lines is None:
                res = np.concatenate((res, extended_lines))
        return res
    def extrapolate_one_line(self, line, y_bottom, y_top, k_average):
        #return two lines, one to the top, the other to the bottom
        x1,y1,x2,y2 = line
        res = []
        
        if y1 < y2:
            x_near_top = x1
            y_near_top = y1
            x_near_bottom = x2
            y_near_bottom = y2
        else:
            x_near_top = x2
            y_near_top = y2
            x_near_bottom = x1
            y_near_bottom = y1
        
        x_bottom  = int((y_bottom - y_near_bottom)/k_average + x_near_bottom)
        x_top  = int((y_top - y_near_top)/k_average + x_near_top)
        
        to_top_line = [x_near_top, y_near_top, x_top, y_top]
        to_bottom_line = [x_near_bottom, y_near_bottom, x_bottom, y_bottom]
        res.append(to_top_line)
        res.append(to_bottom_line)
             
        return np.array(res)
    def connect_top_bottom(self,lines, y_bottom):
        lines = np.array(lines)
        
        x1 = lines[:,0]
        y1 = lines[:,1]
        x2 = lines[:,2]
        y2 = lines[:,3]
        
        y_top = min(y1.min(), y2.min())
        k_average = ((y2-y1)/(x2-x1).astype(np.float32)).mean()
        x_min = min(x1.min(), x2.min())
        x_max = max(x1.max(), x2.max())
        #get x_top
        if k_average < 0:
            x_top = x_max
        else:
            x_top = x_min
        x_bottom  = int((y_bottom - y_top)/k_average + x_top) 
            
        res = np.array((x_bottom, y_bottom, x_top, y_top))[np.newaxis,:]
        return res
    def test_on_one_image(self, img_file_path):
        self.process_image_file_path(img_file_path)
        return
    def test_on_images(self):
        img_file_paths = ['solidWhiteCurve.jpg',
                             'solidWhiteRight.jpg',
                             'solidYellowCurve.jpg',
                             'solidYellowCurve2.jpg',
                             'solidYellowLeft.jpg',
                             'whiteCarLaneSwitch.jpg']
        img_file_paths = ['../test_images/'+ file_path for file_path in img_file_paths]
        for img_file_path in img_file_paths:
            self.process_image_file_path(img_file_path)
#             break
            
        
        return
    def test_on_videos(self, input_video, output_video):
        clip1 = VideoFileClip(input_video)
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(output_video, audio=False)
        return
    def run(self):
#         self.test_on_one_image('../test_images/solidYellowCurve.jpg')
#         self.test_on_images()
#         self.test_on_videos('../solidWhiteRight.mp4','../white.mp4')
#         self.test_on_videos('../solidYellowLeft.mp4','../yellow.mp4')
#         self.test_on_videos('../challenge.mp4','../extra.mp4')

        plt.show()
        
        return






if __name__ == "__main__":   
    obj= LaneDetection()
    obj.run()