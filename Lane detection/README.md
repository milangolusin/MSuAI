**Lane Finding Project**

# test_videos, test_images, and camera_cal folders required for running this #
		
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References) 

# Below paths are valid after executing the code, otherwise images and video are saved in ./docs/ with same names.

[image1]: ./pipeline_stages/undistort_output.png "Undistorted image"
[image2]: ./pipeline_stages/binary_combo_example.jpg "Binary Example image"
[image3]: ./pipeline_stages/road_trapezoid.jpg "Trapezoid shape road image"
[image4]: ./pipeline_stages/warped_straight_lines.jpg "Warped image"
[image5]: ./output_images/test1.jpg "Output image result"
[video1]: ./output_videos/output_challenge01.mp4 "Output video result"

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

    The camera calibration is handled in the calibrate method of the LaneTracker class. I used a set of given "chessboard" images located in the camera_cal folder. 
    Object points: I prepared object points, which represent the (x, y, z) coordinates of the chessboard corners in the real world. I assumed the chessboard is fixed on the (x, y) plane (z=0).
    Image points: I used cv2.findChessboardCorners to detect the (x, y) pixel coordinates of the corners in each calibration image.
    Calibration: When I got the object and image points, I used cv2.calibrateCamera() to calculate the camera matrix (mtx) and distortion coefficients (dist).
    Application: These coefficients are stored as the fields of the class and used in the pipeline method via cv2.undistort() to remove lens distortion from every frame.
    Example of a distortion corrected calibration image is saved to ./pipeline_stages/undistort_output.png
    
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

    In the pipeline method, the first step is resizing the image to 1280x720 and applying cv2.undistort using the calculated calibration data. This ensures that features like straight lane lines 
    do not appear curved because of lens optics. Example image is saved to ./pipeline_stages/undistort_output.png

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

    I used a combination of color space transforms in the pipeline method to isolate lane lines:
    HLS L-Channel: Converting the image to HLS and thresholding the L-channel (> 200) to identify white lane lines. This channel was chosen because white lines are best defined by high lightness.
    LAB B-Channel: Converting the image to LAB and thresholding the B-channel (> 155) to target yellow lane lines, which are more visible in the LAB color space under different 
    lighting conditions. 
    Combining these two binary masks into a single combined_binary image where a pixel is "ON" if it meets either the white or yellow criteria. Example image 
    is saved as ./pipeline_stages/binary_combo_example.jpg.
    Code implementation of this is done in #tresholding section of code:
	hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(undist, cv2.COLOR_BGR2Lab)
        l_channel, b_channel = hls[:, :, 1], lab[:, :, 2]
        
        white_bin = np.zeros_like(l_channel)
        white_bin[(l_channel > 200)] = 1
        yellow_bin = np.zeros_like(b_channel)
        yellow_bin[(b_channel > 155)] = 1
        combined_binary = np.zeros_like(white_bin)
        combined_binary[(white_bin == 1) | (yellow_bin == 1)] = 1

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

    I defined source points that form a trapezoid on the original road and destination points that form a rectangle, to create the birds-eye view ("to make a rectangle from the trapezoid").
    	Source Points: [[w*0.45, h*0.62], [w*0.15, h], [w*0.95, h], [w*0.55, h*0.62]].
    	Destination Points: [[w*0.25, 0], [w*0.25, h], [w*0.75, h], [w*0.75, 0]]. 
    Calculating the transformation matrix M by mapping trapezoid source points to rectangle destination points. This "streches" the image to a birds-eye view, making lane lines look parallel 
    for easier polynomial fitting. Also, calculating the inverse matrix Minv, by swapping the source and destination points, to return to road view.
    	M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)  // Minv later used in draw_line method.
    The idea is to return driver perspective image, not birds-eye view image, which is used for easier lane detection and lane drawing (visualisation).
    Example of a trapezoid image is saved as ./pipeline_stages/road_trapezoid.jpg
    Example of warped road image is saved as ./pipeline_stages/warped_straight_lines.jpg

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

    Lane-line pixels were identified and their positions were fitted using the fit_polynomial method, which is called inside pipeline method. Inside fit_polynomial method:
    1. To find where the lane lines start at the bottom of the image, I calculated a histogram of the lower half of the warped binary image. Since lane pixels are "ON" (value 1), 
    the two largest peaks in the histogram identify the base x-positions of the left and right lane lines.
    2. Once I had the starting x-positions, I used a Sliding Window method search to follow the lines toward the top of the image:
    	-I divided the image into 9 horizontal layers. In each layer, I placed a small window that's centered on the line's current position.
    	-I identified all "ON" pixels inside these windows and added them to a list for that specific lane line.
    	-If a window caught more than 40 pixels, I calculated their average x-position. The next window above it was then shifted to center itself on that new average.
    	-This shifting allows the boxes to go left and right ("moving like a snake"), following the curve of the road.
    3. Once all lane-line pixels were collected for both the left and right lines, I used np.polyfit() to fit a second-order polynomial to the data. Second-order is needed to 
    display curvature of the line. The curve is defined by equation: f(y)=Ay^2+By+C
    y-dependent function is used because the lane lines are mostly vertical in the warped image, which means change in x is 0, which would later break the math (dy/dx) by dividing with 0.
    In pipeline method after fit_polinomial is done: the resulting fit is averaged over 10 frames to ensure the lane detection is smooth. Instead of drawing the lane based 
    only on the current frame, the average of the last 10 frames is taken.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

    Radius of lane curvature is not implemented.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

    I implemented this in the draw_lane method, then generated a blank canvas and drew the polynomial lines.
    I used the inverse perspective matrix (Minv) to warp birds-eye lane back to the original camera perspective.
    I used cv2.addWeighted to overlay this lane onto the original undistorted image. 
    The example image of result is saved as ./output_images/test1.jpg

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

    The code processes all videos in the test_videos folder and saves the results in ./output_videos/ 
    Example video in /docs

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

    Shadows and Lighting: The primary challenge was finding a color threshold that worked for both bright concrete and dark shadows. 
    Combining L-channel (HLS) and B-channel (LAB) proved more robust than using RGB or Grayscale.
    Pipeline Failures: This pipeline might fail on very sharp turns (where the lines leave the side of the trapezoid).
    Improvements: To make it more robust, I would add a sanity check to ensure the left and right lanes are roughly parallel 
    and have a consistent distance between them.
    To make it more robust, I would implement a system that adjusts the thresholding values based on the average brightness of the image (using lower thresholds for night).

