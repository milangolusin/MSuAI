import numpy as np
import cv2
import glob
import os

class LaneTracker:
    def __init__(self):
        self.mtx = None
        self.dist = None
        self.left_fits = []
        self.right_fits = []

    def calibrate(self, calibration_folder, nx=9, ny=6):
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        objpoints, imgpoints = [], []
        images = glob.glob(os.path.join(calibration_folder, 'calibration*.jpg'))
        
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        if objpoints:
            ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)

    def pipeline(self, img, save_diag=False):
        # resize and undistort
        img = cv2.resize(img, (1280, 720))
        h, w = img.shape[:2]
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        
        # define source and destination points
        src = np.float32([[w*0.45, h*0.62], [w*0.15, h], [w*0.95, h], [w*0.55, h*0.62]])
        dst = np.float32([[w*0.25, 0], [w*0.25, h], [w*0.75, h], [w*0.75, 0]])

        if save_diag:
            if not os.path.exists('pipeline_stages'): os.makedirs('pipeline_stages')
            # image 1 - undistorted
            cv2.imwrite('pipeline_stages/undistort_output.png', undist)
            
            # image 2 - road trapezoid
            road_transformed_vis = undist.copy()
            pts = np.array(src, np.int32).reshape((-1, 1, 2))
            cv2.polylines(road_transformed_vis, [pts], True, (0, 0, 255), 5) 
            cv2.imwrite('pipeline_stages/road_trapezoid.jpg', road_transformed_vis)

        # thresholding
        hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(undist, cv2.COLOR_BGR2Lab)
        l_channel, b_channel = hls[:, :, 1], lab[:, :, 2]
        
        white_bin = np.zeros_like(l_channel)
        white_bin[(l_channel > 200)] = 1
        yellow_bin = np.zeros_like(b_channel)
        yellow_bin[(b_channel > 155)] = 1
        combined_binary = np.zeros_like(white_bin)
        combined_binary[(white_bin == 1) | (yellow_bin == 1)] = 1

        # image 3 - binary example
        if save_diag:
            cv2.imwrite('pipeline_stages/binary_combo_example.jpg', combined_binary * 255)

        # perspective transform
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(combined_binary, M, (w, h))

        # image 4 - warped road
        if save_diag:
            warped_road = cv2.warpPerspective(undist, M, (w, h))
            cv2.imwrite('pipeline_stages/warped_straight_lines.jpg', warped_road)

        # polynomial fit
        left_fit, right_fit, _ = self.fit_polynomial(warped)
        
        if left_fit is not None:
            self.left_fits.append(left_fit)
            self.right_fits.append(right_fit)
            if len(self.left_fits) > 10:
                self.left_fits.pop(0)
                self.right_fits.pop(0)
        
        if not self.left_fits: return undist

        avg_left = np.mean(self.left_fits, axis=0)
        avg_right = np.mean(self.right_fits, axis=0)

        result = self.draw_lane(undist, warped, avg_left, avg_right, Minv)

        return result

    def fit_polynomial(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        if np.sum(histogram) < 1000: return None, None, None 

        midpoint = np.int64(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows, margin, minpix = 9, 80, 40
        window_height = np.int64(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        
        left_lane_inds, right_lane_inds = [], []
        leftx_curr, rightx_curr = leftx_base, rightx_base

        for window in range(nwindows):
            y_low, y_high = binary_warped.shape[0]-(window+1)*window_height, binary_warped.shape[0]-window*window_height
            win_xl_l, win_xl_h = leftx_curr - margin, leftx_curr + margin
            win_xr_l, win_xr_h = rightx_curr - margin, rightx_curr + margin
            
            good_l = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= win_xl_l) & (nonzerox < win_xl_h)).nonzero()[0]
            good_r = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= win_xr_l) & (nonzerox < win_xr_h)).nonzero()[0]
            
            left_lane_inds.append(good_l)
            right_lane_inds.append(good_r)
            
            if len(good_l) > minpix: leftx_curr = np.int64(np.mean(nonzerox[good_l]))
            if len(good_r) > minpix: rightx_curr = np.int64(np.mean(nonzerox[good_r]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        if len(left_lane_inds) < 100 or len(right_lane_inds) < 100:
            return None, None, None

        left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
        right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
        return left_fit, right_fit, np.linspace(0, 719, 720)

    def draw_lane(self, undist, warped, left_fit, right_fit, Minv):
        ploty = np.linspace(0, 719, 720)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        line_canvas = np.zeros_like(undist)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)

        cv2.polylines(line_canvas, pts_left, False, (0, 255, 0), thickness=20)
        cv2.polylines(line_canvas, pts_right, False, (0, 255, 0), thickness=20)
        
        newwarp = cv2.warpPerspective(line_canvas, Minv, (1280, 720))
        return cv2.addWeighted(undist, 1, newwarp, 1.0, 0)

# ---EXECUTION---

tracker = LaneTracker()
tracker.calibrate("camera_cal")

# process images and place them in output_images folder
image_files = glob.glob(os.path.join('test_images', '*.jpg'))
output_image_dir = 'output_images'
if not os.path.exists(output_image_dir): os.makedirs(output_image_dir)

# first run of the pipeline (save_diag = True) for pipeline_stages images generating (showing test1.jpg stages if exists)
if len(image_files) > 0:
    diag_img_path = next((f for f in image_files if 'test1.jpg' in f), image_files[0])
    diag_img = cv2.imread(diag_img_path)
    if diag_img is not None:
        tracker.left_fits, tracker.right_fits = [], []
        tracker.pipeline(diag_img, save_diag=True)
print("Diagnostics images processed and saved in pipeline_stages/")

# second run of the pipeline (save_diag = False) for standard processing
for img_path in image_files:
    img = cv2.imread(img_path)
    if img is not None:
        tracker.left_fits, tracker.right_fits = [], [] 
        result = tracker.pipeline(img, save_diag=False)
        save_path = os.path.join(output_image_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, result)
print("All test images processed and saved in output_images/")

# process videos and place them in output_videos folder
video_files = glob.glob(os.path.join('test_videos', '*.mp4'))
output_video_dir = 'output_videos'
if not os.path.exists(output_video_dir): os.makedirs(output_video_dir)

for video_path in video_files:
    tracker.left_fits, tracker.right_fits = [] , []
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 25 
    
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_video_dir, f"output_{video_name}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        result = tracker.pipeline(frame)
        
        if (result.shape[1], result.shape[0]) != (width, height):
            result = cv2.resize(result, (width, height))
        
        out.write(result)

    cap.release()
    out.release()
    print(f"Finished video: {video_name}. Saved to output_videos/")
    
cv2.destroyAllWindows()