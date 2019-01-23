"""
what i added:
orginal comments are removed

source:
https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html


VideoCaputre cap object
cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.





"""

import numpy as np
import cv2 as cv
import time 

### get a video capture object using file slow.flv
cap = cv.VideoCapture('./data/case1-Right.mp4')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
##  stop the algorithm iteration if specified accuracy
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()

## covert to grey color space
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

## goodFeaturesToTrack 
  ### "Determines strong corners on an image."
  ### 

start_time = time.time()

p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(len(p0))
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
frameNum = 0
store  = []
while(1):
    frameNum +=1
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    print("err is")
    print(err)
    
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    if frameNum == 398:
        print("--- %s seconds ---" % (time.time() - start_time))

cv.destroyAllWindows()
cap.release()

