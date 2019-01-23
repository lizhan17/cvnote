import numpy as np
import cv2
import random

MAX_FEATURE = 300
Selected_ratio = 0.5



# brute force matching with SIFT descriptor and Ratio test
def align_sift(img, img_):
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    #brute knn match , set 2 
    matches = bf.knnMatch(des1,des2, k=2)


    good = []
    # ratio test
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if (len(matches[:,0]) >= 4):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    else:
        #print(“Cant find enough key points.”)
        return img

    dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]+100))
    dst[0:img.shape[0], 0:img.shape[1]] = img


    return dst
    d


# Brute-Force Matching with ORB Descriptors and knn match 

def align_orb_bf(img, img_):
    img1_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURE)

    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)


    # brute force matcher, with cross check true
    #bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck =True)
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(des1, des2,k=2)


    good = []
    # ratio test
    for m in matches:
        if m[0].distance < 0.75*m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if (len(matches[:,0]) >= 4):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    else:
        print('here bug!!')
        #return img

    dst = cv2.warpPerspective(img_, H ,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0], 0:img.shape[1]] = img

    return dst


"""
    matches = sorted(matches, key = lambda x:x.distance)

    num_goodmatching = int(len(matches)* Selected_ratio )
    matches = matches[:num_goodmatching]

    # draw matches
  #  img_matches = cv2.drawMatches(img1, kp1,img2, kp2, matches,None)


    # good points

    good_points1 = np.zeros((len(matches),2),dtype = np.float32)
    good_points2 = np.zeros((len(matches),2),dtype = np.float32)
    good_points2

    for i, match in enumerate(matches):
        good_points1[i,:] =kp1[match.queryIdx].pt
        good_points2[i,:] =kp2[match.trainIdx].pt


    h, mask = cv2.findHomography(good_points1, good_points2, cv2.RANSAC)

    #height, width, channels = img2.shape
  
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    # homogenious needs 4 mathces (8 points to compute a 8 para matrix)

    if len(mathces[:,0]) >=4 :
        src = np
"""


cap1 = cv2.VideoCapture('./data/case1-Left.mp4')
cap2 = cv2.VideoCapture('./data/case1-Right.mp4')

frame_width = int(cap1.get(3)) + int(cap2.get(3))
frame_height = int(cap1.get(4))+100

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 32, (frame_width,frame_height))
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    #print(fp)
    #fp +=1
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
   
    # if return value is right 
    if ret1 == True and ret2 == True: 
        img = align_sift(frame1, frame2)
        #img = align_orb_bf(frame1,frame2)

        out.write(img)
        #cv2.imshow('result',img)

    # if press Esc just break
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
