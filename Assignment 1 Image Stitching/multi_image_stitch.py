# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:33:40 2022

@author: bsr
"""

import cv2
import numpy as np
#import glob
from imutils import paths

def find_keypoints (image):
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
    md = cv2.xfeatures2d.SIFT_create()
    keypts1, ftrs1 = md.detectAndCompute(img1, None)
    #keypts2, ftrs2 = md.detectAndCompute(img2, None)
    ftrs1 = np.float32(ftrs1)
    #ftrs2 = np.float32(ftrs2)
    return keypts1, ftrs1

def find_matchers(ftrs1, ftrs2):
    #match = cv2.BFMatcher()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    match = cv2.FlannBasedMatcher(index_params, search_params)
    matches = match.knnMatch(ftrs1,ftrs2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>4:
        return good
    else:
        print("Not enough matches")
    return None

def homography(src_image, dest_image):
    src_kp, ftr1 = find_keypoints(src_image)
    dest_kp, ftr2 = find_keypoints(dest_image)
    good = find_matchers(ftr1, ftr2)
    src_points = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_points = np.float32([dest_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask  = cv2.findHomography(src_points, dst_points, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    return H,matchesMask
def create_Mask(h, w, bar, window, left_biased=True):
    assert bar < w
    m = np.zeros((h, w))
    o = int(window/2)
    try:
        if left_biased:
            m[:,bar-o:bar+o+1]=np.tile(np.linspace(1,0,2*o+1).T, (h, 1))
            m[:,:bar-o] = 1
        else:
            m[:,bar-o:bar+o+1]=np.tile(np.linspace(0,1,2*o+1).T, (h, 1))
            m[:,bar+o:] = 1
    except:
        if left_biased:
            m[:,bar-o:bar+o+1]=np.tile(np.linspace(1,0,2*o).T, (h, 1))
            m[:,:bar-o] = 1
        else:
            m[:,bar-o:bar+o+1]=np.tile(np.linspace(0,1,2*o).T, (h, 1))
            m[:,bar+o:] = 1
    return cv2.merge([m, m, m])

def blending_algo(img2_,img1_warped,img2_width,side):
    h,w,_= img2_.shape
    window=int(img2_width/8)
    bar = img2_width -int(window/2)
    m1 = create_Mask(h, w, bar, window, left_biased = True)
    m2 = create_Mask(h, w, bar, window, left_biased = False)
    no_blend=None
    left=None
    right=None
    if side=='left':
        img2_=cv2.flip(img2_,1)
        img1_warped=cv2.flip(img1_warped,1)
        img2_=(img2_*m1)
        img1_warped=(img1_warped*m2)
        img=img1_warped+img2_
        img=cv2.flip(img,1)
    else:
        img2_=(img2_*m1)
        img1_warped=(img1_warped*m2)
        img=img1_warped+img2_
    return img,no_blend,left,right

def warping_img(img1, img2):
    H,_= homography(img1,img2)
    img1_height,img1_width = img1.shape[:2]
    img2_height,img2_width = img2.shape[:2]
    #print (img1_height, img1_width, img2_height, img2_width)
    corners1 = np.float32([[0,0],[0,img1_height],[img1_width,img1_height],[img1_width,0]]).reshape(-1,1,2)
    corners2 = np.float32([[0,0],[0,img2_height],[img2_width,img2_height],[img2_width,0]]).reshape(-1,1,2)
    corners1_ = cv2.perspectiveTransform(corners1, H)
    corners = np.concatenate((corners1_, corners2), axis=0)
    [minx, miny] = np.int64(corners.min(axis=0).ravel() - 0.5)
    [maxx, maxy] = np.int64(corners.max(axis=0).ravel() + 0.5)
    t = [-minx,-miny]
    if(corners[0][0][0]<0):
        side='left'
        img_width =img2_width+t[0]
    else:
        img_width=int(corners1_[3][0][0])
        side='right'
    img_height=maxy-miny
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 
    img1_warped = cv2.warpPerspective(img1, Ht.dot(H), (img_width,img_height))
    img2_=np.zeros((img_height,img_width,3))
    if side=='left':
        img2_[t[1]:img1_height+t[1],t[0]:img2_width+t[0]] = img2
    else:
        img2_[t[1]:img1_height+t[1],:img2_width] = img2
    img,no_blend,left,right= blending_algo(img2_,img1_warped,img2_width,side)
    img= crop(img,img2_height,corners)
    return img,no_blend,left,right
            
def image_Stitching(images):
    n=int(len(images)/2+0.5)
    left= images[:n]
    right= images[n-1:]
    right.reverse()
    while len(left)>1:
        img1=left.pop()
        img2 =left.pop()
        left_stitched,_,_,_=warping_img(img2,img1)
        left_stitched=left_stitched.astype('uint8')
        left.append(left_stitched)
    while len(right)>1:
        img1=right.pop()
        img2=right.pop()
        right_stitched,_,_,_= warping_img(img2, img1)
        right_stitched=right_stitched.astype('uint8')
        right.append(right_stitched)
    if(right_stitched.shape[1]>=left_stitched.shape[1]):
        stitched,_,_,_= warping_img(left_stitched,right_stitched)
    else:
        stitched,_,_,_= warping_img(right_stitched,left_stitched)
    return stitched

def crop(img,img2_,corners):
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    corners=corners.astype(int)
    if corners[0][0][0]<0:
        n=abs(-corners[1][0][0]+corners[0][0][0])
        img=img[t[1]:img2_+t[1],n:,:]
    else:
        if(corners[2][0][0]<corners[3][0][0]):
            img=img[t[1]:img2_+t[1],0:corners[2][0][0],:]
        else:
            img=img[t[1]:img2_+t[1],0:corners[3][0][0],:]
    return img

if __name__ == '__main__':
    path = ("archive")
    img_path = list(paths.list_images(path))
    images = []
    for i,j in enumerate(img_path):
        img = cv2.imread(j)
        img=cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
        images.append(img)
    stitched_image = image_Stitching(images)
    cv2.imwrite("Stitched_image.jpg", stitched_image)
