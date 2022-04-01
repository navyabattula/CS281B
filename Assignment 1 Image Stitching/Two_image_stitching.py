# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:30:06 2022

@author: bsr
"""
from imutils import paths
import imutils
import cv2
import numpy as np

class utils:
    def loadImages(path,resize):
        '''Load Images from path to array, @param path is the folder which containing images, @param resize is True
        if image is halved in size, otherwise is False'''
        image_path = list(paths.list_images(path))
        list_image = []
        for i,j in enumerate(image_path):
            image = cv2.imread(j)
            if resize==1:
                image=cv2.resize(image,(int(image.shape[1]/4),int(image.shape[0]/4)))
            list_image.append(image)
        return (list_image)
    def trim(frame):
        '''crop frame '''
        #crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        #crop bottom
        elif not np.sum(frame[-1]):
            return trim(frame[:-2])
        #crop left
        elif not np.sum(frame[:,0]):
            return trim(frame[:,1:]) 
        #crop right
        elif not np.sum(frame[:,-1]):
            return trim(frame[:,:-2])    
        return frame
    def padding(img,top,bottom,left,right):
        '''add padding to img'''
        border = cv2.copyMakeBorder(
            img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT
        )
        return border
class features:
    def findAndDescribeFeatures(image,opt='SIFT'):
        '''find and describe features of @image,
        if opt='SURF', SURF algorithm is used.
        if opt='SIFT', SIFT algorithm is used.
        if opt='ORB', ORB algorithm is used.
        @Return keypoints and features of img'''
        #Getting gray image
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if opt=='SURF':
            md = cv2.xfeatures2d.SURF_create()
        if opt=='ORB':
            md = cv2.ORB_create(nfeatures=3000)
        if opt=='SIFT':
            md = cv2.xfeatures2d.SIFT_create()
        #Find interest points and Computing features.
        keypoints, features = md.detectAndCompute(grayImage, None)
        #Converting keypoints to numbers.
        #keypoints = np.float32(keypoints)
        features = np.float32(features)
        return keypoints, features

    def matchFeatures(featuresA, featuresB,ratio=0.75,opt='FB'):
        '''matching features beetween 2 @features.
         If opt='FB', FlannBased algorithm is used.
         If opt='BF', BruteForce algorithm is used.
         @ratio is the Lowe's ratio test.
         @return matches'''
        if opt=='BF':
            featureMatcher = cv2.DescriptorMatcher_create("BruteForce")
        if opt=='FB':
            #featureMatcher = cv2.DescriptorMatcher_create("FlannBased")
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            featureMatcher = cv2.FlannBasedMatcher(index_params, search_params)


        #performs k-NN matching between the two feature vector sets using k=2 
        #(indicating the top two matches for each feature vector are returned).
        matches = featureMatcher.knnMatch(featuresA,featuresB, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance<ratio*n.distance:
                good.append(m)
        if len(good)>4:
            return good
        else:
            raise Exception("Not enought matches") 

    def generateHomography(src_img, dst_img, ransacRep=5.0):
        src_kp,src_features=features.findAndDescribeFeatures(src_img)
        dst_kp,dst_features=features.findAndDescribeFeatures(dst_img)

        good=features.matchFeatures(src_features,dst_features)

        src_points = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_points = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask  = cv2.findHomography(src_points, dst_points, cv2.RANSAC,ransacRep)
        matchesMask = mask.ravel().tolist()
        return H,matchesMask


    def drawKeypoints(img,kp):
        img1=img
        cv2.drawKeypoints(img,kp,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img1
    def drawMatches(src_img,src_kp,dst_img,dst_kp,matches,matchesMask):
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask[:100], # draw only inliers
                       flags = 2)
        return cv2.drawMatches(src_img,src_kp,dst_img,dst_kp,matches[:],None,**draw_params)
class stitch:
    def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
        assert barrier < width
        mask = np.zeros((height, width))

        offset = int(smoothing_window/2)
        try:
            if left_biased:
                mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(1,0,2*offset+1).T, (height, 1))
                mask[:,:barrier-offset] = 1
            else:
                mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(0,1,2*offset+1).T, (height, 1))
                mask[:,barrier+offset:] = 1
        except:
            if left_biased:
                mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(1,0,2*offset).T, (height, 1))
                mask[:,:barrier-offset] = 1
            else:
                mask[:,barrier-offset:barrier+offset+1]=np.tile(np.linspace(0,1,2*offset).T, (height, 1))
                mask[:,barrier+offset:] = 1

        return cv2.merge([mask, mask, mask])

    def panoramaBlending(dst_img_rz,src_img_warped,width_dst,side,showstep=False):
        h,w,_=dst_img_rz.shape
        smoothing_window=int(width_dst/8)
        barrier = width_dst -int(smoothing_window/2)
        mask1 = stitch.blendingMask(h, w, barrier, smoothing_window = smoothing_window, left_biased = True)
        mask2 = stitch.blendingMask(h, w, barrier, smoothing_window = smoothing_window, left_biased = False)

        if showstep:
            nonblend=src_img_warped+dst_img_rz
        else:
            nonblend=None
            leftside=None
            rightside=None

        if side=='left':
            dst_img_rz=cv2.flip(dst_img_rz,1)
            src_img_warped=cv2.flip(src_img_warped,1)
            dst_img_rz=(dst_img_rz*mask1)
            src_img_warped=(src_img_warped*mask2)
            pano=src_img_warped+dst_img_rz
            pano=cv2.flip(pano,1)
            if showstep:
                leftside=cv2.flip(src_img_warped,1)
                rightside=cv2.flip(dst_img_rz,1)
        else:
            dst_img_rz=(dst_img_rz*mask1)
            src_img_warped=(src_img_warped*mask2)
            pano=src_img_warped+dst_img_rz
            if showstep:
                leftside=dst_img_rz
                rightside=src_img_warped


        return pano,nonblend,leftside,rightside

    def warpTwoImages(src_img, dst_img,showstep=False):

        #generate Homography matrix
        H,_=features.generateHomography(src_img,dst_img)

        #get height and width of two images
        height_src,width_src = src_img.shape[:2]
        height_dst,width_dst = dst_img.shape[:2]
        print (height_src, width_src, height_dst, width_dst)

        #extract conners of two images: top-left, bottom-left, bottom-right, top-right
        pts1 = np.float32([[0,0],[0,height_src],[width_src,height_src],[width_src,0]]).reshape(-1,1,2)
        pts2 = np.float32([[0,0],[0,height_dst],[width_dst,height_dst],[width_dst,0]]).reshape(-1,1,2)

        #try:
            #aply homography to conners of src_img
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

            #find max min of x,y coordinate
        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin,-ymin]
        if(pts[0][0][0]<0):
            side='left'
            width_pano=width_dst+t[0]
        else:
            width_pano=int(pts1_[3][0][0])
            side='right'
        height_pano=ymax-ymin

            #Translation 
            #https://stackoverflow.com/a/20355545
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 
        src_img_warped = cv2.warpPerspective(src_img, Ht.dot(H), (width_pano,height_pano))
            #generating size of dst_img_rz which has the same size as src_img_warped
        dst_img_rz=np.zeros((height_pano,width_pano,3))
        if side=='left':
            dst_img_rz[t[1]:height_src+t[1],t[0]:width_dst+t[0]] = dst_img
        else:
            dst_img_rz[t[1]:height_src+t[1],:width_dst] = dst_img
            #blending panorama
        pano,nonblend,leftside,rightside=stitch.panoramaBlending(dst_img_rz,src_img_warped,width_dst,side,showstep=showstep)
        #croping black region
        pano=stitch.crop(pano,height_dst,pts)
        return pano,nonblend,leftside,rightside
        '''except:
            raise Exception("Please try again with another image set!")'''
    def multiStitching(list_images):
        n=int(len(list_images)/2+0.5)
        left=list_images[:n]
        right=list_images[n-1:]
        right.reverse()
        while len(left)>1:
            dst_img=left.pop()
            src_img=left.pop()
            left_pano,_,_,_=stitch.warpTwoImages(src_img,dst_img)
            left_pano=left_pano.astype('uint8')
            left.append(left_pano)

        while len(right)>1:
            dst_img=right.pop()
            src_img=right.pop()
            right_pano,_,_,_=stitch.warpTwoImages(src_img,dst_img)
            right_pano=right_pano.astype('uint8')
            right.append(right_pano)

        #if width_right_pano > width_left_pano, Select right_pano as destination. Otherwise is left_pano
        if(right_pano.shape[1]>=left_pano.shape[1]):
            fullpano,_,_,_=stitch.warpTwoImages(left_pano,right_pano)
        else:
            fullpano,_,_,_=stitch.warpTwoImages(right_pano,left_pano)
        return fullpano

    def crop(panorama,h_dst,conners):
        [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(conners.max(axis=0).ravel() + 0.5)
        t = [-xmin,-ymin]
        conners=conners.astype(int)

        #conners[0][0][0] is the X coordinate of top-left point of warped image
        #If it has value<0, warp image is merged to the left side of destination image
        #otherwise is merged to the right side of destination image
        if conners[0][0][0]<0:
            n=abs(-conners[1][0][0]+conners[0][0][0])
            panorama=panorama[t[1]:h_dst+t[1],n:,:]
        else:
            if(conners[2][0][0]<conners[3][0][0]):
                panorama=panorama[t[1]:h_dst+t[1],0:conners[2][0][0],:]
            else:
                panorama=panorama[t[1]:h_dst+t[1],0:conners[3][0][0],:]
        return panorama
if __name__ == '__main__':
    path = ("archive")
    img_list = utils.loadImages(path, 1)
    pano = stitch.multiStitching(img_list)
    cv2.imwrite("Pano.jpg", pano)

    