import math
import numpy as np
import cv2

# generate a kernel for motion blur.
# dim is kernel size
# angle is kernel angle

def generate_motion_blur_kernel(dim=3,angle=0.,threshold_factor=1.0,divide_by_dim=True,vector=None):
    if vector is None: #use angle as input
        radian = angle/360*math.pi*2 + math.pi/2

        if dim<2:
            return None

        # first, generate xslope and yslope
        offset = (dim-1)/2 # zero-centering
        gby,gbx = np.mgrid[0-offset:dim-offset:1.,0-offset:dim-offset:1.] # assume dim=3, 0:dim -> -1,0,1

        # then mix the slopes according to angle
        gbmix = gbx * math.cos(radian) - gby * math.sin(radian)

    else: #use vector as input
        y,x = vector
        l2 = math.sqrt(x**2+y**2)+1e-2
        ny,nx = y/l2,x/l2

        dim = int(max(abs(x),abs(y)))
        if dim<2:
            return None

        # first, generate xslope and yslope
        offset = (dim-1)/2 # zero-centering
        gby,gbx = np.mgrid[0-offset:dim-offset:1.,0-offset:dim-offset:1.] # assume dim=3, 0:dim -> -1,0,1

        gbmix = gbx * ny - gby * nx

    # threshold it into a line
    kernel = (threshold_factor - gbmix*gbmix).clip(min=0.,max=1.)

    if divide_by_dim: # such that the brightness wont change
        kernel /= np.sum(kernel)
    else:
        pass

    return kernel.astype('float32')

# apply motion blur to the image.
def apply_motion_blur(im,dim,angle):
    kern = generate_motion_blur_kernel(dim,angle)
    if kern is None:
        return im
    imd = cv2.filter2D(im,cv2.CV_32F,kern)
    return imd

def apply_vector_motion_blur(im,vector):
    kern = generate_motion_blur_kernel(vector=vector)
    if kern is None: # if kernel too small to be generated
        return im
    imd = cv2.filter2D(im,cv2.CV_32F,kern)
    return imd

# apply rotation and scaling to an image, while keeping it in the center of the resulting square image.
# img should be an image of shape [HWC]
def rotate_scale(img,angle=0.,scale=1.):
    ih,iw = img.shape[0:2]

    side = int(max(ih,iw) * scale * 1.414) # output image height and width
    center = side//2

    # 1. scale
    orig_points = np.array([[iw/2,0],[0,ih/2],[iw,ih/2]],dtype='float32')
    # x,y coord of top left right corner of input image.

    translated = np.array([[center,center-ih*scale/2],[center-iw*scale/2,center],[center+iw*scale/2,center]],dtype='float32')
    # get affine transform matrix
    at = cv2.getAffineTransform(orig_points,translated)
    at = np.vstack([at,[0,0,1.]])

    # 2. rotate
    rm = cv2.getRotationMatrix2D((center,center),angle,1)
    # per document:
    # coord - rotation center
    # angle – Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
    rm = np.vstack([rm,[0,0,1.]])

    # 3. combine 2 affine transform
    cb = np.dot(rm,at)
    # print(cb)

    # 4. do the transform
    res = cv2.warpAffine(img,cb[0:2,:],(side,side),flags=cv2.INTER_CUBIC)
    return res

# calculate intersection of two integer roi. returns the intersection.
# assume (y,x) or (h,w)
def intersect(tl1,tl2,size1,size2):
    x_tl = max(tl1[1],tl2[1])
    y_tl = max(tl1[0],tl2[0])
    x_br = min(tl1[1]+size1[1], tl2[1]+size2[1])
    y_br = min(tl1[0]+size1[0], tl2[0]+size2[0])

    if x_tl<x_br and y_tl<y_br:
        return [y_tl,x_tl],[y_br,x_br], [y_br-y_tl,x_br-x_tl]
    else:
        return None

# same as above, but return roi-ed numpy array views
def intersect_get_roi(bg,fg,offset=[0,0],verbose=True):
    bgh,bgw,bgc = bg.shape
    fgh,fgw,fgc = fg.shape

    # obtain roi in background coords
    isect = intersect([0,0], offset, [bgh,bgw], [fgh,fgw])
    if isect is None:
        if verbose==True:
            print('(intersect_get_roi)warning: two roi of shape',bg.shape,fg.shape,'has no intersection.')
        return None

    tl,br,sz = isect
    # print(isect)
    bgroi = bg[tl[0]:br[0], tl[1]:br[1]]

    # obtain roi in fg coords
    tl,br,sz = isect = intersect([-offset[0],-offset[1]],[0,0],[bgh,bgw],[fgh,fgw])
    # print(isect)
    fgroi = fg[tl[0]:br[0], tl[1]:br[1]]

    return fgroi,bgroi

# alpha composition.
# bg shape: [HW3] fg shape: [HW4] dtype: float32
def alpha_composite(bg,fg,offset=[0,0],verbose=True):
    isectgr = intersect_get_roi(bg,fg,offset,verbose=verbose)
    if isectgr is None:
        return bg
    else:
        fgroi,bgroi = isectgr

    # print('alphacomp',bgroi.shape,fgroi.shape)
    alpha = fgroi[:,:,3:3+1]
    bgroi[:] = bgroi[:] * (1 - alpha) + fgroi[:,:,0:3] * alpha

    return bg

# pixel displacement
def displace(img,dy,dx):
    assert img.shape[0:2] == dy.shape[0:2]
    assert img.shape[0:2] == dx.shape[0:2]

    ih,iw = img.shape[0:2]
    rowimg,colimg = np.mgrid[0:ih,0:iw]

    rowimg += dy.astype('int32')
    colimg += dx.astype('int32')

    rowimg = np.clip(rowimg,a_max=ih-1,a_min=0)
    colimg = np.clip(colimg,a_max=iw-1,a_min=0)

    res = img[rowimg,colimg]
    return res
