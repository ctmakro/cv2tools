import cv2
import numpy as np

# input to this function must have shape [N H W C] where C = 1 or 3.
def batch_image_to_array(arr):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = num
    height = max(1,int(np.sqrt(patches)*0.9))
    width = int(patches/height+1)

    img = np.zeros((height*(uh+1), width*(uw+1), 3),dtype='float32')

    index = 0
    for row in range(height):
        for col in range(width):
            if index<num:
                channels = arr[index]
                img[row*(uh+1):row*(uh+1)+uh,col*(uw+1):col*(uw+1)+uw,:] = channels
            index+=1
    return img

# automatically scale the image up or down, using nearest neighbour.
def autoscale(img,limit=400.):
    # scales = [0.1,0.125,1./6.,0.2,0.25,1./3.,1./2.] + range(100)
    scales = np.hstack([1./np.linspace(10,2,num=9), np.linspace(1,100,num=100)])

    imgscale = limit/float(max(img.shape[0],img.shape[1]))
    for s in reversed(scales):
        if s<=imgscale:
            imgscale=s
            break

    if imgscale!=1.:
        if imgscale<1.:
            img = cv2.resize(img,dsize=(
                int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),
                interpolation=cv2.INTER_LINEAR) # use bilinear
        else:
            img = cv2.resize(img,dsize=(
                int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),
                interpolation=cv2.INTER_NEAREST) # use nn

    return img,imgscale

def show_autoscaled(img,limit=400.,name=''):
    im,ims = autoscale(img,limit=limit)
    cv2.imshow(name+str(img.shape)+' gened scale:'+str(ims),im)
    cv2.waitKey(1)

def show_batch_autoscaled(arr,limit=400.,name=''):
    img = batch_image_to_array(arr)
    show_autoscaled(img,limit,name)
