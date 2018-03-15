import cv2
import numpy as np

# input to this function must have shape [N H W C] where C = 1 or 3.
def batch_image_to_array(arr):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = num
    height = max(1,int(np.sqrt(patches)*0.9))
    width = int(patches/height+1)

    img = np.zeros((height*(uh+1), width*(uw+1), 3),dtype=arr.dtype)
    if arr.dtype == 'uint8':
        img[:,:,1] = 25
    else:
        img[:,:,1] = .1

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
            # img = cv2.resize(img,dsize=(
            #     int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),
            #     interpolation=cv2.INTER_LINEAR) # use bilinear
            img = resize_autosmooth(
                img, img.shape[0]*imgscale, img.shape[1]*imgscale
            )
        else:
            img = cv2.resize(img,dsize=(
                int(img.shape[1]*imgscale),int(img.shape[0]*imgscale)),
                interpolation=cv2.INTER_NEAREST) # use nn

    return img,imgscale

def resize_of_interpolation(interpolation):
    def resize_interpolation(img, h, w):
        return cv2.resize(img,
            dsize=(int(w), int(h)),
            interpolation=interpolation)
    return resize_interpolation

resize_linear = resize_of_interpolation(cv2.INTER_LINEAR)
resize_cubic = resize_of_interpolation(cv2.INTER_CUBIC)
resize_lanczos = resize_of_interpolation(cv2.INTER_LANCZOS4)

# gaussian pyramid downsample before resize to reduce aliasing
def resize_autosmooth(img, h, w):
    timg = img
    while True:
        # check scale
        scale = h / timg.shape[0]

        if scale > 0.5: # scale them just fine
            return resize_lanczos(timg, h, w)
        else:
            # gaussian downsample
            timg = cv2.pyrDown(timg)

def show_autoscaled(img,limit=400.,name=''):
    im,ims = autoscale(img,limit=limit)
    cv2.imshow(name+str(img.shape)+' gened scale:'+str(ims),im)
    cv2.waitKey(1)

def show_batch_autoscaled(arr,limit=400.,name=''):
    img = batch_image_to_array(arr)
    show_autoscaled(img,limit,name)

# import matplotlib.pyplot as plt
# class plotter:
#     def __init__(self):
#         plt.ion()
#
#         self.x = []
#         self.y = []
#         self.fig = plt.figure()
#         self.ax = self.fig.add_subplot(1,1,1)
#
#     def pushy(self,y):
#         self.y.append(y)
#         if len(self.x)>0:
#             self.x.append(self.x[-1]+1)
#         else:
#             self.x.append(0)
#     def newpoint(self,y):
#         self.pushy(y)
#
#     def show(self):
#         self.ax.clear()
#         self.ax.plot(self.x,self.y)
#         plt.draw()
