import cv2
import numpy as np

def ceil(i):
    ii = int(i)
    if i == ii:
        return ii
    else:
        return ii + 1

# input to this function must have shape [N H W C] where C = 1 or 3.
def batch_image_to_array(arr, margin=1, color=None, aspect_ratio=1.1, width=None, height=None):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = num
    if height is None:
        height = max(1,int(np.sqrt(patches)/aspect_ratio))
    if width is None:
        width = ceil(patches/height)

    assert width*height >= patches

    img = np.zeros((height*(uh+margin), width*(uw+margin), 3),dtype=arr.dtype)
    if color is None:
        if arr.dtype == 'uint8':
            img[:,:,1] = 25
        else:
            img[:,:,1] = .1
    else:
        img += color

    index = 0
    for row in range(height):
        for col in range(width):
            if index<num:
                channels = arr[index]
                img[row*(uh+margin):row*(uh+margin)+uh,col*(uw+margin):col*(uw+margin)+uw,:] = channels
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
resize_nearest = resize_of_interpolation(cv2.INTER_NEAREST)
resize_area = resize_of_interpolation(cv2.INTER_AREA)

# gaussian pyramid downsample before resize to reduce aliasing
# def resize_autosmooth(img, h, w):
#     timg = img
#     while True:
#         # check scale
#         scale = h / timg.shape[0]
#
#         if scale > 0.5: # scale them just fine
#             return resize_linear(timg, h, w)
#         else:
#             # gaussian downsample
#             timg = cv2.pyrDown(timg)

def prefilter_lanczos_kernel(hs, ws):
    hs*=0.95 # over blur a bit
    ws*=0.95

    sx = .5/ws
    sy = .5/hs

    fw = max(2, ceil(sx*4))
    fh = max(2, ceil(sy*4))

    if fw%2==0: fw+=1
    if fh%2==0: fh+=1

    x = np.linspace(-(fw//2), fw//2, fw)
    y = np.linspace(-(fh//2), fh//2, fh)

    x *= ws
    y *= hs

    a = 2
    lanczosx = np.where(x<a, np.sinc(x)*np.sinc(x/a), 0)
    lanczosy = np.where(y<a, np.sinc(y)*np.sinc(y/a), 0)
    lanczos = np.outer(lanczosy, lanczosx).astype('float32')

    return lanczos / lanczos.sum()

# prefilter then resize.
def resize_perfect(img, h, w):

    hs = h/img.shape[0]
    ws = w/img.shape[1]

    if hs < 1 or ws < 1:
        lanczos = prefilter_lanczos_kernel(hs, ws)
        img = cv2.filter2D(img, -1, lanczos)
    else:
        pass

    return resize_linear(img, h, w)
    
resize_autosmooth = resize_perfect

# good for downsampling.
def resize_box(img, h, w):
    # check scale
    scale = h / img.shape[0]

    if scale > 1:
        return resize_linear(img, h, w)
    else:
        return resize_area(img, h, w)

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

if __name__ == '__main__':
    lanczos = prefilter_lanczos_kernel(.1,.5)
    print('sum', lanczos.sum())
    print(lanczos, lanczos.shape)
    show_autoscaled(lanczos*0.5+0.5,limit=50)
    cv2.waitKey(0)
