# pixutils
Important utilities for video processing using python 


Open terminal in the folder where setup.py is present 
run
python setup.py install
or
sudo python setup.py install

use help(*) to get description, documentation and examples of any function
eg: help(dirop), help(Player), help(ImLog)


create a new file demo.py
copy paste the following code and run
copy the location on data_base path and paste it at dbpath
# ------------------------------ Quick start example ------------------------------ #
dbpath = r'/pixutils/data_base'  # data base path
from pixutils import *

# the pixutils will automatically import cv2, dlib and all frequently required modules
# these modules are imported from one_shot_import
# you can add your module over here
# and can import easily with from pixutils import *

'''
# esc   -> close the video
# space -> pause video
# enter -> play video
# d     -> destroy all windows 
if bk!= None: # imshow = win('output', 1, video, bk=dirop(dbpath, 'results', remove=True))
    # s     -> take screen short (cannot support multiple window)

dirop is helpful to support directory level operations like
    automatically creating multiple folders/files recursively (if not found)
    deleting multiple folders/files recursively
    time stamping files/folders
    appending unique hash number at end of folder automatically
    prefix the file name with certain preset value (like frame no) 
'''


tic = clk()  # clock for measuring performance
logpath = dirop(dbpath, 'log', timestamp=True)  # create the log folder using dirop funciton
resultpath = dirop(dbpath, "results")

cam = GetFeed(dirop(dbpath, 'videos/remove_rain.mp4'))  # read video from the location
cam = ThreadIt(cam, fps=300, thread_buff_size=10)  # thread the capture
video = Player(cam, start=50, stop=350, resize=(250, 250))  # double the size of the frame
imshow = win('rain removal', delay=33, video=video, bk=logpath)

for fnos, imgs in video.chunk(13):  # fetch 13 frames chunk
    rain_suppressed = label1(imgs.mean(axis=0), 'rain_suppressed')  # mean will suppress the fast motion noise
    for fno, img in zip(fnos, imgs):
        img = label1(img, (fno, 'with rain'))
        img = label4(img, 'exe_time: %s' % tic.toc())
        img = photoframe([img, rain_suppressed], rcsize=img.shape)
        imshow(img)
        cv2.imwrite(dirop(resultpath, "i.jpg", hash=fno), img)

print('exe_time:%s fps: %s' % tic.toc(fnos))
print('log path: %s' % logpath)
print('result path: %s' % resultpath)
