from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals



from pixutils import *
from pixutils.faceutils import find_face

tic = clk()

'''
# esc   -> close the video
# space -> pause video
# enter -> play video
if bk!= None: # imshow = win('output', 1, video, bk=dirop(dbpath, 'results', remove=True))
    # s     -> take screen short (cannot support multiple window)

dirop is helpful to support directory level operations like
    automatically creating multiple folders/files recursively (if not found)
    deleting multiple folders/files recursively
    time stamping files/folders
    appending unique hash number at end of folder automatically
    prefix the file name with certain preset value (like frame no) 
'''


# ------------------------------------------------------------------------------------------------ #
def frame_pre_process(fno, grab, img):
    if img is not None:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        no_mean_img = cv2.subtract(img, img.mean())  # mean subtracted image
        img = photoframe([img, no_mean_img], 1, img.shape)  # by default this will convert gray image to bgr image (use gray2bgr=False)
        # img = photoframe([img, no_mean_img], 1, img.shape, gray2bgr=False)
    return fno, grab, img


def example1(dbpath):
    # example1:
    #     1. fetch frame from 50 to 150
    #     2. scale it down to half size
    #     3. convert to gray and do mean subtraction
    #     4. stitch the mean subtracted image with the original image
    #     5. run everything in thread
    vpath = join(dbpath, r'videos', 'remove_rain.mp4')
    cam = GetFeed(vpath)
    # cam = ThreadIt(cam)
    video = Player(cam, start=50, stop=150, resize=(600, 600), custom_fn=frame_pre_process)
    # video = Player(cam,start=50,end=150,resize=2,custom_fn=frame_pre_process)
    imshow = win('output', 33, video, bk=dirop(dbpath, 'results', 'temp', timestamp=True))
    tic = clk()
    for fno, img in video.play():
        img = label1(img, 'Space -> pause', (30, 60))
        img = label1(img, 'Enter -> play', (30, 80))
        img = label1(img, 'Esc   -> close', (30, 100))
        img = label1(img, 's -> Screen short', (30, 120))
        img = label1(img, 'Screen short will not work when bk=None', (30, 120))
        imshow(label1(img, fno))
        cv2.imwrite(dirop(dbpath, 'results', 'logs', 'i.jpg', hash=True), img)
        cv2.imwrite(dirop(dbpath, 'results', 'logs', 'i.jpg', timestamp=True), img)
    print('logs path: %s' % dirop(dbpath, 'results', 'logs', ))
    print(tic.toc(fno))
    # <f0>, <f1>, <f2>, <f3>, <f4>, ...


# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #

def example2(dbpath):
    vpath = join(dbpath, r'videos/remove_rain.mp4')
    cam = GetFeed(vpath)
    cam = ThreadIt(cam)  # GetFeed will running in thread
    video = Player(cam)
    imshow = win('output', 100, video)
    for fnos, imgs in video.chunk(5):
        print(fnos)
        rain_removed = imgs.mean(axis=0)
        rain_removed = label1(rain_removed, 'rain suppressed')
        for img in imgs:
            img = label1(img, fnos)
            imshow(label1(photoframe([img, rain_removed], rcsize=img.shape, asgray=True), 'gray image', (img.shape[1], img.shape[0] / 2)))
            imshow(label1(photoframe([img, rain_removed], rcsize=img.shape, asgray=False), 'color image', (img.shape[1], img.shape[0] / 2)))
            # [<f0>,  <f1>,  <f2>,  <f3>,  <f4>,  <f5>],
            # [<f6>,  <f7>,  <f8>,  <f9>,  <f10>, <f11>],
            # [<f12>, <f13>, <f14>, <f15>, <f16>, <f17>], ...


# ------------------------------------------------------------------------------------------------ #
def custom_labeller(impath):
    label = basename(impath).replace('.jpg', '').split('_')[2]
    return label


def example3(dbpath):
    '''
    the video player
    '''
    vpath = dirop(dbpath, 'videoimgs/*.*')
    cam = GetFeed(vpath, labeller=None)
    # cam = GetFeed(vpath, labeller=custom_labeller)
    cam = ThreadIt(cam, fps=300, thread_buff_size=10)
    video = Player(cam)
    imshow = win('video', 1, video)
    for fnos, imgs in video.chunk(5):
        print(fnos)
        for fno, img in zip(fnos, imgs):
            imshow(label1(img, basename(video.impaths(fno))))
            # [<f0>,  <f1>,  <f2>,  <f3>,  <f4>,  <f5>],
            # [<f6>,  <f7>,  <f8>,  <f9>,  <f10>, <f11>],
            # [<f12>, <f13>, <f14>, <f15>, <f16>, <f17>], ...


# ------------------------------------------------------------------------------------------------ #
def detect_face(grab, img):
    if img is not None:
        face_rects = [d2bb(drect) for drect in find_face(img)]
        img = img, face_rects
    return grab, img


def example4(dbpath):
    '''
    Get two feed and run in threads
    Feed1: read the sequence of videoimgs and and detect faces
    Feed2: Get feed from live camera and detect faces
    put bounding box and merge images and show
    '''
    # cam = GetFeed(join(dbpath, r'videos/remove_rain.mp4'))
    cam = GetFeed(join(dbpath, r'videoimgs/*.jpg'))
    cam = ThreadIt(cam, thread_buff_size=20, fps=30, custom_fn=detect_face)
    video1 = Player(cam, start=0)
    # cam = GetFeed(0)
    cam = GetFeed(join(dbpath, r'videos/remove_rain.mp4'))
    cam = ThreadIt(cam, thread_buff_size=10, fps=50, custom_fn=detect_face)
    video2 = Player(cam,start=200)
    imshow1 = win('face1', None, video1, dirop(dbpath, 'results', 'videoimgs', remove=True))
    imshow2 = win('face2', 1, video2, dirop(dbpath, 'results', 'livecam', remove=True))
    tic = clk()
    il1, il2 = ImLog(), ImLog(col=None)
    img = np.zeros((100, 175, 3), 'u1')
    img = label1(img, 'Space -> pause', (20, 40))
    img = label1(img, 'Enter -> play',  (20, 60))
    img = label1(img, 'Esc   -> close', (20, 80))
    for i, ((fno1, (img1, face_rects1)), (fno2s, img2s)) in enumerate(izip(video1.play(), video2.chunk(7))):
        img1, img2 = label1(img1, fno1), label2(img1, fno2s)
        il2.log(img)
        if face_rects1:
            big_face = face_rects1[get_ibig_bb(face_rects1)]
            fg = np.random.randint(0, 100, 3)
            img1 = label1(img1, 'big_face', big_face[:2], fg)
            # img1 = label3(img1, basename(video1.paths[fno1]), face_rect, fg)
            il1.log(img1)

        for img2, face_rects2 in img2s:
            for i, face_rect in enumerate(face_rects2):
                fg, bg = np.random.randint(0, 100, 3), np.random.randint(100, 150, 3)
                img2 = label2(img2, i, face_rect, fg, bg, scale=(1.15, 1.15))
            il1.log(img2)
            il2.log(img2)
        imshow1(photoframe([img1, il2.get()], rcsize=(video2.rc), col=1))
        imshow2(il1.get())
        print(video1.cam.q.qsize(), video2.cam.q.qsize())
    print(tic.toc(fno2s))


# ------------------------------------------------------------------------------------------------ #

class pre_processing:

    def __init__(self):
        self.fcount = 0

    def detect_face(self, fno, grab, img):
        self.fcount+=1
        if self.fcount%5!=0:  # process every 5th frame skip others
            return fno,None,None  # when grab == None that frame will be dropped
        if img is not None:
            imgs = [get_subimage(img, d2bb(drect)) for drect in find_face(img)]
            if not imgs:  # no face is found send entire frame
                return fno,grab, img
            else:
                img = photoframe(imgs)
        return fno, grab, img


def example5(dbpath):
    '''
    Grab the live cam feed at 30 fps
    Run the face detector WITHOUT thread
    Thread buffer size should be 10
    If the face not found skip the frame
    Group by 7 frames
    Merge the 7 frames and create a single image
    Store all the image in the results folder
    Display the with wait-time 100ms
    '''
    cam = GetFeed(0)
    # cam = GetFeed(join(dbpath, r'videos/remove_rain.mp4'))
    cam = ThreadIt(cam, thread_buff_size=10, fps=30)
    video = Player(cam, custom_fn=pre_processing().detect_face)
    imshow = win('face', 1, video, bk=dirop(dbpath, r'results\bk', remove=True))  # delete the results folder if exist
    tic = clk()
    for fnos, imgs in video.chunk(7):
        img = photoframe(imgs, rcsize=None)
        # print(video.cam.q.qsize())
        imshow(img)
    print(tic.toc(fnos))


# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #

def example6(dbpath):
    from pixutils.faceutils import face_landmarks
    full_face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
    vpath = join(dbpath, r'videos/remove_rain.mp4')
    vpath = 0
    cam = GetFeed(vpath)
    cam = ThreadIt(cam)
    video = Player(cam, start=150)
    imshow = win('video output', 1, video)
    tic = clk()
    for i, (fno, img) in enumerate(video.play()):
        mask = np.zeros_like(img)
        for drect in find_face(img, 1):
            lrect, landmarks = face_landmarks(img, drect, full_face)
            cv2.drawContours(mask, [landmarks], 0, (255, 255, 255), -1)
            img = put_bb(img, d2bb(drect))
        mask = cv2.bitwise_and(img, mask)
        imshow(mask)
    print(tic.toc(fno))


def example7(dbpath):
    def simplePlayer(vpath, waittime=1):
        cam = cv2.VideoCapture(vpath)
        while True:
            grab, img = cam.read()
            if img is not None:
                cv2.imshow('outvideo', img)
                if cv2.waitKey(waittime) == 27:
                    break
            else:
                break

    simplePlayer(vpath=join(dbpath, r'videos/remove_rain.mp4'))


# ------------------------------------------------------------------------------------------------ #


if __name__ == '__main__':
    example4(dbpath)
    # example1(dbpath)
    # example2(dbpath)
    # example3(dbpath)
    # example5(dbpath)
    # example6(dbpath)
    # example7(dbpath)
    # dirop(dbpath, 'results', remove=True)  # clear all logs