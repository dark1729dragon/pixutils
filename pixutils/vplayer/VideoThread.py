from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals



from .videoutils import *


class CamGrabber(Thread):
    '''
    cam = cv2.VideoCapture(0)
    video = Player(cam)
    cam = ThreadIt(cam,300, thread_buff_size=0)
    imshow = win(video)
    for fno, img in video.play():
        imshow('show_video', img, 700)
    '''
    def __init__(self, cam, thread_buff_size, fps, custom_fn):
        super(CamGrabber, self).__init__()
        self.fps = fps
        self.thread_buff_size = thread_buff_size
        self.q = Queue(self.thread_buff_size)
        self.cam = cam
        self.custom_fn = custom_fn
        self._stop = False

    def run(self):
        while True:
            if self._stop:
                if Queue is not None:
                    self.q = Queue(self.thread_buff_size)
                    self.q.put((False, None))
                break
            else:
                if self.custom_fn is None:
                    self.q.put(self.cam.read())
                else:
                    self.q.put(self.custom_fn(*self.cam.read()))
            time.sleep(1.0 / self.fps)

    def get_queue(self):
        return self.q


class ThreadIt2(Thread):
    def __init__(self,(source, cam), fps=30, thread_buff_size=50, custom_fn=None):
        self.cam, self.fps, self.thread_buff_size = cam, fps, 0 if thread_buff_size == np.inf else thread_buff_size
        self.get = self.cam.get
        self.set = self.cam.set
        self.paths = self.cam.paths if source == 'imgs' else None
        self.release = self.cam.release
        self.custom_fn = custom_fn
        super(ThreadIt2, self).__init__()

    def initThread(self):
        self.cam = CamGrabber(self.cam, self.thread_buff_size, self.fps, self.custom_fn)
        self.q = self.cam.get_queue()
        self.cam.setDaemon(True)
        self.cam.start()
        self._stop = False
        self.setDaemon(True)
        self.start()
        self.frame = None, None

    def read(self, stop=False):
        if stop:
            self.cam.release()
            return False, None
        if not self.q.empty():
            # print(self.q.qsize())
            return self.q.get()
        else:
            return None, None

    def stop(self):
        self.cam._stop = True
        self.read(stop=True)


def ThreadIt((source, cam),*a,**kw):
    cam = ThreadIt2((source, cam),*a,**kw)
    return source, cam


if __name__ == '__main__':
    from pixutils import *

    vpath = 0
    cam = GetFeed(vpath)
    cam = ThreadIt(cam)
    video = Player(cam)
    imshow = win('out video', 100, video=video)
    tic = clk()
    for fno, img in video.play():
        print(cam[1].q.qsize())
        imshow(img)