from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals



from .videoutils import *

def default_labeller(x):
    try:
        return int(x.split('_')[1].split('.')[0])
    except:
        try:
            return int(basename(x).split('_')[1].split('.')[0])
        except:
            return 0

class Im2Video():
    '''
    def labeller(impath):
        return impath.replace('.jpg','').split('_')[1]
    vpath = join(dbpath, r'videoimgs/*.*')
    cam = Imgs2Video(vpath, labeller)
    video = Player(cam)
    cam = ThreadIt(cam)
    imshow = win(video)
    for fno, img in video.play():
        imshow('show_video', img, 1)
    '''
    def __init__(self, opaths, labeller=None):
        labeller = labeller or default_labeller
        if type(opaths) not in (list, tuple):
            paths = glob(opaths)
        else:
            paths = opaths
        if not paths:
            raise Exception('No file found in %s' % opaths)
        paths = [(int(labeller(path)), path) for path in paths]
        self.paths = sorted(paths, key=lambda x: x[0])
        self.frameno, self.paths = list(zip(*self.paths))
        self.row, self.col = cv2.imread(self.paths[0]).shape[:2]
        self.index = -1

    def release(self):
        pass

    def read(self):
        self.index += 1
        if len(self.paths) <= self.index:
            return False, None
        try:
            return True, cv2.imread(self.paths[self.index])
        except:
            return None, None

    def get(self, i):
        if i == 3:
            return self.col
        elif i == 4:
            return self.row
        elif i == 5:
            return 30
        elif i == 7:
            return len(self.paths)

    def set(self, i, start_frame):
        self.index += (start_frame - 1)



def GetFeed(vpath, *a, **kw):
    if type(vpath) == int:
        return 'stream', cv2.VideoCapture(vpath)
    elif type(vpath) in (list,tuple) or '*' in vpath:
        return 'imgs', Im2Video(vpath, *a, **kw)
    else:
        assert exists(vpath), 'Video File missing: %s' % vpath
        return 'video', cv2.VideoCapture(vpath)