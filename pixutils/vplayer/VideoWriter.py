from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals



from .videoutils import *


class WriteVideo():
    def __init__(self):
        self.writter = None
        self.resize_method = cv2.INTER_LINEAR
        pass

    def write(self, img):
        if img.dtype != np.uint8:
            img = float2img(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if (img.shape[1],img.shape[0]) != self.framesize:
            img = cv2.resize(img, self.framesize, interpolation=self.resize_method)
        self.writter.write(img)

    def resize(self, img):
        return cv2.resize(img, self.framesize)

    def setup(self, vpath, fps, framesize, resize_method=cv2.INTER_LINEAR, coder=cv2.VideoWriter_fourcc(b'M', b'J', b'P', b'G')):
        if self.writter is None:
            self.resize_method = resize_method
            if type(framesize) == np.ndarray:
                self.framesize = framesize.shape[:2][::-1]
            elif type(framesize) in (list, tuple):
                self.framesize = framesize[::-1]
            else:
                self.framesize = framesize.width_col, framesize.height_row
            self.writter = cv2.VideoWriter(vpath, fourcc=coder, fps=fps, frameSize=self.framesize)
            return self