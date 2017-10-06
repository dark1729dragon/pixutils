from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals


from .utils3 import *

common_bk_path = None  # edit this variable to change common settings of backup

__dirhash = defaultdict(int)  # private variable


def dirop(*dirpath, **kw):
    '''
    This function will recursively create folder, if the folder does not exist.
    :param dirpath: can be anything including int, string, bool, list; except --> :,|,<,>,,?,*
    :param kw: ts, hash, remove, mkdir, sep
    :return:
    Examples
    --------
        test_dir, out = 'test', ['e1', 'e2']
        print('-' * 150)
        print('os.path.join      ->', join(dbpath, r'results///desktop\\\img', 'log', 'out'))
        print('dirop join        ->', dirop(dbpath, r'results\\\desktop///img', 'log', 'out'))  # this will create dir if not exist
        print('deleting folder   ->', dirop(dbpath, r'results\desktop/img', remove=True))  # this will delete the folder if exist
        print('hashing           ->', dirop(dbpath, r'results\log', out, hash=True))
        print('hashing           ->', dirop(dbpath, r'results\log', out, hash=True))
        print('time stamp hash   ->', dirop(dbpath, 'results', test_dir, 'log', out, out, timestamp='stmp'))
        print('skipping creating ->', dirop(dbpath, 'results', test_dir, 'log', out, out, timestamp=True, mkdir=False))
        cv2.imwrite(dirop(dbpath, r'results\\\desktop///img', 'log', 'gray.jpg', hash=True), 128 + np.zeros((100, 100), 'u1'))
        cv2.imwrite(dirop(dbpath, r'results///desktop\\\img', 'log', 'gray.jpg', hash='zero'), 128 + np.zeros((100, 100), 'u1'))
        cv2.imwrite(dirop(dbpath, r'results\\\desktop///img', 'log', 'white.jpg', timestamp=True), 255 + np.zeros((100, 100), 'u1'))
        print('-' * 150)
        print('See the result at: %s' % dirop(dbpath, 'results', mkdir=False))
    '''
    global __dirhash
    mkdir, ts, hsh, remove,sep = kw.get('mkdir', True), kw.get('timestamp'), kw.get('hash'), kw.get('remove'), kw.get('sep', os.sep)
    dirpath = list(map(str, dirpath))
    path = join(*dirpath)
    path = path.replace('\\',sep)
    path = path.replace('/', sep)
    in_name, file_ext = os.path.splitext(path)
    if file_ext:  # the input is file
        dir_path = dirname(in_name)
        if ts is not None:
            __dirhash[dir_path] += 1
            path = '%s%s_%s_%s%s' % (in_name, '' if ts is True else ts, dt.now().strftime('%I%M%p%S'), __dirhash[dir_path], file_ext)
        elif hsh is not None:
            __dirhash[dir_path] += 1
            path = '%s%s_%s%s' % (in_name,'' if hsh is True else hsh, __dirhash[dir_path], file_ext)
        elif remove is True and exists(path):
            print('Deleting  path: %s' % path)
            os.remove(path)
            __dirhash[path] = 0
        if mkdir and not exists(dir_path) and dir_path:
            os.makedirs(dir_path)
        return path
    else:  # the input is file folder
        if ts is not None:
            __dirhash[in_name] += 1
            path = '%s%s_%s_%s' % (in_name,'' if ts is True else ts, dt.now().strftime('%I%M%p%S'), __dirhash[in_name])
        elif hsh is not None:
            __dirhash[in_name] += 1
            path = '%s%s_%s' % (in_name,'' if hsh is True else hsh, __dirhash[in_name])
        elif remove is True and exists(path):
            print('Deleteing path: %s' % path)
            shutil.rmtree(path, ignore_errors=True)
            __dirhash[in_name] = 0
        dir_path = path
        if mkdir and not exists(dir_path) and not remove:
            os.makedirs(dir_path)
        return path


class win():
    '''
    # from projectConfig import *
    # this will show the image or video
    # space -> pause video
    # enter -> play video
    # s     -> take screen short (cannot support multiple window)
    # esc   -> close the video
    '''

    def __init__(self, winname=None, delay=None, video=None, bk=common_bk_path, winsize=None):
        '''
        if ' ':
            impaths = glob(join(dbpath,'videoimgs',r'*.jpg'))
            for impath in impaths:
                img = cv2.imread(impath)
                img = label2(img, basename(impath))
                if win('img', 0)(img) == 'esc':
                    break
        if ' ':
            video = Player(GetFeed(dirop(dbpath, r'videos', 'remove_rain.mp4')))
            imshow = win('video', 1, video, dirop(dbpath, 'results/bk', remove=True))
            for fno, img in video.play():
                label1(img, fno)
                imshow(img)  # condition to close internally handled.
        '''
        if winname is not None:
            cv2.namedWindow(winname, 0)
            if winsize is not None: cv2.resizeWindow(winname, *winsize)
        self.video = video
        self.winname = winname
        self.delay, self.pdelay = delay, delay
        self.bk = bk
        if self.bk is not None:
            print('Backup folder path: %s' % self.bk)
        if delay is None:
            if bk is None: self.__call__ = lambda x: cv2.imshow(winname, x)
            else: self.__call__ = self.show_no_delay
        elif video is None:
            self.__call__ = self.show_delay

    def show_no_delay(self, img):
        if img is None:
            print('skipping display image is None')
            return
        cv2.imwrite(dirop(self.bk, self.winname, 'i.jpg', timestamp=True), img)
        cv2.imshow(self.winname, img)

    def show_delay(self, img):
        if img is None:
            print('skipping display image is None')
            return
        if self.bk:
            cv2.imwrite(dirop(self.bk, self.winname, 'i.jpg', timestamp=True), img)
        cv2.imshow(self.winname, img)
        key = cv2.waitKey(self.delay) & 255
        if key in (10, 141, 13):
            self.delay = self.pdelay
            self.pdelay = self.delay
            key = 'enter'
        elif key == 27:
            print('Esc pressed: closing window')
            if self.video:
                self.video.stop()
            key = 'esc'
        elif key == 32:
            self.delay = 0
            key = 'space'
        elif key == 115 and self.bk:
            cv2.imwrite(dirop(self.bk, '%s_ss' % self.winname, 'i.jpg', timestamp=True), img)
        elif key == 99:
            cv2.destroyAllWindows()
        return key

    def wait(self, waittime=None):
        key = cv2.waitKey(self.delay) & 255 if waittime is None else cv2.waitKey(waittime) & 255
        if key in (10, 141, 13):
            self.delay = self.pdelay
            self.pdelay = self.delay
            key = 'enter'
        elif key == 27:
            print('Esc pressed: closing window')
            if self.video:
                self.video.stop()
            key = 'esc'
        elif key == 32:
            self.delay = 0
            key = 'space'
        elif key == 99:
            cv2.destroyAllWindows()
        return key

    def __call__(self, img):
        if img is None:
            print('skipping imshow image is None')
            return
        if self.bk:
            cv2.imwrite(dirop(self.bk, self.winname, 'i.jpg', timestamp=True), img)
        cv2.imshow(self.winname, img)
        key = cv2.waitKey(self.delay) & 255
        if key in (10, 141, 13):
            self.delay = self.pdelay
            self.pdelay = self.delay
            key = 'enter'
        elif key == 27:
            print('Esc pressed: closing window')
            if self.video:
                self.video.stop()
            key = 'esc'
        elif key == 32:
            self.delay = 0
            key = 'space'
        elif key == 115 and self.bk:
            cv2.imwrite(dirop(self.bk, '%s_ss' % self.winname, 'i.jpg', timestamp=True), img)
        elif key == 99:
            cv2.destroyAllWindows()
        return key


def photoframe(imgs, col=None, rcsize=None, resize_method=cv2.INTER_LINEAR, fit=False, asgray=False, nimgs=0):
    '''
    # This method pack the array of images in a visually pleasing manner.
    # If the col is not specified then the row and col are equally divided
    # This method can automatically pack images of different size. Default stitch size is 128,128
    # when fit is True final photo frame size will be rcsize
    #          is False individual image size will be rcsize
    # Examples
    # --------
        video = Player(GetFeed(join(dbpath, 'videos', r'remove_rain.mp4')), custom_fn=None)
        for fnos, imgs in video.chunk(4):
            i1 = photoframe(imgs, col=None)
            i2 = photoframe(imgs, col=4)
            i3 = photoframe(imgs, col=4, rcsize=(200,300),nimgs=7)
            i4 = photoframe(imgs, col=3, nimgs=7)
            i5 = photoframe(imgs, col=4, rcsize=imgs[0].shape)
            i6 = photoframe(imgs, col=6, rcsize=imgs[0].shape, fit=True)
            i7 = photoframe(imgs, col=4, rcsize=imgs[0].shape, fit=True, asgray=True)
            for i, img in enumerate([i1, i2, i3, i4, i5, i6, i7], 1):
                print(i, img.shape)
                win('i%s' % i, )(img)
            win('totoal')(photoframe([i1, i2, i3, i4, i5, i6, i7]))
            if win().wait(waittime) == 'esc':
                break
    '''
    imgs = list(imgs)
    if len(imgs):
        imrow, imcol = (128, 128) if rcsize is None else rcsize[:2] # fetch first two vals
        nimgs = max(nimgs, len(imgs))
        col = int(np.ceil(nimgs ** .5)) if col is None else int(col)
        row = nimgs / col
        row = int(np.ceil(row + 1)) if (col * row) - nimgs else int(np.ceil(row))
        if fit:
            imrow /= row; imcol /= col
        imrow, imcol = int(imrow), int(imcol)
        resshape = (imrow, imcol) if asgray else (imrow, imcol, 3)
        imgs = zip_longest(list(range(row*col)), imgs, fillvalue=np.zeros(resshape,imgs[0].dtype))
        resimg = []
        for i, imggroup in groupby(imgs, lambda k: k[0] // col):
            rowimg = []
            for i, img in imggroup:
                if img.dtype != np.uint8:
                    img = float2img(img)
                if asgray:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if tuple(img.shape) != resshape:
                    img = cv2.resize(img, (imcol, imrow), interpolation=resize_method)
                rowimg.append(img)
            resimg.append(cv2.hconcat(rowimg))
        return cv2.vconcat(resimg)


class ImLog:
    '''
    # imsize: convert all image to specific size # (optimised)
    # fit   : fit the image to specific window size
    # maxfit: the window size will be size of the maximum image
    # max   : all images will be scale to max image size
    video = Player(GetFeed(join(dbpath, 'videos', r'remove_rain.mp4')), start=20, stop=300, resize=(300,600))
    if ' ':
        il = ImLog('fit', rcsize=(486, 752), resize_method=cv2.INTER_CUBIC)
        il_1 = ImLog('max', col=3)
        il_2 = ImLog('imsize', rcsize=(128, 256))
        il_3 = ImLog('maxfit', col=3)
        for fnos, imgs in video.chunk(5):
            for fno, img in zip(fnos, imgs):
                il.log(label1(img, fno))
                il_1.log(label1(img, fno))
                il_2.log(label1(img, fno))
                il_3.log(label1(img, fno))
            for logformat, i in ('fit', il), ('max', il_1), ('imsize', il_2), ('maxfit', il_3):
                imlog = i.get()
                print(imgs.shape, imlog.shape)
                # (5, 444, 333, 3)(256, 768, 3)
                win(logformat)(label1(imlog, logformat))
            if win().wait(waittime) == 'esc':
                break
    if ' ':
        il = ImLog('maxfit', col=3)
        for fnos, imgs in video.chunk(5):
            for fno, img in zip(fnos, imgs):
                il.log(label1(img, fno))
                il.winsize((500, 750))
            i = il.get()
            print(imgs.shape, i.shape)
            # (5, 444, 333, 3)(500, 750, 3)
            if win('out', 0)(i) == 'esc':
                break
    '''

    def __init__(self, stitch_format='max', col=None, rcsize=(320, 240), resize_method=cv2.INTER_LINEAR, asgray=False, nimgs=0):
        self.my_list = []
        self.setup(stitch_format, col, rcsize, resize_method, asgray, nimgs)

    def setup(self, log_format, col=None, rcsize=(256, 256), resize_method=cv2.INTER_LINEAR, asgray=False, nimgs=0):
        self.ncol = col
        self.row, self.col = rcsize
        self.resize_method = resize_method
        self.asgray = asgray
        self.nimgs = nimgs
        # imsize: convert all image to specific rcsize # (optimised)
        # fit   : fit the image to specific window rcsize
        # maxfit: the window rcsize will be rcsize of the maximum image
        # max   : all images will be scale to max image rcsize
        self.log_format = log_format.lower()
        return self

    def winsize(self, windowsize):
        self.row, self.col = windowsize
        self.log_format = 'fit'
        return

    def log(self, img):
        if self.log_format == 'imsize':
            img = cv2.resize(img, (self.row, self.col), interpolation=self.resize_method)
        r, c = img.shape[:2]
        if self.log_format in ('max', 'maxfit'):
            self.row = r if self.row < r else self.row
            self.col = c if self.col < c else self.col
        self.my_list.append(img)

    def get(self):
        if self.row == 0 or self.col == 0:
            raise Exception('size variable is not initialized')
        if self.log_format in ('fit', 'maxfit'):
            imgs = photoframe(self.my_list, self.ncol, (self.row, self.col), self.resize_method, True, self.asgray, self.nimgs)
        else:
            imgs = photoframe(self.my_list, self.ncol, (self.row, self.col), self.resize_method, False, self.asgray, self.nimgs)
        self.my_list = []
        return imgs

    def get_imgs(self):
        return self.my_list


class clk:
    '''
    use to time stamp
    video = Player(GetFeed(r'v.mp4'),resize=(290, 412))
    imshow = win('out',1, video)
    tic = clk()
    for fno, img in video.play():
        imshow(img)
        print(tic.toc())
    print('exe time: %ss        fps: %s' % tic.toc(fno))
    '''
    def __init__(self):
        # from datetime import datetime as dt
        self.ttic = dt.now()
        self.tic = self.ttic

    def time(self):
        return dt.now()

    def toc(self, fno=None):
        if fno is None:
            exe_time = (dt.now() - self.tic).total_seconds()
            self.tic = dt.now()
            return round(exe_time,4)
        exe_time = (dt.now() - self.ttic).total_seconds()
        self.tic = dt.now()
        if type(fno) in (list, tuple, np.ndarray):
            fno = fno[-1]
        return round(exe_time,4), round((fno / exe_time),4)