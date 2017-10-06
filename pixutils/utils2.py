from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals

# ----------------------------------------------default-------------------------------------------- #
foreground = (90, 43, 69)  # foreground, -1
# ------------------------------------------------------------------------------------------------- #

from .BridgeIt import *


def get_subimage(img, bb, scale=None, bias=(0, 0, 0, 0)):
    '''
    Crop the image inside the bounding box coordinate and return the subimage
    :param img:
    :param bb: (x0,y0,x1,y1)
    :param scale: .5,.5
    :param bias: (left_bias,top_bias,right_bias,bottom_bias); this is not an x0,y0,x1,y1
    :return: sub image
    Examples
    --------
        img = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
        simg = get_subimage(img, bb=r2bb((270, 276, 150, 175)))  # r2bb -> convert open cv rect to bounding box
        win('sub img')(simg)
        # (.75,1.15) -> (scale_width, scale_height)
        # (-10, 17, 5, 20) -> (left_bias, top_bias, right_bias, bottom_bias)
        img = put_subimage(img, simg, r2bb((120, 164, 182, 292)), (.75, 1.15), (-10, 17, 5, 20), fit=True)
        win('img', 0)(img)
    '''
    x0, y0, x1, y1 = bb_bias(bb, scale, bias, img.shape)
    img = img[y0:y1, x0:x1]
    return None if img.size == 0 else img


def put_subimage(img, subimg, bb, scale=None, bias=(0, 0, 0, 0), fit=False, resize_method=cv2.INTER_LINEAR):
    '''
    This will place the subimage inside the roi.
    When fit is True this will scale the subimg to match the put_bb log_format
    Examples
    --------
        img = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
        roi = im2bb(img, scale=(.5, .5))  # get the bounding box of complete image
        simg = get_subimage(img, bb=roi)
        win('sub img')(simg)
        # (.75,1.15) -> (scale_width, scale_height)
        # (-10, 17, 5, 20) -> (left_bias, top_bias, right_bias, bottom_bias)
        img = put_subimage(img, simg, r2bb((120, 164, 182, 292)), (.75, 1.15), (-10, 17, 5, 20), fit=True)
        win('img', 0)(img)

    '''
    x0, y0, x1, y1 = bb_bias(bb, scale, bias, img.shape)
    if fit:
        w, h = x1 - x0, y1 - y0
        subimg = cv2.resize(subimg, (w, h), interpolation=resize_method)
    img[y0:y1, x0:x1] = subimg
    return img


def get_mask_subimage(img, bb, mask, scale=None, bias=(0, 0, 0, 0)):
    '''
    if ' ':
        oimg = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
        img = oimg[..., 1].copy()
        mask = np.zeros_like(img)
        for drect in find_face(img, 1):
            mask = put_bb(mask, d2bb(drect), thickness=-1, scale=(.75, 1.5))
            mask = put_bb(mask, d2bb(drect), thickness=-1, scale=(1.5, .75))
            win('mask')(mask)
            face = get_mask_subimage(img, d2bb(drect), mask, scale=(1.5, 1.5))
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
            oimg = put_subimage(oimg, face, d2bb(drect), scale=(1.5, 1.5))
        win('gray_face', 0)(oimg)
    if ' ':
        img = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
        mask = np.zeros_like(img[..., 0])
        for drect in find_face(img, 1):
            mask = put_bb(mask, d2bb(drect), thickness=-1, color=(255, 255, 255))
            img = get_mask_subimage(img, d2bb(drect), mask=mask)
        win('ly face', 0)(img)
    '''
    if len(img.shape) == 2:
        img = cv2.bitwise_and(img, mask)
    else:
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, mask)
    return get_subimage(img, bb, scale, bias)


def get_grid(win, ngrids=None, gridshape=None):
    '''
    Split the image into grids and return the grid rectangle corrdinate
    :param img:
    :param grid: number of grid in row, col
    :return: list of rectangle bounding boxes
    Examples
    --------
        oimg = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
        img = oimg.copy()
        imshow = win('image', 0)
        bbs = get_grid(img, ngrids=(6, 3))
        for bb in bbs:
            img = put_bb(img, bb)
        imshow(img)
        img = oimg.copy()
        bbs = get_grid(img, gridshape=(30, 60))  # shape -> row_size, col_size
        for bb in bbs:
            img = put_bb(img, bb, color=(255, 255, 255))
        imshow(img)
    '''
    if ngrids:
        (r, c), (winr, winc) = ngrids, win.shape[:2]
        r, c, w, h = list(range(r)), list(range(c)), int(winr / r), int(winc / c)
    else:
        winr, winc = win.shape[:2]
        r, c = int(winr / gridshape[0]), int(winc / gridshape[1])
        r, c, w, h = list(range(r)), list(range(c)), int(winr / r), int(winc / c)
    bbs = [r2bb((h * j, w * i, h, w)) for i in r for j in c]
    return bbs


def resize(img, size, method=cv2.INTER_LINEAR):
    '''
    Examples
    --------
    print resize(np.zeros(50,50), 4,cv2.INTER_NEAREST).shape
    print resize(np.zeros(50,50), .5,cv2.INTER_NEAREST).shape
    print resize(np.zeros(50,50), (600,600),cv2.INTER_NEAREST).shape

    '''
    r, c = size if type(size) in (tuple, list) else (img.shape[0] * size, img.shape[1] * size)
    return cv2.resize(img, (int(c), int(r)), interpolation=method)


def put_circle(img, pt, radius=3, color=(0, 0, 255), thickness=-1):
    cv2.circle(img, tuple((int(pt[0]), int(pt[1]))), int(radius), color, thickness)
    return img


def put_bb(img, bb, color=(255, 255, 255), thickness=3, scale=None, bias=(0, 0, 0, 0)):
    x0, y0, x1, y1 = bb_bias(bb, scale, bias, win=img.shape)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    return img


def label1(img, text, loc=(30, 30), fg=foreground, bg=None, textsize=.5, thickness=3, scale=None, bias=(0, 0, 0, 0)):
    '''
    NOTE:
        when loc get 4 values label will automatically draw bounding box
        rect = r2bb((30,30,100,200))
        win('label with bounding box', 0)   (label1(np.zeros((300,300),'u1'),'bb',   loc=rect))
        win('label without bounding box', 0)(label1(np.zeros((300,300),'u1'),'no bb',loc=rect[:2]))
    '''
    bg = (255 - fg[0], 255 - fg[1], 255 - fg[2]) if bg is None else bg
    if len(loc) == 4:
        x0, y0, x1, y1 = bb_bias(loc, scale, bias, win=img.shape)
        cv2.rectangle(img, (x0, y0), (x1, y1), bg, thickness)
    else:
        x0, y0 = loc[:2]
        x0, y0 = int(x0), int(y0)
    text = str(text)
    rt, tt = (y0 - 18, y0 - 6)
    cv2.rectangle(img, (x0, y0), (x0 + 10 * len(text), rt), bg, -1)
    cv2.putText(img, text, (x0, tt), 4, textsize, fg, 1)
    return img


def label2(img, text, loc=(30, 30), fg=foreground, bg=None, textsize=.5, thickness=3, scale=None, bias=(0, 0, 0, 0)):
    '''
    NOTE:
        when loc get 4 values label will automatically draw bounding box
        rect = r2bb((30,30,100,200))
        win('label with bounding box', 0)   (label2(np.zeros((300,300),'u1'),'bb',   loc=rect))
        win('label without bounding box', 0)(label2(np.zeros((300,300),'u1'),'no bb',loc=rect[:2]))
    '''
    bg = (255 - fg[0], 255 - fg[1], 255 - fg[2]) if bg is None else bg
    if len(loc) == 4:
        x0, y0, x1, y1 = bb_bias(loc, scale, bias, win=img.shape)
        cv2.rectangle(img, (x0, y0), (x1, y1), bg, thickness)
    else:
        x0, y0 = loc[:2]
        x0, y0 = int(x0), int(y0)
    text = str(text)
    rt, tt = (y0 + 18, y0 + 12)
    cv2.rectangle(img, (x0, y0), (x0 + 10 * len(text), rt), bg, -1)
    cv2.putText(img, text, (x0, tt), 4, textsize, fg, 1)
    return img


def label3(img, text, loc=None, fg=foreground, bg=None, textsize=.5, thickness=3, scale=None, bias=(0, 0, 0, 0)):
    '''
    NOTE:
        when loc get 4 values label will automatically draw bounding box
        rect = r2bb((30,30,100,200))
        win('label with bounding box', 0)   (label3(np.zeros((300,300),'u1'),'bb',   loc=rect))
        win('label without bounding box', 0)(label3(np.zeros((300,300),'u1'),'no bb',loc=rect[:2]))
    '''
    bg = (255 - fg[0], 255 - fg[1], 255 - fg[2]) if bg is None else bg
    if loc and len(loc) == 4:
        x0, y0, x1, y1 = bb_bias(loc, scale, bias, win=img.shape)
        cv2.rectangle(img, (x0, y0), (x1, y1), bg, thickness)
    else:
        x0, y1 = (30, img.shape[0] - 60) if loc is None else loc[:2]
        x0, y1 = int(x0), int(y1 + 30)
    text = str(text)
    rt, tt = (y1 - 18, y1 - 6)
    cv2.rectangle(img, (x0, y1), (x0 + 10 * len(text), rt), bg, -1)
    cv2.putText(img, text, (x0, tt), 4, textsize, fg, 1)
    return img


def label4(img, text, loc=None, fg=foreground, bg=-1, textsize=.5, thickness=3, scale=None, bias=(0, 0, 0, 0)):
    '''
    NOTE:
        when loc get 4 values label will automatically draw bounding box
        rect = r2bb((30,30,100,200))
        win('label with bounding box', 0)   (label4(np.zeros((300,300),'u1'),'bb',   loc=rect))
        win('label without bounding box', 0)(label4(np.zeros((300,300),'u1'),'no bb',loc=rect[:2]))
    '''
    bg = (255 - fg[0], 255 - fg[1], 255 - fg[2]) if bg is None else bg
    if loc and len(loc) == 4:
        x0, y0, x1, y1 = bb_bias(loc, scale, bias, win=img.shape)
        cv2.rectangle(img, (x0, y0), (x1, y1), bg, thickness)
    else:
        x0, y1 = (30, img.shape[0] - 60) if loc is None else loc[:2]
        x0, y1 = int(x0), int(y1 + 30)
    text = str(text)
    rt, tt = (y1 + 18, y1 + 12)
    cv2.rectangle(img, (x0, y1), (x0 + 10 * len(text), rt), bg, -1)
    cv2.putText(img, text, (x0, tt), 4, textsize, fg, 1)
    return img


def get_ibig_bb(bbs):
    '''
    For given list of rectangles return index of the biggest bounding box by its area
    '''
    size = len(bbs)
    assert size, 'The input is empty'
    if size == 1:
        return 0
    else:
        wight_hights = [(x1 - x0, y1 - y0) for x0, y0, x1, y1 in bbs]
        maxarea, imaxbb = 0, 0
        for i, (w, h) in enumerate(wight_hights):
            area = w * h
            if area > maxarea:
                maxarea, imaxbb = area, i
        return imaxbb


def mode(results):
    '''
    Calculate the mode of the input
    :param results:
    :return:
    '''
    mode, counts = np.unique(results, return_counts=True)
    maxcount = counts.max()
    imode = np.where(counts == maxcount)
    return mode[imode]


class LabelEncoder2:
    def __init__(self):
        self.labels = []

    def encode(self, label):
        '''
        Encode only one item
        :param label:
        :return:
        '''
        if label not in self.labels:
            self.labels.append(label)
        return self.labels.index(label)

    def decode(self, elabel):
        '''
        Decode only one item
        :param elabel:
        :return:
        '''
        return self.labels[elabel]

    def encodes(self, labels):
        '''
        Encodes list of items
        :param labels:
        :return:
        '''
        elabels = []
        for label in labels:
            if label not in self.labels:
                self.labels.append(label)
            elabels.append(self.labels.index(label))
        return np.array(elabels)

    def decodes(self, elabels):
        '''
        Decode list of items
        :param elabels:
        :return:
        '''
        return [self.labels[elabel] for elabel in elabels]

    def restore(self, labels, elabels):
        nnew = max(elabels) + 1
        nold = len(self.labels)
        if nold < nnew:
            new = [None] * nnew
            new[:len(self.labels)] = self.labels
            for label, elabel in zip(labels, elabels):
                new[elabel] = label
            self.labels = new


def shiftbb(bigbb, querybb, scale=None, bias=(0, 0, 0, 0), win=None):
    '''
    Returns the location of small bounding box inside the big bounding box
    Eg: location of eye rectangles inside a face rectangle in a image
    Examples
    --------
        img = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
        face_rect = (228, 228, 377, 377)
        face = get_subimage(img,face_rect).copy()
        eye_rects = (10, 20, 60, 60),(85,25,125,60)
        for eye_rect in eye_rects:
            face = put_bb(face,eye_rect)
            eye_in_face = shiftbb(face_rect,eye_rect)
            img = put_bb(img, eye_in_face)
        win("eye in face")(face)
        win("eye in img",0)(img)
    '''
    qx0, qy0, qx1, qy1 = querybb
    bx, by, _, _ = bb_bias(bigbb, scale, bias, win)
    return bx + qx0, by + qy0, bx + qx1, by + qy1


def border(img, (left_border, top_border, right_border,bottom_border), val=0):
    '''
    This will add border to the given image
        img = np.zeros((300, 300), 'u1')
        img = border(img, (100, 200, 300, 400), 125)
        win("gray image bordering", )(img.copy())
        img = np.zeros((300, 300, 3), 'u1')
        img = border(img, (100, 200, 300, 400), 125)
        win("color image bordering", )(img.copy())
        img = np.zeros((300, 300, 3), 'u1')
        i = border(img, (100, 200, 300, 400), [(125, 68, 27), (56, 77, 159), (100, 20, 189), (68, 10, 99)])
        print i.shape
        win("3channel image multi bordering", 0)(i)
    '''
    try:
        a, b, c, d = val
    except:
        a, b, c, d = val, val, val, val
    if len(img.shape) == 2:
        p = np.empty((img.shape[0], left_border), img.dtype)
        p[:] = a
        img = cv2.hconcat([p, img])
        p = np.empty((top_border, img.shape[1]), img.dtype)
        p[:] = b
        img = cv2.vconcat([p, img])
        p = np.empty((img.shape[0], right_border), img.dtype)
        p[:] = c
        img = cv2.hconcat([img, p])
        p = np.empty((bottom_border, img.shape[1]), img.dtype)
        p[:] = d
        img = cv2.vconcat([img, p])
        return img
    else:
        p = np.empty((img.shape[0], left_border, 3), img.dtype)
        p[:] = a
        img = cv2.hconcat([p, img])
        p = np.empty((top_border, img.shape[1], 3), img.dtype)
        p[:] = b
        img = cv2.vconcat([p, img])
        p = np.empty((img.shape[0], right_border, 3), img.dtype)
        p[:] = c
        img = cv2.hconcat([img, p])
        p = np.empty((bottom_border, img.shape[1], 3), img.dtype)
        p[:] = d
        img = cv2.vconcat([img, p])
        return img
