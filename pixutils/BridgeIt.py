from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals

from pixutils.one_shot_import import *

# ------------------ these modules will take some time to import but for simple implementation clumped together ------------------ #
'''
These functions supports to convert one form of data to other.
Example: r2bb convert opencv rectange (x,y,w,h) to bounding box format (x0,y0,x1,y1)
'''


def bb_bias((x0, y0, x1, y1), scale=None, bias=(0, 0, 0, 0), win=None):
    '''
    This function will increase or decrease the size of rectangle roi by the bias factor
    This function can also be used to fit the rectangle inside the image
        It normalize the coordinates and avoid negative values
        bias: (left_bias,top_bias,right_bias,bottom_bias)
    Generally used funcitons
    Note:
    The bb need 4 points (x0,y0,x1,y1) so if your have (x,y,w,h) use r2bb to convert to point
        Example bb_bias(img,r2bb(x,y,w,h))
    The scale need 2 independent values
    The bias need 4 independent bias values (This is not a rectangle coordinate)
    Examples
    --------
    img = np.zeros((1469, 1531),'u1')
    bb = bb_bias((0, 0, 100, 100),(2,3), (22, 33, 27, 4),win=img.shape)
    print(bb)
    # (0, 0, 123, 196)
    bb = bb_bias((-9, -8, 150, 103),win=img.shape)  # fit the bbangle inside image
    print(bb)
    # (0, 0, 150, 103)
    bb = bb_bias(r2bb((120, 134, 184, 127)),(2,.5), (16, 34, 18, -37))
    # (12, 131, 378, 266)
    print(bb)
    '''
    x0b, y0b, x1b, y1b = bias
    if scale is not None:
        a, b = 1 - scale[0], 1 + scale[0]
        c, d = 1 - scale[1], 1 + scale[1]
        if win is None:
            x0, y0, x1, y1 = (x1 * a + x0 * b) / 2 - x0b,\
                             (y1 * c + y0 * d) / 2 - y0b,\
                             (x1 * b + x0 * a) / 2 - x1b, \
                             (y1 * d + y0 * c) / 2 - y1b
            return int(x0), int(y0), int(x1), int(y1)
        else:
            x0, y0, x1, y1 = max(0, (x1 * a + x0 * b) / 2 - x0b),\
                             max(0, (y1 * c + y0 * d) / 2 - y0b), \
                             min(win[1], (x1 * b + x0 * a) / 2 - x1b), \
                             min(win[0], (y1 * d + y0 * c) / 2 - y1b)
            return int(x0), int(y0), int(max(x0, x1)), int(max(y0, y1))
    else:
        if win is None:
            x0, y0, x1, y1 = x0 - x0b, y0 - y0b, x1 + x1b, y1 + y1b
            return int(x0), int(y0), int(x1), int(y1)
        else:
            x0, y0, x1, y1 = max(0, x0 - x0b), max(0, y0 - y0b), min(win[1], x1 + x1b), min(win[0], y1 + y1b)
            return int(x0), int(y0), int(max(x0, x1)), int(max(y0, y1))


def float2img(img, pixmin=0, pixmax=255, dtype=0):
    '''
    convert img to (0 to 255) range
    '''
    return cv2.normalize(img, None, pixmin, pixmax, 32, dtype)


def img2float(img, pixmin=0, pixmax=1, dtype=5):
    '''
    convert image to (0.0 to 1.0) range
    '''
    return cv2.normalize(img, None, pixmin, pixmax, 32, dtype)


def d2bb(d, asint=False):
    '''
    # Convert dlib rectangle to points
    # Note: output will be always tuple
    for i in find_face(img):
        print(d2bb(i))
    '''
    return (int(d.left()), int(d.top()), int(d.right()), int(d.bottom())) if asint else \
        (d.left(), d.top(), d.right(), d.bottom())


def d2r(d, asint=False):
    '''
    # Convert dlib rectangle to rectangle
    # Note: output will be always tuple
    for i in find_face(img):
        print(d2r(i))
    '''
    x0, y0, x1, y1 = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom())) if asint else \
        (d.left(), d.top(), d.right(), d.bottom())
    return x0, y0, x1 - x0, y1 - y0


def r2bb((x, y, w, h)):
    '''
    # Rectangle to bounding box converter
    # Converts opencv rectangle (x,y,w,h) to bounding box (x0,y0,x1,y1)
    # Note: output will be always tuple
    # Examples
    # --------
    print(r2bb((267,132,67,92)))
    # (267, 132, 334, 224)
    '''
    # dtype = type(rect)
    # return np.array((x, y, x + w, y + h)) if dtype == np.ndarray else dtype((x, y, x + w, y + h))
    return x, y, x + w, y + h


def bb2r((x0, y0, x1, y1)):
    '''
    # Point to rectangle converter
    # Converts opencv point (x0,y0,x1,y1) to rectangle (x,y,w,h)
    # Note: output will be always tuple
    # Examples
    # --------
    print(bb2r((267,132,367,392)))
    # (267, 132, 100, 260)
    '''
    # dtype = type(rect)
    # return np.array((x0, y0, x1 - x0, y1 - y0)) if dtype == np.ndarray else dtype((x0, y0, x1 - x0, y1 - y0))
    return x0, y0, x1 - x0, y1 - y0


def r2d((x, y, w, h)):
    '''
    # Convert rectangle to dlib rectangle
    # Note: output will be always tuple
    # Examples
    # --------
    print(r2d((267,132,67,92)))
    # [(267, 132) (334, 224)]
    # '''
    return dlib.rectangle(x, y, x + w, y + h)


def bb2d((x0, y0, x1, y1)):
    '''
    # Convert point to dlib rectangle
    # Examples
    # --------
    print(bb2d((267,132,67,92)))
    # [(267, 132) (67, 92)]
    # [(267, 132) (67, 92)]
    '''
    return dlib.rectangle(x0, y0, x1, y1)


def im2bb(img, scale=None, bias=(0, 0, 0, 0), win_fit=True):
    '''
    # Return starting and ending point of image
    img = resize(cv2.imread(impath), (63,127))
    print(im2bb(img))  # (0, 0, 127, 63)
    img = put_bb(img, im2bb(img))
    win('im2bb', 0)(img)
    '''
    return bb_bias((0, 0, img.shape[1], img.shape[0]), scale, bias, win=img.shape if win_fit else None)


def wh(img):
    '''
    # return widhth and height of the image
    img = resize(cv2.imread(impath), (63,127))
    print(wh(img))  # (127, 63)
    '''
    h, w = img.shape[:2]
    return w, h


def plt2cv(fig):
    '''
    # convert matplotlib plot to opencv image
    import matplotlib.pyplot as plt
    impaths = glob(r'*.jpg')
    fig, axs = plt.subplots(3,3)
    for impath,ax in zip(impaths,axs.ravel()):
        img = plt.imread(impath)
        ax.imshow(img)
    imgs = plt2cv(fig)
    plt.show()
    win('plt2cv', 0)(imgs)
    '''
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.array(fig.canvas.buffer_rgba(), dtype='u1').reshape(h, w, 4)
    return img[..., [2, 1, 0]]


def rectmid((a, b, c, d)):
    return int((a + c) / 2), int((b + d) / 2)


def splitpath(path, sep=os.sep):
    '''
    this function is used to recursively split the path and return img array
    example:
        # path = r'\\192.168.1.2\pixutils'
        # print(splitpath(path))
        ['192.168.1.2', 'pixutils']
    :param your_path:
    :return:
    '''
    path = path.replace('\\',sep)
    path = path.replace('/', sep)
    return [p for p in path.split(sep) if p]


def im2txt(img, bookpath=None, resolution=1.0, sep=''):
    '''
    This function create text representation of pixels
    useful to visualize how machine sees the image
    Example:
    --------
        for pix in im2txt(img,resolution=1/256.0,sep=''):
            print pix
        im2txt(img, 'temp.txt',resolution=.1,sep='-')

    :param img: input image
    :param bookpath: text path or (None -> return the text)
    :param resolution: resolution=.1 -> 255*.1 = 25; resolution=1/256.0
    :param sep:
    :return: if bookpath is None:
                returns list of rows of image
             else:
                returns file path (bookpath)  
    '''
    img = np.round((img * float(resolution)))
    if img.min() >= 0:
        val = str(len(str(img.max())) - 2)
        np.savetxt(bookpath or 'temp.txt', img, fmt=str(b'%' + val + 'd'), delimiter=str(sep))
    else:
        val = str(len(str(img.max())) - 1)
        np.savetxt(bookpath or 'temp.txt', img, fmt=str('%' + val + 'd'), delimiter=str(sep))
    if bookpath is None:
        with open('temp.txt', b'r') as book:
            bookpath = book.read().split('\n')
        dirop('temp.txt', remove=True)
    return bookpath

