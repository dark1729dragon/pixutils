from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals



from pixutils import *
from pixutils.faceutils import find_face

def example1(dbpath):
    '''
    1. read image
    2. crop the image form x0, y0 = 270, 276; with width and height of 150,175
    3. put the image at x0,y0 = 120,164; with width and height of 182,292
    4. before placing the image increase the size of bounding box by the scale .75, 1.15
    5. decrease the size of bounding box by 10 pixel in left
    6. increase the size of bounding box by 17 pixel in top
    7. increase the size of bounding box by 5 pixel in right
    8. increase the size of bounding box by 20 pixel in bottom
    '''
    img = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
    simg = get_subimage(img, bb=r2bb((270, 276, 150, 175)))  # r2bb -> convert open cv rect to bounding box
    win('sub img')(simg)
    # (.75,1.15) -> (scale_width, scale_height)
    # (-10, 17, 5, 20) -> (left_bias, top_bias, right_bias, bottom_bias)
    img = put_subimage(img, simg, r2bb((120, 164, 182, 292)), (.75, 1.15), (-10, 17, 5, 20), fit=True)
    win('img', 0)(img)


def example2(dbpath):
    '''
    1. read image
    2. crop the image by (.5,.5) of its size
    3. put the image at x0,y0 = 120,164; with width and height of 182,292
    4. before placing the image increase the size of bounding box by the scale .75, 1.15
    5. decrease the size of bounding box by 10 pixel in left
    6. increase the size of bounding box by 17 pixel in top
    7. increase the size of bounding box by 5 pixel in right
    8. increase the size of bounding box by 20 pixel in bottom
    '''
    img = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
    roi = im2bb(img, scale=(.5, .5))  # get the bounding box of complete image
    simg = get_subimage(img, bb=roi)
    win('sub img')(simg)
    # (.75,1.15) -> (scale_width, scale_height)
    # (-10, 17, 5, 20) -> (left_bias, top_bias, right_bias, bottom_bias)
    img = put_subimage(img, simg, r2bb((120, 164, 182, 292)), (.75, 1.15), (-10, 17, 5, 20), fit=True)
    win('img', 0)(img)


def example3(dbpath):
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


def example4(dbpath):
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


def example5(dbpath):
    if ' ':
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
    if '':
        face = np.zeros((1469, 1531), 'u1')
        rr = r2bb((103, 120, 743, 694))
        ss = r2bb((58, 71, 272, 281))
        scale, bias = (1.15, .35), (-12, 11, 51, -7)
        pp = bb_bias(rr, scale, bias)
        print(pp)
        simg = get_subimage(face.copy(), rr, scale, bias)
        simg = put_bb(simg, ss)
        win('simg')(simg)
        ii = shiftbb(rr, ss, scale, bias, win=face.shape)
        face = put_bb(face, ii)
        face = get_subimage(face, rr, scale, bias)
        print(simg.shape, face.shape)
        win('face', 0)(face)
        i, j = np.where(simg == 255)
        print(i.min(), j.min(), i.max(), j.max())
        i, j = np.where(face == 255)
        print(i.min(), j.min(), i.max(), j.max())

if __name__ == '__main__':
    example1(dbpath)
    # example2(dbpath)
    # example3(dbpath)
    # example4(dbpath)
    # dirop(dbpath, 'results', remove=True)  # clear all logs