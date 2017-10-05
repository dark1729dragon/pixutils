from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals



from .utils2 import *


def get_angle(o_pt, n_pt):
    slope = n_pt - o_pt
    return cv2.fastAtan2(slope[1], slope[0])  # y/x


def get_dist(o_pt, n_pt):
    '''
    x = np.vstack([np.arange(0,10), np.arange(50,60)*2]).T
    y = np.vstack([np.arange(100,110), np.arange(250,260)*2]).T
    for xx,yy in zip(x,y):
        print(get_distance(xx,yy))
    :param pp:
    :param cp:
    :return:
    # cp = x0,y0; cp = x1,y1
    '''
    slope = n_pt - o_pt
    return cv2.norm(slope)


def translate(image, x, y):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # return the translated image
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(rho, theta):
    x = rho * np.cos(theta * np.pi / 180.0)
    y = rho * np.sin(theta * np.pi / 180.0)
    return x, y


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def dirsearch(rootpath):
    '''
    return all the file names inside the dir
    '''
    return [join(root, book) for root, dirs, books in os.walk(rootpath) for book in books if book]