from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals



from pixutils import *
find_face = dlib.get_frontal_face_detector()

fshape = 96, 96

######################### FACE ALIGNMENT ############################3
imgDim = fshape[0]
TEMPLATE = np.float32([(0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943), (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066), (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778), (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149), (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107), (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279), (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421), (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744), (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053), (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323), (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851), (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854), (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114), (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193), (0.516221448289, 0.396200446263),
                       (0.517118861835, 0.473797687758), (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668), (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208), (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656), (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002), (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083), (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225), (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267), (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656), (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172), (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073), (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768), (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516), (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972), (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
                       (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727), (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612), (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691), (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626), (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
landmarkIndices = OUTER_EYES_AND_NOSE
npLandmarkIndices = np.array(landmarkIndices)
######################### FACE POSE ESTIMATOR ##########################3
'''
    # landmarks_3D is the 3D points of the head landmarks, obtained from antrophometric measurement on the human head.
    landmarks_3D = [RIGHT_SIDE(0), GONION_RIGHT(4), MENTON(8), GONION_LEFT(12), LEFT_SIDE(16),
    FRONTAL_BREADTH_RIGHT(17),FRONTAL_BREADTH_LEFT(26), SELLION(27), NOSE(30), SUB_NOSE(33),
    RIGHT_EYE(36), RIGHT_TEAR(39), LEFT_TEAR(42),LEFT_EYE(45), STOMION(62)]
'''

landmarks_3D = np.float32([[-100.0, -77.5, -5.0], [-110.0, -77.5, -85.0], [0.0, 0.0, -122.7], [-110.0, 77.5, -85.0], [-100.0, 77.5, -5.0], [-20.0, -56.1, 10.0], [-20.0, 56.1, 10.0], [0.0, 0.0, 0.0], [21.1, 0.0, -48.0], [5.0, 0.0, -52.0], [-20.0, -65.5, -5.0], [-10.0, -40.5, -5.0], [-10.0, 40.5, -5.0], [-20.0, 65.5, -5.0], [10.0, 0.0, -75.0]])
FACE_POSE_ESTIMATOR = [0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62]
AXIS = np.float32([[75, 0, 0], [0, 75, 0], [0, 0, 75]])

skin_fp1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 46, 35, 34, 33, 32, 31, 36]
skin_fp2 = [31, 32, 33, 34, 35, 54, 55, 56, 57, 58, 59, 48]
full_face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]


# ALL_POINTS = range(0, 68)  # Used for debug only

###################################### Face Detector ##############################################


def face_landmarks(img, drect=None, featurepoints=None):
    '''
    drect, mask = find_face(img, 1)[0], np.zeros_like(img)
    lrect, landmarks = face_landmarks(img, drect, full_face)
    cv2.drawContours(mask, [landmarks], 0, (255, 255, 255), -1)
    imshow('vide output', mask, 1)
    '''
    detected_landmarks = pose_predictor(img, drect).parts()
    landmarks = np.array([[p.x, p.y] for p in detected_landmarks])
    (l, m), (n, o) = landmarks.min(axis=0), landmarks.max(axis=0)
    return ((l, m, n, o), landmarks) if featurepoints is None else ((l, m, n, o), landmarks[featurepoints])


def get_all_face_info(img, upsample_num_times=1, featurepoints=full_face):
    '''
    faces = get_all_face_info(img, 1)
    for rect, lrect, landmarks in faces:
        img = put_rect(img, rect)
        img = put_rect(img, lrect)
        print(landmarks)
    imshow('mask', mask, 0)
    imshow('img', img, 0)
    '''
    faces = []
    for drect in find_face(img, upsample_num_times):
        mask = np.zeros_like(img)
        lrect, landmarks = face_landmarks(img, drect, featurepoints)
        cv2.drawContours(mask, [landmarks], 0, (255, 255, 255), -1)
        faces.append((d2bb(drect), lrect, mask, landmarks))
    return faces


def face_pose_estimator(landmarks, cam_matrix, lens_distortion, featurepoints=FACE_POSE_ESTIMATOR):
    '''
    for fno, img in video.play():
        for drect in find_face(img,1):
            lrect, landmarks = face_landmarks(img,drect)
            linept, imgpts = face_pose_estimator(landmarks, video.get_cam_matrix, video.get_lens_distortion)
            cv2.line(img, linept, tuple(imgpts[0]), (0, 0, 255), 1)  # RED
            cv2.line(img, linept, tuple(imgpts[1]), (0, 255, 0), 1)  # GREEN
            cv2.line(img, linept, tuple(imgpts[2]), (255, 0, 0), 1)  # BLUE
        imshow('img  51 temp1', img, 1)
    '''
    landmarks_2D = landmarks[[featurepoints]].astype('f4')
    success, rotation_vector, translation_vector = cv2.solvePnP(landmarks_3D, landmarks_2D, cam_matrix, lens_distortion)
    # Projecting the 3D points into the image plane
    imgpts, jac = cv2.projectPoints(AXIS, rotation_vector, translation_vector, cam_matrix, lens_distortion)
    face_poses = (landmarks_2D[7][0], landmarks_2D[7][1]), imgpts.reshape(-1, 2)
    return face_poses


def align_face(rgbImg, landmarks):
    H = cv2.getAffineTransform(landmarks[npLandmarkIndices].astype('f4'), imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
    alignedFace = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
    return alignedFace

if __name__ == '__main__':
    if 0:
        full_face = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]
        vpath = join(dbpath, r'videos/remove_rain.mp4')
        cam = GetFeed(vpath)
        # cam = ThreadIt(cam)
        video = Player(cam, start=150)
        imshow = win('video output',1,video)
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
    if ' ':
        vpath = join(dbpath, r'videos/remove_rain.mp4')
        cam = GetFeed(vpath)
        cam = ThreadIt(cam)
        video1 = Player(cam, start=150)

        vpath = 0
        def detect_face1(grab, img):
            if img is not None:
                face_rects = [d2bb(drect) for drect in find_face(img)]
                img = img, face_rects
            return grab, img

        vpath = 0
        cam = GetFeed(vpath)
        cam = ThreadIt(cam, thread_buff_size=50, fps=50, custom_fn=detect_face1)
        video2 = Player(cam)
        imshow = win('face', 1, video2, dirop(dbpath,'results', r'temp',timestamp=True))
        for i, ((fno1, img1), (fno2, (img2, face_rects))) in enumerate(zip(video1.play(), video2.play())):
            for face_rect in face_rects:
                img2 = put_bb(img2, face_rect, thickness=3)
            img = photoframe([img1, img2], rcsize=img1.shape)
            print(video2.cam.q.qsize())
            imshow(img)