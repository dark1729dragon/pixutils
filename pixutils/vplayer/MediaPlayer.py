from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals




from .videoutils import *
class Player:
    '''
    ____________________________________Example____________________________________
    # Get two feed and run in threads
    # Feed1: read the sequence of videoimgs and and detect faces
    # Feed2: Get feed from live camera and detect faces
    # put bounding box and merge images and show
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
    ____________________________________Example____________________________________
    '''
    def __init__(self, cam, start=0, stop=None, resize=None, resize_method=cv2.INTER_LINEAR, custom_fn=None, hide_print=False):
        (self.source, self.cam), self.custom_fn, self.resize, self.start = cam, custom_fn, resize, 0
        if start and self.source != 'stream':
            self.cam.set(1, start)
            self.start = start
        self.width_col, self.height_row, self.fps, self.total_frames = video_meta_data(self.cam, self.source)
        self.end = min(stop or np.inf, self.total_frames)
        self.resize_method = resize_method
        if resize:
            if type(resize) in (tuple, list):
                self.height_row, self.width_col = resize
            else:
                self.height_row, self.width_col = int(self.height_row * resize), int(self.width_col * resize)
        self.cam_matrix = get_cam_matrix((self.height_row, self.width_col))
        self.lens_distortion = get_lens_distortion()
        try:  # iniatzing theaded cam
            self.cam.initThread()
        except:
            pass
        if not hide_print:
            print('Soruce                : %s' % self.source)
            print('Start frame           : %s' % self.start)
            print('Stop frame            : %s' % self.end)
            print('Resize                : %s' % str(self.resize))
            print('Resize Method         : %s' % 'NA' if self.resize is None else self.resize_method)
            print('height_row, width_col : %s, %s' % (self.height_row, self.width_col))
        self._stop = False
        self.write_width_col, self.write_height_row = None, None
        self.wh = self.width_col, self.height_row
        self.rc = self.height_row, self.width_col

    def __chunk_non_optimized(self, chunk_size):
        chunks, frameno = [], []
        crnt_frame = self.start
        while True:
            grabbed, img = self.cam.read()
            if self.custom_fn is not None: crnt_frame, grabbed, img = self.custom_fn(crnt_frame, grabbed, img)
            if grabbed is None: continue
            if self._stop or img is None or crnt_frame >= self.end:
                self.stop(closeThread=True)
                print('video ended (frameno): %s' % crnt_frame)
                break
            if self.resize is not None: img = cv2.resize(img, (self.width_col, self.height_row),interpolation=self.resize_method)

            # video reader start here
            chunks.append(img)
            frameno.append(crnt_frame)
            crnt_frame += 1
            if len(frameno) == chunk_size:
                yield np.array(frameno), np.array(chunks)
                chunks, frameno = [], []

    def __chunk_optimized(self, chunk_size):
        chunks, frameno, count, crnt_frame = None, [], 0, self.start
        while True:
            grabbed, img = self.cam.read()
            if self.custom_fn is not None: crnt_frame, grabbed, img = self.custom_fn(crnt_frame, grabbed, img)
            if grabbed is None: continue
            if self._stop or img is None or crnt_frame >= self.end:
                self.stop(closeThread=True)
                print('video ended (frameno): %s' % crnt_frame)
                break
            if self.resize is not None: img = cv2.resize(img, (self.width_col, self.height_row), interpolation=self.resize_method)

            # video reader start here
            try:
                chunks[count, ...] = img
            except:
                if chunks is not None: raise
                else:
                    # initialize chunk
                    shape = [chunk_size]
                    shape.extend(img.shape)
                    chunks = np.empty(shape, 'u1')
                    chunks[count, ...] = img
            crnt_frame += 1
            count += 1
            frameno.append(crnt_frame)
            if count == chunk_size:
                yield np.array(frameno), chunks
                frameno, count = [], 0

    def chunk(self, chunk_size):
        '''
        optimized version is not tested properly
        use the following command to call optimized video chunk Player
        for fnos, imgs in video.play_video_chunk_optimized(<chunk_size>):
            img = photoframe(imgs, rcsize=imgs[0].shape)
            print(video.cam.q.qsize())
            imshow(img)
        :param chunk_size:
        :return:
        '''
        return self.__chunk_non_optimized(chunk_size)
        # else:
        #     return self.play_video_chunk_optimized(chunk_size)

    def queue(self, queue_size):
        temp, frameno = deque(maxlen=queue_size), deque(maxlen=queue_size)
        crnt_frame = self.start
        while True:
            grabbed, img = self.cam.read()
            if self.custom_fn is not None: crnt_frame, grabbed, img = self.custom_fn(crnt_frame, grabbed, img)
            if grabbed is None: continue
            if self._stop or img is None or crnt_frame >= self.end:
                self.stop(closeThread=True)
                print('video ended (frameno): %s' % crnt_frame)
                break
            if self.resize is not None: img = cv2.resize(img, (self.width_col, self.height_row),interpolation=self.resize_method)

            # video reader start here
            temp.append(img)
            frameno.append(crnt_frame)
            crnt_frame += 1
            if len(temp) == queue_size:
                yield np.array(frameno), np.array(temp)

    def play(self, _=None):
        crnt_frame = self.start
        while True:
            grabbed, img = self.cam.read()
            if self.custom_fn is not None: crnt_frame, grabbed, img = self.custom_fn(crnt_frame, grabbed, img)
            if grabbed is None: continue
            crnt_frame += 1
            if self._stop or img is None or crnt_frame > self.end:
                self.stop(closeThread=True)
                print('video ended (frameno): %s' % crnt_frame)
                break
            if self.resize is not None: img = cv2.resize(img, (self.width_col, self.height_row),interpolation=self.resize_method)
            yield crnt_frame, img
        self.cam.release()

    def stop(self, closeThread=False):
        self._stop = True
        if closeThread:
            try:
                self.cam.stop()
            except:
                pass

    def impaths(self, fno):
        try:
            return self.cam.paths[fno]
        except:
            return str(fno)