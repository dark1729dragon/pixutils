from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals



from pixutils import *



def example1(dbpath):
    print('Demo: dirop')
    test_dir, out = 'test', ['e1','e2']
    if ' ':
        print('-' * 150)
        print('os.path.join      ->', join(dbpath, r'results///desktop\\\img', 'log', 'out'))
        print('dirop join        ->', dirop(dbpath, r'results\\\desktop///img', 'log', 'out'))  # this will create dir if not exist
        print('deleting folder   ->', dirop(dbpath, r'results\desktop/img', remove=True))  # this will delete the folder if exist
        print('hashing           ->', dirop(dbpath, r'results\log', out, hash=True))
        print('hashing           ->', dirop(dbpath, r'results\log', out, hash=True))
        print('time stamp hash   ->', dirop(dbpath, 'results', test_dir, 'log', out, out, timestamp=True))
        print('skipping creating ->', dirop(dbpath, 'results', test_dir, 'log', out, out, timestamp=True, mkdir=False))
        cv2.imwrite(dirop(dbpath, r'results\\\desktop///img', 'log', 'gray.jpg', hash=True), 128 + np.zeros((100, 100), 'u1'))
        cv2.imwrite(dirop(dbpath, r'results///desktop\\\img', 'log', 'gray.jpg', hash=True), 128 + np.zeros((100, 100), 'u1'))
        cv2.imwrite(dirop(dbpath, r'results\\\desktop///img', 'log', 'white.jpg', timestamp=True), 255 + np.zeros((100, 100), 'u1'))
        print('-' * 150)
        print('See the result at: %s' % dirop(dbpath, 'results', mkdir=False))


def example2(dbpath):
    if ' ':
        img = cv2.imread(join(dbpath, 'imgs', 'lenna.png'))
        win('image', 0, winsize=(250, 500), bk=dirop(dbpath, 'results', 'bk', hash=True))(img)
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


def example3(dbpath, eg1=True, waittime = 0):
    if eg1:
        video = Player(GetFeed(join(dbpath, 'videos', r'remove_rain.mp4')), custom_fn=None)
    else:
        video = Player(GetFeed(join(dbpath, 'videos', r'remove_rain.mp4')), custom_fn=lambda fno,k,img:(fno,k,cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)))
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


def example4(dbpath, waittime):
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


if __name__ == '__main__':
    example1(dbpath)
    # example2()
    # example3()
    # example4()
    # dirop(dbpath, 'results', remove=True)  # clear all logs