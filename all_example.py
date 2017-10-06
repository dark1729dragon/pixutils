from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals

from __builtin__ import help
import argparse
from pixutils import *
# the pixutils will automatically import cv2, dlib and all frequently required modules
# these modules are imported from one_shot_import
# you can add your module over here
# and can import easily with from pixutils import *

'''
# esc   -> close the video
# space -> pause video
# enter -> play video
if bk!= None: # imshow = win('output', 1, video, bk=dirop(dbpath, 'results', remove=True))
    # s     -> take screen short (cannot support multiple window)

dirop is helpful to support directory level operations like
    automatically creating multiple folders/files recursively (if not found)
    deleting multiple folders/files recursively
    time stamping files/folders
    appending unique hash number at end of folder automatically
    prefix the file name with certain preset value (like frame no) 
'''




from pixutils.examples import MediaPlayer_examples, io_example, utils2_example

if '':
    help(float2img)
    print('_' * 150)
    help(img2float)
    print('_' * 150)
    help(d2bb)
    print('_' * 150)
    help(d2r)
    print('_' * 150)
    help(r2bb)
    print('_' * 150)
    help(bb2r)
    print('_' * 150)
    help(mode)
    print('_' * 150)
    help(r2d)
    print('_' * 150)
    help(bb2d)
    print('_' * 150)
    help(im2bb)
    print('_' * 150)
    help(wh)
    print('_' * 150)
    help(plt2cv)
    print('_' * 150)
    # help(rectmid)
    # print('_' * 150)
    help(splitpath)
    print('_' * 150)
    help(resize)
    print('_' * 150)
    help(put_circle)
    print('_' * 150)
    help(put_bb)
    print('_' * 150)
    help(get_ibig_bb)
    print('_' * 150)
    help(shiftbb)
    print('_' * 150)
    help(bb_bias)
    print('_' * 150)
    help(label1)
    print('_' * 150)
    help(label2)
    print('_' * 150)
    help(label3)
    print('_' * 150)
    help(label4)
    print('_' * 150)
    help(get_subimage)
    print('_' * 150)
    help(put_subimage)
    print('_' * 150)
    help(get_mask_subimage)
    print('_' * 150)
    help(get_grid)
    print('_' * 150)
    help(dirop)
    print('_' * 150)
    help(photoframe)
    print('_' * 150)
    help(Player)
    print('_' * 150)


parser = argparse.ArgumentParser(description='Example of pixutils module')
parser.add_argument('--dbpath', default=dirop(os.sep, os.sep.join(splitpath(__file__)[:-1]), r'data_base',mkdir=False),
                    help='data base path eg: /pixutils_dev/data_base')
args = parser.parse_args()
# dbpath = dirop(os.sep, os.sep.join(splitpath(__file__)[:-1]), r'data_base',mkdir=False)
dbpath = dirop(args.dbpath,mkdir=False)
print('dbpath: %s' % dbpath)
MediaPlayer_examples.example4(dbpath)
MediaPlayer_examples.example1(dbpath)
MediaPlayer_examples.example2(dbpath)
MediaPlayer_examples.example3(dbpath)
MediaPlayer_examples.example5(dbpath)
MediaPlayer_examples.example6(dbpath)
MediaPlayer_examples.example7(dbpath)
io_example.example1(dbpath)
io_example.example2(dbpath)
io_example.example3(dbpath, eg1=True, waittime=0)
io_example.example3(dbpath, eg1=False, waittime=0)
io_example.example4(dbpath, waittime=0)
utils2_example.example1(dbpath)
utils2_example.example2(dbpath)
utils2_example.example3(dbpath)
utils2_example.example4(dbpath)
utils2_example.example5(dbpath)
utils2_example.example6(dbpath)
