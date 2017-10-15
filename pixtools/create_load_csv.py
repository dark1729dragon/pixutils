from __future__ import nested_scopes, generators, division, absolute_import, with_statement, print_function, unicode_literals

from pixutils import *
import pandas as pd

class CreateCsv():
    """
    This will crate a csv which will help to load image easily form folder
    eg1:
        create a train.csv file which has list of images used for training
        in csv the images for training starts from the row labels,status...
        the first column is labels name of the image
        the second column is status <pass, miss, skip>
            pass: the image found and can be used for training
            miss: the image is missing from the folder
            skip: the image is not used for training
        the all other column has the relative the image path

        def labeller(impath):
            val = basename(dirname(impath))
            return val

        root = r'faces/lfw_home/lfw_funneled_sample'
        impaths = []
        csvpath = 'db/result/train.csv'
        impaths.extend(glob(r'faces/lfw_home/lfw_funneled_sample/*/*.*'))
        create_csv(root, impaths, labeller, csvpath)
        for label, impath in load_csv(csvpath=r'db/result/train.csv'):
            print(label, impath)
    """

    def __init__(self, root, impaths, labeller, csvpath):
        self.root = root
        self.impaths = impaths
        self.labeller = labeller
        self.csvpath = dirop(csvpath, remove=True)
        self.ppath = defaultdict(list)
        self.fpath = defaultdict(list)
        self.meta = OrderedDict()
        self.samplesperclass, self.nsamplesperclass = None, None

    def path2label(self, impaths=None):
        for impath in impaths or self.impaths:
            label = self.labeller(impath)
            if isfile(impath):
                path = self.ppath
            else:
                path = self.fpath
            impath = impath.replace(self.root, '')
            impath = impath.strip('\\')
            impath = impath.strip('/')
            path[label].append(impath)
        assert self.ppath, 'No image found, please check the path' % self.root
        try:
            self.ppath = OrderedDict(sorted(self.ppath.items(), key=lambda (k, v): float(k)))
        except:
            self.ppath = OrderedDict(sorted(self.ppath.items(), key=lambda (k, v): k))
        try:
            self.fpath = OrderedDict(sorted(self.fpath.items(), key=lambda (k, v): float(k)))
        except:
            self.fpath = OrderedDict(sorted(self.fpath.items(), key=lambda (k, v): k))

    def build_meta(self):
        self.meta['root'] = self.root
        self.meta['nmissingpaths'] = len(self.fpath.keys())
        self.meta["nclass"] = len(self.ppath.keys())
        self.meta['nimgs'] = len(np.hstack(self.ppath.values()))
        self.samplesperclass, self.nsamplesperclass = self.sort_keys()

    def sort_keys(self):
        # cryptic coding for fun :P
        sizedict = OrderedDict()
        samplesize = defaultdict(list)
        for k, v in self.ppath.items():
            samplesize[len(v)].append(k)

        def x((k, v)):
            sizedict[k] = len(v)
            return k

        samplesperclass = OrderedDict(sorted(samplesize.items(), key=x))
        return samplesperclass, sizedict

    def write(self):
        pd.DataFrame(self.meta.values(), index=self.meta.keys()).T.to_csv(self.csvpath, index_label='meta', mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        pd.DataFrame(self.nsamplesperclass.values(), index=self.nsamplesperclass.keys()).to_csv(self.csvpath, index_label='nsamplesperclass', mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        pd.DataFrame(self.samplesperclass.values(), index=self.samplesperclass.keys()).to_csv(self.csvpath, index_label='samplesperclass', mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        pd.DataFrame().to_csv(self.csvpath, mode=b'a')
        self.build_csv_data().to_csv(self.csvpath, mode=b'a')

    def build_csv_data(self):
        path = pd.DataFrame(self.ppath.values())
        path['labels'] = self.ppath.keys()
        path.insert(0, 'status', 'pass')
        path.set_index('labels', inplace=True)
        if self.fpath:
            misspath = pd.DataFrame(self.fpath.values())
            misspath['labels'] = self.fpath.keys()
            misspath.insert(0, 'status', 'miss')
            path = pd.concat([misspath, path], axis=0)
        return path


class LoadCsv:
    def path2label(self, csvpath):
        imeta, idata = None, None
        with open(csvpath, b'r') as book: lines = book.read().split('\n')
        for index, line in enumerate(lines):
            if imeta is None and 'meta,root,nmissingpaths,nclass,nimgs' in line:
                imeta = index
            if idata is None and line.startswith('labels,status'):
                idata = index
                break
        assert imeta is not None, print(r'failed to decode meta data')
        assert idata is not None, print(r'failed to decode data')
        meta = pd.read_csv(csvpath, dtype='O', skiprows=range(imeta + 2, len(lines)))
        df = pd.read_csv(csvpath, dtype='O', index_col='labels', skiprows=idata)
        df = df[df.status != 'skip']
        df = df.drop('status', 1)
        respath = []
        for label, impaths in df.iterrows():
            for impath in impaths:
                if impath is not np.nan:
                    impath = join(meta.root[0], impath)
                    if isfile(impath):
                        respath.append((label, impath))
                    else:
                        print("file_missing: %s" % impath)
        return respath


def create_csv(root, impaths, labeller, csvpath):
    '''
    :param root: root folder path
    :param impaths: list of paths of the images
    :param labeller: function to image path to label name
    :param csvpath: destination csv path
    :return:
    '''
    csv = CreateCsv(root, impaths, labeller, csvpath)
    csv.path2label()
    csv.build_meta()
    csv.write()
    print('csv_path: %s' % csv.csvpath)


def load_csv(csvpath):
    '''
    this will load the images form the csvpath
    :return:
    '''
    csv = LoadCsv()
    return csv.path2label(csvpath)

if __name__ == '__main__':
    def labeller(impath):
        val = basename(dirname(impath))
        return val


    root = r'faces/lfw_home/lfw_funneled_sample'
    impaths = []
    csvpath = 'db/result/train.csv'
    impaths.extend(glob(r'faces/lfw_home/lfw_funneled_sample/*/*.*'))
    create_csv(root, impaths, labeller, csvpath)
    for label, impath in load_csv(csvpath=r'db/result/train.csv'):
        print(label, impath)