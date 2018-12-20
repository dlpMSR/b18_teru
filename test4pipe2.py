import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import os.path

from time import sleep
from datetime import datetime

import chainer
from chainer import cuda
from chainer import datasets
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import serializers
from chainer import training
from chainer.training import extensions

import glob
from itertools import chain

from PIL import Image
from PIL import ImageOps


width, height = 128, 74
hidden = 100
epoch_num = 3
batchsize = 10
gpu_id = -1


class Autoencoder(chainer.Chain):
    def __init__(self, n_in, n_units):
        super(Autoencoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_in)

    def __call__(self, x):
        y = self.fwd(x)
        loss = F.mean_squared_error(x, y)
        return loss

    def fwd(self, x):
        h = F.sigmoid(self.l1(x))
        y = F.sigmoid(self.l2(h))
        return y


def train_autoencoder():
    train = load_images('./train_data')
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    model = Autoencoder(width * height, hidden)

    if gpu_id >= 0:
        chainer.cuda.get_device(gpu_id).use()
        model.to_gpu()
        xp = cuda.cupy

    opt = chainer.optimizers.Adam()
    opt.setup(model)
    loss_list = []
    train_num = len(train)

    for epoch in range(epoch_num):
        for i in range(0, train_num, batchsize):
            batch = train_iter.next()
            if gpu_id >= 0:
                x = xp.asarray([s for s in batch])
            else:
                x = np.asarray([s for s in batch])
            loss = model(x)
            model.cleargrads()
            loss.backward()
            opt.update()
        loss_list.append(loss.array)
        print('epoch' + str(int(epoch+1)) + ':' + str(loss.data))

    NAME_OUTPUTDIRECTORY = 'exp' + datetime.now().strftime("%Y%m%d%H%M")
    FILENAME_MODEL = 'teru_Autoencoder.model'
    FILENAME_RESULT = 'result.txt'
    output_path = os.path.join('./result', NAME_OUTPUTDIRECTORY)
    os.mkdir(output_path)
    # モデルを保存
    model.to_cpu()
    serializers.save_npz(os.path.join(output_path, FILENAME_MODEL), model)
    # テキストファイルに出力
    with open(os.path.join(output_path, FILENAME_RESULT), mode='w') as f:
        f.write('width:'+str(width)+'\n')
        f.write('height:'+str(height)+'\n')
        f.write('hidden:'+str(hidden)+'\n')
        f.write('compression-rate:' +
                str(round((hidden/(width*height))*100, 2))+'%')
    # ロス値のグラフを出力
    plt.plot([i for i, x in enumerate(loss_list, 1)], loss_list)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'loss'))
    #plot(xp.asnumpy(x), a[0] + '/' + a[1] + 'data')
    #plot(xp.asnumpy(y.data), a[0] + '/' + a[1] + 'reconstdata')


def test_autoencoder(Model):
    a = setup_dir('./result/' + '{0:%Y%m%d}'.format(datetime.datetime.now()),
                  '{0:%Y%m%d%H}'.format(datetime.datetime.now()))
    gpu_id = 0
    model = Autoencoder(width*height, hidden)
    serializers.load_npz(Model, model)

    def resize(img):
        img = Image.fromarray(img.transpose(1, 2, 0))
        img = img.resize((width, height), Image.BICUBIC)
        return np.asarray(img).transpose(2, 0, 1)

    def transform(img):
        img = img[:3, ...]
        img = resize(img.astype(np.uint8))
        img = img.astype(np.float32)
        img = img[0, :, :]
        img = img / 255
        img = img.reshape(-1)
        return img

    test = './test.jpg'
    image = Image.open(test)
    # img.show()
    y = model.fwd(transform(image))


def load_images(IMG_DIR):
    image_files = glob.glob('{}/*.png'.format(IMG_DIR))
    dataset = chainer.datasets.ImageDataset(image_files)

    def resize(img):
        img = Image.fromarray(img.transpose(1, 2, 0))
        img = img.resize((width, height), Image.BICUBIC)
        return np.asarray(img).transpose(2, 0, 1)

    def transform(img):
        img = img[:3, ...]
        img = resize(img.astype(np.uint8))
        img = img.astype(np.float32)
        img = img[0, :, :]
        img = img / 255
        img = img.reshape(-1)
        return img

    transformed_d = chainer.datasets.TransformDataset(dataset, transform)
    return transformed_d


def generate_image(data, savename, device=-1):

    for i in range(16):

        plt.figure()
        plt.axis('off')
        plt.imshow(data[i].reshape((height, width)),
                   cmap=cm.gray, interpolation='nearest')
        plt.savefig(saveName + 'data' + str(int(i)) + '.png')


def main():
    train_autoencoder()


if __name__ == '__main__':
    main()
