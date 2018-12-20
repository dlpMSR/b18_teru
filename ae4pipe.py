import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import os.path

import time
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
hidden = 2048
epoch_num = 100
batchsize = 64
gpu_id = 0


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
    NAME_OUTPUTDIRECTORY = 'exp' + datetime.now().strftime("%Y%m%d%H%M")
    FILENAME_MODEL = 'teru_Autoencoder.model'
    FILENAME_RESULT = 'result.txt'
    output_path = os.path.join('./result', NAME_OUTPUTDIRECTORY)
    os.mkdir(output_path)

    train = setup_images_dataset('./train_data')
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
        print('epoch'+str(int(epoch+1))+':'+str(loss.data))
        # ロス値のグラフを出力
        plt.grid()
        plt.tight_layout()
        plt.plot([i for i, x in enumerate(loss_list, 1)], loss_list)
        plt.savefig(os.path.join(output_path, 'loss'))
    # 入力・出力画像のサンプルを保存
    y = model.fwd(x)
    save_image(x, 'input', output_path, gpu_id)
    save_image(y.array, 'reconst', output_path, gpu_id)
    # モデルを保存
    model.to_cpu()
    serializers.save_npz(os.path.join(output_path, FILENAME_MODEL), model)
    # 条件をテキストファイルに出力
    with open(os.path.join(output_path, FILENAME_RESULT), mode='w') as f:
        f.write('width:'+str(width)+'\n')
        f.write('height:'+str(height)+'\n')
        f.write('hidden:'+str(hidden)+'\n')
        f.write('compression-rate:' +
                str(round((hidden/(width*height))*100, 2))+'%\n')


def test_autoencoder():
    model = Autoencoder(width*height, hidden)
    serializers.load_npz('./test/teru_Autoencoder.model', model)

    def load_image(image_path):
        img = Image.open(image_path)
        img = img.resize((width, height), Image.BICUBIC)
        img = np.asarray(img).transpose(2, 0, 1)
        img = img.astype(np.float32)
        img = img[0, :, :]
        img = img / 255
        img = img.reshape(-1)
        return img

    img = load_image('./test/test.png')
    x = np.array([img])
    y = model.fwd(x)

    im = y.array
    im = im.reshape(height, width)
    im = im * 255
    im = im.astype(np.uint8)
    pil_img = Image.fromarray(im)
    pil_img.show()
    pil_img.save('./test/output.png')


def setup_images_dataset(IMG_DIR):
    image_files = glob.glob('{}/*.jpg'.format(IMG_DIR))
    dataset = chainer.datasets.ImageDataset(image_files)

    def resize(img):
        img = Image.fromarray(img.transpose(1, 2, 0))
        img = img.resize((width, height), Image.BICUBIC)
        return np.asarray(img).transpose(2, 0, 1)

    def transform(img):
        img = resize(img.astype(np.uint8))
        img = img.astype(np.float32)
        img = img[0, :, :]
        img = img / 255
        img = img.reshape(-1)
        return img

    transformed_d = chainer.datasets.TransformDataset(dataset, transform)
    return transformed_d


def save_image(data, savename, output_path, device=-1):
    destination = os.path.join(output_path, savename)
    os.mkdir(destination)
    if device >= 0:
        data = cuda.cupy.asnumpy(data)
    for i in range(10):
        im = data[i].reshape(height, width)
        im = im * 255
        pil_img = Image.fromarray(np.uint8(im)).convert('RGB')
        pil_img.save(os.path.join(destination, str(int(i+1))+'.png'))


def main():
    test_autoencoder()
    #train_autoencoder()


if __name__ == '__main__':
    main()
