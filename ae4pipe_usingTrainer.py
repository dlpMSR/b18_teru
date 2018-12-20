import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import sys, os.path

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

import cupy

#n_in, n_units
class Autoencoder(chainer.Chain):
    def __init__(self, n_in, n_units):
        super(Autoencoder, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_in)
        
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return h2
    

def train():
    epoch_num = 100
    batchsize = 64
    gpu_id = -1

    train = load_images('./train_data')
    train = datasets.TupleDataset(train, train)
    train_iter = chainer.iterators.SerialIterator(train, batchsize)

    model = L.Classifier(Autoencoder(256*256, 128), lossfun = F.mean_squared_error)
    model.compute_accuracy = False
    
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (epoch_num, 'epoch'), out='result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    model.to_cpu()
    serializers.save_npz('sushi_hotdog.model', model)

'''
    train_num = len(train)
    for epoch in range(epoch_num):
        for i in range(0, train_num, batchsize):
            batch = train_iter.next()
            x = np.asarray([s for s in batch])
            y = model(x)
            loss = F.mean_squared_error(x, y)
            model.cleargrads()
            loss.backward()
            opt.update()
        print('epoch' + str(int(epoch+1)) + ':' + str(loss.data))
        plot(x, y, './pict/teru_')

    model.to_cpu()
    serializers.save_npz('teru_Autoencoder.model', model)
'''
   
def load_images(IMG_DIR):
    image_files = glob.glob('{}/*.jpg'.format(IMG_DIR))
    dataset = chainer.datasets.ImageDataset(image_files)

    def resize(img):
        width, height = 256, 256
        img = Image.fromarray(img.transpose(1, 2, 0))
        img = img.resize((width, height), Image.BICUBIC)
        return np.asarray(img).transpose(2, 0, 1)

    def transform(img):
        img = img[:3, ...]
        img = resize(img.astype(np.uint8))
        img = img.astype(np.float32)
        img = img[0,:,:]
        img = img / 255
        img = img.reshape(-1)
        return img

    transformed_d = chainer.datasets.TransformDataset(dataset, transform)
    return transformed_d


def plot(testData, testReconst, saveName):
    for i in range(10):
        data    = testData[i]
        reconst = testReconst[i].array
        
        plt.axis('off')
        plt.imshow(data.reshape(128, 128), cmap = cm.gray, interpolation = 'nearest')
        plt.savefig(saveName + 'data' + str(int(i)) + '.png')
        
        plt.axis('off')
        plt.imshow(reconst.reshape(128, 128), cmap = cm.gray, interpolation = 'nearest')
        plt.savefig(saveName + 'reconst' + str(int(i)) + '.png')


def main():
    train()


if __name__ == '__main__':
    main()
