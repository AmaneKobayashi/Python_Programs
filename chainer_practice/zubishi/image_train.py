import chainer
import chainer.functions as F
import chainer.links as L
import os
import glob
from itertools import chain
import chainer
from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset
from chainer.datasets import split_dataset_random
from chainer import iterators, training, optimizers, datasets, serializers
from chainer.training import extensions, triggers
from chainer.dataset import concat_examples
import chainer.functions as F
import chainer.links as L
import numpy
import sys
sys.path.append('C:\Python_Programs\my_module')
import convert_RGB
from fortran_RGB import sub_convert_RGB
from PIL import Image

from chainer.backends import cuda 
gpu_id=0
chainer.cuda.get_device_from_id(gpu_id).use()


class MyChain(chainer.Chain):

    def __init__(self):
        super(MyChain, self).__init__(
        conv1 = L.Convolution2D(None, 16, 5, pad=2).to_gpu(),
        conv2 = L.Convolution2D(None, 32, 5, pad=2).to_gpu(),
        l3 = L.Linear(None, 256).to_gpu(),
        l4 = L.Linear(None, 3).to_gpu(),
    )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=5, stride=2, pad=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=5, stride=2, pad=2)
        h = F.dropout(F.relu(self.l3(h)))
        y = self.l4(h)
        return y


chainer.config.train = True

# 画像フォルダのパス
IMG_DIR = 'classed_image'

# 各キャラクターごとのフォルダ
dnames = glob.glob('{}/*'.format(IMG_DIR))

# 画像ファイルパス一覧
fnames = [glob.glob('{}/*.tif'.format(d)) for d in dnames]
fnames = list(chain.from_iterable(fnames))

# それぞれにフォルダ名から一意なIDを付与
labels = [os.path.basename(os.path.dirname(fn)) for fn in fnames]
dnames = [os.path.basename(d) for d in dnames]
labels = [dnames.index(l) for l in labels]
#d = LabeledImageDataset(list(zip(fnames, labels)))

#print("labels :", labels)
#print("dnames :", dnames)
#print("fnames :", fnames)
#exit()

imageData=[]
for i in range(len(fnames)):
    np_img=Image.open(fnames[i])
    np_img = numpy.asarray(np_img,dtype="float32")
    
    fortran_flag=1
    if(fortran_flag == 1):
        img_rgb=sub_convert_RGB.convert_rgb_chainer(np_img,2,5)
        img_rgb=numpy.rot90(img_rgb)
        img_rgb=numpy.flipud(img_rgb)
    else:
        img_rgb=convert_RGB.convert_RGB_chainer(np_img,2,5)
    imageData.append(img_rgb)
    print(i,fnames[i])

imageData=cuda.to_gpu(imageData, device=0)
labels=numpy.asarray(labels,dtype="int")
labels=cuda.to_gpu(labels, device=0)
train=[(x,y) for x,y in zip(imageData,labels)]
train, valid = split_dataset_random(train, 300, seed=0)
#imagedata, label=zip(*train)
#print(label)
#print(len(label))
#print()
#imagedata, label=zip(*valid)
#print(label)
#print(len(label))
#exit()

epoch = 500
batch = 10

model = L.Classifier(MyChain()).to_gpu()
optimizer = optimizers.Adam()
optimizer.setup(model)

train_iter = iterators.SerialIterator(train, batch)
valid_iter = iterators.SerialIterator(valid, batch, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(valid_iter, model))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
trainer.extend(extensions.LogReport())
#trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.PlotReport(['main/loss'], 'epoch', file_name='main_loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy'], 'epoch', file_name='main_accuracy.png'))
trainer.extend(extensions.PlotReport(['validation/main/loss'], 'epoch', file_name='validation_loss.png'))
trainer.extend(extensions.PlotReport(['validation/main/accuracy'], 'epoch', file_name='validation_accuracy.png'))
trainer.run()

serializers.save_npz("mymodel.npz", model)