import os
import shutil
import glob
import chainer
from itertools import chain
from chainer import iterators, training, optimizers, datasets, serializers
from chainer.dataset import concat_examples
import chainer.functions as F
import chainer.links as L
import numpy
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


chainer.config.train = False

# モデルの読み込み
model = L.Classifier(MyChain())
serializers.load_npz("mymodel.npz", model)

image_files = [glob.glob('predicted_image/in/*.tif')]
d = chainer.datasets.ImageDataset(list(chain.from_iterable(image_files)))

def transform(data):
    data = numpy.asarray(data,dtype="float32")
    data = cuda.to_gpu(data, device=0)
    return data

model = L.Classifier(MyChain()).to_gpu()
serializers.load_npz("mymodel.npz", model)

test = chainer.datasets.TransformDataset(d, transform)

folder_names = ['predicted_image/other', 'predicted_image/zubishi',
                'predicted_image/zubishi_mix']

print("len(test):" + str(len(test)))
for i in range(len(test)):
    x = test[i]
    y = F.softmax(model.predictor(x[None, ...]))
    #shutil.copy(image_files[0][i], folder_names[int(y.data.argmax())])
    print("{0}: {1}: {2}".format(image_files[0][i],
                            folder_names[int(y.data.argmax())], y.data.max()))
