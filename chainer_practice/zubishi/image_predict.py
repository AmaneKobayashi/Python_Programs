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
import sys
sys.path.append('C:\Python_Programs\my_module')
import convert_RGB
from fortran_RGB import sub_convert_RGB

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
image_files = list(chain.from_iterable(image_files))

model = L.Classifier(MyChain()).to_gpu()
serializers.load_npz("mymodel.npz", model)

imageData=[]
for i in range(len(image_files)):
    np_img=Image.open(image_files[i])

    fortran_flag=1
    if(fortran_flag == 1):
        img_rgb=sub_convert_RGB.convert_rgb_chainer(np_img,2,5)
        img_rgb=numpy.rot90(img_rgb)
        img_rgb=numpy.flipud(img_rgb)
    else:
        img_rgb=convert_RGB.convert_RGB_chainer(np_img,2,5)
    imageData.append(img_rgb)
    print(i,image_files[i])

imageData=cuda.to_gpu(imageData, device=0)
test = imageData

folder_names = ['predicted_image/other', 'predicted_image/zubishi',
                'predicted_image/zubishi_mix']

print("len(test):" + str(len(test)))
for i in range(len(test)):
    x = test[i]
    y = F.softmax(model.predictor(x[None, ...]))
    shutil.copy(image_files[i], folder_names[int(y.data.argmax())])
    print("{0}: {1}: {2}".format(image_files[i],
                            folder_names[int(y.data.argmax())], y.data.max()))
