import copy
import numpy as np
import xml.etree.ElementTree as ET
import os

import chainer
from chainer.datasets import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import VOCBboxDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD512
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

#from my_vott_voc_dataset import MyVoTTVOCDataset
#from my_bbox_label_name import voc_labels

import cv2
cv2.setNumThreads(0)

voc_labels=('win', 'win_blank', 'block_small', 'block_large')

class MyVoTTVOCDataset(VOCBboxDataset):

    def _get_annotations(self, i):
        id_ = self.ids[i]

        # Pascal VOC形式のアノテーションデータは，XML形式で配布されています
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))

        # XMLを読み込んで，bboxの座標・大きさ，bboxごとのクラスラベルなどの
        # 情報を取り出し，リストに追加していきます
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')

            # bboxの座標値が0-originになるように1を引いています
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(voc_labels.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # オリジナルのPascal VOCには，difficultという
        # 属性が画像ごとに真偽値で与えられていますが，今回は用いません
        # （今回のデータセットでは全画像がdifficult = 0に設定されているため）
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool)
        return bbox, label, difficult

class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def forward(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report({'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss}, self)

        return loss


class Transform(object):

    def __init__(self, coder, size, mean):
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        img, bbox, label = in_data

        img = random_distort(img)

        if np.random.randint(2):
            img, param = transforms.random_expand(img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        img, param = random_crop_with_bbox_constraints(img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(bbox, y_slice=param['y_slice'], x_slice=param['x_slice'], allow_outside_center=False, return_param=True)
        label = label[param['index']]

        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        img, params = transforms.random_flip(img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(bbox, (self.size, self.size), x_flip=params['x_flip'])

        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label


def main():

    # cuDNNのautotuneを有効にする
    #chainer.cuda.set_max_workspace_size(512 * 1024 * 1024)
    #chainer.config.autotune = True

    gpu_id = 0
    batchsize = 6
    out_num = 'results'
    log_interval = 1, 'epoch'
    epoch_max = 500
    initial_lr = 0.0001
    lr_decay_rate = 0.1
    lr_decay_timing = [200, 300, 400]

    # モデルの設定
    model = SSD512(n_fg_class=len(voc_labels), pretrained_model='imagenet')
    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)

    # GPUの設定
    chainer.cuda.get_device_from_id(gpu_id).use()
    model.to_gpu()

    # データセットの設定
    train_dataset = MyVoTTVOCDataset('C:\Python_Programs\chainer_practice\Telescope_image', 'train')
    valid_dataset = MyVoTTVOCDataset('C:\Python_Programs\chainer_practice\Telescope_image', 'val')

    # データ拡張
    transformed_train_dataset = TransformDataset(train_dataset, Transform(model.coder, model.insize, model.mean))

    # イテレーターの設定
    train_iter = chainer.iterators.MultiprocessIterator(transformed_train_dataset, batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid_dataset, batchsize, repeat=False, shuffle=False)

    # オプティマイザーの設定
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    # アップデーターの設定
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # トレーナーの設定
    trainer = training.Trainer(updater, (epoch_max, 'epoch'), out_num)
    trainer.extend(extensions.ExponentialShift('lr', lr_decay_rate, init=initial_lr), trigger=triggers.ManualScheduleTrigger(lr_decay_timing, 'epoch'))
    trainer.extend(DetectionVOCEvaluator(valid_iter, model, use_07_metric=False, label_names=voc_labels), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'lr', 'main/loss', 'main/loss/loc', 'main/loss/conf', 'validation/main/map', 'elapsed_time']), trigger=log_interval)

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'main/loss/loc', 'main/loss/conf'],
                'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['validation/main/map'],
                'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=(10, 'epoch'))

    # 途中で止めた学習を再開する場合は、trainerにスナップショットをロードして再開する
    # serializers.load_npz('results/snapshot_epoch_100.npz', trainer)

    # 学習実行
    trainer.run()

    # 学習データの保存
    model.to_cpu()
    serializers.save_npz('my_ssd_model.npz', model)

if __name__ == '__main__':
    main()