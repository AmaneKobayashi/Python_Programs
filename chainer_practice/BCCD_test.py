from chainercv.links import SSD512
from chainer import serializers
from chainercv.visualizations import vis_bbox
from chainercv import utils
from matplotlib import pyplot as plt
import sys

#from my_bbox_label_name import voc_labels

imag=sys.argv[1]
print("imag = " + imag)

voc_labels=('rbc', 'wbc', 'platelets')

def inference(image_filename):
    img = utils.read_image(image_filename, color=True)

    bboxes, labels, scores = model.predict([img])

    bbox, label, score = bboxes[0], labels[0], scores[0]

    #fig = plt.figure(figsize=(5.12,5.12), dpi=100)
    #ax = plt.subplot(1,1,1)

    ax=vis_bbox(img, bbox, label, label_names=voc_labels)
    ax.set_axis_off()
    ax.figure.tight_layout()    #vis_bbox(img, bbox, label, label_names=voc_labels)
    plt.show()


model = SSD512(n_fg_class=len(voc_labels))

serializers.load_npz('C:\Python_Programs\chainer_practice\BCCD_Dataset\my_ssd_model.npz', model)

inference(imag)