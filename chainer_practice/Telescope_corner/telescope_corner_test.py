from chainercv.links import SSD300
from chainer import serializers
from chainercv.visualizations import vis_bbox
from chainercv import utils
from matplotlib import pyplot as plt
import sys
import time

#from my_bbox_label_name import voc_labels

imag=sys.argv[1]
print("imag = " + imag)

voc_labels=('corner')

def inference(image_filename):
    img = utils.read_image(image_filename, color=True)
    t2=time.time()
    bboxes, labels, scores = model.predict([img])
    t3=time.time()
    print("t3-t2 = "  + str(t3-t2))

    bbox, label, score = bboxes[0], labels[0], scores[0]

    #fig = plt.figure(figsize=(5.12,5.12), dpi=100)
    #ax = plt.subplot(1,1,1)

    ax=vis_bbox(img, bbox, label, label_names=voc_labels)
    ax.set_axis_off()
    ax.figure.tight_layout()    #vis_bbox(img, bbox, label, label_names=voc_labels)
    #print(voc_labels[label[0]])
    #print(score[0])
    #print(bbox[0,0])
    plt.show()


model = SSD300(n_fg_class=len(voc_labels))

serializers.load_npz('C:\Python_Programs\chainer_practice\Telescope_corner\my_ssd_model_SSD300.npz', model)

inference(imag)