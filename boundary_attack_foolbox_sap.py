import foolbox
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50

from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
import robustml
import sys
from defense import *
from inceptionv3 import model as inceptionv3_model

# from robustml_model import InputTransformations
from defense import *

from robustml_model_sap import SAP
# print(foolbox.__file__)

# instantiate model

sess = tf.InteractiveSession()

# defense = 'jpeg'  # 'bitdepth | jpeg | crop | quilt | tv' ############# change ##############################
# inputmodel = InputTransformations(sess,defense)
sapmodel = SAP(sess)

x = tf.placeholder(tf.float32, (32,32,3))
x_exp = tf.expand_dims(x, axis=0)
eval_logits = sapmodel.model(x_exp)

model = foolbox.models.TensorFlowModel(x_exp, eval_logits, (0, 1))
attack = foolbox.attacks.BoundaryAttack(model)
# imagenet_path = '../imagenetval'

# provider = robustml.provider.ImageNet(imagenet_path, (299,299,3))
cifar_path = './cifar10_data/test_batch'
provider =  robustml.provider.CIFAR10(cifar_path)

start = 40 
end = 60
wrongexample = 0
totalImages = 0
succImages = 0
faillist = []
# l2thresh = 0.05 * 299
lithresh = 0.031
for i in range(start,end):
    inputs, targets = provider[i]
    # inputs = defend_jpeg(inputs.reshape(299,299,3))
    # print(np.max(inputs))
    # print(np.min(inputs))
#     image, label = defend_jpeg(foolbox.utils.imagenet_example()/255.0)

    print('evaluating %d of [%d, %d]' % (i, start, end))
    logits = model.predictions(inputs)
#     logits = kmodel.predict(preprocess_input(inputs* 255.0))
    if np.argmax(logits) != targets:
        wrongexample += 1
        print('skip the wrong example ', i)
        sys.stdout.flush()
        continue
    successflag = False
    totalImages += 1
    adversarial = attack(inputs, targets, iterations=5000)
    # l2norm = np.linalg.norm(( adversarial - inputs).astype(np.float32))**2/(299*299*3)
    l2real = np.linalg.norm(( adversarial - inputs).astype(np.float32))
    lireal = np.max(adversarial)
    # print("l2normsqure Error: {}".format(l2norm))
    print("l2real Error: {}".format(l2real))
    print("lireal Error: {}".format(lireal))

    logitsadv = model.predictions(adversarial)
    if lireal <= lithresh and np.argmax(logitsadv) != targets:
        successflag = True
        succImages += 1
        print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
        sys.stdout.flush()
    else:
        faillist.append(i)
        print('faillist: ',faillist)



print(wrongexample)
print(faillist)

sess.close()
























