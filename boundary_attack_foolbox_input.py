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

from robustml_model import InputTransformations
from defense import *
print(foolbox.__file__)

# instantiate model

sess = tf.InteractiveSession()

defense = 'jpeg'  # 'bitdepth | jpeg | crop | quilt | tv' ############# change ##############################
inputmodel = InputTransformations(sess,defense)

model = foolbox.models.TensorFlowModel(inputmodel._input, inputmodel._logits, (0, 1))
attack = foolbox.attacks.BoundaryAttack(model)
imagenet_path = '../imagenetval'

provider = robustml.provider.ImageNet(imagenet_path, (299,299,3))

start = 140
end = 160
wrongexample = 0
totalImages = 0
succImages = 0
faillist = []
l2thresh = 0.05 * 299
for i in range(start,end):
    inputs, targets = provider[i]
    inputs = defend_jpeg(inputs.reshape(299,299,3))
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
    l2norm = np.linalg.norm(( adversarial - inputs).astype(np.float32))**2/(299*299*3)
    l2real = np.linalg.norm(( adversarial - inputs).astype(np.float32))
    print("l2normsqure Error: {}".format(l2norm))
    print("l2real Error: {}".format(l2real))
    logitsadv = model.predictions(adversarial)
    if l2real <= l2thresh and np.argmax(logitsadv) != targets:
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
























