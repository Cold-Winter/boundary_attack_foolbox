from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras.datasets import mnist
import pickle
import time
import datetime
import os
from PIL import Image
import json
import sys


import tensorflow as tf

from model import Model


import robustml


def orthogonal_perturbation(delta, prev_sample, target_sample):
	prev_sample = prev_sample.reshape(32, 32, 3)
	# Generate perturbation
	perturb = np.random.randn(32, 32,3)
	perturbdiff= get_inf_diff(perturb, np.zeros_like(perturb))
	perturb = perturb/perturbdiff


# 	print(get_inf_diff(perturb, np.zeros_like(perturb)))

	perturb *= delta * np.max(get_inf_diff(target_sample, prev_sample))
	# Project perturbation onto sphere around target
	diff = (target_sample - prev_sample).astype(np.float32)
	diff /= get_inf_diff(target_sample, prev_sample)

# 	diff = diff.reshape(3, 32, 32)
# 	perturb = perturb.reshape(3, 32, 32)

	diff = np.transpose(diff, (2, 0, 1))
	perturb = np.transpose(perturb, (2, 0, 1))

	for i, channel in enumerate(diff):
		perturb[i] -= np.dot(perturb[i], channel) * channel
	# Check overflow and underflow
# 	mean = [103.939, 116.779, 123.68]
	mean = [0.0, 0.0, 0.0]
	perturb = np.transpose(perturb, (1, 2, 0))
# 	perturb = perturb.reshape(32, 32, 3)
	overflow = (prev_sample + perturb) - np.concatenate((np.ones((32, 32, 1)) * (255. - mean[0]), np.ones((32, 32, 1)) * (255. - mean[1]), np.ones((32, 32, 1)) * (255. - mean[2])), axis=2)
	overflow = overflow.reshape(32, 32, 3)
	perturb -= overflow * (overflow > 0)
	underflow = np.concatenate((np.ones((32, 32, 1)) * (0. - mean[0]), np.ones((32, 32, 1)) * (0. - mean[1]), np.ones((32, 32, 1)) * (0. - mean[2])), axis=2) - (prev_sample + perturb)
	underflow = underflow.reshape(32, 32, 3)
	perturb += underflow * (underflow > 0)
	return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
	perturb = (target_sample - prev_sample).astype(np.float32)
	perturb /= get_inf_diff(target_sample, prev_sample)
	perturb *= epsilon
	return perturb

def get_converted_prediction(sample, classifier):
	sample = sample.reshape(32, 32, 3)
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] += mean[0]
	sample[..., 1] += mean[1]
	sample[..., 2] += mean[2]
	sample = sample[..., ::-1].astype(np.uint8)
	sample = sample.astype(np.float32).reshape(1, 32, 32, 3)
	sample = sample[..., ::-1]
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] -= mean[0]
	sample[..., 1] -= mean[1]
	sample[..., 2] -= mean[2]
	label = decode_predictions(classifier.predict(sample), top=1)[0][0][1]
	return label

def draw(sample, classifier, folder):
	label = get_converted_prediction(np.copy(sample), classifier)
	sample = sample.reshape(32, 32, 3)
	# Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] += mean[0]
	sample[..., 1] += mean[1]
	sample[..., 2] += mean[2]
	sample = sample[..., ::-1].astype(np.uint8)
	# Convert array to image and save
	sample = Image.fromarray(sample)
	id_no = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
	# Save with predicted label for image (may not be adversarial due to uint8 conversion)
	sample.save(os.path.join("images", folder, "{}_{}.png".format(id_no, label)))

def preprocess(sample_path):
	img = image.load_img(sample_path, target_size=(32, 32))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

# def get_diff(sample_1, sample_2):
# 	sample_1 = sample_1/255
# 	sample_2 = sample_2/255
# 	sample_1 = sample_1.reshape(3, 32, 32)
# 	sample_2 = sample_2.reshape(3, 32, 32)
# 	diff = []
# 	for i, channel in enumerate(sample_1):
# 		diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
# 	return np.array(diff)
def get_inf_diff(sample_1, sample_2):
	sample_1 = sample_1/255
	sample_2 = sample_2/255
	sample_1 = np.transpose(sample_1, (2, 0, 1))
	sample_2 = np.transpose(sample_2, (2, 0, 1))
# 	sample_1 = sample_1.reshape(3, 32, 32)
# 	sample_2 = sample_2.reshape(3, 32, 32)
	diff = []
	for i, channel in enumerate(sample_1):
		diff.append(np.max(np.abs((channel - sample_2[i]).astype(np.float32))))
	return np.array(diff)

def boundary_attack():




	l2thresh = 0.05 * np.sqrt(32*32)


	with open('config.json') as config_file:
		config = json.load(config_file)

	model_file = tf.train.latest_checkpoint(config['model_dir'])
	if model_file is None:
		print('No model found')
		sys.exit()

	model = Model(mode='eval')

	saver = tf.train.Saver()

	linfty = 0.031

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	saver.restore(sess, model_file)


	cifar_path = './cifar10_data/test_batch'

	provider = robustml.provider.CIFAR10(cifar_path)
	real_logits = tf.nn.softmax(model.pre_softmax)

	start = 50
	end = 100
	wrongexample = 0
	totalImages = 0
	succImages = 0
	for i in range(start,end):
		inputs, targets = provider[i]
		logits = sess.run(real_logits, feed_dict={model.x_input: [inputs],model.y_input: targets})
# 		logits = model.outlogits(inputs.reshape(1,32,32,3))
		print('evaluating %d of [%d, %d]' % (i, start, end))
		sys.stdout.flush()

		if np.argmax(logits) != targets:
			wrongexample += 1
			print('skip the wrong example ', i)
			sys.stdout.flush()
			continue

		totalImages += 1
		target_tem = i+1
		while True:
			target_x, target_y = provider[target_tem]
			if target_y != targets:
				break
			target_tem += 1


		initial_sample = inputs * 255
		target_sample = target_x * 255

		#folder = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
		#os.mkdir(os.path.join("images", folder))
# 		draw(np.copy(initial_sample), classifier, folder)
# 		attack_class = np.argmax(model.outlogits(initial_sample.reshape(1,32,32,3)/255))
# 		target_class = np.argmax(model.outlogits(target_sample.reshape(1,32,32,3)/255))
		attack_class = np.argmax(sess.run(real_logits, feed_dict={model.x_input: initial_sample.reshape(1,32,32,3),model.y_input: targets}))
		target_class = np.argmax(sess.run(real_logits, feed_dict={model.x_input: target_sample.reshape(1,32,32,3),model.y_input: targets}))


		adversarial_sample = initial_sample
		n_steps = 0
		n_calls = 0
		epsilon = 1.
		delta = 0.1

		# Move first step to the boundary
		while True:
			trial_sample = adversarial_sample + forward_perturbation(epsilon * get_inf_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
# 			prediction =  model.outlogits(trial_sample.reshape(1, 32, 32, 3)/255)
			prediction = sess.run(real_logits, feed_dict={model.x_input: trial_sample.reshape(1,32,32,3),model.y_input: targets})
			n_calls += 1
			if np.argmax(prediction) == attack_class:
				adversarial_sample = trial_sample
				break
			else:
				epsilon *= 0.9

# 		while True:
# 			print("Step #{}...".format(n_steps))
# 			print("\tDelta step...")
		successflag = False
		for attack_step in range(600):
			d_step = 0
			while True:
				d_step += 1
# 				print("\t#{}".format(d_step))
				trial_samples = []
				for i in np.arange(300):
					trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
					trial_samples.append(trial_sample)
# 				predictions = model.outlogits(np.array(trial_samples).reshape(-1, 32, 32, 3)/255)
				predictions = sess.run(real_logits, feed_dict={model.x_input: np.array(trial_samples).reshape(-1,32,32,3),model.y_input: targets})
				n_calls += 10
				predictions = np.argmax(predictions, axis=1)
				d_score = np.mean(predictions == attack_class)
				if d_score > 0.0:
					if d_score < 0.3:
						delta *= 0.9
					elif d_score > 0.7:
						delta /= 0.9
					adversarial_sample = np.array(trial_samples)[np.where(predictions == attack_class)[0][0]]
					break
				else:
					delta *= 0.9
# 			print("\tEpsilon step...")
			e_step = 0
			while True:
				e_step += 1
# 				print("\t#{}".format(e_step))
				trial_sample = adversarial_sample + forward_perturbation(epsilon * get_inf_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)

# 				prediction =  model.outlogits(trial_sample.reshape(1, 32, 32, 3)/255)
				prediction = sess.run(real_logits, feed_dict={model.x_input: trial_sample.reshape(1,32,32,3),model.y_input: targets})

				n_calls += 1
				if np.argmax(prediction) == attack_class:
					adversarial_sample = trial_sample
					epsilon /= 0.5
					break
				elif e_step > 500:
						break
				else:
					epsilon *= 0.5
			n_steps += 1
# 			chkpts = [1, 5, 10, 50, 100, 500, 1000]
# 			if (n_steps in chkpts) or (n_steps % 500 == 0):
# 				print("{} steps".format(n_steps))
# 				draw(np.copy(adversarial_sample), classifier, folder)
			diff = np.max(get_inf_diff(adversarial_sample, target_sample))
			realdiff = np.sum((adversarial_sample/255 - target_sample/255 )**2)**0.5
			realinfdiff = diff
			if e_step > 500:
				print("{} steps".format(n_steps))
				print("Mean Squared Error: {}".format(diff))
				sys.stdout.flush()
# 				draw(np.copy(adversarial_sample), classifier, folder)
				break

			if realinfdiff <= linfty:
				successflag = True
				succImages += 1
				print('clipimage succImages: '+str(succImages)+'  totalImages: '+str(totalImages))
				sys.stdout.flush()
				break
			if attack_step % 50 == 0:
				print("Mean Squared Error: {}".format(diff))
				print("{} steps".format(n_steps))
				print("Real Mean Squared Error: {}".format(realdiff))
				print("Real INf dis: {}".format(realinfdiff))
# 				print("Calls: {}".format(n_calls))
				print("Attack Class: {}".format(attack_class))
				print("Target Class: {}".format(target_class))
				print("Adversarial Class: {}".format(np.argmax(prediction)))
				sys.stdout.flush()
	print(wrongexample)





if __name__ == "__main__":
	boundary_attack()





