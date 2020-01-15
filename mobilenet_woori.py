import cv2
import numpy as np
import os, sys
from time import strftime, localtime
import random
import time
from openvino.inference_engine import IENetwork, IEPlugin

plugin = IEPlugin("CPU", "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64")
plugin.add_cpu_extension("/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so")

model_xml = "/home/intel/Desktop/sample/FP32/mobilenet-ssd.xml"
model_bin = "/home/intel/Desktop/sample/FP32/mobilenet-ssd.bin"

print('Loading network files:\n\t{}\n\t{}'.format(model_xml, model_bin))

net = IENetwork(model=model_xml, weights=model_bin)
print("net inputs: {}, outputs: {}".format(net.inputs['data'].shape, net.outputs['detection_out'].shape))


net.batch_size = 1

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

exec_net = plugin.load(network=net)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit('camera error')

labels = ["plane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog", "horse", "motorcycle", "person", "plant", "sheep", "sofa", "train", "monitor"]


captured = False
detect_count = 0
while True:
	ret, frame = cap.read()
	if not ret: continue

	ch = cv2.waitKey(1) & 0xFF
	if ch == 27: break

	
	n, c, h, w = net.inputs[input_blob].shape

	images = np.ndarray(shape=(n, c, h, w))
	images[0] = cv2.resize(frame, (300,  300)).transpose((2,0,1))

	res = exec_net.infer(inputs={input_blob: images})
	detections = res[out_blob][0][0]

	for i, detect in enumerate(detections):

		image_id = float(detect[0])
		label_index = int(detect[1])
		confidence = float(detect[2])

		if image_id < 0 or confidence == 0.:
			continue 


		if confidence > 0.7:
			detected = labels[label_index-1]
			print(detected, detect_count)
			if detected == "cat" or detected == "dog" or detected == "plane" or detected == "horse" or detected == "car":
				detect_count += 1
				if detect_count > 15:
					font = cv2.FONT_HERSHEY_SIMPLEX
					msg = "{} image saved".format(detected)
					cv2.putText(frame, msg, (30, 30), font, 1.5, (255,255,255), 2, cv2.LINE_AA ) 
					if not captured:
						cv2.imwrite("{}{}.jpeg".format(detected, time.time()), frame)
					
						captured = True

			else:
				detect_count = 0
				captured = False
	

	cv2.imshow('view', frame)

cv2.destroyAllWindows()
cap.release()













