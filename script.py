import cv2
import numpy as np
from keras import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

inference_model = models.load_model("best_model.hdf5")

#wget ...
image_input = cv2.imread("test.jpg")
image_input = cv2.resize(image_input,dsize=(64,280))
image_input = plt.imshow(image_input)
plt.show()

image_input = np.expand_dims(np.asarray(image_input), axis=0)

preds = inference_model.predict(image_input)
# label_map = ["BCS","CHN","CLN","DGT","ODT","SKK","SON","TSN"]
i=0
data = ""
for pred in preds[0]:
  data +=("%s:%.4f\n"%(label_map[i],pred))
  i+=1

result = open('result.txt','w')
result.write(data)
result.close()