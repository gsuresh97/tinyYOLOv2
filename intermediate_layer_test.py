import tensorflow as tf
import os
import numpy as np
# import net
import weights_loader
import cv2
import warnings
warnings.filterwarnings('ignore')


def sigmoid(x):
  return 1. / (1. + np.exp(-x))



def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out



def iou(boxA,boxB):
  # boxA = boxB = [x1,y1,x2,y2]

  # Determine the coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  # Compute the area of intersection
  intersection_area = (xB - xA + 1) * (yB - yA + 1)
 
  # Compute the area of both rectangles
  boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
  # Compute the IOU
  iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

  return iou



def non_maximal_suppression(thresholded_predictions,iou_threshold):

  nms_predictions = []

  # Add the best B-Box because it will never be deleted
  nms_predictions.append(thresholded_predictions[0])

  # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
  # thresholded_predictions[i][0] = [x1,y1,x2,y2]
  i = 1
  while i < len(thresholded_predictions):
    n_boxes_to_check = len(nms_predictions)
    #print('N boxes to check = {}'.format(n_boxes_to_check))
    to_delete = False

    j = 0
    while j < n_boxes_to_check:
        curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
        if(curr_iou > iou_threshold ):
            to_delete = True
        #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
        j = j+1

    if to_delete == False:
        nms_predictions.append(thresholded_predictions[i])
    i = i+1

  return nms_predictions



def preprocessing(input_img_path,input_height,input_width):

  input_image = cv2.imread(input_img_path)

  # Resize the image and convert to array of float32
  resized_image = cv2.resize(input_image,(input_height, input_width), interpolation = cv2.INTER_CUBIC)
  image_data = np.array(resized_image, dtype='f')

  # Normalization [0,255] -> [0,1]
  image_data /= 255.

  # BGR -> RGB? The results do not change much
  # copied_image = image_data
  #image_data[:,:,2] = copied_image[:,:,0]
  #image_data[:,:,0] = copied_image[:,:,2]

  # Add the dimension relative to the batch size needed for the input placeholder "x"
  image_array = np.expand_dims(image_data, 0)  # Add batch dimension

  return image_array

def intermediate_inference(sess,preprocessed_image, model='voc'):
  if model is not 'voc' and model is not 'coco':
    print("Model {} does not exist.".format(model))
    return
  
  if model is 'voc':
    import voc_net as net
  else:
    import coco_net as net
    init_g = tf.global_variables_initializer()
    sess.run(init_g)

  # Forward pass of the preprocessed image into the network defined in the net.py file
  predictions = sess.run(net.o1,feed_dict={net.x:preprocessed_image})

  return predictions


### MAIN ##############################################################################################################

def main(_):

	# Definition of the paths
  weights = 'coco'
  if weights is 'voc':
    import voc_net as net
  else:
    import coco_net as net
  voc_paths = ['./yolov2-tiny-voc.weights']
  coco_paths = ['./yolov2-tiny.weights']
  input_img_path = './dog.jpg'

  # If you do not have the checkpoint yet keep it like this! When you will run test.py for the first time it will be created automatically
  ckpt_folder_path = './ckpt/'

  # Definition of the parameters
  input_height = 416
  input_width = 416
  
  # Definition of the session
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Check for an existing checkpoint and load the weights (if it exists) or do it from binary file
  # print('Looking for a checkpoint...')
  saver = tf.train.Saver()
  if weights is 'coco':
    _ = weights_loader.load(sess,coco_paths[0],ckpt_folder_path,saver,'coco')
  else:
    _ = weights_loader.load(sess,voc_paths[0],ckpt_folder_path,saver,'voc')

  # Preprocess the input image
  print('Preprocessing...')
  preprocessed_image = preprocessing(input_img_path,input_height,input_width)

  # Compute the predictions on the input image
  print('Computing predictions...')
  predictions = intermediate_inference(sess,preprocessed_image, weights)
  for i in range(16):
    inter_predictions = predictions[0, :, :, i]
    inter_predictions = inter_predictions + abs(np.amin(inter_predictions))
    inter_predictions = inter_predictions / np.amax(inter_predictions)
    inter_predictions *= 255
    if weights is 'coco':
      cv2.imwrite('cocos/img'+str(i)+".jpg",inter_predictions)
    else:
      cv2.imwrite('vocs/img'+str(i)+".jpg",inter_predictions)


if __name__ == '__main__':
     tf.app.run(main=main) 

#######################################################################################################################
