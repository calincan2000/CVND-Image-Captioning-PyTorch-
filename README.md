# CVND---Image-Captioning-Project


![Image Captioning Model](https://github.com/calincan2000/CVND-Image-Captioning-PyTorch-/blob/master/images/image-captioning.png)
Project Overview
In this project, you will create a neural network architecture to automatically generate captions from images.

After using the Microsoft Common Objects in COntext (MS COCO) dataset to train your network, you will test your network on novel images!

![Encoder-Decoder](https://github.com/calincan2000/CVND-Image-Captioning-PyTorch-/blob/master/images/encoder-decoder.png)

The encoder that we provide to you uses the pre-trained ResNet-50 architecture (with the final fully-connected layer removed) to extract features from a batch of pre-processed images. The output is then flattened to a vector, before being passed through a Linear layer to transform the feature vector to have the same size as the word embedding.


LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on!





# Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).
