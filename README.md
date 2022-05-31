# PCB-object-detection-using-Tensorflow-object-detection-api

Implementation of Faster RCNN and RESNET using Tensorflow object detection api2 for PCB component detection like resistors, capacitors etc.
This repository also contains custom scripts for the following:
  1. Visualisation of the bounding boxes.
  2. Extracting bounding boxes and saving them folder named after the image and with folder for each individual images for further implementation of OCR on the components
  3. Optical Character Recognition using easyOCR

### Installation of the Object detection api

The complete installation and setup of the api can be found at "https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html"

### Database

FICS-PCB: A MULTI-MODAL IMAGE DATASET FOR AUTOMATED PRINTED CIRCUIT BOARD VISUAL INSPECTION
link: "https://www.trust-hub.org/#/data/fics-pcb"

### Data preparation

For people wanting to directly use the data in the required format of PASCAL VOC, the csv files from the above dataset has been changed to the PASCAL VOC format and can be found in "workspace\training_demo-fcis\images\all_images"

### Utils

The utils.py has custom scripts for extracting bounding boxes from the main image into idividual images and segregate them into folders acccording to their classes.
A Easy OCR script to run text detection these images, but I stopped working on OCR. Feel free to use it to make custom pipelines.

### Results

Results.csv file contains the results in COCO parameters with different augumentations and image preprocessing used.
Here is an example of the one of the better inference results.

![s29_front](https://user-images.githubusercontent.com/38318593/170691674-a9cc3e5a-c98c-44f0-9ea0-ff87403f4803.jpg) ![image](https://user-images.githubusercontent.com/38318593/170691356-51e6757c-ed93-4e3d-a4cc-32134e32efec.png) 

### Notes and suggestions

1. Ensure gpu installation for running smoothly and quickly: I have included my pc specs below for comparison
2. Things you should do to improve results, which I could not because of memory constraints:
    1. Increase batch size to improve batch normalisation
    2. Give model a higher resolution image froom the dataset as the components are very small 
    3. Play aound with the learning rate
3. Get more images. This dataset of 50 is very samll for any meaningfull success, if you get more images, highly appreciate if you do share it with me at "kashyapmahesh97@gmail.com"

### PC specs
1. CPU: Intel(R) Xeon(R) W-3223 CPU
2. Ram: 64 gb ddr4
3. GPU: Nvidia RTX quadro 4000 (8gb vram)
