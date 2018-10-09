# Number Plate Detection
Detecting numberplates on cars using transfer learning with the ssd_mobilenet_v2, the coco API and the tensorflow object detection API

# Prerequisites
- ![tensorflow](https://github.com/tensorflow)
- ![tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

# Usage
put images to test in the test_images/ directory and then either run the juypter notebook object_detection_tutorial.ipynb or run plate-detection.py

The model will then detect license plates in the given images and add a box with the according probability to it. Samples will look like this:
![sample1](https://github.com/Lugges991/MachineLearning/blob/master/plate-detection/sample1.png)
![sample2](https://github.com/Lugges991/MachineLearning/blob/master/plate-detection/sample2.png)
![sample3](https://github.com/Lugges991/MachineLearning/blob/master/plate-detection/sample3.png)
![sample4](https://github.com/Lugges991/MachineLearning/blob/master/plate-detection/sample4.png)

# Notes
If you change the images in the test_images directory, make sure to use thenaming convention inplace: image{image-nr}.jpg
Also update the for loop either in plate-detection.py or in objec-detection-tutorial.py depending on what you want to execute.

.

.

.
shout outs to ![@datitran](https://github.com/datitran) for the xml_to_csv.py and generate_tfrecords.py! I know Im a lazy copy cat ;))
