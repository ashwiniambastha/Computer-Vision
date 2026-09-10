
Hard Hat Sample - v2 augmented-416x416
==============================

This dataset was exported via roboflow.com on November 7, 2022 at 10:49 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 240 images.
Workers are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 40 percent of the image
* Random rotation of between -15 and +15 degrees
* Random Gaussian blur of between 0 and 1.5 pixels


