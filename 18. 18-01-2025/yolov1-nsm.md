# NMS
 Ashwini(Heyh)

## Yolo v1

[Yolo v1 research paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

Yolo steps

- YOLO divides an input image into an S × S grid. From research paper it is S=7 i.e. 7x7
- If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object. 
- Each grid cell predicts B bounding boxes and confidence scores for those boxes. From research paper it is B=2 i.e. 2 box per grid
- YOLO predicts multiple bounding boxes per grid cell.
- These confidence scores reflect how confident the model is that the box contains an object and how accurate it thinks the predicted box is.
- They used PASCAL VOC which has 20 labelled classes so C = 20
- Final prediction is a 7 × 7 × 30 tensor.
    - 7x7 grid will produce 7x7 result
    - each grid has 2 boxes and each box predicts [x,y,w,h,objectness_conf/box_conf]. Total is 10 values
    - objectness_conf/box_conf in [x,y,w,h,objectness_conf/box_conf] is Pr(Object) * IOU(Truth prediction). If no object then they want Pr(Object) = 0, else Pr(Object) == IOU(Truth prediction)
    - Each grid only produces one set of class probabilities regardless of number of boxes B.
    - For each grid cell they output 20 class probabilities i.e. total no of class.
    - 20 + 10 = 30. Final output dimension is 7x7x30.
    - At training time we only want one bounding box predictor to be responsible for each object and they select prediction of box which has the highest current IOU with the ground truth. 
    - This leads to specialization between the bounding box predictors. It means bbox which has vertical rectangle will be speciased in person and horizontal one will be specialised in cars and others.
    - Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall.
    - 98 bounding box per image rather than 2000 from selective search.
    - During test time, they multiply the class probability they multiply class probabilty of grid * Probability of object to get the probability of class 

    - network design is inspired by Google net



### Yolo output

NMS :  NMS is used to identify and remove redundant or incorrect bounding boxes and to output a single bounding box for each object in the image.
1. Lots of detection as output
2. Remove the boxes with low probability score.
3. Apply NMS
-  Remove objects that have high IOU with the bounding boxes that have the highest probability.

![nms](./nsm.png)
