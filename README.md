# README

Multi Class Object Detection using PyTorch Faster RCNN ResNet50 FPN V2

PyTorch recently released an improved version of the Faster RCNN object detection model. They call it the Faster RCNN ResNet50 FPN V2. This model is miles ahead in terms of detection quality compared to its predecessor, the original Faster RCNN ResNet50 FPN. In this repo, we will discover what makes the new Faster RCNN model better, why it is better, and what kind of detection results we can expect from it.
To improve the Faster RCNN ResNet50 (to get the V2 version) model, changes were made to both:

* The ResNet50 backbone
* The object detection modules of Faster RCNN

### Pretraining ResNet50 Backbone

Pretraining the ResNet50 backbone is an essential task in improving the performance of the entire object detection model. The ResNet50 (as well as many other classification models) model was trained with a new training recipe. These include, but are not limited to:

* Learning rate optimizations.
* Longer training.
* Augmentation such as TrivialAugment, Random Erasing,  MixUp, and CutMix.
* Repeated Augmentation
* EMA
* Weight Decay Tuning

With these new techniques, the ResNet50 Accuracy@1 jumps to 80.858% from the previous 76.130%.

Training the Faster RCNN ResNet50 FPN V2 Model
As mentioned earlier, most of the improvements to train the entire object detection model were taken from the aforementioned paper.

The contributors to these improvements call these improvements as per post-paper optimization. These include:

* FPN with batch normalization.
* Using two convolutional layers in the Region Proposal * Network (RPN) instead of one. In other words, using a heavier FPN module.
* Using a heavier box regression head. To be specific, using four convolutional layers with Batch Normalization followed by linear layer. Previously, a two layer MLP head without Batch Normalization was used.
* No Frozen Batch Normalizations were used.

Using the above recipe improves the mAP from the previous 37.0% to 46.7%, a whopping 9.7% increase in mAP.

## Directory Structure

* The input directory contains all input images and videos that we will run inference on.

* The outputs directory contains the detection outputs that we obtain after running inference.
Directly inside the project directory, we have 5 Python files and one README.md file. We will get into the details of the Python files further in the blog post.

* The README file contains the links to some of the images and videos that you can download yourself and run the inference on.


#### Executing the detect_image.py for Image Inference

`python detect_image.py --input input/image_1.jpg --model v2 /`

#### Executing detect_video.py for Video Inference

`python detect_video.py --input input/video_1.mp4 --model v2 /`


## Important Links

* https://github.com/pytorch/vision/pull/5763
* https://github.com/pytorch/vision/pull/5444
* https://github.com/pytorch/vision/issues/5307
* https://github.com/pytorch/vision/issues/3995
* Improving the Backbones - How to Train State-Of-The-Art Models Using TorchVision’s Latest Primitives => https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
* Direct Link to new ResNet50 Training recipe.



## Images / Videos Credits and Attributions

* `input/`
  * `image_1.jpg`: Photo by Kent Zhong: https://www.pexels.com/photo/group-of-elephant-walking-on-brown-field-11528780/
    * https://www.pexels.com/photo/group-of-elephant-walking-on-brown-field-11528780/
  * `image_2.jpg`: Photo by Suraj Arya: https://www.pexels.com/photo/people-riding-on-wooden-boat-on-lake-10544422/
    * https://www.pexels.com/photo/people-riding-on-wooden-boat-on-lake-10544422/  
  * `video_1.mp4`: Video by Alex Pelsh: https://www.pexels.com/video/a-busy-downtown-intersection-6896028/
    * https://www.pexels.com/video/a-busy-downtown-intersection-6896028/
  * `video_2.mp4`: Video by Kelly: https://www.pexels.com/video/drone-footage-of-the-street-without-heavy-traffic-3696014/
    * https://www.pexels.com/video/drone-footage-of-the-street-without-heavy-traffic-3696014/
  * `video_3.mp4`: Video by ŠIBaj TV: https://www.pexels.com/video/drone-footage-of-basketball-court-6244065/.
    * https://www.pexels.com/video/drone-footage-of-basketball-court-6244065/
  * `faster-rcnn-small-object-detection.jpg`: https://www.pexels.com/photo/aerial-footage-of-tourists-by-the-beach-12318345/

