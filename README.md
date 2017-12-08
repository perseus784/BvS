# BvS
**Dawn of AI**

An Image classifier to identify whether the given image is Batman or Superman.

## Building a simple Neural Network:

**Step 1:** Prepare the image data & preprocess it.

Firstly, we will have to collect huge amount of data to get atleast a significant amount of accuracy. I've collected 300 images from google. While it cannot be considered as decent data at all, it is enough to demonstrate the process.

<p align="center">
<img src="/media/image_collection.png" alt="All bat" width="650" height="300">
</p>

These Images may come of as different resolutions and formats. We don't need higher resolution images. So, we convert all of them into a standard 30 * 30 resolution in .jpg format.

A simple program to do this can be found [**here**](https://github.com/perseus784/BvS/blob/master/image_process.py).  
Input -> Folder containing collection of images.  

<p align="center">
<img src="/media/convert.png" alt="Conversion" width="650" height="300">
</p>


**Step 2:** Convert Images into Data matrices.  

Our program cannot directly take image inputs. So, we need to convert it into a format which it understands.  
*Numbers!*. Yes, it can handle numbers better than us (Unless you are an asian).
<p align="center">
<img src="/media/club.png" alt="Conversion to table" width="800" height="350">
</p>  

It is done [**here**](https://github.com/perseus784/BvS/blob/master/data_prep.py).  

**Step 3:** Create a Neural Network Tensorflow graph.


**Step 4:** Train the model using the data matrices.


**Step 5:** Test the model that was created with a new set of data.
