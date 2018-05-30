# BvS
**Dawn of AI**  
An Image classifier to identify whether the given image is Batman or Superman using a CNN with high accuracy.  

# What are we gonna do:  
* We will build a 3 layered **Community Standard CNN Image classifier** to classify whether the given image is a image of Batman or Superman.
* Learn how to build a model from scratch in Tensoflow which is accurate.
* How to train and test it.
* How to save and use it further.  

Indepth explanation of each section:  
[Medium post with detailed step by step explanation](fill in) for deeper understanding of CNNs and architecture of the network.

# Data:

### Collect data:
* [Google Images Downloader](https://github.com/hardikvasa/google-images-download).It's fast, easy, simple and efficient.
* I've collected 300 images each for Supes and Batsy respectively, But more data is highly preferable. Try to collect as much clean data as possible.  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/image_collection.png" width="800" height="400">
</p>  

### Augmentation:
* 300 is not a number at all in Deep learning. So, we must Augment the images to get more images from whatever we collected.  
* You can use the following to do it easily, [Augmentor](https://github.com/mdbloice/Augmentor).  
* [This](https://github.com/perseus784/BvS/blob/master/augment.py) is the code I've used for augmenting my images.   
* Same image, augmented using various transformations. I had 3500 images each after augmentation for each class.    
* *Careful: While Augmenting, be careful about what kind of transformation you use. You can mirror flip a Bat Logo but cannot make it upside down.*  

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/augment.png" width="800" height="400">
</p> 

### Standardize:
* After Augmentation, Make a folder named rawdata in the current working directory.
* Create folders with their respective class names and put all the images in their respective folders.
* Run [this](https://github.com/perseus784/BvS/blob/master/preprocessing.py) file in the same directory as rawdata.
* This will resize all the images to a standard resolution and same format and put it in a new folder named data.  
**Note:** As I embedded it in *trainer.py*, it is unnecessary to run it explicitly.  

<p align="left">
<img src="https://github.com/perseus784/BvS/blob/master/media/convert.png" width="400" height="200">
<img src="https://github.com/perseus784/BvS/blob/master/media/file_structure.png" width="300" height="400">
</p>  


# Architecture:
### A Simple Architecture: 
> For detailed explanation of Architecture and CNNs read the medium [post]()  

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/convolution_nn_medium_post.png" width="800" height="400">
</p>  

In code:

          #level 1 convolution
          network=model.conv_layer(images_ph,5,3,16,1)
          network=model.pooling_layer(network,5,2)
          network=model.activation_layer(network)

          #level 2 convolution
          network=model.conv_layer(network,4,16,32,1)
          network=model.pooling_layer(network,4,2)
          network=model.activation_layer(network)

          #level 3 convolution
          network=model.conv_layer(network,3,32,64,1)
          network=model.pooling_layer(network,3,2)
          network=model.activation_layer(network)

          #flattening layer
          network,features=model.flattening_layer(network)

          #fully connected layer
          network=model.fully_connected_layer(network,features,1024)
          network=model.activation_layer(network)
          
          #output layer      
          network=model.fully_connected_layer(network,1024,no_of_classes)

### A Brief Architecture:
With dimentional informations:  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/brief_architecture.png" width="800" height="400">
</p>

# Training:
* Clone this repo.
* Put the images in thier respective folders in *rawdata*.  
  
       rawdata/batman: 3810 images
       rawdata/superman: 3810 images

Our file structure should look like this,  
<p align="left">
<img src="https://github.com/perseus784/BvS/blob/master/media/file_placement.png" width="500" height="400">
<img src="https://github.com/perseus784/BvS/blob/master/media/fstr.png" width="300" height="400">
</p>  

***data*** folder will be generated automatically by trainer.py

* **Configuration:**  If you want to edit something, you can do it using [this](https://github.com/perseus784/BvS/blob/master/config.py) file. 

          raw_data='rawdata'
          data_path='data'
          height=100
          width=100
          all_classes = os.listdir(data_path)
          number_of_classes = len(all_classes)
          color_channels=3
          epochs=300
          batch_size=10
          model_save_name='checkpoints\\'


* **Run** [trainer.py](https://github.com/perseus784/BvS/blob/master/trainer.py).
* Wait for few hours.
* For me it took **8 hrs for 300 epochs**. I did it in my laptop which has **i5 processors, 8 Gigabytes of RAM, Nvidia geforce 930M 2GB setup**. You can end the process anytime if saturated, as the model will be saved frequently.  
  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/train_info.png" width="800" height="400">
</p>  

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/training_loss.png" width="800" height="400">
</p>  

### Saving our model:  
Once training is over, we can see a folder named checkpoints is created which contains our model for which we trained. These two simple lines does that for us in tensorflow:  

          saver = tf.train.Saver(max_to_keep=4)
          saver.save(session, model_save_name)  
          
You can get my pretrained model [here.](https://drive.google.com/open?id=18ZzIYCkdTfYQQ1-tzpcfMuxzDwOJ0CU6)   

# Inference time:  
To run a simple prediction,
* Edit the image name in [predict.py](https://github.com/perseus784/BvS/blob/master/predict.py).
* Download the model files and extract in the same folder.
* Run [predict.py](https://github.com/perseus784/BvS/blob/master/predict.py).  

          image='sup.jpg'
          img=cv2.imread(image)
          session=tf.Session()
          img=cv2.resize(img,(100,100))
          img=img.reshape(1,100,100,3)
          labels = np.zeros((1, 2))
          # Creating the feed_dict that is required to be feed the io:
          feed_dict_testing = {im_ph: img, label_ph: labels}
          result=session.run(network, feed_dict=feed_dict_testing)
          print(result)  

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/output_screeenshot.png" width="800" height="400">
</p>   

You can see the results as [1,0]{Batman}, [0,1]{Superman} corresponding to the index.  
*Please note that this is not one-hot encoding.*  

# Accuracy:
It is actually pretty good. It is almost right all the time. I even gave it an image with both Batman and Superman, it actually gave me values which are almost of same magnitude(after removing the sigmoid layer that we added just before).  

*Comment out **network=tf.nn.sigmoid(network)** in predict.py to see the real magnitudes as this will only give squashed outputs.*  

From here on you can do whatever you want with those values.  
Initially loading the model will take some time(70 seconds) but once the model is loaded, you can put a for loop or something to throw in images and *get output in a second or two!*

# Tensorboard:
I have added some additional lines in the training code for tensorboard options. Using tensorboard we can track progress of our training even while training and after. You can also see your network structure and all the other components inside it.*It is very useful for visualizing the things happening.*
To start it, just go to the directory and open command line,  

    tensorboard --logdir checkpoints
    
You should see the following,

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/Inkedtensorboard_start_LI.jpg" width="800" height="400">
</p>

Now type the same address in in your browser. Your tensorboard is now started. Play with it.

# Graph Structure Visualization:  
Yeah, you can see our entire model with dimensions in each layer and operations here!

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/tensorboard_graph.png" width="800" height="400">
</p>

# Future Implementations:
While this works for Binary classification, it will also work for Multiclass classification but not as well. We might need to alter architecture and build a larger model depending on the number of classes we want.

> So, that's how Batman wins!
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/lego-batman-movie-tuxedo.jpg" alt="Batwin" width="800" height="400">
</p>
