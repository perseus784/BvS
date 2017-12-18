# BvS
**Dawn of AI**  
An Image classifier to identify whether the given image is Batman or Superman.  

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/giphy.gif" alt="bvs" width="600" height="400">
</p>  

# Building a simple Neural Network:

## **Step 1:** Prepare the image data & preprocess it.

Firstly, we will have to collect huge amount of data to get atleast a significant amount of accuracy. I've collected 300 images from google. While it cannot be considered as decent data at all, it is enough to demonstrate the process.

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/image_collection.png" alt="All bat" width="650" height="300">
</p>

These Images may come of as different resolutions and formats. We don't need higher resolution images. So, we convert all of them into a standard 30 * 30 resolution in .jpg format.

> A simple program to do this can be found [**here**](https://github.com/perseus784/BvS/blob/master/image_process.py).  
> Input -> Folder containing collection of images.  

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/convert.png" alt="Conversion" width="650" height="300">
</p>

## **Step 2:** Convert Images into Data matrices.  

Our program cannot directly take image inputs. So, we need to convert it into a format which it understands.  
*Numbers!*. Yes, it can handle numbers better than us (Unless you are an asian).  
> It is done [**here**](https://github.com/perseus784/BvS/blob/master/data_prep.py).  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/club.png" alt="Conversion to table" width="900" height="350">
</p>  
 
## **Step 3:** Create a Neural Network Tensorflow graph.  
Okay, this is gonna be long but interesting. Lets begin!  

> ***A Neural Network is a Bio-inspired mathematical model based on how our brains work.***  

***How do we learn?*** It was a mystery for millions of years ever since we were conscious. Our brains consist of billions of neurons. These neurons communicate to each other by Synapses. By recent advancements, it is found that whenever we learn new stuff these synapses between these neurons gets strong. *Thus, we learn!*  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/brain-cell-neuron.gif" alt="neuron" width="500" height="350">
</p>  
This process is inspired and applied in machine learning. While in early periods there was neither enough data nor the processing power to do the math. Luckily, we are in the golden age of Information.  
Firstly, we create a simple perceptron. A rudimentary model without any complexities. It has,  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/perc.png" alt="neuron" width="600" height="300">
</p>  

    Inputs -> x1, x2, x3.
    Weights -> w1, w2, w3.
    Formatted Output -> y.
    Weights are nothing but likeablity of that respective input to be chosen.
    Weight is high for an input branch which gives max. output.
    Bias will give a basic shift to the output, it avoids nullification of a cell.
*An Analogy:*  
> Inputs are from our senses or previous neurons.  
> Weights are the synapes that connects the neurons.  
> Function cells are neuron cells. 

    Each Neuron cell has the functional equation,  
     Y= Wx+ b
    Where Y is output,
          W is Weight,
          b is bias.  
The whole thing can be represented mathematically as,  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/Perceptron.jpg" alt="Perceptron" width="1000" height="600">
</p>  

*Here comes Deep Neural Networks:*  
If we add more layers in a neural network to make it more accurate and handle larger data, it's a deep neural network.  
It works the same way as shown above, except it has more layers of it's repeated selfs.  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/neural_network.jpg" alt="DNN" width="700" height="350">
</p>
 
> A Neural Network with three hidden layers and one output layer is built for our case[**[link]**](https://github.com/perseus784/BvS/blob/master/neural_network.py):
   
    #build the skeleton dictionaries
    hidden_lr1={'weight':tf.Variable(tf.random_normal([image_shape,n_hl1])),'biases':tf.Variable(tf.random_normal([n_hl1]))}
    hidden_lr2={'weight':tf.Variable(tf.random_normal([n_hl1,n_hl2])),'biases':tf.Variable(tf.random_normal([n_hl2]))}
    hidden_lr3={'weight':tf.Variable(tf.random_normal([n_hl2,n_hl3])),'biases':tf.Variable(tf.random_normal([n_hl3]))}
    output_layer={'weight':tf.Variable(tf.random_normal([n_hl3,groups])),'biases':tf.Variable(tf.random_normal([groups]))}

    #Operation -> Y=Wx+b
    l1=tf.add(tf.matmul(data,hidden_lr1['weight']),hidden_lr1['biases'])
    l1=tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_lr2['weight']), hidden_lr2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_lr3['weight']), hidden_lr3['biases'])
    l3 = tf.nn.relu(l3)
     
    #Y can be obtained from this layer 
    output = tf.add(tf.matmul(l3, output_layer['weight']), output_layer['biases'])

  

## **Step 4:** Train the model.  

Now that we have created our Neural Network model, we throw in some inputs to it and get some output.  
Awful surprise, the outputs are nowhere realted to the expected inputs at all!   
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/dispmnt.jpg" alt="disp" width="550" height="300">
</p>  

*So, what is the problem?*  
Our Neural Network needs to be trained. Like our own brains, the model should be trained before it can take some test.  
<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/kungfu.gif" alt="kungfu" width="650" height="350">
</p>

**Training:**  
Training the model means teaching it why it is wrong and how it can rectify the mistake. This can be done using a cost fucntion.  
A ***Cost Function*** is the measurement squared error between the actual and the predicted outputs.  
Paraphrasing: By how much magnitude our model is wrong about the predicted output?
    
    Squared error for a single prediction = (predicted - actual)²
    Error for overall prediction:
                 Cost = ∑ ( predicted - actual )²

Once the Cost is calculated, we have to minimize it.  
This minimisation can be done easily using many methods. Now it's an optimization problem.  
> **Gradient Descent** is the one most significant technique in model optimization.
### Gradient Descent:  
We have to adjust those weights to give us a reduced total average cost or loss.
For simplicity, We take one input and one output in this example.  
To what values of inputs the the output is minimum?  
```
If we start at random and take larger steps, the minima can never be found since steps are too big.  
If we take smaller steps, the system may never converge.  
```
To avoid this, we start at a random point of input and take big leaps. Using the gradient we can find the direction of the slope where it gives minimum output.
Once a minima is reached, the upcoming steps should be smaller in size and find a minimal gradient.  
Again for the next few steps, even smaller steps are taken and the minimum of that funtion is found. 
This point is said to be the optimum point and the weights are adjusted according to this point.

<p align="center">
<img src="https://github.com/perseus784/BvS/blob/master/media/Sketch.png" alt="grad" width="1000" height="400">
</p>

There are many gradient descent methods evolved from this idea itself.  
Mainly there are,
> - Simple Gradient Descent.
> - Stochastic Gradient Descent.
> - AdaGrad -> Adapdtive Gradient Descent, AdaDelta -> Adaptive Delta.
> - Adam -> Adaptive Momentum Gradient Descent.

By proof it is best to use Adam optimizer due to it's quick convergence.

**The last piece of the puzzle**  
Though we have found optimum values, we have to tune the whole network sequence to adjust the weights in each layer.
This tuning of the whole network is done by ***Back Propagation***.  

*Back Propagation is the step where we actually train the Network to our data.*   
### Back Propagation:

> We are gonna do some serious stuff and it's called Math.

By backpropogation we mean to tune the weights for a likely ouput. We are gonna do that by tuning the weights right from Output layer wayback to Input layer including all the hidden layers. This process is done using the partial derivative *Chain rule*.

#### Chain Rule:  
We know the optimum cost using the Gradient Descent method. Now, we can tune the previous weight of the layer by using a derivative.

      For a perceptron: 
            Applying chain rule, dC/dX = dC/dW * dW/dX
      
      For a 2-layered network:
            
            dC/dX= dC/dW3 * dW3/dA2 * dA2/dW2 * dW2/dA1 * dA1/dW1 * dW1/dX
            where A1,A2 are activation functions.
            W1,W2,W3 are weights that connect each layer respectively.
            


## **Step 5:** Test the model.
