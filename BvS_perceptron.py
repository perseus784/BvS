#import image_process as ip
import pickler as pkl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

from scipy.ndimage import imread
'''
bat_directory="img_datasets/bat_img_dataset/"
sup_directory="img_datasets/sup_img_dataset/"
bat_train="img_datasets/bat_test/"
sup_train="img_datasets/sup_test/"
'''

image_shape=2700
grp=2
rate=0.001
steps=1000

def data_dict(bat_directory,sup_directory):
    #convert images to data points
    bat_dir = os.listdir(bat_directory)
    sup_dir = os.listdir(sup_directory)
    total_data={"images":[],"labels":[]}
    for i, j in zip(bat_dir, sup_dir):
        bat = imread(bat_directory+i, flatten=False)
        total_data["images"].append(bat)
        total_data["labels"].append("bat")
        sup = imread(sup_directory+j, flatten=False)
        total_data["images"].append(sup)
        total_data["labels"].append("sup")
    return total_data


#tensor flowgraph

img_ph=tf.placeholder(tf.float32,[None,2700])
                                #[number of images,features clubbed]
label_ph=tf.placeholder(tf.int64,[None])
                                 #n labels for n images
weight=tf.Variable(tf.zeros([2700,2]))
                               #two types of images
biases=tf.Variable(tf.zeros([2]))

operation=tf.add(tf.matmul(img_ph,weight),biases)

#cost fucntion
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=operation,labels=tf.one_hot(label_ph,depth=2)))

train_step=tf.train.AdamOptimizer().minimize(loss=loss)

correct_prediction = tf.equal(tf.argmax(operation, 1), label_ph)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#ALLEN, till here.................................

def label_changer(dictionary):
    for index,value in enumerate(dictionary["labels"]):
        if value=="bat":
            dictionary["labels"][index]=0
        if value=="sup":
            dictionary["labels"][index] = 1
    return dictionary

if __name__=="__main__":
    '''dic=data_dict(bat_directory,sup_directory)
    test_dic=data_dict(bat_test,sup_test)
    dic["images"]=np.array(dic["images"])
    dic=label_changer(dic)
    test_dic=label_changer(test_dic)
    pkl.write(dic, "train.pkl")
    pkl.write(test_dic, "test.pkl")'''

    #load training and testing data
    dic=pkl.retrive("train.pkl")
    test_dic=pkl.retrive("test.pkl")

    l = len(dic["images"])#202 images
    lt=len(test_dic["images"])#10 images

    print(dic["images"].shape)# 202 images , 30 * 30 pixel area , 3 colors
    dic["images"]=(np.array(dic["images"])).reshape(l,-1)# 202 images, 30*30*3=2700 features for each image
    print(np.array(test_dic["images"]).shape)

    test_dic["images"]=(np.array(test_dic["images"])).reshape(lt,-1)# 10 images, 30*30*3=2700 features for each image
    print(test_dic["images"].shape)

    dic["labels"] = (np.array(dic["labels"]))
    test_dic["labels"]=(np.array(test_dic["labels"]))
    print(dic["labels"].shape)# 202 labels for 202 images 0->bat 1->sup
    image_shape=len(dic["images"][0])
    print(image_shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(steps):
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            indices=np.random.choice(dic["images"].shape[0],lt)
            print(indices)
            images_batch = dic['images'][indices]
            print(images_batch.shape)
            labels_batch = dic['labels'][indices]
            print((labels_batch.shape))
# ALLEN from here I cannot understand the stuff happens
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={
                    img_ph: images_batch, label_ph: labels_batch})
                print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

            sess.run(train_step, feed_dict={img_ph: images_batch,label_ph: labels_batch})
        test_accuracy = sess.run(accuracy, feed_dict={
                img_ph: test_dic['images'],
                label_ph: test_dic['labels']})
        print('Test accuracy {:g}'.format(test_accuracy))



