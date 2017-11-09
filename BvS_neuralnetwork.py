import tensorflow as tf
import os
import numpy as np
import pickler as pkl
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
groups=2

image_shape=2700
n_hl1=500
n_hl2=600
n_hl3=500

# load training and testing data
dic = pkl.retrive("train.pkl")
test_dic=pkl.retrive("test.pkl")

img_ph=tf.placeholder(tf.float32,[None,image_shape])
                                #[number of images,features clubbed]
label_ph=tf.placeholder(tf.float32,[None,groups])
                                 #n labels for n images
def neural_network(data):

    hidden_lr1={'weight':tf.Variable(tf.random_normal([image_shape,n_hl1])),'biases':tf.Variable(tf.random_normal([n_hl1]))}
    hidden_lr2={'weight':tf.Variable(tf.random_normal([n_hl1,n_hl2])),'biases':tf.Variable(tf.random_normal([n_hl2]))}
    hidden_lr3={'weight':tf.Variable(tf.random_normal([n_hl2,n_hl3])),'biases':tf.Variable(tf.random_normal([n_hl3]))}
    output_layer={'weight':tf.Variable(tf.random_normal([n_hl3,groups])),'biases':tf.Variable(tf.random_normal([groups]))}

    #o=Wx+b
    l1=tf.add(tf.matmul(data,hidden_lr1['weight']),hidden_lr1['biases'])
    l1=tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_lr2['weight']), hidden_lr2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_lr3['weight']), hidden_lr3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weight']), output_layer['biases'])

    return output

def data_manipulator(x):
    global dic
    l = len(dic["images"])  # 202 images
    lt=len(test_dic["images"])#10 images

    dic["images"] = (np.array(dic["images"])).reshape(l, -1)  # 202 images, 30*30*3=2700 features for each image
    dic["labels"] = (np.array(dic["labels"]))

    test_dic["images"] =(np.array(test_dic["images"])).reshape(lt, -1)# 10 images, 30*30*3=2700 features for each image
    test_dic["labels"]=(np.array(test_dic["labels"]))

    if x=='train':
        label_dic = {'labels': []}
        for i in dic['labels']:
            if i == 0:
                label_dic['labels'].append([0, 1])
            elif i == 1:
                label_dic['labels'].append([1, 0])
        label_dic["labels"] = (np.array(label_dic["labels"]))
        return dic['images'], label_dic['labels']

    if x=='test':
        label_dic = {'labels': []}
        for i in test_dic['labels']:
            if i == 0:
                label_dic['labels'].append([0, 1])
            elif i == 1:
                label_dic['labels'].append([1, 0])
        label_dic["labels"] = (np.array(label_dic["labels"]))
        return test_dic['images'], label_dic['labels']

def train(data):
    predict=neural_network(data)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=label_ph))
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    generations=50

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for gen in range(generations):
            images,labels=data_manipulator('train')
            #just to print loss UNNECESSARY`
            gen_loss=sess.run(cost,feed_dict={img_ph:images,label_ph:labels})
            sess.run(optimizer,feed_dict={img_ph:images,label_ph:labels})
            print('loss.....',gen_loss)

        test_images,test_labels=data_manipulator('test')
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(label_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('accuracy', accuracy.eval({img_ph:test_images, label_ph:test_labels}))

train(img_ph)