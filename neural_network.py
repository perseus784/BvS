import tensorflow as tf

def neural_network(data):
    image_shape = 2700
    n_hl1 = 500
    n_hl2 = 600
    n_hl3 = 500
    groups = 2

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