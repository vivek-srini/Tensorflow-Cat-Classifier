from tensorflow.python.framework import ops
import math
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
files=glob.glob('*.jpg')
m=len(files)
x = np.zeros((m,64, 64, 3))
i=0
for file in files:
    
    image=Image.open(file)
    image=image.resize((64,64))
    image=np.array(image)
    x[i]=image
    i=i+1
x_flatten=x.reshape(x.shape[0],-1).T
x_train=x_flatten/255.
y_1=np.ones((1,1000))
y_2=np.zeros((1,1000))
y_train=np.append(y_1,y_2,axis=1)
y_train=y_train.ravel()
def one_hot(labels,C):
    C=tf.constant(C,name='C')
    one_hot_matrix=tf.one_hot(indices=labels,depth=C,axis=0)
    with tf.Session() as sess:
        one_hot=sess.run(one_hot_matrix)
        sess.close()
    return one_hot
y_train=one_hot(y_train,2)
def init_placeholders(n_x,n_y):
    X=tf.placeholder(tf.float32,[n_x,None],name='X')
    Y=tf.placeholder(tf.float32,[n_y,None],name='Y')
    return X,Y

def initialize_parameters():
    W1=tf.get_variable('W1',[100,12288],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b1=tf.get_variable('b1',[100,1],initializer=tf.zeros_initializer())
    W2=tf.get_variable('W2',[50,100],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2=tf.get_variable('b2',[50,1],initializer=tf.zeros_initializer())
    W3=tf.get_variable('W3',[2,50],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b3=tf.get_variable('b3',[2,1],initializer=tf.zeros_initializer())
    parameters={'W1':W1,
                'W2':W2,
                'W3':W3,
                'b1':b1,
                'b2':b2,
                'b3':b3
            }
    return parameters

def forward_propagation(parameters,X):
    W1=parameters['W1']
    W2=parameters['W2']
    W3=parameters['W3']
    b1=parameters['b1']
    b2=parameters['b2']
    b3=parameters['b3']
    Z1=tf.add(tf.matmul(W1,X),b1)
    A1=tf.nn.relu(Z1)
    Z2=tf.add(tf.matmul(W2,A1),b2)
    A2=tf.nn.relu(Z2)
    Z3=tf.add(tf.matmul(W3,A2),b3)
    return Z3

def compute_cost(Z3,Y):
    logits=tf.transpose(Z3)
    labels=tf.transpose(Y)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost

def random_mini_batches(X,Y,mini_batch_size):
    minibatches=[]
    m=X.shape[1]
    permutation=list(np.random.permutation(m))
    X_shuffled=X[:,permutation]
    Y_shuffled=Y[:,permutation].reshape((Y.shape[0],m))
    num_complete_mini_batches=math.floor(m/mini_batch_size)
    for k in range(num_complete_mini_batches):
        X_mini_batch=X_shuffled[:,k*mini_batch_size:(k+1)*mini_batch_size]
        Y_mini_batch=Y_shuffled[:,k*mini_batch_size:(k+1)*mini_batch_size]
        minibatch=(X_mini_batch,Y_mini_batch)
        minibatches.append(minibatch)
    if m%mini_batch_size!=0:
        
        X_mini_batch=X_shuffled[:,num_complete_mini_batches*mini_batch_size:]
        Y_mini_batch=Y_shuffled[:,num_complete_mini_batches*mini_batch_size:]
        minibatch=(X_shuffled,Y_shuffled)
        minibatches.append(minibatch)
    return minibatches


def model(X_train,Y_train,mini_batch_size,learning_rate,num_epochs):
    ops.reset_default_graph()
    costs=[]
    (n_x,m)=X_train.shape
    n_y=Y_train.shape[0]
    X,Y=init_placeholders(n_x,n_y)
    parameters=initialize_parameters()
    Z3=forward_propagation(parameters,X)
    cost=compute_cost(Z3,Y)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost=0
            num_minibatches=int(m/mini_batch_size)
            minibatches=random_mini_batches(X_train,Y_train,mini_batch_size)
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y)=minibatch
                _,minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                epoch_cost+=minibatch_cost/num_minibatches
            if epoch%5==0:
                print("Cost in epoch",epoch,"=",epoch_cost)
                costs.append(epoch_cost)
        plt.plot(costs)
        parameters=sess.run(parameters)
        correct_prediction=tf.equal(tf.argmax(Z3),tf.argmax(Y))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
        print("Training Set accuracy:",accuracy.eval({X:X_train,Y:Y_train}))
        return parameters
    
parameters=model(x_train,y_train,256,0.0001,300) 
x_train=x_train.astype('float32')
Z3=forward_propagation(parameters,x_train)
A3=tf.nn.softmax(Z3,0)
sess=tf.Session()
out=sess.run(A3)             
print(out)  

        