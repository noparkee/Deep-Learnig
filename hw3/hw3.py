import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class nn_linear_layer:
    
    # linear layer.
    # randomly initialized by creating matrix W and bias b          # layer1 - W: 4x2   b: 4x1
    def __init__(self, input_size, output_size, std=1):         
        self.W = np.random.normal(0,std,(output_size,input_size))
        self.b = np.random.normal(0,std,(output_size,1))
    
    ######
    ## Q1
    def forward(self,x):            
        x = np.array(x)
        xre = x.reshape(x.shape[0], x.shape[1], 1)
        lin = self.W @ xre + self.b
        return lin
    
    ######
    ## Q2
    ## returns three parameters 
    def backprop(self,x,dLdy):
        # b_dLdW_2,b_dLdb_2,b_dLdx_2=layer2.backprop(x=a1_out,dLdy=b_nce_smax_out)     
        
        x = x.reshape(x.shape[0], x.shape[1], 1)
        dLdx = dLdy @ self.W

        dLdb_j = dLdy @ np.eye(self.b.shape[0])
        dLdb = np.sum(dLdb_j, axis=0) / x.shape[0]

        dLdW_j = dLdy.reshape(dLdy.shape[0], dLdy.shape[2], dLdy.shape[1]) @ x.reshape(x.shape[0], x.shape[2], x.shape[1])
        dLdW = np.sum(dLdW_j, axis=0) / x.shape[0]
        
        #print(dLdW.shape)
        #print(dLdb.shape)
        #print(dLdx.shape)
        
        return dLdW, dLdb ,dLdx

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

class nn_activation_layer:
    
    def __init__(self):
        pass
    
    ######
    ## Q3
    def forward(self,x):
        return 1 / (1 + np.exp(-x))
    
    ######
    ## Q4
    def backprop(self,x,dLdy):    
        # b_act_out=act.backprop(x=l1_out,dLdy=b_dLdx_2)
        dydx = np.zeros(shape=(20, 4, 4))
        for i in range(x.shape[0]):
            dydx[i][0][0] = ( np.exp(-x[i][0]) / ((1 + np.exp(-x[i][0]))**2) )
            dydx[i][1][1] = ( np.exp(-x[i][1]) / ((1 + np.exp(-x[i][1]))**2) )
            dydx[i][2][2] = ( np.exp(-x[i][2]) / ((1 + np.exp(-x[i][2]))**2) )
            dydx[i][3][3] = ( np.exp(-x[i][3]) / ((1 + np.exp(-x[i][3]))**2) )
        
        dLdx = dLdy @ dydx
               
        return dLdx


class nn_softmax_layer:
    def __init__(self):
        pass
    ######
    ## Q5
    def forward(self,x):
        # input x 가 (20, 2, 1)
        sum = []
        p = []
        smax_lst = []

        for i in range (x.shape[0]):
            sum.append(np.exp(x[i][0]) + np.exp(x[i][1]))
            p.append(np.exp(x[i][0]) / sum [i])
            smax_ins = [p[i], 1-p[i]]
            smax_lst.append(smax_ins)
        
        smax = np.array(smax_lst)       # smax (20, 2, 1)
               
        return smax         
    
    ######
    ## Q6
    def backprop(self,x,dLdy):
        # b_nce_smax_out=smax.backprop(l2_out,b_cent_out)
        dydx = np.zeros(shape=(20, 2, 2))
        for i in range (x.shape[0]):
            qnsah = (np.exp(x[i][0]) + np.exp(x[i][1])) ** 2
            dydx[i][0][0] = np.exp(x[i][0] + x[i][1]) / qnsah
            dydx[i][0][1] = - np.exp(x[i][0] + x[i][1]) / qnsah
            dydx[i][1][0] = - np.exp(x[i][0] + x[i][1]) / qnsah
            dydx[i][1][1] = np.exp(x[i][0] + x[i][1]) / qnsah
        
        dLdx = dLdy @ dydx
        
        #print("dLdx shape")
        #print(dLdx.shape)
            
        return dLdx     

class nn_cross_entropy_layer:
    def __init__(self):
        pass
        
    ######
    ## Q7
    def forward(self,x,y):      # input x: (20, 2, 1) y: (20, 1)
        one_hot = np.eye(2)[y]
        sum = 0
        for i in range (x.shape[0]):
            #sum += np.dot (one_hot[i], np.log(x[i]))
            sum += y[i] * np.log(x[i][0]) + (1-y[i]) * np.log(x[i][1])  
        return - sum / x.shape[0]
        
    ######
    ## Q8
    def backprop(self,x,y):
        # b_cent_out=cent.backprop(smax_out,y_train)
        dLdx = np.zeros(shape=(20, 1, 2))
        
        for i in range(x.shape[0]):
            if y[i] == 0:
                dLdx[i] = [-1/x[i][0], 0]
            else:
                dLdx[i] = [0, -1/x[i][1]]
        #print("dLdx shape")
        #print(dLdx.shape)
        return dLdx     

# number of data points for each of (0,0), (0,1), (1,0) and (1,1)
num_d=5

# number of test runs
num_test=40

## Q9. Hyperparameter setting
## learning rate (lr)and number of gradient descent steps (num_gd_step)
## This part is not graded (there is no definitive answer).
## You can set this hyperparameters through experiments.
lr=0.2                ###
num_gd_step= 2000      ###

# dataset size
batch_size=4*num_d

# number of classes is 2
num_class=2

# variable to measure accuracy
accuracy=0

# set this True if want to plot training data
show_train_data=True

# set this True if want to plot loss over gradient descent iteration
show_loss=True

for j in range(num_test):
    
    # create training data
    m_d1=(0,0)
    m_d2=(1,1)
    m_d3=(0,1)
    m_d4=(1,0)

    sig=0.05
    s_d1=sig**2*np.eye(2)

    d1=np.random.multivariate_normal(m_d1,s_d1,num_d)
    d2=np.random.multivariate_normal(m_d2,s_d1,num_d)
    d3=np.random.multivariate_normal(m_d3,s_d1,num_d)
    d4=np.random.multivariate_normal(m_d4,s_d1,num_d)

    # training data, and has dimension (4*num_d,2,1)
    x_train_d = np.vstack((d1,d2,d3,d4))
    # training data lables, and has dimension (4*num_d,1)
    y_train_d = np.vstack((np.zeros((2*num_d,1),dtype='uint8'),np.ones((2*num_d,1),dtype='uint8')))
    
    # plotting training data if needed
    if (show_train_data) & (j==0):
        plt.grid()
        plt.scatter(x_train_d[range(2*num_d),0], x_train_d[range(2*num_d),1], color='b', marker='o')
        plt.scatter(x_train_d[range(2*num_d,4*num_d),0], x_train_d[range(2*num_d,4*num_d),1], color='r', marker='x')
        plt.show()
                        
    # create layers

    # hidden layer
    # linear layer
    layer1= nn_linear_layer(input_size=2,output_size=4,)
    # activation layer
    act=nn_activation_layer()
                            
                            
    # output layer
    # linear
    layer2= nn_linear_layer(input_size=4,output_size=2,)
    # softmax
    smax=nn_softmax_layer()
    # cross entropy
    cent=nn_cross_entropy_layer()


    # variable for plotting loss
    loss_out=np.zeros((num_gd_step))

    for i in range(num_gd_step):
        
        # fetch data
        x_train = x_train_d
        y_train = y_train_d
        
        # create one-hot vectors from the ground truth labels
        y_onehot = np.zeros((batch_size,num_class))
        y_onehot[range(batch_size),y_train.reshape(batch_size,)]=1

        ################
        # forward pass
        
        # hidden layer
        # linear
        l1_out=layer1.forward(x_train)                  ######### 여기서 차원이 갑자기 들어나.
        
        # activation
        a1_out=act.forward(l1_out)
        
        # output layer
        # linear
        l2_out=layer2.forward(a1_out)
        # softmax
        smax_out=smax.forward(l2_out)
        # cross entropy loss
        loss_out[i]=cent.forward(smax_out,y_train)
        
        ################
        # perform backprop
        # output layer
        
        # cross entropy
        b_cent_out=cent.backprop(smax_out,y_train)
        # softmax
        b_nce_smax_out=smax.backprop(l2_out,b_cent_out)
            
        # linear
        b_dLdW_2,b_dLdb_2,b_dLdx_2=layer2.backprop(x=a1_out,dLdy=b_nce_smax_out)
    
        # backprop, hidden layer
        # activation
        b_act_out=act.backprop(x=l1_out,dLdy=b_dLdx_2)
        # linear
        b_dLdW_1,b_dLdb_1,b_dLdx_1=layer1.backprop(x=x_train,dLdy=b_act_out)
    
        ################
        # update weights: perform gradient descent
        layer2.update_weights(dLdW=-b_dLdW_2*lr,dLdb=-b_dLdb_2.T*lr)
        layer1.update_weights(dLdW=-b_dLdW_1*lr,dLdb=-b_dLdb_1.T*lr)

        if (i+1) % 2000 ==0:
            print('gradient descent iteration:',i+1)
                
    # set show_loss to True to plot the loss over gradient descent iterations
    if (show_loss) & (j==0):
        plt.figure(1)
        plt.grid()
        plt.plot(range(num_gd_step),loss_out)
        plt.xlabel('number of gradient descent steps')
        plt.ylabel('cross entropy loss')
        plt.show()
    
    
    ################
    # training done
    # now testing

    predicted=np.ones((4,))

    # predicting label for (1,1)
    l1_out=layer1.forward([[1,1]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[0] = np.argmax(smax_out)
    print('softmax out for (1,1)',smax_out,'predicted label:',int(predicted[0]))

    # predicting label for (0,0)
    l1_out=layer1.forward([[0,0]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[1] = np.argmax(smax_out)
    print('softmax out for (0,0)',smax_out,'predicted label:',int(predicted[1]))

    # predicting label for (1,0)
    l1_out=layer1.forward([[1,0]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[2] = np.argmax(smax_out)
    print('softmax out for (1,0)',smax_out,'predicted label:',int(predicted[2]))

    # predicting label for (0,1)
    l1_out=layer1.forward([[0,1]])
    a1_out=act.forward(l1_out)
    l2_out=layer2.forward(a1_out)
    smax_out=smax.forward(l2_out)
    predicted[3] = np.argmax(smax_out)
    print('softmax out for (0,1)',smax_out,'predicted label:',int(predicted[3]))

    print('total predicted labels:',predicted.astype('uint8'))
        
    accuracy += (predicted[0]==0)&(predicted[1]==0)&(predicted[2]==1)&(predicted[3]==1)

    if (j+1)%10==0:
        print('test iteration:',j+1)

print('accuracy:',accuracy/num_test*100,'%')






