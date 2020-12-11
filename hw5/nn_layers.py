import numpy as np
from skimage.util.shape import view_as_windows

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # x.shape=(batch size, input channel size, in width, in height)
        # = (batch size, 1, 28, 28)
        # self.W.shape = (num_filters, in_ch_size, filter_width, filter_height)
        # = (28, 1, 3, 3)       # 각 배치마다 filter 한 개
        # self.b.shape = (1, num_filters, 1, 1)
        # = (1, 28, 1, 1)
        
        b = np.squeeze(self.b)

        # (batch size, filter_num, outw, outh)
        out = np.zeros(shape = (x.shape[0], self.W.shape[0], x.shape[2] - self.W.shape[2] + 1, x.shape[3] - self.W.shape[3] + 1)) 
        for i in range (x.shape[0]):        # 데이터 수
            y = view_as_windows(x[i], (self.W.shape[1], self.W.shape[2], self.W.shape[3]))  # (채널, 가로, 세로), 
            y = y.reshape((y.shape[1], y.shape[2], -1))     # (30, 30, 27)

            out_f = np.zeros(shape = (out.shape[1], out.shape[2], out.shape[3]))    # (8, 30, 30)
            
            for j in range (self.W.shape[0]):       # 필터 수
                out_ins = y.dot(self.W[j].reshape((-1, 1)))     # (30, 30, 1)       # reshape한 W = (27, 1)
                out_ins = np.squeeze(out_ins,axis=2)        # (30, 30)
                out_f[j] = out_ins + b[j]       # bias 더하기
            
            out[i] = out_f

        return out

    def backprop(self, x, dLdy):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # x.shape = (50, 1, 28, 28)
        # self.b.shape = (1, 28, 1, 1)
        # self.W.shape = (28, 1, 3, 3)
        # dLdy.shape = (50, 28, 26, 26)

        bat_siz, num_f, out_w, out_h = dLdy.shape
        _, depth, fil_w, fil_h = self.W.shape   

        W_flip = np.zeros(self.W.shape)     # (28, 1, 3, 3)                                     
        for i in range (num_f):     # W 상하좌우 뒤집기
            for j in range (depth):
                W_flip[i, j] = np.flip(self.W[i, j])

        n = x.shape[2] - (dLdy.shape[2])        # 28 - 26 = 2
        dLdy_pad = np.zeros(shape = (bat_siz, num_f, out_w + 2 * n, out_h + 2 * n))     # (50, 28, 30, 30)
        for i in range (bat_siz):
            for j in range (num_f):
                dLdy_pad[i, j] = np.pad(dLdy[i, j], ((n, n), (n, n)), 'constant', constant_values = 0)
        
        dLdx = np.zeros(x.shape)
        for i in range(bat_siz):
            for f in range (num_f):
                for d in range (depth):
                    w = view_as_windows(dLdy_pad[i, f], (fil_w, fil_h))               # 여기선 filter가 W_flip
                    w = w.reshape((w.shape[0], w.shape[1], -1))
                    
                    dLdx_ins = w.dot(W_flip[f, d].reshape((-1, 1)))
                    dLdx_ins = np.squeeze(dLdx_ins)         
                    dLdx[i, d] += dLdx_ins

        dLdW = np.zeros(self.W.shape)
        for i in range (bat_siz):
            for f in range (num_f):
                for d in range (depth):
                    #dLdW[f, d] += x[i, d] * dLdy[i, f]
                    w = view_as_windows(x[i, d], (out_w, out_h))        
                    w = w.reshape((w.shape[0], w.shape[1], -1))         

                    dLdW_ins = w.dot(dLdy[i, f].reshape((-1, 1)))       
                    dLdW_ins = np.squeeze(dLdW_ins)                     
                    dLdW[f, d] += dLdW_ins

        dLdb = np.sum(dLdy, axis = (0, 2, 3)) 
        dLdb = dLdb.reshape(self.b.shape)

        return dLdx, dLdW, dLdb

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # x.shape = (50, 28, 26, 26)
        # (batch size, num_filter, w / strid, h / strid)
        out = np.zeros(shape = (x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2))
        
        for i in range (x.shape[0]):    # 각 데이터 마다
            out_f = np.zeros(shape = (out.shape[1], out.shape[2], out.shape[3]))
            
            for j in range (x.shape[1]):    # 각 depth 마다
                y = view_as_windows(x[i][j], (self.pool_size, self.pool_size), step = self.stride)      # y = (13, 13, 2, 2)
                max_p = np.max(y, axis=(2, 3))  # 작은 (2, 2)중 가장 큰 거
                out_f[j] = max_p
            
            out[i] = out_f

        return out

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # x.shape = (50, 28, 26, 26)
        # dLdy.shape = (50, 1, 4732)
        
        dLdy = dLdy.reshape(x.shape[0], x.shape[1], x.shape[3]//2, x.shape[2]//2)

        dLdx = np.zeros(x.shape)    
        
        for i in range (dLdx.shape[0]):     # bacth 사이즈 만큼 반복 >> for문 안에서는 (3, 32, 32)
            dLdx_i = np.zeros(shape = (dLdx.shape[1], dLdx.shape[2], dLdx.shape[3]))        # (3, 32, 32)

            for j in range (dLdx.shape[1]):     # channel 갯수 만큼 반복 >> for문 안에서는 한 장! (32, 32)

                # max index를 알아내서, 그 부분은 dLdy값을, 다른 부분은 0을!
                y = view_as_windows(x[i][j], (self.pool_size, self.pool_size), step = self.stride)      # (16, 16, 2, 2) - (2, 2)크기 만큼 (16, 16)개
                y = y.reshape(y.shape[0], y.shape[1], -1)   # (16, 16, 4)
                
                dLdx_j = np.zeros(shape = (dLdx.shape[2], dLdx.shape[3]))       # (32, 32)

                for y1 in range(y.shape[0]):        # 16번 바복
                    for y2 in range(y.shape[1]):    # 16번 반복
                        max_idx = y[y1, y2].argmax()
                        
                        for y3 in range (y.shape[2]):   # 4번 이하 반복
                            if y3 == max_idx:
                                col = 2 * y2 + (y3 % 2)
                                row = 2 * y1 + (y3 % 2)
                                dLdx_j[row, col] = dLdy[i, j, y1, y2]
                                break
                
                dLdx_i[j] = dLdx_j

            dLdx[i] = dLdx_i       

        return dLdx



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # x.shape = (50, 28, 13, 13)
        # self.W.shape = (output_size, input_size) = (128, 4732)
        # self.b.shape = (output_size, )

        #print("1")
        b = self.b.reshape(-1, 1)       # (128, 1)  / (10, 1)
        #print("2")
        xre = x.reshape(x.shape[0], -1, 1)      # (50, 4732, 1) / (50, 128, 1)
        #print("3")
        # (50, 128, 1)  / (50, 10, 1)
        out = self.W @ xre + b
        #print("4")

        return out

    def backprop(self,x,dLdy):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # dLdy = (50, 1, 10) / (50, 1, 128)
        # x = (50, 128, 1) / (50, 28, 13, 13)
        # W = (10, 128) / (128, 4732)
        # b = (10, ) /  (128, )

        b = self.b.reshape(-1, 1)
        x = x.reshape(x.shape[0], -1, 1)    # batch size만 유지하고, 펼치기

        # (50, 1, 128) / (50, 1, 4732)
        dLdx = dLdy @ self.W
        
        # (1, 10) / (1, 128)
        dLdb = np.sum(dLdy @ np.eye(b.shape[0]), axis=0) / x.shape[0]

        # (10, 128) / (128, 4732)
        dLdW = np.sum((dLdy.reshape(dLdy.shape[0], dLdy.shape[2], dLdy.shape[1])) @ x.reshape(x.shape[0], x.shape[2], x.shape[1]), axis=0) / x.shape[0]

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        
        # x.shape = (50, 28, 26, 26)

        out = np.where(x < 0, 0, x)
        
        return out
    
    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########
        
        # x.shape = (50, 128, 1) / (50, 28, 26, 26)
        # dLdy.shape = (50, 1, 128) / (50, 28, 26, 26)
        
        # (50, 1, 128) / 

        x_shape = x.shape
        dLdy_shape = dLdy.shape
        
        x = x.reshape(-1, 1)        # 쭉 펼치기
        dLdy = dLdy.reshape(-1, 1)
    
        for i in range (x.shape[0]):
            if x[i][0] <= 0:
                dLdy[i][0] = 0
        
        dLdx = dLdy.reshape(dLdy_shape)

        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # x.shape = (50, 10, 1)

        #max_x = np.max(x)
        #exp = np.exp(x - max_x)
        exp = np.exp(x)
        exp_sum = exp.sum(axis=1)
        
        out = np.zeros(x.shape)
        for i in range (x.shape[0]):
            for j in range (x.shape[1]):
                out[i][j] = np.exp(x[i][j]) / exp_sum[i]

        return out

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # dLdy = (50, 1, 10)
        # x.shape = (50, 10, 1)
        
        #max_x = np.max(x)
        #exp = np.exp(x - max_x)     
        exp = np.exp(x)
        exp_sum = exp.sum(axis=1)

        # (50, 10, 10)
        dydx = np.zeros((x.shape[0], dLdy.shape[2], x.shape[1]))
        for k in range (x.shape[0]):
            for i in range (dydx.shape[1]): # y
                for j in range (dydx.shape[2]): # x
                    if i==j:
                        dydx[k][i][j] = (np.exp(x[k][i]) / exp_sum[k]) * (1 - (np.exp(x[k][i]) / exp_sum[k]))
                    else:
                        dydx[k][i][j] = - (np.exp(x[k][i]) / exp_sum[k]) * (np.exp(x[k][j]) / exp_sum[k])


        dLdx = dLdy @ dydx

        return dLdx

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):

        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # y는 train label
        # x = (50, 10, 1)
        y = y.reshape(-1, 1)
        
        loss_sum = 0
        for i in range(x.shape[0]):
            label = y[i]
            loss_sum += np.log(x[i][label])
        
        out = - loss_sum/x.shape[0]

        return out

    def backprop(self, x, y):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # x.shape = (50, 10, 1)
        # x는 softmax 결과 값
        dLdx = np.zeros((x.shape[0], x.shape[2], x.shape[1]))

        for i in range (x.shape[0]):
            label = y[i]
            dLdx[i][0][label] = -1 / x[i][label]
            
        return dLdx
