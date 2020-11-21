import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        # filter가 W        
        # out.shape=(batch size, num filter, out width, out height)
        # x.shape=(batch size, input channel size, in width, in height)
        # W.shape=(num filter,in channel size,filt width,filt height)
        # W = (8, 3, 3, 3)
        # b = (1, 8, 1, 1)
        # x = (8, 3, 32, 32)        
        
        b = np.squeeze(self.b)

        out = np.zeros(shape = (x.shape[0], self.W.shape[0], x.shape[2] - self.W.shape[2] + 1, x.shape[3] - self.W.shape[3] + 1))   # (8, 8, 30, 30)
        for i in range (x.shape[0]):        # 데이터 수
            y = view_as_windows(x[i], (self.W.shape[1], self.W.shape[2], self.W.shape[3]))  # (채널, 가로, 세로), y = (1, 30, 30, 3, 3, 3)
            y = y.reshape((y.shape[1], y.shape[2], -1))     # (30, 30, 27)

            out_f = np.zeros(shape = (out.shape[1], out.shape[2], out.shape[3]))    # (8, 30, 30)
            
            for j in range (self.W.shape[0]):       # 필터 수
                out_ins = y.dot(self.W[j].reshape((-1, 1)))     # (30, 30, 1)       # reshape한 W = (27, 1)
                out_ins = np.squeeze(out_ins,axis=2)        # (30, 30)
                out_f[j] = out_ins + b[j]       # bias 더하기
            
            out[i] = out_f
       
        return out

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        # x.shape=(batch size, input channel size, in width, in height)     # x = (8, 3, 32, 32)   
        # dLdy.shape=(batch size, num filter, out width, out height)        # dLdy = (8, 8, 30, 30)

        # W.shape=(num filter,in channel size,filt width,filt height)       # W = (8, 3, 3, 3)
        # b.shape=(1,num filter,1,1)                                        # b = (1, 8, 1, 1)

        # dLdx.shape=(batch size, input channel size, in width, in height)          > (8, 3, 32, 32)
        # dLdW.shape=(num filter, in channel size, filter width, filter height)     > (8, 3, 3, 3)
        # dLdb.shape=(1,num filter, 1,1)                                            > (1, 8, 1, 1)
                
        bat_siz, num_f, out_w, out_h = dLdy.shape
        _, depth, fil_w, fil_h = self.W.shape   

        W_flip = np.zeros(self.W.shape)                                         # (8, 3, 3, 3)
        for i in range (num_f):     # W 상하좌우 뒤집기
            for j in range (depth):
                W_flip[i, j] = np.flip(self.W[i, j])        
        
        n = x.shape[2] - (dLdy.shape[2])
        dLdy_pad = np.zeros(shape = (bat_siz, num_f, out_w + 2 * n, out_h + 2 * n))     # (8, 8, 32, 32) 
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
                    w = view_as_windows(x[i, d], (out_w, out_h))        # (3, 3, 30, 30)        # 여기선 filter가 dLdy[i, f]
                    w = w.reshape((w.shape[0], w.shape[1], -1))         # (3, 3, 900)

                    #print(dLdy[i, f].shape)                             # (30, 30)
                    dLdW_ins = w.dot(dLdy[i, f].reshape((-1, 1)))       # (3, 3, 1)
                    dLdW_ins = np.squeeze(dLdW_ins)                     # (3, 3)
                    dLdW[f, d] += dLdW_ins

        dLdb = np.sum(dLdy, axis = (0, 2, 3)) 
        dLdb = dLdb.reshape(self.b.shape)

        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        # x.shape=(batch size, input channel size, in width, in height)
        # print(x.shape)    x = (8, 3, 32, 32)

        out = np.zeros(shape = (x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2))
        
        for i in range (x.shape[0]):
            out_f = np.zeros(shape = (out.shape[1], out.shape[2], out.shape[3]))
            
            for j in range (x.shape[1]):
                y = view_as_windows(x[i][j], (self.pool_size, self.pool_size), step = self.stride)      # y = (16, 16, 2, 2)    
                max_p = np.max(y, axis=(2, 3))  # (16, 16)
                out_f[j] = max_p
            
            out[i] = out_f

        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        # x.shape=(batch size, input channel size, in width, in height)
        # dLdy.shape=(batch size, input channel size, out width, out height)
        # dLdx.shape=(batch size, input channel size, in width, in height)
        
        dLdx = np.zeros(shape = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))     # (8, 3, 32, 32)
        
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

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')