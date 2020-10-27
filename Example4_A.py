#----------------------------------------------------------------------#
#                     EXAMPLE 4 (RECTANGULAR DOMAIN)                   #
#                                                                      #
#   This program solve the Poisson equation:                           #
#                        u_{xx} + u_{yy} = f,                          #
#   where                                                              #
#              f = int( 2*delta(x-Xs)*delta(y-Ys) )                    #
#   with Dirichlet BC over an arbitrary domain and the exact solution: #
#                u(x,y) = 1.0             r <  r0                      #
#                u(x,y) = 1.0 + log(2*r)  r <= r0                      # 
#----------------------------------------------------------------------# 
#   How to run:                                                        #
#                       >> python3 Example4_A                          #
#----------------------------------------------------------------------#
#  Cite as:                                                            #
#  R. Itza Balam, F. Hernandez-Lopez, J. Trejo-Sanchez, M. Uh Zapata.  #
#  An immersed boundary neural network for solving elliptic equations  #
#  with singular forces on arbitrary domains, Mathematical Biosciences #
#  and Engineering (2020).                                             #
#                                                                      #
#----------------------------------------------------------------------#

import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import math

np.random.seed(1234)
tf.set_random_seed(1234)

Nfeval = 1
Nloss  = 0.0

########################################################################
# ---------------------------------------------------------------------#
#                  SUBROUTINE: Physics Informed DNN                    #
# ---------------------------------------------------------------------#
########################################################################
    
class PhysicsInformedNN:
    
    # .================================================================.
    # |  [1]                    Definitions                            |
    # .================================================================.
    
    def __init__(self, X_u, u, X_f, rhs, layers, lb, ub):
        
        #.----------------------------------------------.
        #|                    INPUTS                    |
        #.______________________________________________.
        
        #------------------------------
        # Training points  
        self.xu = X_u[:,0:1]
        self.yu = X_u[:,1:2]
        self.u  = u
        #------------------------------
        # Collocation points
        self.xf  = X_f[:,0:1]
        self.yf  = X_f[:,1:2]
        self.rhs = rhs
        #------------------------------
        # Layers 
        self.layers = layers
        #------------------------------
        # Limits 
        self.lb = lb
        self.ub = ub

        # .----------------------------------------------.
        # |      CALL INITIAL NETWORK INPUTS: W & b      |
        # .______________________________________________.
        
        self.weights, self.biases = self.initialize_NN(layers)
              
        # .----------------------------------------------.
        # |    TENSOR FLOW CONFIGURATION: CPU or GPU     |
        # .______________________________________________.
                
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        # .----------------------------------------------.
        # |           TENSOR FLOW VARIABLES              |
        # .______________________________________________.
        
        #------------------------------
        #  Training points
        self.xu_tf = tf.placeholder(tf.float32,shape=[None,self.xu.shape[1]])
        self.yu_tf = tf.placeholder(tf.float32,shape=[None,self.yu.shape[1]])        
        self.u_tf  = tf.placeholder(tf.float32,shape=[None,self.u.shape[1]])
        #------------------------------
        #  Collocations points        
        self.xf_tf  = tf.placeholder(tf.float32,shape=[None,self.xf.shape[1]])
        self.yf_tf  = tf.placeholder(tf.float32,shape=[None,self.yf.shape[1]])
        self.rhs_tf = tf.placeholder(tf.float32,shape=[None,self.rhs.shape[1]])
                
        # .----------------------------------------------.
        # |          LOSS FUNCTION & OPTIMIZER           |
        # .______________________________________________.
        
        #------------------------------                         
        self.u_pred   = self.net_u(self.xu_tf,self.yu_tf) 
        self.f_pred   = self.net_f(self.xf_tf,self.yf_tf)
        #------------------------------      
        self.loss = tf.reduce_mean(tf.square(self.u_tf   - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.rhs_tf - self.f_pred))
        #------------------------------
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                         method = 'L-BFGS-B', 
                         options = {'maxiter': 50000,
                                    'maxfun' : 50000,
                                    'maxcor' : 50,
                                    'maxls'  : 50,
                                    'ftol'   : 1.0 * np.finfo(float).eps})
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    # .================================================================.
    # | [2]     INITIAL VALUES: weights(W) & biases(b) randomly        |
    # .================================================================.

    #-----------------------------------------------                    
    def initialize_NN(self, layers):        
        weights = []
        biases  = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    #-----------------------------------------------     
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    # .================================================================.
    # | [3]                  DEEP NEURAL NETWORK                       |
    # .================================================================.
    
    #-----------------------------------------------
    #  Network using composition: tanh(H*W+b)       
    def neural_net(self, X, weights, biases):
        #--------------
        num_layers = len(weights) + 1
        #--------------
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        #--------------
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        #--------------
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    #-----------------------------------------------
    #  Network u          
    def net_u(self,x,y):
        u = self.neural_net(tf.concat([x,y],1), self.weights, self.biases)
        return u
    #-----------------------------------------------
    #  Network f
    def net_f(self,x,y):
        u    = self.net_u(x,y)
        u_x  = tf.gradients(u,x)[0]
        u_y  = tf.gradients(u,y)[0]
        u_xx = tf.gradients(u_x,x)[0]
        u_yy = tf.gradients(u_y,y)[0]
        f = u_xx + u_yy
        return f
    #-----------------------------------------------
    #  Loss function call % print   
    def callback(self, loss):
        global Nfeval
        global Nloss
        print(' ', Nfeval, ' Loss: ', loss)
        Nfeval += 1
        Nloss = loss

    # .================================================================.
    # | [4]           TRAINING: OPTIMIZER SOLUTION W & b               |
    # .================================================================.
            
    def train(self):
        
        tf_dict = {self.xu_tf:   self.xu, 
                   self.yu_tf:   self.yu, 
                   self.u_tf :   self.u,
                   self.xf_tf:   self.xf, 
                   self.yf_tf:   self.yf,
                   self.rhs_tf:  self.rhs}
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        

    # .================================================================.
    # | [5]           PREDICTION: u(X_star) & f(X_star)                |
    # .================================================================.
        
    def predict(self, X_star):         
        u_star = self.sess.run(self.u_pred, {self.xu_tf: X_star[:,0:1], self.yu_tf: X_star[:,1:2]})
        return u_star
        
########################################################################
# ---------------------------------------------------------------------#
#                          SCRIPT SOLUTION                             #
# ---------------------------------------------------------------------#
######################################################################## 
    
if __name__ == "__main__": 

    # .================================================================.
    # | [S.1]                    PREPARATION                           |
    # .================================================================.
    
    # .----------------------------------------------.
    # |                  PARAMETERS                  |
    # .______________________________________________.

    #-----------------------------------------------
    #  Number of training points (from boundary data)
    N_u = 32
    #-----------------------------------------------
    #  Interface: Peskin method h=ds or h=(ul-lb)/Nh 
    Ns = 32     
    Nh = 32   
    #-----------------------------------------------
    #  Number of neurons and layers
    Ne = 20
    layers = [2,Ne,Ne,Ne,1]
    
    # .----------------------------------------------.
    # |               READ DATA & MESH               |
    # |       Read data: "Elliptic_discon.mat"       |
    # .______________________________________________.
    
    #----------------------------------
    CaseEx4 = 1
    if (CaseEx4==1):
       data = scipy.io.loadmat('input/Input_Ex4_Circle.mat')
       r0 = 0.5 # Interface radius
       x0 = 0.0 # Interface center (x0,y0)
       y0 = 0.0
    if (CaseEx4==2):
       data = scipy.io.loadmat('input/Input_Ex4_General.mat') 
       r0 = 0.25 # Interface radius
       x0 = 0.3  # Interface center (x0,y0)
       y0 = 0.3
    #----------------------------------    
    X     = data['x'].flatten()[:,None]
    Y     = data['y'].flatten()[:,None]    
    Exact  = np.real(data['usol']).T
    xp     = data['xp'].flatten()[:,None]
    yp     = data['yp'].flatten()[:,None]
    Exactp = np.real(data['usolp']).T
    trip   = np.real(data['trip']) 
    xbdy   = data['xbdy'].flatten()[:,None]
    ybdy   = data['ybdy'].flatten()[:,None]
    ubdy   = np.real(data['ubdy']).T 
    
    N_b  = len(xbdy)  # Total number of boundary points
    N_f  = len(X)     # Total number of collocation points 
    N_p  = len(xp)    # Total number of predicted points

    # .----------------------------------------------.
    # |                  Interface                   |
    # .______________________________________________.
    
    th = np.linspace(0,2*np.pi,num=Ns+1)
    #----------
    xs = np.zeros((Ns,1),float)
    ys = np.zeros((Ns,1),float)
    for s in range(0,Ns):
        xs[s]  = r0*np.cos(th[s]) + x0
        ys[s]  = r0*np.sin(th[s]) + y0

    print('  ')
    print('---------------------------------------------')
    print('              ELLIPTIC PROBLEM               ')
    print('---------------------------------------------')
    print('  TRAINING:                                  ')
    print('  Training known solution points   N_u =',N_u)
    print('  Training collocation points      N_f =',N_f)
    print('  Boundary points                  N_b =',N_b)
    print('  Number of interface points       N_s =',Ns)    
    print('---------------------------------------------')
    print('  NEURAL NETWORK:                            ')
    print('  layers =',layers)    
    print('---------------------------------------------')
    print('  PREDICTED:                                 ')
    print('  Points Np = ',N_p)
    print('---------------------------------------------')
    print('  ')
    
    # .----------------------------------------------.
    # |         PREDICTED POINTS & SOLUTION          |
    # .______________________________________________. 
    
    Xp_star = np.hstack((xp,yp))
    up_star = Exactp.flatten()[:,None]

    #----------------
    # Domain bounds
    lb = Xp_star.min(0)
    ub = Xp_star.max(0)
    
    # .----------------------------------------------.
    # |        POINTS: TRAINING & COLLOCATION        |
    # .______________________________________________.
    
    #-----------------------------------------------
    # Collocation points:
    X_f = np.hstack((X,Y))
    X_f_train = np.vstack(X_f)
    #----------------
    xf = X_f_train[:,0:1]
    yf = X_f_train[:,1:2]
          
    #-----------------------------------------------
    # Training boundary data: (xu_i,yu_i,u_i)
    xx1 = np.hstack((xbdy,ybdy));
    uu1 = ubdy.T
    X_bdy = np.vstack([xx1])
    u_bdy = np.vstack([uu1])
    
    idx = np.random.choice(X_bdy.shape[0],N_u,replace=False)
    X_u_train = X_bdy[idx,:]
    u_train   = u_bdy[idx,:]
    #----------------
    xu = X_u_train[:,0]
    yu = X_u_train[:,1]

    # .----------------------------------------------.
    # |          RIGHT-HAND SIDE: rhs(Nf)            |
    # .______________________________________________. 

    #---------------------------
    # h value for delta
    ds = r0*(2.0*np.pi/Ns)
    dx = np.abs(ub[0]-lb[0])/Nh
    h  = 1.0*dx
    
    #---------------------------
    J1  = np.zeros((Ns,1),float) 
    rhs = np.zeros((N_f,1),float)   
    for l in range(0,N_f):
        #____________________
        som1 = 0.0
        for s in range(0,Ns):
            #----------------
            xk = xf[l]-xs[s]
            yk = yf[l]-ys[s]
            #----------------
            J1[s] = 1.0/r0   
            #----------------
            deltax  = 0.0
            deltay  = 0.0
            if np.abs(xk) < 2*h: 
               deltax  = (1.0 + np.cos(np.pi*xk/(2.0*h)))/(4.0*h)    
            if np.abs(yk) < 2*h: 
               deltay  = (1.0 + np.cos(np.pi*yk/(2.0*h)))/(4.0*h)   
            #----------------
            ds = r0*2.0*np.pi/Ns
            #----------------
            som1 = som1 + J1[s]*deltax*deltay*ds          
        #____________________
        rhs[l] = som1     
    #---------------------------
    rhs_train = rhs

    # .================================================================.
    # | [S.2]                    APPROXIMATION                         |
    # .================================================================.
    
    # .----------------------------------------------.
    # |          MODEL TRAINING: Optimizer           |
    # .______________________________________________.

    #-----------------------------------------------
    #  Function: PhysicsInformedNN      
    model = PhysicsInformedNN(X_u_train, u_train,    \
                              X_f_train, rhs_train,  \
                              layers, lb, ub)
    #-----------------------------------------------
    #  Optimization solution: W & b 
    start_time = time.time()                
    model.train()
    elapsed = time.time() - start_time
    
    # .----------------------------------------------.
    # |            PREDICTIONS: u(Xp_star)           |
    # .______________________________________________.
        
    up_pred = model.predict(Xp_star)
        
    # .----------------------------------------------.
    # |                    NORMS                     |
    # .______________________________________________.
                              
    error_u   = np.linalg.norm(up_star-up_pred,2)/np.linalg.norm(up_star,2)
    error_max = np.linalg.norm(up_star-up_pred,np.inf)   

    # .================================================================.
    # | [S.3]                  DISPLAYING                              |
    # .================================================================.
    
    print('---------------------------------------------')
    print('  TRAINING:                                  ')
    print('  Training points     N_u =',N_u)
    print('  Collocation points  N_f =',N_f)
    print('  Predicted points    N_p =',N_p)
    print('---------------------------------------------')
    print('  NEURAL NETWORK:                            ')
    print('  layers =',layers)
    print('---------------------------------------------')
    print('  INTERFACE:                                 ')
    print('  Number of interface points:  N_s =',Ns)
    print('  -----------------')
    print('  Perskin method with h =', h)    
    print('---------------------------------------------')
    print('  N_u   N_f   N_s    L2-norm     Max-norm' )
    print('  %3s  %3s  %3s  %e  %e' % (N_u,N_f,Ns,error_u,error_max))
    print('---------------------------------------------')            
    print('   Training time: %.4f' % (elapsed))
    print('---------------------------------------------') 
    print('  ')

    # .================================================================.
    # | [S.4]                    SAVE DATA                             |
    # .================================================================.
     
    # -------------------------------
    # Final results    
    file1 = open("output/dataA/Results_Ex4.txt","w")
    file1.write('-------------------------------------------------\n')
    file1.write('Nu = %s\n' % N_u)
    file1.write('Nf = %s\n' % N_f)
    file1.write('Nb = %s\n' % N_b)
    file1.write('Np = %s\n' % N_p)
    file1.write('layers = %s\n' % layers)
    file1.write('Points at the interface Ns = %s\n' % Ns)
    file1.write('Perskin method with h = %f\n' % h)    
    file1.write('L2-norm  error: %.3e\n' % error_u)
    file1.write('Max-norm error: %.3e\n' % error_max)
    file1.write('Training time : %.4f (sec)\n' % elapsed)
    file1.write('Iterations    : %s\n' % Nfeval)
    file1.write('Loss          : %f\n' % Nloss)
    file1.write('-------------------------------------------------\n')
    file1.close()
 
    # -------------------------------
    # Training data (xu,yu)
    file1 = open("output/dataA/xu.txt","w")
    for l in range(0,N_u):
        file1.write('%f  ' % xu[l])
        file1.write('%f  ' % yu[l])
        file1.write('%f\n' % u_train[l])
    file1.close()
    # -------------------------------
    # Training data (xf,yf)
    file2 = open("output/dataA/xf.txt","w")
    for l in range(0,N_f):
        file2.write('%f  ' % xf[l])
        file2.write('%f  ' % yf[l])
        file2.write('%f\n' % rhs[l])
    file2.close() 
    # -------------------------------
    # Solution (xp,yp)
    file3 = open("output/dataA/xp.txt","w")
    for l in range(0,N_p):
        file3.write('%f  ' % xp[l])
        file3.write('%f  ' % yp[l]) 
        file3.write('%f  ' % up_star[l]) 
        file3.write('%f\n' % up_pred[l])
    file3.close()
    # -------------------------------
    # Interface (xs,ys)
    file4 = open("output/dataA/xs.txt","w")
    for s in range(0,Ns):
        file4.write('%f  ' % xs[s])
        file4.write('%f\n' % ys[s])
    file4.close()
    # -------------------------------
    # Triangular mesh (only to display)
    file5 = open("output/dataA/trip.txt","w")
    N_c = len(trip[:,0])
    for l in range(0,N_c):
        file5.write('%i  ' % trip[l,0])
        file5.write('%i  ' % trip[l,1]) 
        file5.write('%i\n' % trip[l,2]) 
    file5.close()    
