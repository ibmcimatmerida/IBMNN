#----------------------------------------------------------------------#
#                     EXAMPLE 4 (RECTANGULAR DOMAIN)                   #
#                                                                      #
#   This program solve the Poisson equation:                           #
#                        u_{xx} + u_{yy} = f,                          #
#   where                                                              #
#              f = int( 2*delta(x-Xs)*delta(y-Ys) )                    #
#   with Dirichlet BC over the rectangular region [-1,1]x[-1,1] and    #
#   the exact solution:                                                #
#                u(x,y) = 1.0             r <  r0                      #
#                u(x,y) = 1.0 + log(2*r)  r <= r0                      # 
#----------------------------------------------------------------------# 
#   How to run:                                                        #
#                       >> python3 Example4_R                          #
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
        u_y  = tf.gradients(u,y)[0]
        u_x  = tf.gradients(u,x)[0]
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

    #  .---------------------------------------------------------------.
    #  | PARAMETERS:                                                   |
    #  | --------                                                      |
    #  | N_u    : Number of training points                            |
    #  | N_f    : Number of collocation points                         |
    #  | --------                                                      |
    #  | Ne     : Number of neurons per layer                          | 
    #  | layers : Vector with neurons and layers                       |                                  
    #  | --------------------------------------------------------------|
    #  | MESHGRID:                                                     |
    #  |                             DIRICHLET                         |
    #  |                              [0:1,:]                          |
    #  |                        (xx1,uu1) <<u(-1,x)>>                  |
    #  |                                                               |
    #  |                    0   1   2  . . .Nx-3 Nx-2 Nx-1             |
    #  |              0   [ o   o   o  . . .  o   o   o ]              |
    #  |  DIRICHLET   1   [ o   *   *  . . .  *   *   o ]    NEUMANN   |
    #  |  (xx2,uu2)   2   [ o   *   *  . . .  *   *   o ]   (xx3,uu3)  |
    #  | <<u(y,-1)>>  .   [ .   .   .         .   .   . ]   <<u(y,1)>> |
    #  |   [:,0:1]    .   [ .   .   .         .   .   . ]    [:,-1:]   |
    #  |              .   [ .   .   .         .   .   . ]              |
    #  |              Ny-3[ o   *   *  . . .  *   *   o ]              |
    #  |              Ny-2[ o   *   *  . . .  *   *   o ]              |
    #  |              Ny-1[ o   o   o  . . .  o   o   o ]              |
    #  |                                                               |
    #  |                              [-1,:]                           |
    #  |                       (xx4,uu4) <<u(1,x)>>                    |
    #  |                             DIRICHLET                         |
    #  .---------------------------------------------------------------. 
    
    # .================================================================.
    # | [S.1]                    PREPARATION                           |
    # .================================================================.
    
    # .----------------------------------------------.
    # |                  PARAMETERS                  |
    # .______________________________________________.

    NN = 80
    #-----------------------------------------------
    N_u = NN     # No. of training points
    N_f = NN*NN  # No. of collocation points
    #-----------------------------------------------
    Nx = NN+1    # No. of points in the x-direction (reference meshgrid)
    Ny = NN+1    # No. of points in the y-direction (reference meshgrid)
    Ns = NN      # No. of divisions in the interface
    #-----------------------------------------------
    Npx = NN+1   # No. of points in the x-direction (predicted meshgrid)
    Npy = NN+1   # No. of points in the y-direction (predicted meshgrid) 
    #-----------------------------------------------
    Ne = 20      # No. of neurons
    layers = [2,Ne,Ne,Ne,1]

    print('  ')
    print('---------------------------------------------')
    print('              ELLIPTIC PROBLEM               ')
    print('---------------------------------------------')
    print('  TRAINING:                                  ')
    print('  Boundary training points     N_u =',N_u)
    print('  Interior collocation points  N_f =',N_f)
    print('---------------------------------------------')
    print('  NEURAL NETWORK:                            ')
    print('  layers =',layers)
    print('---------------------------------------------')
    print('  PREDICTED MESH GRID:',Nx*Ny)
    print('  Grid points in the x-direction   Nx = ',Nx)
    print('  Grid points in the y-direction   Ny = ',Ny)
    print('---------------------------------------------')
    print('  INTERFACE:                                 ')
    print('  Points at the interface          Ns = ',Ns)
    print('---------------------------------------------')
    print('  ')
    
    # .----------------------------------------------.
    # |                  MESH: 2D domain             |
    # .______________________________________________.

    #--------------------------------
    # 2D Domain
    xIni = -1.0
    xFin =  1.0
    yIni = -1.0
    yFin =  1.0
    #----------
    # Meshgrid for boundary training
    x = np.linspace(xIni,xFin,num=Nx)
    y = np.linspace(yIni,yFin,num=Ny)
    X,Y = np.meshgrid(x,y) 
    #----------
    # Meshgrid for prediction
    xp = np.linspace(xIni,xFin,num=Npx)
    yp = np.linspace(yIni,yFin,num=Npy)    
    XP,YP = np.meshgrid(xp,yp)    

    # .----------------------------------------------.
    # |          Interface & Jump condition          |
    # .______________________________________________.
    
    r0 = 0.5
    th = np.linspace(0,2*np.pi,num=Ns+1) # theta
    #----------
    xs  = np.zeros((Ns,1),float)
    ys  = np.zeros((Ns,1),float)
    J1  = np.zeros((Ns,1),float)
    for s in range(0,Ns):
        xs[s] = r0*np.cos(th[s])
        ys[s] = r0*np.sin(th[s])
        J1[s] = 2.0
    #----------
    ds = r0*2*np.pi/Ns
    
    # .----------------------------------------------.
    # |             EXACT SOLUTION: Exact            |
    # .______________________________________________.
    
    #--------------------------------
    Exact = np.zeros((Ny,Nx),float)
    for j in range(0,Ny):
        for i in range(0,Nx):
            rr = np.sqrt(x[i]**2+y[j]**2)
            if rr < r0:
               Exact[j,i] = 1.0
            else:
               Exact[j,i] = 1.0 + np.log(2.0*rr)
    #--------------------------------
    Exactp = np.zeros((Npy,Npx),float)
    for j in range(0,Npy):
        for i in range(0,Npx):
            rr = np.sqrt(xp[i]**2+yp[j]**2)
            if rr < r0:
               Exactp[j,i] = 1.0
            else:
               Exactp[j,i] = 1.0 + np.log(2.0*rr)

    # .----------------------------------------------.
    # |      MESH, EXACT & BOUNDS (row format)       |
    # .______________________________________________. 
    
    #----------------
    # Meshgrid
    X_star  = np.hstack((X.flatten()[:,None],Y.flatten()[:,None]))
    Xp_star = np.hstack((XP.flatten()[:,None],YP.flatten()[:,None]))
    #----------------
    # Exact solution 
    u_star  = Exact.flatten()[:,None]
    up_star = Exactp.flatten()[:,None]
    #----------------
    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    # .----------------------------------------------.
    # |        POINTS: TRAINING & COLLOCATION        |
    # .______________________________________________.
    
    #-----------------------------------------------
    # [1] Meshgrid at the boundary (to extract training data)        
    xx1 = np.hstack((X[0:1,:].T, Y[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], Y[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], Y[:,-1:]))
    uu3 = Exact[:,-1:]
    xx4 = np.hstack((X[-1:,:].T, Y[-1:,:].T))
    uu4 = Exact[-1:,:].T
    #----------------
    X_bdy = np.vstack([xx1,xx2,xx3,xx4])
    u_bdy = np.vstack([uu1,uu2,uu3,uu4])
    N_bdy = len(X_bdy) # Total number of boundary points

    #-----------------------------------------------
    # [2] # Collocation points: (X_f_train,yf_j)
    #----------------    
    # Random 2D points [Latin-hiper-cube]:
    X_f = lb + (ub-lb)*lhs(2,N_f)
    X_f_train = np.vstack(X_f)
    #----------------
    xf = X_f_train[:,0:1]
    yf = X_f_train[:,1:2]      
    Nf = len(xf) # Total number of collocation points
    
    #-----------------------------------------------
    # [3] Training boundary data: (X_u_train,u_train)
    idx = np.random.choice(X_bdy.shape[0],N_u,replace=False)
    X_u_train = X_bdy[idx,:]
    u_train   = u_bdy[idx,:]
    #----------------
    xu = X_u_train[:,0]
    yu = X_u_train[:,1]

    # .----------------------------------------------.
    # |          RIGHT-HAND SIDE: rhs(Nf)            |
    # |               Perskin's method               |
    # .______________________________________________. 

    #---------------------------
    # h value for delta
    h = ds
    #---------------------------
    rhs = np.zeros((Nf,1),float)
    minf = 100000.0
    for l in range(0,Nf):
        #____________________
        som1 = 0.0
        for s in range(0,Ns):
            #----------------
            xk = xf[l]-xs[s]
            yk = yf[l]-ys[s]
            #----------------
            dd = np.sqrt(xk**2+yk**2)
            minf = min(minf,dd)
            #----------------
            if np.abs(xk) < 2*h: 
               deltax  = (1.0 + np.cos(np.pi*xk/(2.0*h)))/(4.0*h)
            else:
               deltax  = 0.0
            if np.abs(yk) < 2*h: 
               deltay  = (1.0 + np.cos(np.pi*yk/(2.0*h)))/(4.0*h)
            else:
               deltay  = 0.0
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
        
    #-----------------------------------------------
    # Solution u (1D row formulation)
    up_pred = model.predict(Xp_star)                  
    #-----------------------------------------------
    # Solution u (2D matrix formulation)
    UP_pred = griddata(Xp_star,up_pred.flatten(),(XP,YP),method='cubic')
        
    # .----------------------------------------------.
    # |          NORMS: L2 (relative) & Max          |
    # .______________________________________________.
          
    error_u   = np.linalg.norm(up_star-up_pred,2)/np.linalg.norm(up_star,2)
    error_max = np.linalg.norm(up_star-up_pred,np.inf)   

    # .================================================================.
    # | [S.3]                  DISPLAYING                              |
    # .================================================================.
    
    print('---------------------------------------------')
    print('  TRAINING:                                  ')
    print('  Number of random training points  N_u =',N_u)  
    print('  Number of collocation points      N_f =',Nf)
    print('---------------------------------------------')
    print('  NEURAL NETWORK:                            ')
    print('  layers =',layers)
    print('---------------------------------------------')
    print('  PREDICTED MESH GRID:',Nx*Ny)
    print('  Points in the x-direction         Nx  =',Nx)
    print('  Points in the y-direction         Ny  =',Ny)
    print('---------------------------------------------')
    print('  INTERFACE:                                 ')
    print('  Number of interface points        N_s =',Ns)
    print('  -----------------')
    print('  Perskin method with h =', h)
    print('  Minimum distance xf to the interface =', minf)
    print('---------------------------------------------')
    print('   N_u    N_f    L2-norm         Max-norm' )
    print('  %3s   %3s   %e    %e' % (N_u,N_f,error_u,error_max))
    print('---------------------------------------------')            
    print('   Training time: %.4f' % (elapsed))
    print('---------------------------------------------') 
    print('  ')

    # .================================================================.
    # | [S.4]                    SAVE DATA                             |
    # .================================================================.
     
    # -------------------------------
    # Final results    
    file1 = open("output/dataR/Results_Ex4.txt","w")
    file1.write('-------------------------------------------------\n')
    file1.write('Example 3: DNN + IBM\n')
    file1.write('-------------------------------------------------\n')
    file1.write('Nu = %s\n' % N_u)
    file1.write('Nf = %s '  % Nf )
    file1.write('layers = %s\n' % layers)
    file1.write('Points at the interface Ns = %s\n' % Ns)
    file1.write('Perskin method with h = %f\n' % h)
    file1.write('L2-norm  error: %.3e\n' % error_u)
    file1.write('Max-norm error: %.3e\n' % error_max)
    file1.write('Training time : %.4f (sec)\n' % elapsed)
    file1.write('Iterations    : %s\n' % Nfeval)
    file1.write('Loss          : %f\n' % Nloss)
    file1.write('Predicted points in the x-direction: Nx = %s\n' % Npx)
    file1.write('Predicted points in the x-direction: Ny = %s\n' % Npy)
    file1.write('-------------------------------------------------\n')
    file1.close()
 
    # -------------------------------
    # Training data (xu,yu)
    file2 = open("output/dataR/xu.txt","w")
    for l in range(0,N_u):
        file2.write('%f  ' % xu[l])
        file2.write('%f  ' % yu[l])
        file2.write('%f\n' % u_train[l])
    file2.close()
        
    # -------------------------------
    # Training data (xf,yf)
    file3 = open("output/dataR/xf.txt","w")
    for l in range(0,Nf):
        file3.write('%f  ' % xf[l])
        file3.write('%f  ' % yf[l])
        file3.write('%f\n' % rhs[l])
    file3.close()

    # -------------------------------
    # Predicted data Grid(x,y)
    file4 = open("output/dataR/xp_x.txt","w")
    for l in range(0,Npx):
        file4.write('%f  ' % xp[l])
    file4.close()
    #------------
    file4 = open("output/dataR/xp_y.txt","w")
    for l in range(0,Npy):
        file4.write('%f\n' % yp[l])
    file4.close()
    
    # -------------------------------
    # Analytical solution uA(x,y)
    file5 = open("output/dataR/xp_uA.txt","w")
    for j in range(0,Npy):
        for i in range(0,Npx):
            file5.write('%f  ' % Exactp[j,i])
        file5.write('\n')    
    file5.close()
    
    # -------------------------------
    # Numerical solution uN(x,y)
    file6 = open("output/dataR/xp_uN.txt","w")
    for j in range(0,Npy):
        for i in range(0,Npx):
            file6.write('%f  ' % UP_pred[j,i])
        file6.write('\n')                       
    file6.close()
    # -------------------------------
    # Interface
    file7 = open("output/dataR/xs.txt","w")
    for s in range(0,Ns):
        file7.write('%f  ' % xs[s])
        file7.write('%f\n' % ys[s])
    file7.close()