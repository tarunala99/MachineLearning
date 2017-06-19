import numpy as np
import struct
import math

def parse_images(filename):
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    sx,sy = struct.unpack('>ii', f.read(8))
    X = []
    for i in range(size):
        im =  struct.unpack('B'*(sx*sy), f.read(sx*sy))
        X.append([float(x)/255.0 for x in im]);
    return np.array(X);

def parse_labels(filename):
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], 
                                    dtype=np.float64)
    f = open(filename,"rb");
    magic,size = struct.unpack('>ii', f.read(8))
    return one_hot(np.array(struct.unpack('B'*size, f.read(size))), 10)

def error(y_hat,y):
    return float(np.sum(np.argmax(y_hat,axis=1) != 
                        np.argmax(y,axis=1)))/y.shape[0]



# function calls to load data (uncomment to load MINST data)



# helper functions for loss and neural network activations
softmax_loss = lambda yp,y : (np.log(np.sum(np.exp(yp))) - yp.dot(y), 
                              np.exp(yp)/np.sum(np.exp(yp)) - y)
f_tanh = lambda x : (np.tanh(x), 1./np.cosh(x)**2)
f_relu = lambda x : (np.maximum(0,x), (x>=0).astype(np.float64))
f_lin = lambda x : (x, np.ones(x.shape))


# set up a simple deep neural network for MNIST task


##### Implement the functions below this point ######

def softmax_gd(X, y, Xt, yt, epochs=10, alpha = 0.5):
    """ 
    Run gradient descent to solve linear softmax regression.
    
    Inputs:
        X: numpy array of training inputs
        y: numpy array of training outputs
        Xt: numpy array of testing inputs
        yt: numpy array of testing outputs
        epochs: number of passes to make over the whole training set
        alpha: step size
        
    Outputs:
        Theta: 10 x 785 numpy array of trained weights
    """
    theta=np.random.randn(10,785) ##make them more compaitable
    a,b=theta.shape
    count1=0
    for count1 in range(0,a):
        count2=0
        for count2 in range(0,b):
            theta[count1,count2]=0
            count2=count2+1
        count1=count1+1
    
    a,b=X.shape
    a1,b1=Xt.shape
    count=0
    for t in range(0,epochs):
        g=0
        yui,opi=y.shape
    	yten=np.random.randn(yui,opi) 
    	count1=0
    	for count1 in range(0,yui):
        	count2=0
        	for count2 in range(0,opi):
            		yten[count1,count2]=0
            		count2=count2+1
        	count1=count1+1
        count100=0
        for i in range(1,a+1):
            X1=np.concatenate((X[i-1],np.array([1])), axis=0)
            ylaw=np.inner(theta,X1)
            count100=count100+(softmax_loss(ylaw,y[i-1]))[0]
            yten[i-1]=ylaw
        print str(error(yten,y))+"error training"
        print str(count100/a)+"loss training"
        yui,opi=yt.shape
    	yten=np.random.randn(yui,opi) 
    	count1=0
    	for count1 in range(0,yui):
        	count2=0
        	for count2 in range(0,opi):
            		yten[count1,count2]=0
            		count2=count2+1
        	count1=count1+1
        count100=0
        for i in range(1,a1+1):
            X1=np.concatenate((Xt[i-1],np.array([1])), axis=0)
            ylaw=np.inner(theta,X1)
            count100=count100+(softmax_loss(ylaw,yt[i-1]))[0]
            yten[i-1]=ylaw
        print str(error(yten,yt))+"error testing"
        print str(count100/yui)+"loss testing"
        for i in range(1,a+1):
            X1=np.concatenate((X[i-1],np.array([1])), axis=0)
            ylaw=np.inner(theta,X1)
            dca=np.random.randn(ylaw.shape[0])
            count3=0
            count4=0
            for count3 in range(0,ylaw.shape[0]):
                count4=math.exp(ylaw[count3])+count4
            count5=0
            for count5 in range(0,ylaw.shape[0]):
                o=math.exp(ylaw[count5])
                q=o/count4
                dca[count5]=q-y[i-1,count5]
            X2=np.transpose(X1)
            grad=np.outer(dca,X2)
            g=grad/a+g
        theta=theta-(alpha*g)
    return theta

def softmax_sgd(X,y, Xt, yt, epochs=10, alpha = 0.01):
    """ 
    Run stoachstic gradient descent to solve linear softmax regression.
    
    Inputs:
        X: numpy array of training inputs
        y: numpy array of training outputs
        Xt: numpy array of testing inputs
        yt: numpy array of testing outputs
        epochs: number of passes to make over the whole training set
        alpha: step size
        
    Outputs:
        Theta: 10 x 785 numpy array of trained weights
    """
    theta=np.random.randn(10,785) ##make them more compaitable
    a,b=theta.shape
    count1=0
    for count1 in range(0,a):
        count2=0
        for count2 in range(0,b):
            theta[count1,count2]=0
            count2=count2+1
        count1=count1+1
    
    a,b=X.shape
    a1,b1=Xt.shape
    count=0
    for t in range(0,epochs):
        yui,opi=y.shape
    	yten=np.random.randn(yui,opi) 
    	count1=0
    	for count1 in range(0,yui):
        	count2=0
        	for count2 in range(0,opi):
            		yten[count1,count2]=0
            		count2=count2+1
        	count1=count1+1
        count100=0
        for i in range(1,a+1):
            X1=np.concatenate((X[i-1],np.array([1])), axis=0)
            ylaw=np.inner(theta,X1)
            count100=count100+(softmax_loss(ylaw,y[i-1]))[0]
            yten[i-1]=ylaw
        print str(error(yten,y))+"error training"
        print str(count100/a)+"loss training"
        yui,opi=yt.shape
    	yten=np.random.randn(yui,opi) 
    	count1=0
    	for count1 in range(0,yui):
        	count2=0
        	for count2 in range(0,opi):
            		yten[count1,count2]=0
            		count2=count2+1
        	count1=count1+1
        count100=0
        for i in range(1,a1+1):
            X1=np.concatenate((Xt[i-1],np.array([1])), axis=0)
            ylaw=np.inner(theta,X1)
            count100=count100+(softmax_loss(ylaw,yt[i-1]))[0]
            yten[i-1]=ylaw
        print str(error(yten,yt))+"error testing"
        print str(count100/yui)+"loss testing"
        for i in range(1,a+1):
            X1=np.concatenate((X[i-1],np.array([1])), axis=0)
            ylaw=np.inner(theta,X1)
            dca=np.random.randn(ylaw.shape[0])
            count3=0
            count4=0
            for count3 in range(0,ylaw.shape[0]):
                count4=math.exp(ylaw[count3])+count4
            count5=0
            for count5 in range(0,ylaw.shape[0]):
                o=math.exp(ylaw[count5])
                q=o/count4
                dca[count5]=q-y[i-1,count5]
            X2=np.transpose(X1)
            grad=np.outer(dca,X2)
            theta=theta-alpha*grad
    return theta


def nn(x, W, b, f):

    """
    Compute output of a neural network.
    
    Input:
        x: numpy array of input
        W: list of numpy arrays for W parameters
        b: list of numpy arraos for b parameters
        f: list of activation functions for each layer
        
    Output:
        z: list of activationsn, where each element in the list is a tuple:
           (z_i, z'_i)
           for z_i and z'_i each being a numpy array of activations/derivatives
    """
    a=len(f)
    list=[]
    for c in range(0,a):
    	y=W[c].dot(x)+b[c]
    	z=(f[c](y))[0]
    	zp=(f[c](y))[1]
    	list.append(f[c](y))
    	x=z
    return list ## the function returns many values. Which values are to be considered ???????
    ## the second part is the gradient of the value 
	



def nn_loss(x, y, W, b, f):
    """
    Compute loss of a neural net prediction, plus gradients of parameters
    
    Input:
        x: numpy array of input
        y: numpy array of output
        W: list of numpy arrays for W parameters
        b: list of numpy arrays for b parameters
        f: list of activation functions for each layer
        
    Output tuple: (L, dW, db)
        L: softmax loss on this example
        dW: list of numpy arrays for gradients of W parameters
        db: list of numpy arrays for gradients of b parameters
    """
    list=nn(x, W, b, f)
    ylaw=(list[len(list)-1])[0]
    taru=softmax_loss(ylaw,y)
    loss=taru[0]
    listg=[]
    listg.append(taru[1])
    listb=[]
    listw=[]
    i=len(list)
    while(i>0):
        t=listg[len(listg)-1]*list[i-1][1]
        listb.append(t)
        c=(np.transpose(W[i-1])).dot(t)
        listg.append(c)
        if(i==1):
            k=x
        else:
            k=list[i-2][0]
        d=np.outer(t,k) 
        listw.append(d)
        i=i-1
    listg.reverse()
    listb.reverse()
    listw.reverse()
    return (loss,listb,listw)


            
def nn_sgd(X,y, Xt, yt, W, b, f, epochs=10, alpha = 0.01):
    """ 
    
    Run stoachstic gradient descent to solve linear softmax regression.
    
    Inputs:
        X: numpy array of training inputs
        y: numpy array of training outputs
        Xt: numpy array of testing inputs
        yt: numpy array of testing outputs
        W: list of W parameters (with initial values)
        b: list of b parameters (with initial values)
        f: list of activation functions
        epochs: number of passes to make over the whole training set
        alpha: step size
        
    Output: None (you can directly update the W and b inputs in place)
    """
    a,c=X.shape
    ren,ten=Xt.shape
    print a
    for r in range(0,epochs):   
        print r
        count=0
        for d in range(1,a+1):
            s,t,u=nn_loss(X[d-1],y[d-1],W,b,f)
            count=count+s
            for q in range(0,len(t)):
                W[q]=W[q]-alpha*u[q]
                b[q]=b[q]-alpha*t[q]
        print count/a
        count=0
        for d in range(1,ren+1):
            s,t,u=nn_loss(Xt[d-1],yt[d-1],W,b,f)
            count=count+s
        print count/ren
    return W,b


#X_train = parse_images("train-images-idx3-ubyte")
#y_train = parse_labels("train-labels-idx1-ubyte")
#X_test = parse_images("t10k-images-idx3-ubyte")
#y_test = parse_labels("t10k-labels-idx1-ubyte")
#print softmax_sgd(X_train,y_train,X_test,y_test,10,0.5)
np.random.seed(0)
layer_sizes = [784, 200, 100, 10]
W = [0.1*np.random.randn(n,m) for m,n in zip(layer_sizes[:-1], layer_sizes[1:])]
b = [0.1*np.random.randn(n) for n in layer_sizes[1:]]
f = [f_relu]*(len(layer_sizes)-2) + [f_lin]
#nn_sgd(X_train,y_train,X_test,y_test,W,b,f,epochs=5,alpha = 0.01)