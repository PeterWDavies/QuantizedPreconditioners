
import argparse
import os
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import libsvm
import math
from libsvm.svmutil import *
from quantization import QSGD, LQSGD, HadamardQuantizer, Noquant
from scipy.special import expit

def compressall(quantizer, gb): # compresses a numpy array of batch gradients
    N, d = gb.shape
    ans = np.zeros((N,d))
    for i in range(N):
        ans[i] = quantizer.compress(gb[i])
    return ans

def preconquant(Mi, minbits):  #quantize local approximate preconditioners, decode at root node, add, quantize, and broadcast
    n, d = Mi[0].shape
    N = len(Mi)
    numbits = 0
    quantsum=np.zeros((n,d))
    for i in range(N-1):    #simulates the process of node i quantizing its preconditioner, and root node N-1 decoding
        fmin = 9e50

        for j in range(n):
            for k in range(d):
                if (Mi[i][j][k] != 0):
                    fmin  = min(fmin, abs(Mi[i][j][k]))
        sidelength = fmin/2 # heuristic for choosing side length based on smallest non-zero absolute value in matrix
        Milastcoords = np.round(Mi[N-1]/sidelength).astype(int)
        coords = np.round(Mi[i]/sidelength).astype(int)

        diff = coords - Milastcoords

        diffmod = None
        shiftmod= None
        for bits in range(minbits, minbits+100):    #using error detection (full version of Davies et al. 2021) to adaptively choose #bits
            diffmod =  diff%(2**bits)
            shiftmod =  (diffmod+ 2**(bits-2))%(2**bits)
            if (shiftmod.max() <= 2**(bits-1)):     #detects if decode if successful by distance criterion
                numbits += bits
                quantized = Mi[N-1]+ sidelength*(shiftmod - 2**(bits-2))
                quantsum += quantized
                break
    quantsum += Mi[N-1]
    fmin = 9e50

    for j in range(n): 
        for k in range(d):
              if (Mi[i][j][k] != 0):
                   fmin  = min(fmin, abs(Mi[i][j][k]))
    sidelength = fmin/2     #setting side length for return broadcast
    quantsumcoords = np.round(quantsum/sidelength)
    for i in range(N-1):     #root node broadcasts sum
        coords = np.round(N*Mi[i]/sidelength)
        diff = coords - Milastcoords
        for bits in range(minbits, minbits+100):
            diffmod =  diff%(2**bits)
            shiftmod =  (diffmod+ 2**(bits-2))%(2**bits)
            if (shiftmod.max() <= 2**(bits-1)):
                numbits += bits
                break
    return quantsumcoords*sidelength, numbits/(2*(N-1))
            
               
def numpy_dataset(y, x): # return the dataset in numpy array form
    n = len(x)
    d=0
    for i in range(n):
        for j in zip(x[i].keys()):
            d= max(max(j)+1,d)
    A = np.zeros((n,d))
    for i in range(n):
        for j,k in zip(x[i].keys(),x[i].values()):
            A[i][j] = k
    b = np.array(y).reshape(n,1)
    A, b = A.astype('float32'), b.astype('float32')
    idx = np.argwhere(np.all(A[..., :] == 0, axis=0))
    A = np.delete(A, idx, axis=1)

    return A, b

def hessian(A,w,b): #computes Hessian of A at point w, under logistic regression cost function with labels b
    logit = np.dot(A, w)
    pred_prob = expit(logit)
    data_weighted = np.array(pred_prob * (1. - pred_prob) * A, dtype=np.float32)
    hessian = np.dot(A.T, data_weighted )
    return hessian

def grad(A,w,b):    #computes gradient of A at point w, under logistic regression cost function with labels b
    logit = np.dot(A, w)
    pred_prob = expit(logit)
    grad = np.dot((pred_prob - b).T , A)
    return grad

def loss(A,w,b):    #computes cost at point w, under logistic regression cost function with labels b
    logit = np.dot(A, w)
    pred_prob = expit(logit)
    loss = -(np.dot(b.T, np.log(pred_prob))+ np.dot((1-b).T, np.log(1-pred_prob)))
    return loss[0][0]

def matrixquant(A,quantizer):   #quantizes symmetric matrix A using input quantizer
    d=A.shape[0]
    Avec = A[0,:]
    for i in range(1,d):
        Avec = np.concatenate((Avec,A[i,i:d]))
    if(quantizer == "full"):
        qAvec = Avec
    else:
        qAvec =  quantizer.compress(Avec)

    qA = np.empty((d,d))
    si=0
    ei=d
    for i in range(0,d):    #only quantizes lower triangular due to symmetry
        qA[i,i:d]=qAvec[si:ei]
        si=ei
        ei = ei+(d-i-1)
    i_lower = np.tril_indices(d, -1)
    qA[i_lower] = qA.T[i_lower]
    return qA

def experiment(A,b,N,n,d,qlevel,Pbits,batch_size,iterations,seed):
    np.random.seed(seed)
    
    # print to stdout
    print("-----")
    print("0. seed = ", seed)
    print("1. data points (S) = {}, d = {}, workers (n) = {}".format(n,d,N))
    print("2. qlevel = ", qlevel)
    print("-----")


    M =A.T.dot(A)
    if(LA.matrix_rank(M)== d):
        Minv = np.linalg.inv(M)
    else:
        Minv = np.eye(d)
 
    # quantizer setup
    qsgd = QSGD(k = qlevel)
    noquant = Noquant(dimension = d, qlevel = qlevel)

    # arrays for plotting
    best = {("GD","QSGD"):[],("GLM","QSGD"):[],("Newton","QSGD"):[],("GD","full_gradient"):[],("GLM","full_gradient"):[],("Newton","full_gradient"):[]} 
    ler = {("GD","QSGD"):[],("GLM","QSGD"):[],("Newton","QSGD"):[],("GD","full_gradient"):[],("GLM","full_gradient"):[],("Newton","full_gradient"):[]} 
    cost = {("GD","QSGD"):[],("GLM","QSGD"):[],("Newton","QSGD"):[],("GD","full_gradient"):[],("GLM","full_gradient"):[],("Newton","full_gradient"):[]} 
    for quantizer in ["QSGD", "full_gradient"]: 
            for precon in ["GLM", "GD", "Newton"]:
                best[(precon, quantizer)] = [np.nan]    #initialises costs
    indices = np.arange(n)
    np.random.shuffle(indices)
    Mi=[]
    for i in range(N):      #divides data amont nodes
        Ai = A[indices[i*batch_size:(i+1)*batch_size]]
        Mi.append(Ai.T.dot(Ai))
        
    Mbar, bits = preconquant(Mi, Pbits)     
    
    if(LA.matrix_rank(Mbar)== d):   #if Mbar invertible, computes preconditioner Mbarinv; otherwise, uses indentity matrix (i.e. no preconditioning)
        Minvbar = np.linalg.inv(Mbar)
    else:
        Minvbar = np.eye(d)
        
    for j in range(50):     #iterates through exponentially decreasing learning rate
        cost = {("GD","QSGD"):[],("GLM","QSGD"):[],("Newton","QSGD"):[],("GD","full_gradient"):[],("GLM","full_gradient"):[],("Newton","full_gradient"):[]} 
        lr = 2**(-j)
        print('lr = {}'.format(lr))
        for quantizer in [noquant, qsgd]:
              
            for precon in ["GLM", "GD"]:
                name = quantizer.name
                
                # intitialize weight
                w = np.zeros((d,1))         
    
        
                prevgrad = np.zeros((N,d))
                prevavg = np.zeros(d)
                if(j>0):    #test if fastest stable convergence has already occurred (to save time)
                    if( not math.isnan(sum(best[(precon,name)]) ) and ler[(precon,name)]< j-2 and ler[(precon,name)]>= 0):
                        continue
                for it in range(iterations):
                    gb = np.zeros((N,d))
                    for i in range(N):  # get gradients
                        gb[i] = grad(A[indices[i*batch_size:(i+1)*batch_size]],w,b[indices[i*batch_size:(i+1)*batch_size]]).flatten()     
        
                    gb -= prevgrad  #quantize and sum gradient differences
                    compressed = compressall(quantizer,gb)
                    compressed += prevgrad
                    prevgrad = compressed
                    
                    avg = np.zeros(d)
                    for i in range(N):
                        avg += compressed[i]
                    avg /= N
                    avg -=prevavg
                    if(N > 2):
                        avg = quantizer.compress(avg)  
                    gradient = avg + prevavg    #find gradient for this step by adding average difference to last round's
                    prevavg=gradient
 
                    if (precon == "GLM"):
                        if(quantizer!=noquant): gradient = Minvbar.dot(gradient) #apply quantized preconditioner
                        else:
                            gradient = Minv.dot(gradient)   #apply full preconditioner
                    
                    
                    # calculate loss (for last iteration)
                    Loss = loss(A,w,b)
                    cost[(precon,name)].append(Loss)
                    if math.isnan(Loss):
                        Loss = 9e50
        
                    # take step
                    w -= lr*gradient.reshape(d,1) 
        
                if len(best[(precon,name)]) != iterations:  #if no prior complete set of results, use this one
                    best[(precon,name)] = cost[(precon,name)]
                    ler[(precon,name)] = j
                    
                    #if this set has lower total cost, or previous is NaN, also use this one
                elif sum(cost[(precon,name)]) < sum(best[(precon,name)]) or math.isnan(sum(best[(precon,name)]) ):
                    best[(precon,name)] = cost[(precon,name)]
                    ler[(precon,name)] = j   

                
            for precon in ["Newton"]:
                name = quantizer.name

                if(j>0):
                    if( not math.isnan(sum(best[(precon,name)]) ) and ler[(precon,name)]< j-2 and ler[(precon,name)]>= 0):
                        continue
                # intitialize weight
                w = np.zeros((d,1))         
    
    
                prevhess = np.zeros((N,d,d))
                for i in range(N):
                        prevhess[i] = np.zeros((d,d))  
        
                prevgrad = np.zeros((N,d))
                prevavg = np.zeros(d)
                
                hess = np.zeros((N,d,d))
                if(quantizer == noquant):
                    for i in range(N):
                        hess[i] = hessian(A,w,b)
                else:
                     for i in range(N):
                        hess[i] = Mbar/2            #initialize Hessian estimate using quantized preconditioner           
                hessdiff = np.zeros((N,d,d))
                quanthess = np.zeros((N,d,d))
                for it in range(iterations):
                    gb = np.zeros((N,d))
                    hesstotal = np.zeros((d,d))
                    invhess = np.eye(d)
                    
                    for i in range(N):
                        gb[i] = grad(A[indices[i*batch_size:(i+1)*batch_size]],w,b[indices[i*batch_size:(i+1)*batch_size]]).flatten() 
                    

                    for i in range(N):  #quantize, sum, and broadcast local Hessian updates
                        hess[i] = hessian(A[indices[i*batch_size:(i+1)*batch_size]],w,b[indices[i*batch_size:(i+1)*batch_size]])
                        hessdiff[i] = hess[i]-prevhess[i]
                        if(name != "full_gradient"):
                            quanthess[i] = matrixquant(hessdiff[i],quantizer)
                        else:
                            quanthess[i] = hessdiff[i]
                        hess[i]= prevhess[i]+quanthess[i]
                        prevhess[i] = prevhess[i]+quanthess[i]
                        hesstotal += hess[i]
                    if(LA.matrix_rank(hesstotal)== d):  #compute preconditioner
                        invhess = np.linalg.inv(hesstotal)
                    else:
                        invhess = np.eye(d)
                        print("Hessian is singular")
                        
                    gb -= prevgrad
                    compressed = compressall(quantizer,gb)
                    compressed += prevgrad
                    prevgrad = compressed
                    
                    # getting gradient, allow other methods to use full gradient in first step
                    avg = np.zeros(d)
                    for i in range(N):
                        avg += compressed[i]
                    avg /= N
                    avg -=prevavg
                    if(N > 2):
                        avg = quantizer.compress(avg)  
                    gradient = avg + prevavg
                   
                    prevavg=gradient
                    gradient = invhess.dot(gradient) #precondition
                    
                    # ||w-w_star||, loss
                    Loss = loss(A,w,b)
                    cost[(precon,name)].append(Loss)

                    # take step
                    w -= lr*gradient.reshape(d,1) 
        

                if len(best[(precon,name)]) != iterations:
                    best[(precon,name)] = cost[(precon,name)]
                    ler[(precon,name)] = j
                if sum(cost[(precon,name)]) < sum(best[(precon,name)]) or math.isnan(best[(precon,name)][iterations-1]):
                    best[(precon,name)] = cost[(precon,name)]
                    ler[(precon,name)] = j    
   
    return best,ler, bits

def main():
    parser = argparse.ArgumentParser(description='Superlinear Convergence for general number of workers')
    parser.add_argument('--bits', type=int, default= 8, metavar='QLEVEL', help='bits (default: 4)')
    parser.add_argument('--Pbits', type=int, default=12, metavar='QLEVEL', help='bits (default: 4)')
    parser.add_argument('--iterations', type=int, default=100, metavar='ITER', help='iterations (default: 25)')
    parser.add_argument('--dataset', type=str, default='german_numer.txt', metavar='DATASET', help='dataset (default: cpusmall_scale)')
    parser.add_argument('--workers', type=int, default=5, metavar='WORKERS', help='workers (default: 2)')
    parser.add_argument('--nseeds', type=int, default=5, metavar='nseeds', help='number of seeds (default: 1)')
    parser.add_argument('--save', default=False,action='store_true')
    
    #Pbits is precon bits.
    
    
    args = parser.parse_args()
    # dataset
    bits = args.bits
    Pbits = args.Pbits
    qlevel = 2** bits
    y, x = svm_read_problem(args.dataset)
    A, b = numpy_dataset(y,x)
    bmin = min(b)
    bmax = max(b)
    b= (b-bmin)/(bmax-bmin) #rescale labels to {0,1}
    A =A.astype(np.float32)
    n, d = A.shape
    N = args.workers # number of worker nodes
    batch_size = int(n/N) # batches you want to use for the comparison
    iterations = args.iterations
    
    # repeating the experiment for five seeds
    quants = ["QSGD","full_gradient"]
    precons = ["GD", "GLM", "Newton"]
    seeds = list(range(args.nseeds))
    pbitavg = 0
    bestavg = {("GD","QSGD"):[],("GLM","QSGD"):[],("Newton","QSGD"):[],("GD","full_gradient"):[],("GLM","full_gradient"):[],("Newton","full_gradient"):[]}
    
    for quantizer in quants:
        for precon in precons:
            bestavg[(precon, quantizer)] = np.array([0.0]*iterations)

    for seed in seeds:
        best,ler, pbits = experiment(A,b,N,n,d,qlevel,Pbits,batch_size,iterations,seed)
        pbitavg += pbits
        for quantizer in quants:
            for precon in precons:
                bestavg[(precon, quantizer)] += best[(precon, quantizer)]
   
    pbitavg = np.round(pbitavg/args.nseeds,1)
    for quantizer in ["QSGD", "full_gradient"]:
        for precon in ["GLM", "GD", "Newton"]:
            bestavg[(precon, quantizer)] = np.round(np.array(bestavg[(precon, quantizer)])/args.nseeds,1)

    ''' Plots '''
    iteration = np.array(range(iterations))
    start_iter, end_iter = max(0,iterations-400), iterations
    plt.plot(iteration[start_iter:end_iter],bestavg[("GD","QSGD")][start_iter:end_iter],label="QSGD")
    plt.plot(iteration[start_iter:end_iter],bestavg[("GLM","QSGD")][start_iter:end_iter],label="QPGD-GLM")

    plt.plot(iteration[start_iter:end_iter],bestavg[("GD","full_gradient")][start_iter:end_iter],label="Full-GD",linestyle='dashed')
    plt.plot(iteration[start_iter:end_iter],bestavg[("Newton","QSGD")][start_iter:end_iter],label="Q-Newton")
    plt.plot(iteration[start_iter:end_iter],bestavg[("Newton","full_gradient")][start_iter:end_iter],label="Full-Newton",linestyle='dashed')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Results on the "german_numer" dataset \n samples = {}, nodes = {}, dim = {}\n grad/Hessian bits = {}, precon. bits = {}'.format(n,N,d,bits, pbitavg))
    plt.legend()

    print(ler)  #indices of final learning rate used by each method
    if not os.path.isdir('out'):
        os.makedirs('out')
    plt.savefig('out/{}_n_{}_qb_{}.pdf'.format(args.dataset,args.workers,args.bits), bbox_inches='tight')  
    plt.show()
    plt.close()
    

if __name__ == '__main__':
    main()
