# This code is mostly based on Matthieu Courbariaux's
# Binary Neural Network code, accessable from: 
# https://github.com/MatthieuCourbariaux/BinaryNet
# Jintao has modify the code so it is able to implemented on-chip.

from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T
from theano.tensor import _shared

import lasagne

import cPickle as pickle
import gzip

import binary_net

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict
import pdb
#theano.config.compute_test_value = 'warn'  # 'off' # Use 'warn' to activate this feature

def quantization (array,num_bits):# DAC Quantization
	# This quantization will limit input array in range [0, 1)
	
    max_num = 1.0 - 0.5**num_bits 
    min_num = 0
    num_levels = (2. ** num_bits)
    
    array_res = np.empty_like(array)
    array_res[:] = array  
    array_res = array_res.reshape(array_res.size,)

    levels = np.linspace(min_num,max_num,num_levels)
    cnt = 0    
    for i in np.nditer(array):
        tmp = np.abs(levels - i)
        index = (tmp == np.min(np.abs(levels - i)))        
        array_res[cnt] = levels[index]
        cnt = cnt + 1        
    return array_res.reshape((array.shape))


if __name__ == "__main__":
    
    # BN parameters
    batch_size = 100 # was 100
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # MLP parameters
    num_units = 96 # Jintao: was 4096
    print("num_units = "+str(num_units))
    n_hidden_layers = 3 # was 3
    print("n_hidden_layers = "+str(n_hidden_layers))
    
    # Training parameters
    num_epochs = 100 # was 1000
    print("num_epochs = "+str(num_epochs))
    
    # Dropout parameters
    dropout_in = 0. # 0. means no dropout, Jintao: was .2
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = 0. # Jintao : was .5
    print("dropout_hidden = "+str(dropout_hidden))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
	
    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Decaying LR 
    LR_start = .003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    #save_path = "mnist_parameters.npz"
    #save_path = "mnist_"
    save_path = None
    print("save_path = "+str(save_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading MNIST dataset...')
    
    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = False)
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = False)
    test_set = MNIST(which_set= 'test', center = False)
    
    # bc01 format    
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
	# Jintao make input from 0 to 1
	# Jintao change size to 10*10 (100) to fit in chip.
    #train_set.X = 2* train_set.X.reshape(-1, 1, 28, 28) - 1.
    #valid_set.X = 2* valid_set.X.reshape(-1, 1, 28, 28) - 1.
    #test_set.X = 2* test_set.X.reshape(-1, 1, 28, 28) - 1.
    train_set.X = train_set.X.reshape(-1, 1, 10, 10)
    train_set.X = quantization(train_set.X, 5)
    valid_set.X = valid_set.X.reshape(-1, 1, 10, 10)
    valid_set.X = quantization(valid_set.X, 5)	
    test_set.X = test_set.X.reshape(-1, 1, 10, 10) 
    test_set.X = quantization(test_set.X, 5)	

	# flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])    
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])
    
    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('Building the MLP...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    #input.tag.test_value = np.random.randn(3, 1, 10, 10).astype('float32')
    
    chip_input = T.matrix('chip_inputs')
    #chip_input.tag.test_value = np.random.randn(3, 96).astype('float32')
    
    chip_nonideal = T.matrix('chip_nonideals')
	
    target = T.matrix('targets')
    #target.tag.test_value = np.asarray([9, 0, 4]).astype('float32')	
    
    LR = T.scalar('LR', dtype=theano.config.floatX)
    #LR.tag.test_value = 0.001
	# Jintao: Input Layer dimension change
    #mlp = lasagne.layers.InputLayer(
    #        shape=(None, 1, 10, 10),
			#shape=(None, 1, 28, 28), 
    #       input_var=input)
            
    #mlp = lasagne.layers.DropoutLayer(
    #        mlp, 
    #        p=dropout_in)
    
    l1_in = lasagne.layers.InputLayer(shape=(None, 1, 10, 10)
		, input_var=input)
	#Jintao: tear-up the layers to get binarized weight/ output for testing. 
    l1_dl = binary_net.DenseLayer(l1_in,
		binary=binary,
		stochastic=stochastic,
		H=H,
		W_LR_scale=W_LR_scale,
		nonlinearity=lasagne.nonlinearities.identity,
		num_units=num_units)
								
    l1_bn = lasagne.layers.BatchNormLayer(l1_dl, 
	    epsilon=epsilon,
	    alpha=alpha)
		
    l1_nl = lasagne.layers.NonlinearityLayer(l1_bn,
		nonlinearity=activation)

    l2_dl = binary_net.DenseLayer(l1_nl,
		binary=binary,
		stochastic=stochastic,
		H=H,
		W_LR_scale=W_LR_scale,
		nonlinearity=lasagne.nonlinearities.identity,
		num_units=num_units)
								
    l2_bn = lasagne.layers.BatchNormLayer(l2_dl, 
		epsilon=epsilon,
		alpha=alpha)
	
    l2_nl = lasagne.layers.NonlinearityLayer(l2_bn,
		nonlinearity=activation)
		
    l3_dl = binary_net.DenseLayer(l2_nl,
		binary=binary,
		stochastic=stochastic,
		H=H,
		W_LR_scale=W_LR_scale,
		nonlinearity=lasagne.nonlinearities.identity,
		num_units=num_units)
								
    l3_bn = lasagne.layers.BatchNormLayer(l3_dl, 
		epsilon=epsilon,
		alpha=alpha)
	
    lnonideal = lasagne.layers.InputLayer(shape=(None, num_units), input_var = chip_nonideal)	
    l3_merge = lasagne.layers.ElemwiseMergeLayer([l3_bn, lnonideal], T.add)
	
    l3_nl = lasagne.layers.NonlinearityLayer(l3_merge,
		nonlinearity=activation)

    # attempt to add in an additional layer to inject numbers
    # assume activation = 0
    
    #l4 = lasagne.layers.InjectionLayer(l3_nl, filename="inputs/test.txt")
    lchip = lasagne.layers.InputLayer(shape=(None, num_units), input_var = chip_input)
    l4 = lasagne.layers.ElemwiseMergeLayer([l3_nl, lchip], T.add)
    #pdb.set_trace()
    #for k in range(n_hidden_layers):

        #mlp = binary_net.DenseLayer(
         #       mlp, 
         #       binary=binary,
         #       stochastic=stochastic,
         #       H=H,
         #       W_LR_scale=W_LR_scale,
         #       nonlinearity=lasagne.nonlinearities.identity,
         #       num_units=num_units)                  
        
        #mlp = lasagne.layers.BatchNormLayer(
         #       mlp,
         #       epsilon=epsilon, 
         #       alpha=alpha)

        #mlp = lasagne.layers.NonlinearityLayer(
         #       mlp,
         #       nonlinearity=activation)
                
        #mlp = lasagne.layers.DropoutLayer(
         #       mlp, 
         #       p=dropout_hidden)
    
    mlp = binary_net.DenseLayer(
                #mlp,
				l4,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)    
    
    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon, 
            alpha=alpha)
    #pdb.set_trace() 
	#When training, use stochastic approach
    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
    
		
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    #train_fn = theano.function([input, target, LR], loss, updates=updates)
    train_fn = theano.function([input, chip_input, chip_nonideal, target, LR], loss, updates=updates)
    Output1 = lasagne.layers.get_output(l3_bn, deterministic=False)
    Output2 = lasagne.layers.get_output(l3_nl, deterministic=False)
    Output3 = lasagne.layers.get_output(l4   , deterministic=False)
 
    get_intermediate_activation = theano.function([input, chip_input, chip_nonideal], [Output1, Output2, Output3])
    #Outputs1, Outputs2, Outputs3 = get_intermediate_activation(test_set.X)
    #pdb.set_trace()
    #get_Wb = lasagne.layers.get_
    #Grad1 = T.grad(loss, wrt=Output1)
    #Grad2 = T.grad(loss, Output2)
    #Grad3 = T.grad(loss, wrt=Output3)
    #get_gradient = theano.function([input, chip_input], [Grad2])
	
    # Compile a second function computing the validation loss and accuracy:
    #val_fn = theano.function([input, target], [test_loss, test_err])
    val_fn = theano.function([input, chip_input, chip_nonideal, target], [test_loss, test_err])

    print('Training...')
    if save_path is not None:
        os.mkdir(save_path)
    inputname = 'inputs/test'
	# emulate chip nonideality
    np.random.seed(1234)
    chip_nonideal_rng = np.random.ranf((1, num_units)).astype('float32')
    chip_nonideal_rng = chip_nonideal_rng * 1.0 - 0.5
    chip_nonideal_rng = np.repeat(chip_nonideal_rng, batch_size, axis=0)
	
    #pdb.set_trace()	
    binary_net.train(
            train_fn, get_intermediate_activation, #get_gradient,
            val_fn,
            mlp,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            inputname,
            chip_nonideal_rng,			
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y,
            save_path,
            shuffle_parts)
			
filename = "mnist_"
#os.mkdir(filename)
#os.chdir(filename)
#Params_val = lasagne.layers.get_all_param_values(mlp)
#Params  = lasagne.layers.get_all_params(mlp)
#Output1 = lasagne.layers.get_output(l1, deterministic=True)

#Out1Func = theano.function([input], Output1)
#Outputs1 = Out1Func(test_set.X)

#Out2Func = theano.function([input], Output2)
#Outputs2 = Out2Func(test_set.X)

#out2=T.tensor4()
#Out3Func = theano.function([out2], Output3)
#Outputs3 = Out3Func(Outputs2)

#out3=T.tensor4()
#Out4Func = theano.function([out3], Output4)
#Outputs4 = Out2Func(Outputs3)

#np.save("Params_val.npy", Params_val)
#np.save("Params.npy", Params)
#np.save("Outputs1.npy", Outputs1)
#np.save("Outputs1.npy", Outputs1)
#np.save("Outputs2.npy", Outputs2)
#np.save("Outputs3.npy", Outputs3)

#os.chdir("..")