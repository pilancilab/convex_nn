import numpy as np
import dill
import pickle
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import time
import scipy
from scipy.sparse.linalg import LinearOperator
import torch
import sklearn.linear_model
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import argparse
import random




def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--GD', nargs=1, type=int, required=True)
    parser.add_argument('--CVX', nargs=1, type=int, required=True)
    parser.add_argument('--n_epochs', nargs=2, type=int, required=True)
    parser.add_argument('--solver_cvx', type=str, nargs=1, default="adam")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(a=args.seed)
    np.random.seed(seed=args.seed)
    torch.manual_seed(seed=args.seed)

    return args

ARGS=parse_args()

# In[2]:


class FCNetwork(nn.Module):
    def __init__(self, H, num_classes=10, input_dim=3072):
        self.num_classes = num_classes
        super(FCNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, H, bias=False), nn.ReLU())
        self.layer2 = nn.Linear(H, num_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.layer2(self.layer1(x))
        return out
    
# functions for generating sign patterns
def check_if_already_exists(element_list, element):
    # check if element exists in element_list
    # where element is a numpy array
    for i in range(len(element_list)):
        if np.array_equal(element_list[i], element):
            return True
    return False

class PrepareData(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
            
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PrepareData3D(Dataset):
    def __init__(self, X, y, z):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        else:
            self.X = X
            
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
        else:
            self.y = y
        
        if not torch.is_tensor(z):
            self.z = torch.from_numpy(z)
        else:
            self.z = z
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.z[idx]

def generate_conv_sign_patterns(A2, P, verbose=False): 
    # generate convolutional sign patterns
    n, c, p1, p2 = A2.shape
    A = A2.reshape(n,int(c*p1*p2))
    fsize=9*c
    d=c*p1*p2;
    fs=int(np.sqrt(9))
    unique_sign_pattern_list = []  
    u_vector_list = []             

    for i in range(P): 
        # obtain a sign pattern
        ind1=np.random.randint(0,p1-fs+1)
        ind2=np.random.randint(0,p2-fs+1)
        u1p= np.zeros((c,p1,p2))
        u1p[:,ind1:ind1+fs,ind2:ind2+fs]=np.random.normal(0, 1, (fsize,1)).reshape(c,fs,fs)
        u1=u1p.reshape(d,1)
        sampled_sign_pattern = (np.matmul(A, u1) >= 0)[:,0]
        unique_sign_pattern_list.append(sampled_sign_pattern)
        u_vector_list.append(u1)

    if verbose:
        print("Number of unique sign patterns generated: " + str(len(unique_sign_pattern_list)))
    return len(unique_sign_pattern_list),unique_sign_pattern_list, u_vector_list


def generate_sign_patterns(A, P, verbose=False):
    # generate sign patterns
    n, d = A.shape
    sign_pattern_list = []  # sign patterns
    u_vector_list = []             # random vectors used to generate the sign paterns
    umat = np.random.normal(0, 1, (d,P))
    sampled_sign_pattern_mat = (np.matmul(A, umat) >= 0)
    for i in range(P):
        sampled_sign_pattern = sampled_sign_pattern_mat[:,i]
        sign_pattern_list.append(sampled_sign_pattern)
        u_vector_list.append(umat[:,i])
    if verbose:
        print("Number of sign patterns generated: " + str(len(sign_pattern_list)))
    return len(sign_pattern_list),sign_pattern_list, u_vector_list

def one_hot(labels, num_classes=10):
    y = torch.eye(num_classes) 
    return y[labels.long()] 



#=====================================STANDARD NON-CONVEX NETWORK=====================================


def loss_func_primal(yhat, y, model, beta):
    loss = 0.5 * torch.norm(yhat - y)**2
    
    ## l2 norm on first layer weights, l1 squared norm on second layer
    for layer, p in enumerate(model.parameters()):
        if layer == 0:
            loss += beta/2 * torch.norm(p)**2
        else:
            loss += beta/2 * sum([torch.norm(p[:, j], 1)**2 for j in range(p.shape[1])])
    
    return loss

def validation_primal(model, testloader, beta, device):
    test_loss = 0
    test_correct = 0

    for ix, (_x, _y) in enumerate(testloader):
        _x = Variable(_x).float().to(device)
        _y = Variable(_y).float().to(device)

        output = model.forward(_x)
        yhat = model(_x).float()

        loss = loss_func_primal(yhat, one_hot(_y).to(device), model, beta)

        test_loss += loss.item()
        test_correct += torch.eq(torch.argmax(yhat, dim=1), torch.squeeze(_y)).float().sum()

    return test_loss, test_correct

# solves nonconvex problem
def sgd_solver_pytorch_v2(ds, ds_test, num_epochs, num_neurons, beta, 
                         learning_rate, batch_size, solver_type, schedule, 
                          LBFGS_param, verbose=False, 
                        num_classes=10, D_in=3*1024, test_len=10000, 
                          train_len=50000, device='cuda'):
    
    device = torch.device(device)
    # D_in is input dimension, H is hidden dimension, D_out is output dimension.
    H, D_out = num_neurons, num_classes
    # create the model
    model = FCNetwork(H, D_out, D_in).to(device)
    
    if solver_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#,
    elif solver_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)#,
    elif solver_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)#,
    elif solver_type == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=LBFGS_param[0], max_iter=LBFGS_param[1])#,
        
    # arrays for saving the loss and accuracy    
    losses = np.zeros((int(num_epochs*np.ceil(train_len / batch_size))))
    accs = np.zeros(losses.shape)
    losses_test = np.zeros((num_epochs+1))
    accs_test = np.zeros((num_epochs+1))
    times = np.zeros((losses.shape[0]+1))
    times[0] = time.time()
    
    losses_test[0], accs_test[0] = validation_primal(model, ds_test, beta, device) # loss on the entire test set
    
    if schedule==1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=verbose,
                                                           factor=0.5,
                                                           eps=1e-12)
    elif schedule==2:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        
    iter_no = 0
    for i in range(num_epochs):
        for ix, (_x, _y) in enumerate(ds):
            #=========make input differentiable=======================
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)
            
            #========forward pass=====================================
            yhat = model(_x).float()
            
            loss = loss_func_primal(yhat, one_hot(_y).to(device), model, beta)/len(_y)
            correct = torch.eq(torch.argmax(yhat, dim=1), torch.squeeze(_y)).float().sum()/len(_y)
            
           
            optimizer.zero_grad() # zero the gradients on each pass before the update
            loss.backward() # backpropagate the loss through the model
            optimizer.step() # update the gradients w.r.t the loss

            losses[iter_no] = loss.item() # loss on the minibatch
            accs[iter_no] = correct
        
            iter_no += 1
            times[iter_no] = time.time()
        
        # get test loss and accuracy
        losses_test[i+1], accs_test[i+1] = validation_primal(model, ds_test, beta, device) # loss on the entire test set

        if i % 1 == 0:
            print("Epoch [{}/{}], loss: {} acc: {}, test loss: {} test acc: {}".format(i, num_epochs,
                    np.round(losses[iter_no-1], 3), np.round(accs[iter_no-1], 3), 
                    np.round(losses_test[i+1], 3)/test_len, np.round(accs_test[i+1]/test_len, 3)))
        if schedule>0:
            scheduler.step(losses[iter_no-1])
            
    return losses, accs, losses_test/test_len, accs_test/test_len, times, model


#=====================================CONVEX NETWORK=====================================


class custom_cvx_layer(torch.nn.Module):
    def __init__(self, d, num_neurons, num_classes=10):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(custom_cvx_layer, self).__init__()
        
        # P x d x C
        self.v = torch.nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)
        self.w = torch.nn.Parameter(data=torch.zeros(num_neurons, d, num_classes), requires_grad=True)

    def forward(self, x, sign_patterns):
        sign_patterns = sign_patterns.unsqueeze(2)
        x = x.view(x.shape[0], -1) # n x d
        
        Xv_w = torch.matmul(x, self.v - self.w) # P x N x C
        
        # for some reason, the permutation is necessary. not sure why
        DXv_w = torch.mul(sign_patterns, Xv_w.permute(1, 0, 2)) #  N x P x C
        y_pred = torch.sum(DXv_w, dim=1, keepdim=False) # N x C
        
        return y_pred
    
def get_nonconvex_cost(y, model, _x, beta, device):
    _x = _x.view(_x.shape[0], -1)
    Xv = torch.matmul(_x, model.v)
    Xw = torch.matmul(_x, model.w)
    Xv_relu = torch.max(Xv, torch.Tensor([0]).to(device))
    Xw_relu = torch.max(Xw, torch.Tensor([0]).to(device))
    
    prediction_w_relu = torch.sum(Xv_relu - Xw_relu, dim=0, keepdim=False)
    prediction_cost = 0.5 * torch.norm(prediction_w_relu - y)**2
    
    regularization_cost = beta * (torch.sum(torch.norm(model.v, dim=1)**2) + torch.sum(torch.norm(model.w, p=1, dim=1)**2))
    
    return prediction_cost + regularization_cost
def loss_func_cvxproblem(yhat, y, model, _x, sign_patterns, beta, rho, device):
    _x = _x.view(_x.shape[0], -1)
    
    # term 1
    loss = 0.5 * torch.norm(yhat - y)**2
    # term 2
    loss = loss + beta * torch.sum(torch.norm(model.v, dim=1))
    loss = loss + beta * torch.sum(torch.norm(model.w, dim=1))
    
    # term 3
    sign_patterns = sign_patterns.unsqueeze(2) # N x P x 1
    
    Xv = torch.matmul(_x, torch.sum(model.v, dim=2, keepdim=True)) # N x d times P x d x 1 -> P x N x 1
    DXv = torch.mul(sign_patterns, Xv.permute(1, 0, 2)) # P x N x 1
    relu_term_v = torch.max(-2*DXv + Xv.permute(1, 0, 2), torch.Tensor([0]).to(device))
    loss = loss + rho * torch.sum(relu_term_v)
    
    Xw = torch.matmul(_x, torch.sum(model.w, dim=2, keepdim=True))
    DXw = torch.mul(sign_patterns, Xw.permute(1, 0, 2))
    relu_term_w = torch.max(-2*DXw + Xw.permute(1, 0, 2), torch.Tensor([0]).to(device))
    loss = loss + rho * torch.sum(relu_term_w)
    
    return loss

def validation_cvxproblem(model, testloader, u_vectors, beta, rho, device):
    test_loss = 0
    test_correct = 0
    test_noncvx_cost = 0

    with torch.no_grad():
        for ix, (_x, _y) in enumerate(testloader):
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)
            _x = _x.view(_x.shape[0], -1)
            _z = (torch.matmul(_x, torch.from_numpy(u_vectors).float().to(device)) >= 0)

            output = model.forward(_x, _z)
            yhat = model(_x, _z).float()

            loss = loss_func_cvxproblem(yhat, one_hot(_y).to(device), model, _x, _z, beta, rho, device)

            test_loss += loss.item()
            test_correct += torch.eq(torch.argmax(yhat, dim=1), _y).float().sum()

            test_noncvx_cost += get_nonconvex_cost(one_hot(_y).to(device), model, _x, beta, device)

    return test_loss, test_correct, test_noncvx_cost
def sgd_solver_cvxproblem(ds, ds_test, num_epochs, num_neurons, beta, 
                       learning_rate, batch_size, rho, u_vectors, 
                          solver_type, LBFGS_param, verbose=False,
                         n=60000, d=3072, num_classes=10, device='cpu'):
    device = torch.device(device)

    # create the model
    model = custom_cvx_layer(d, num_neurons, num_classes).to(device)
    
    if solver_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif solver_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#,
    elif solver_type == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)#,
    elif solver_type == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)#,
    elif solver_type == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), history_size=LBFGS_param[0], max_iter=LBFGS_param[1])#,
    
    # arrays for saving the loss and accuracy 
    losses = np.zeros((int(num_epochs*np.ceil(n / batch_size))))
    accs = np.zeros(losses.shape)
    noncvx_losses = np.zeros(losses.shape)
    
    losses_test = np.zeros((num_epochs+1))
    accs_test = np.zeros((num_epochs+1))
    noncvx_losses_test = np.zeros((num_epochs+1))
    
    times = np.zeros((losses.shape[0]+1))
    times[0] = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=verbose,
                                                           factor=0.5,
                                                           eps=1e-12)
    
    model.eval()
    losses_test[0], accs_test[0], noncvx_losses_test[0] = validation_cvxproblem(model, ds_test, u_vectors, beta, rho, device) # loss on the entire test set
    
    iter_no = 0
    print('starting training')
    for i in range(num_epochs):
        model.train()
        for ix, (_x, _y, _z) in enumerate(ds):
            #=========make input differentiable=======================
            _x = Variable(_x).to(device)
            _y = Variable(_y).to(device)
            _z = Variable(_z).to(device)
            
            #========forward pass=====================================
            yhat = model(_x, _z).float()
            
            loss = loss_func_cvxproblem(yhat, one_hot(_y).to(device), model, _x,_z, beta, rho, device)/len(_y)
            correct = torch.eq(torch.argmax(yhat, dim=1), _y).float().sum()/len(_y) # accuracy
            #=======backward pass=====================================
            optimizer.zero_grad() # zero the gradients on each pass before the update
            loss.backward() # backpropagate the loss through the model
            optimizer.step() # update the gradients w.r.t the loss

            losses[iter_no] = loss.item() # loss on the minibatch
            accs[iter_no] = correct
            noncvx_losses[iter_no] = get_nonconvex_cost(one_hot(_y).to(device), model, _x, beta, device)/len(_y)
        
            iter_no += 1
            times[iter_no] = time.time()
        
        model.eval()
        # get test loss and accuracy
        losses_test[i+1], accs_test[i+1], noncvx_losses_test[i+1] = validation_cvxproblem(model, ds_test, u_vectors, beta, rho, device) # loss on the entire test set
        
        if i % 1 == 0:
            print("Epoch [{}/{}], TRAIN: noncvx/cvx loss: {}, {} acc: {}. TEST: noncvx/cvx loss: {}, {} acc: {}".format(i, num_epochs,
                    np.round(noncvx_losses[iter_no-1], 3), np.round(losses[iter_no-1], 3), np.round(accs[iter_no-1], 3), 
                    np.round(noncvx_losses_test[i+1], 3)/10000, np.round(losses_test[i+1], 3)/10000, np.round(accs_test[i+1]/10000, 3)))
        
        scheduler.step(losses[iter_no-1])
        
    return noncvx_losses, accs, noncvx_losses_test/10000, accs_test/10000, times, losses, losses_test/10000





# cifar-10 -- using the version downloaded from "http://www.cs.toronto.edu/~kriz/cifar.html"
import os
directory = os.path.dirname(os.path.realpath(__file__))

import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

train_dataset = datasets.CIFAR10(
    directory, train=True, download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    normalize,
]))

test_dataset = datasets.CIFAR10(
    directory, train=False, download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    normalize,
]))



# data extraction
print('Extracting the data')
dummy_loader= torch.utils.data.DataLoader(
    train_dataset, batch_size=50000, shuffle=False,
    pin_memory=True, sampler=None)
for A, y in dummy_loader:
    pass
Apatch=A.detach().clone()

A = A.view(A.shape[0], -1)
n,d=A.size()



# problem parameters
P, verbose = 4096, True # SET verbose to True to see progress
GD_only=ARGS.GD[0]
CVX_only=ARGS.CVX[0]
beta = 1e-3 # regularization parameter
num_epochs1, batch_size =  ARGS.n_epochs[0], 1000 #
num_neurons = P # number of neurons is equal to number of hyperplane arrangements


# create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True, sampler=None)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1000, shuffle=False,
    pin_memory=True)





# SGD solver for the nonconvex problem
if CVX_only==0:

    solver_type = "sgd" # pick: "sgd", "adam", "adagrad", "adadelta", "LBFGS"
    schedule=0 # learning rate schedule (0: Nothing, 1: ReduceLROnPlateau, 2: ExponentialLR)
    LBFGS_param = [10, 4] # these parameters are for the LBFGS solver
    learning_rate = 1e-2
    
    ## SGD1 constant    
    print('SGD1-training-mu={}'.format(learning_rate))
    results_noncvx_sgd1 = sgd_solver_pytorch_v2(train_loader, test_loader, num_epochs1, num_neurons, beta, 
                             learning_rate, batch_size, solver_type, schedule, 
                              LBFGS_param, verbose=True, 
                            num_classes=10, D_in=d, train_len=n )
    

    ## SGD2 constant    
    learning_rate = 5e-3
    print('SGD2-training-mu={}'.format(learning_rate))
    results_noncvx_sgd2 = sgd_solver_pytorch_v2(train_loader, test_loader, num_epochs1, num_neurons, beta, 
                             learning_rate, batch_size, solver_type, schedule,
                              LBFGS_param, verbose=True, 
                            num_classes=10, D_in=d, train_len=n )
  
    ## SGD3 constant
    learning_rate = 1e-3
    print('SGD3-training-mu={}'.format(learning_rate))
    results_noncvx_sgd3 = sgd_solver_pytorch_v2(train_loader, test_loader, num_epochs1, num_neurons, beta, 
                             learning_rate, batch_size, solver_type, schedule,
                              LBFGS_param, verbose=True, 
                            num_classes=10, D_in=d, train_len=n )
    
   
# Solver for the convex problem
if GD_only ==0:

    rho = 1e-2 # coefficient to penalize the violated constraints
    solver_type = ARGS.solver_cvx[0] # pick: "sgd", "adam", "adagrad", "adadelta", "LBFGS"
    LBFGS_param = [10, 4] 
    batch_size = 1000
    num_epochs2, batch_size = ARGS.n_epochs[1], 1000 
 
    
    
    #  Convex
    print('Generating sign patterns')
    num_neurons,sign_pattern_list, u_vector_list = generate_sign_patterns(A, P, verbose)
    sign_patterns = np.array([sign_pattern_list[i].int().data.numpy() for i in range(num_neurons)])
    u_vectors = np.asarray(u_vector_list).reshape((num_neurons, A.shape[1])).T
    
    ds_train = PrepareData3D(X=A, y=y, z=sign_patterns.T)
    ds_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False,
        pin_memory=True)


    #  Convex1
    learning_rate = 1e-6 # 1e-6 for sgd    
    print('Convex Random1-mu={}'.format(learning_rate))
    results_cvx1 = sgd_solver_cvxproblem(ds_train, test_loader, num_epochs2, num_neurons, beta, 
                            learning_rate, batch_size, rho, u_vectors, solver_type, LBFGS_param, verbose=True, 
                                             n=n, device='cuda')

    #  Convex2
    learning_rate = 5e-7 # 1e-6 for sgd    
    print('Convex Random2-mu={}'.format(learning_rate))
    results_cvx2 = sgd_solver_cvxproblem(ds_train, test_loader, num_epochs2, num_neurons, beta, 
                            learning_rate, batch_size, rho, u_vectors, solver_type, LBFGS_param, verbose=True, 
                                             n=n, device='cuda')
    
    
    
    #  Convex with convolutional patterns
    print('Generating conv sign patterns')
    num_neurons,sign_pattern_list, u_vector_list = generate_conv_sign_patterns(Apatch, P, verbose)
    sign_patterns = np.array([sign_pattern_list[i].int().data.numpy() for i in range(num_neurons)])
    u_vectors = np.asarray(u_vector_list).reshape((num_neurons, A.shape[1])).T
    
    ds_train = PrepareData3D(X=A, y=y, z=sign_patterns.T)
    ds_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    #  Convex Conv1
    learning_rate = 1e-6       
    print('Convex Conv1-mu={}'.format(learning_rate))
    results_cvx_conv1 = sgd_solver_cvxproblem(ds_train, test_loader, num_epochs2, num_neurons, beta, 
                            learning_rate, batch_size, rho, u_vectors, solver_type, LBFGS_param, verbose=True, 
                                             n=n, device='cuda')

    #  Convex Conv2 
    learning_rate = 5e-7       
    print('Convex Conv2-mu={}'.format(learning_rate))
    results_cvx_conv2 = sgd_solver_cvxproblem(ds_train, test_loader, num_epochs2, num_neurons, beta, 
                            learning_rate, batch_size, rho, u_vectors, solver_type, LBFGS_param, verbose=True, 
                                             n=n, device='cuda')
    
    
# plots and saves the results
import pickle
from datetime import datetime
now = datetime.now() 
if GD_only==1 and CVX_only==0:


    
    results_noncvx_sgd1v2=results_noncvx_sgd1[:5]
    results_noncvx_sgd2v2=results_noncvx_sgd2[:5]
    results_noncvx_sgd3v2=results_noncvx_sgd3[:5]

 

    
    print('Saving the objects')
    torch.save([num_epochs1,results_noncvx_sgd1v2, results_noncvx_sgd2v2, results_noncvx_sgd3v2
                    ],'results_fig_gdonly_stepsize_cifar10_'+now.strftime("%d-%m-%Y_%H-%M-%S")+'.pt')
    

    
elif GD_only==0 and CVX_only==1:

    print('Saving the objects')
    torch.save([num_epochs2, results_cvx1,results_cvx2, 
                    results_cvx_conv1,results_cvx_conv2],'results_fig_cvxonly_stepsize_cifar10_'+now.strftime("%d-%m-%Y_%H-%M-%S")+'.pt')
    
else:

    results_noncvx_sgd1v2=results_noncvx_sgd1[:5]
    results_noncvx_sgd2v2=results_noncvx_sgd2[:5]
    results_noncvx_sgd3v2=results_noncvx_sgd3[:5]
    print('Saving the objects')
    torch.save([num_epochs1,num_epochs2,results_noncvx_sgd1v2, results_noncvx_sgd2v2, results_noncvx_sgd3v2, results_cvx1,results_cvx2, 
                    results_cvx_conv1,results_cvx_conv2],'results_fig_all_stepsize_cifar10_'+now.strftime("%d-%m-%Y_%H-%M-%S")+'.pt')

    import matplotlib.pyplot as plt

    skip=1#int(num_epochs1/num_epochs2)
    mark_sgd=10
    mark_cvx=30
    
    marker_size_sgd=10
    marker_size_cvx=12
    

    plt.gcf().set_facecolor("white")
    #fig,ax = plt.subplots()
    
    # plot
    fsize=24
    fsize_legend=15
    
    plt.rcParams.update({'font.size': 24})
    plt.xlabel('Time(s)',fontsize=fsize);  plt.grid()
    
    plot_no = 1 # select --> 0: cost, 1: accuracy
    
    
    
    num_all_iters1 = results_noncvx_sgd1v2[4].shape[0] - 1
    num_all_iters2 = results_cvx1[4].shape[0] - 1
    
    iters_per_epoch1 = num_all_iters1 // num_epochs1
    iters_per_epoch2 = num_all_iters2 // num_epochs2
    
    epoch_times_noncvx1 = results_noncvx_sgd1v2[4][0:num_all_iters1+1:iters_per_epoch1]-results_noncvx_sgd1v2[4][0]
    epoch_times_noncvx2 = results_noncvx_sgd2v2[4][0:num_all_iters1+1:iters_per_epoch1]-results_noncvx_sgd2v2[4][0]
    epoch_times_noncvx3 = results_noncvx_sgd3v2[4][0:num_all_iters1+1:iters_per_epoch1]-results_noncvx_sgd3v2[4][0]
    
    
    epoch_times_cvx1 = results_cvx1[4][0:num_all_iters2+1:iters_per_epoch2]-results_cvx1[4][0]
    epoch_times_cvx2 = results_cvx2[4][0:num_all_iters2+1:iters_per_epoch2]-results_cvx2[4][0]
    
    epoch_times_cvx_conv1= results_cvx_conv1[4][0:num_all_iters2+1:iters_per_epoch2]-results_cvx_conv1[4][0]
    epoch_times_cvx_conv2= results_cvx_conv2[4][0:num_all_iters2+1:iters_per_epoch2]-results_cvx_conv2[4][0]
    
    
    plt.grid()
    
    # To plot results in the validation set
    plt.plot( epoch_times_noncvx1[::skip],results_noncvx_sgd1v2[plot_no+2][::skip],'--', color='darkred', markevery=mark_sgd,linewidth=3.0, markersize=marker_size_sgd,label="SGD-$\mu=1e-2$")
    plt.plot( epoch_times_noncvx2[::skip],results_noncvx_sgd2v2[plot_no+2][::skip],'--', color='red', markevery=mark_sgd,linewidth=3.0, markersize=marker_size_sgd,label="SGD-$\mu=5e-3$")
    plt.plot( epoch_times_noncvx3[::skip],results_noncvx_sgd3v2[plot_no+2][::skip],'--', color='lightcoral', markevery=mark_sgd,linewidth=3.0, markersize=marker_size_sgd,label="SGD-$\mu=1e-3$")
    
    
    plt.plot( epoch_times_cvx1,results_cvx1[plot_no+2],  'o--', color='g', markevery=mark_cvx,linewidth=3.0, markersize=marker_size_cvx,label="Convex-Random-$\mu=1e-6$")
    plt.plot( epoch_times_cvx2,results_cvx2[plot_no+2],  'o--', color='lime', markevery=mark_cvx,linewidth=3.0, markersize=marker_size_cvx,label="Convex-Random-$\mu=5e-7$")
    
    plt.plot( epoch_times_cvx_conv1,results_cvx_conv1[plot_no+2],  'o--', color='b', markevery=mark_cvx,linewidth=3.0, markersize=marker_size_cvx,label="Convex-Conv-$\mu=1e-6$")
    plt.plot( epoch_times_cvx_conv2,results_cvx_conv1[plot_no+2],  'o--', color='lightblue', markevery=mark_cvx,linewidth=3.0, markersize=marker_size_cvx,label="Convex-Conv-$\mu=5e-7$")
    

    plt.legend(prop={'size': fsize_legend})
    plt.ylabel("Test Accuracy",fontsize=fsize)
    plt.ylim(0.3, 0.6)
    plt.xlim(0, 4500)
    
    plt.grid()
    plt.savefig('cifar_multiclass_stepsize_testacc.png', format='png', bbox_inches='tight')
    
    
    plt.figure()
    # To plot training  acc
    
    plt.xlabel('Time(s)',fontsize=fsize)  
    plt.grid()

    p11=results_noncvx_sgd1v2[1].reshape(-1,1)
    p12=results_noncvx_sgd2v2[1].reshape(-1,1)
    p13=results_noncvx_sgd3v2[1].reshape(-1,1)
    
    p21=results_cvx1[1].reshape(-1,1)
    p22=results_cvx2[1].reshape(-1,1)
    
    p31=results_cvx_conv1[1].reshape(-1,1)
    p32=results_cvx_conv2[1].reshape(-1,1)
    

    
    n=50000
    batch_size1=1000
    batch_size2=1000
    
    plt.plot(epoch_times_noncvx1[:-1][::skip],p11[np.arange(num_epochs1)*int(n/batch_size1)][::skip],'-',color='darkred', markevery=mark_sgd,linewidth=3, markersize=marker_size_sgd,label="SGD-$\mu=1e-2$")
    plt.plot(epoch_times_noncvx2[:-1][::skip],p12[np.arange(num_epochs1)*int(n/batch_size1)][::skip],'-',color='red', markevery=mark_sgd,linewidth=3, markersize=marker_size_sgd,label="SGD-$\mu=5e-2$")
    plt.plot(epoch_times_noncvx3[:-1][::skip],p13[np.arange(num_epochs1)*int(n/batch_size1)][::skip],'-',color='lightcoral', markevery=mark_sgd,linewidth=3, markersize=marker_size_sgd,label="SGD-$\mu=1e-3$")
    
    plt.plot( epoch_times_cvx1[:-1],p21[np.arange(num_epochs2)*int(n/batch_size2)] ,'o-',color='g', markevery=mark_cvx,linewidth=3, markersize=marker_size_cvx,label="Convex-Random-$\mu=1e-6$")
    plt.plot( epoch_times_cvx2[:-1],p22[np.arange(num_epochs2)*int(n/batch_size2)] ,'o-',color='lime', markevery=mark_cvx,linewidth=3, markersize=marker_size_cvx,label="Convex-Random-$\mu=5e-7$")
    
    plt.plot( epoch_times_cvx_conv1[:-1],p31[np.arange(num_epochs2)*int(n/batch_size2)] ,'o-', color='b',markevery=mark_cvx,linewidth=3, markersize=marker_size_cvx,label="Convex-Conv-$\mu=1e-6$")
    plt.plot( epoch_times_cvx_conv2[:-1],p32[np.arange(num_epochs2)*int(n/batch_size2)] ,'o-', color='lightblue',markevery=mark_cvx,linewidth=3, markersize=marker_size_cvx,label="Convex-Conv-$\mu=5e-7$")
    

    plt.xlim(0, 4500)
    
    plt.ylabel("Training Accuracy",fontsize=fsize)
    plt.grid()
    matplotlib.pyplot.grid(True, which="both")
    plt.savefig('cifar_multiclass_stepsize_tracc.png', format='png', bbox_inches='tight')

    
    # To plot training loss

    plt.figure()
    
    plt.xlabel('Time(s)',fontsize=fsize)  
    plt.grid()
    p11=results_noncvx_sgd1v2[0].reshape(-1,1)
    p12=results_noncvx_sgd2v2[0].reshape(-1,1)
    p13=results_noncvx_sgd3v2[0].reshape(-1,1)
    
    p21=results_cvx1[5].reshape(-1,1)
    p22=results_cvx2[5].reshape(-1,1)
    
    p31=results_cvx_conv1[5].reshape(-1,1)
    p32=results_cvx_conv2[5].reshape(-1,1)
    

    
    n=50000
    batch_size1=1000
    batch_size2=1000
    
    plt.semilogy(epoch_times_noncvx1[:-1][::skip],p11[np.arange(num_epochs1)*int(n/batch_size1)][::skip],'-',color='darkred', markevery=mark_sgd,linewidth=3, markersize=marker_size_sgd,label="SGD-$\mu=1e-2$")
    plt.semilogy(epoch_times_noncvx2[:-1][::skip],p12[np.arange(num_epochs1)*int(n/batch_size1)][::skip],'-',color='red', markevery=mark_sgd,linewidth=3, markersize=marker_size_sgd,label="SGD-$\mu=5e-2$")
    plt.semilogy(epoch_times_noncvx3[:-1][::skip],p13[np.arange(num_epochs1)*int(n/batch_size1)][::skip],'-',color='lightcoral', markevery=mark_sgd,linewidth=3, markersize=marker_size_sgd,label="SGD-$\mu=1e-3$")
    
    plt.semilogy( epoch_times_cvx1[:-1],p21[np.arange(num_epochs2)*int(n/batch_size2)] ,'o-',color='g', markevery=mark_cvx,linewidth=3, markersize=marker_size_cvx,label="Convex-Random-$\mu=1e-6$")
    plt.semilogy( epoch_times_cvx2[:-1],p22[np.arange(num_epochs2)*int(n/batch_size2)] ,'o-',color='lime', markevery=mark_cvx,linewidth=3, markersize=marker_size_cvx,label="Convex-Random-$\mu=5e-7$")
    
    plt.semilogy( epoch_times_cvx_conv1[:-1],p31[np.arange(num_epochs2)*int(n/batch_size2)] ,'o-', color='b',markevery=mark_cvx,linewidth=3, markersize=marker_size_cvx,label="Convex-Conv-$\mu=1e-6$")
    plt.semilogy( epoch_times_cvx_conv2[:-1],p32[np.arange(num_epochs2)*int(n/batch_size2)] ,'o-', color='lightblue',markevery=mark_cvx,linewidth=3, markersize=marker_size_cvx,label="Convex-Conv-$\mu=5e-7$")
    

    plt.xlim(0, 4500)
    
    plt.ylabel("Objective Value",fontsize=fsize)
    plt.grid()
    matplotlib.pyplot.grid(True, which="both")
    plt.savefig('cifar_multiclass_stepsize_obj.png', format='png', bbox_inches='tight')





