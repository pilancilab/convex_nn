## This is a basic CVXPY based implementation on a toy dataset for the paper 
## "Neural Networks are Convex Regularizers: Exact Polynomial-time Convex Optimization Formulations for Two-layer Networks"
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0,x)
def drelu(x):
    return x>=0
n=10
d=3
X=np.random.randn(n,d-1)
X=np.append(X,np.ones((n,1)),axis=1)

y=((np.linalg.norm(X[:,0:d-1],axis=1)>1)-0.5)*2
beta=1e-4


dmat=np.empty((n,0))

## Finite approximation of all possible sign patterns
for i in range(int(1e2)):
    u=np.random.randn(d,1)
    dmat=np.append(dmat,drelu(np.dot(X,u)),axis=1)

dmat=(np.unique(dmat,axis=1))


# Optimal CVX
m1=dmat.shape[1]
Uopt1=cp.Variable((d,m1))
Uopt2=cp.Variable((d,m1))

## Below we use hinge loss as a performance metric for binary classification
yopt1=cp.Parameter((n,1))
yopt2=cp.Parameter((n,1))
yopt1=cp.sum(cp.multiply(dmat,(X*Uopt1)),axis=1)
yopt2=cp.sum(cp.multiply(dmat,(X*Uopt2)),axis=1)
cost=cp.sum(cp.pos(1-cp.multiply(y,yopt1-yopt2)))/n+beta*(cp.mixed_norm(Uopt1.T,2,1)+cp.mixed_norm(Uopt2.T,2,1))
constraints=[]
constraints+=[cp.multiply((2*dmat-np.ones((n,m1))),(X*Uopt1))>=0]
constraints+=[cp.multiply((2*dmat-np.ones((n,m1))),(X*Uopt2))>=0]
prob=cp.Problem(cp.Minimize(cost),constraints)
prob.solve()
cvx_opt=prob.value
print("Convex program objective value (eq (8)): ",cvx_opt)
