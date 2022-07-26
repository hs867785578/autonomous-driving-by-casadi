import casadi
import numpy as np
from math import *
import matplotlib.pyplot as plt
import time

'''

This code uses Orthogonal Collocation Direct Transcription(OCDT) to find the
numerical solution of the optimal control problem. It transforms solving the optimal
control problem into solving the Nonlinear Programing NLP problem.

Author:Han Shuo
Data:2022.3.26

'''

# OCDT parameter
Nfe = 10
K_radau = 3
I = np.arange(1,Nfe+1)
I1 = np.arange(1,Nfe)
J = np.arange(1,K_radau+1)
K = np.arange(0,K_radau+1)
tao = np.array([0.0, 0.1550510257216822, 0.6449489742783178, 1.0])
dljtauk = np.zeros((4,4))
dljtauk[0,:] = [-9.0000, 10.0488, -1.3821, 0.3333]
dljtauk[1,:] = [-4.1394, 3.2247, 1.1678, -0.2532]
dljtauk[2,:] = [1.7394, -3.5678, 0.7753, 1.0532]
dljtauk[3,:] = [-3, 5.5320, -7.5320, 5.0000]
dljtauk = dljtauk.transpose()
# print(dljtauk)

#vehicle parameter
L_wheelbase = 2.8
a_max = 0.3
v_max = 2.0
phy_max = 0.72
w_max = 0.54

#decision variable
x = casadi.SX.sym('x', 11, 4)
y = casadi.SX.sym('y', 11, 4)
theta = casadi.SX.sym('theta', 11, 4)
v = casadi.SX.sym('v', 11, 4)
phy = casadi.SX.sym('phy', 11, 4)

a = casadi.SX.sym('a', 11, 4)
w = casadi.SX.sym('w', 11, 4)
tf = casadi.SX.sym('tf')
hi = casadi.SX.sym('hi')

X = casadi.transpose(x[1:,:])#Casadi is different from numpy, its reshape is depend on column
X = casadi.reshape(X,-1,1)
Y = casadi.transpose(y[1:,:])
Y = casadi.reshape(Y,-1,1)
THETA = casadi.transpose(theta[1:,:])
THETA = casadi.reshape(THETA,-1,1)
V = casadi.transpose(v[1:,:])
V = casadi.reshape(V,-1,1)
PHY = casadi.transpose(phy[1:,:])
PHY = casadi.reshape(PHY,-1,1)
A = casadi.transpose(a[1:,1:])
A = casadi.reshape(A,-1,1)
W = casadi.transpose(w[1:,1:])
W = casadi.reshape(W,-1,1)

#equality constrains
g = []
gl = []
gu = []
#dynamics contrains
for i in I:
    for k in J:
        temp_gx     = dljtauk[0,k]*x[i,0]     + dljtauk[1,k]*x[i,1]     + dljtauk[2,k]*x[i,2]     + dljtauk[3,k]*x[i,3]     - hi*v[i,k]*casadi.cos(theta[i,k])
        temp_gy     = dljtauk[0,k]*y[i,0]     + dljtauk[1,k]*y[i,1]     + dljtauk[2,k]*y[i,2]     + dljtauk[3,k]*y[i,3]     - hi*v[i,k]*casadi.sin(theta[i,k])
        temp_gtheta = dljtauk[0,k]*theta[i,0] + dljtauk[1,k]*theta[i,1] + dljtauk[2,k]*theta[i,2] + dljtauk[3,k]*theta[i,3] - hi*casadi.tan(phy[i,k])*v[i,k]/L_wheelbase
        temp_gv     = dljtauk[0,k]*v[i,0]     + dljtauk[1,k]*v[i,1]     + dljtauk[2,k]*v[i,2]     + dljtauk[3,k]*v[i,3]     - hi*a[i,k]
        temp_gphy   = dljtauk[0,k]*phy[i,0]   + dljtauk[1,k]*phy[i,1]   + dljtauk[2,k]*phy[i,2]   + dljtauk[3,k]*phy[i,3]   - hi*w[i,k]
        g.append(temp_gx)
        gl.append(0.0)
        gu.append(0.0)
        g.append(temp_gy)
        gl.append(0.0)
        gu.append(0.0)
        g.append(temp_gtheta)
        gl.append(0.0)
        gu.append(0.0)
        g.append(temp_gv)
        gl.append(0.0)
        gu.append(0.0)
        g.append(temp_gphy)
        gl.append(0.0)
        gu.append(0.0)

#boundary continuous contrains
for i in I1:
    temp_eqx = 0
    temp_eqy = 0
    temp_eqtheta = 0
    temp_eqv = 0
    temp_eqphy = 0
    for j in K:
        temp_eqx_ = 1
        temp_eqy_ = 1
        temp_eqtheta_ = 1
        temp_eqv_ = 1
        temp_eqphy_ = 1
        for k in K:
            if j != k:
                temp_eqx_ *= (1-tao[k])/(tao[j]-tao[k])
                temp_eqy_ *= (1-tao[k])/(tao[j]-tao[k])
                temp_eqtheta_ *= (1-tao[k])/(tao[j]-tao[k])
                temp_eqv_ *= (1-tao[k])/(tao[j]-tao[k])
                temp_eqphy_ *= (1-tao[k])/(tao[j]-tao[k])
        temp_eqx += temp_eqx_ * x[i,j]
        temp_eqy += temp_eqy_ * y[i,j]
        temp_eqtheta += temp_eqtheta_ * theta[i,j]
        temp_eqv += temp_eqv_ * v[i,j]
        temp_eqphy += temp_eqphy_ * phy[i,j]

    g.append(x[i+1,0]-temp_eqx)
    gl.append(0.0)
    gu.append(0.0)
    g.append(y[i+1,0]-temp_eqy)
    gl.append(0.0)
    gu.append(0.0)
    g.append(theta[i+1,0]-temp_eqtheta)
    gl.append(0.0)
    gu.append(0.0)
    g.append(v[i+1,0]-temp_eqv)
    gl.append(0.0)
    gu.append(0.0)
    g.append(phy[i+1,0]-temp_eqphy)
    gl.append(0.0)
    gu.append(0.0) 

#inequatility constrains
#box contrains
for i in I:
    for j in K:
        g.append(v[i,j])
        gl.append(-v_max)
        gu.append(v_max)

        g.append(phy[i,j])
        gl.append(-phy_max)
        gu.append(phy_max)
    
for i in I:
    for j in J:
        g.append(a[i,j])
        gl.append(-a_max)
        gu.append(a_max)

        g.append(w[i,j])
        gl.append(-w_max)
        gu.append(w_max)

#init contrains
g.append(x[1,0])
gl.append(1.03)
gu.append(1.03)

g.append(y[1,0])
gl.append(2.41)
gu.append(2.41)

g.append(theta[1,0])
gl.append(-0.03)
gu.append(-0.03)

g.append(v[1,0])
gl.append(0.1)
gu.append(0.1)

g.append(phy[1,0])
gl.append(-0.08)
gu.append(-0.08)

#terminal contrains
g.append(x[Nfe,K_radau])
gl.append(5.31)
gu.append(5.31)

g.append(y[Nfe,K_radau])
gl.append(2.41)
gu.append(2.41)

g.append(theta[Nfe,K_radau])
gl.append(-3.1416)
gu.append(-3.1416)

g.append(v[Nfe,K_radau])
gl.append(0.0)
gu.append(0.0)

g.append(phy[Nfe,K_radau])
gl.append(0.0)
gu.append(0.0)

g.append(a[Nfe,K_radau])
gl.append(0.0)
gu.append(0.0)

g.append(w[Nfe,K_radau])
gl.append(0.0)
gu.append(0.0)

g.append(tf)
gl.append(0.0)
gu.append(inf)

g.append(hi-tf/Nfe)
gl.append(0.0)
gu.append(0.0)

#cost function
F = 0
for i in I:
    for j in J:
        F += w[i,j]*w[i,j]
F = F + 10*tf

#nlp problem
nlp = {'x':casadi.vertcat(X,Y,THETA,V,PHY,A,W,tf,hi), 'f':F, 'g':casadi.vertcat(*g)}

#solve the nlp problem
S = casadi.nlpsol('S', 'ipopt', nlp)

#init solution
init_soution = np.load('init/init_planning.npz')
x = init_soution['x']
y = init_soution['y']
theta = init_soution['theta']
v = init_soution['v']
phy = init_soution['phy']
a = init_soution['a']
w = init_soution['w']
tf = init_soution['tf']
hi = init_soution['hi']

init_x = x[1:,:].reshape(-1).tolist()
init_y = y[1:,:].reshape(-1).tolist()
init_theta = theta[1:,:].reshape(-1).tolist()
init_v = v[1:,:].reshape(-1).tolist()
init_phy = phy[1:,:].reshape(-1).tolist()
init_a = a[1:,1:].reshape(-1).tolist()
init_w = w[1:,1:].reshape(-1).tolist()
init_tf = tf.tolist()
init_hi = hi.tolist()

init = init_x+init_y+init_theta+init_v+init_phy+init_a+init_w+init_tf+init_hi

t_start = time.perf_counter() 
r = S(x0=init, lbg = gl, ubg=gu)
t_end =  time.perf_counter() 
runtime = t_end - t_start
print('computing time:',runtime)


#print result
res = r['x']
X_star = res[0:40]
Y_star = res[40:80]
THETA_star = res[80:120]
V_star = res[120:160]
PHY_star = res[160:200]
A_star = res[200:230]
W_star = res[230:260]
TF_star = res[260:261]
HI_star = res[261:262]

# print(X_star[-2:-1])
# print(Y_star[-2:-1])
# print(THETA_star[-2:-1])
# print(V_star[-2:-1])
# print(PHY_star[-2:-1])
# print(A_star[-2:-1])
# print(W_star[-2:-1])

#plot
X_star=X_star.elements()
Y_star=Y_star.elements()

plt.plot(X_star,Y_star)
plt.show()

