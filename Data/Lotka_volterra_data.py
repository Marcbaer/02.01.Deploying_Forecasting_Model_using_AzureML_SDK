
"""
Created on Wed Jan  9 13:46:38 2019

@author: marcbaer
"""

'''This script integrates the Lotka - Volterra Differential Equations and plots the resulting dynamical system'''

from matplotlib import pyplot as plt  
import numpy as np                    
from scipy.integrate import odeint 
import pickle

# Parameters.
a = 0.5
b = 0.01
c = 0.1
d = 0.01

y_total=[]
y_hist=[]


for j in range(0,10,1):
    if j==0:
        
        t0 = 0.0       # Initial time.
        t1 = 10000.0     # Final time.
        y0 = [10, 25]  # Initial population of species 0 and 1.
    else:
        t0 = 0.0       # Initial time.
        t1 = 10000.0     # Final time.
        y0 = [1, 2]  # Initial population of species 0 and 1.
    
        
    y=[]
    F = lambda y, t: [
        y[0] * (a - b * y[1]),   # How population 0 changes.
        y[1] * (-c + d * y[0]),  # How population 1 changes.
    ]
    
    
    t = np.linspace(t0, t1, 10000)
    y = odeint(F, y0, t)
    
    if j==0:
        y_total=y
    else:    
        y_total=np.concatenate((y_total,y[1:,:]),axis=0)
        y_total=np.array(y_total)
    
    
    
    
t0 = 0.0       # Initial time
t1 = 100000.0 
t = np.linspace(t0, t1, 100000)

point=0
size=400

plt.plot(t[point:point+size], y_total[point:point+size, 0], label='prey')
plt.plot(t[point:point+size], y_total[point:point+size, 1], label='predator')
plt.xlabel('time')
plt.ylabel('population')
#plt.grid(linestyle=':')
plt.title('Lotka-Volterra equations')
plt.legend(loc=1)
plt.savefig('./Figures/Lotka_Predictions_data_first400.pdf')
plt.show()

shift=1
sequence_length=12
pred_mode=1

total_length=sequence_length+shift

result = []
for index in range(len(y) - total_length):
        
        i=y[index: index + total_length]
        k=i[:sequence_length]
        j=np.array(i[total_length-1])
        j=j.reshape(1,2)
        k=np.append(k,j,axis=0)
        result.append(k)
        
result = np.array(result)

pickle.dump(y, open('./Data/Lotka_Volterra2.p', "wb"))

d1 = pickle.load(open("./Data/Lotka_Volterra2.p", "rb"))



