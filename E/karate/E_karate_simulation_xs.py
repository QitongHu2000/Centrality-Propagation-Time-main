# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp, odeint
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.1
eta=0.3
alpha = 1
t_0=1000
t_1=20
n=100
from scipy.io import loadmat

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def simulation(A,source):
    def F(A,x):
        return np.asmatrix(-B*x+alpha*np.multiply(1-x,A*x))
    
    def Fun(t,x):
        x=np.asmatrix(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        return dx
    
    def Fun_1(x,t,source):
        x=np.asmatrix(x).T
        dx=F(A,x).tolist()
        dx=[dx[i][0] for i in range(len(dx))]
        dx[source]=0
        return dx
    
    def sim_first(A):
        x_0=np.zeros(np.shape(A)[0])
        sol=solve_ivp(Fun, [0,t_0], x_0, rtol=1e-10, atol=1e-10)
        xs=sol.y.T
        t=sol.t
        x=xs[-1,:].tolist()
        return x, t[-1]
    
    def sim_second(A,x,t_0,source):
        x[source]+=gamma
        ts=np.linspace(t_0,t_0+t_1,n)
        xs=odeint(Fun_1,x,ts, rtol=1e-10,atol=1e-10,args=(source,))
        return np.asmatrix(xs), np.array(ts)
    
    x,t=sim_first(A)
    xs,ts=sim_second(A,x.copy(),t,source)
    return x, xs, ts

G_edge = nx.karate_club_graph()
G_edge=nx.convert_node_labels_to_integers(G_edge)
A_edge=np.asmatrix(nx.to_numpy_array(G_edge))
G_edge=nx.from_numpy_array(A_edge)
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
A_edge=np.asmatrix(nx.to_numpy_array(G_edge))
degrees=np.sum(A_edge,axis=1)
G_edge_length=len(G_edge.nodes())

params=[B for B in np.linspace(0.4,1,5)]

for B in params:
    print(B)
    simulation_xs=list()
    simulation_ts=list()
    
    for source in G_edge.nodes:
        x1, xs_perturbed, ts_perturbed =simulation(A_edge,source)
        entropy=list()
        Delta_xs=xs_perturbed-xs_perturbed[0,:]
        Delta_x=xs_perturbed[-1,source]-x1[source]
        entropy=np.diag(Delta_xs*Delta_xs.T)/Delta_x**2
        simulation_xs.append(entropy.tolist())
        simulation_ts.append((ts_perturbed-t_0).tolist())
    
    save_dict(simulation_xs, 'states/E_karate_xs_'+str(int(100*B)))
    save_dict(simulation_ts, 'states/E_karate_ts_'+str(int(100*B)))