# -*- coding: utf-8 -*-

import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

params=[0.4,0.7,1]
positions=[[0.5,0.35,0.425,0.3],
           [0.5,0.35,0.425,0.3],
           [0.35,0.35,0.3,0.3]]
xys=[(10,3000),(10,1),(10,0.5)]

for (k,B) in enumerate(params):
    simulation_xs=load_dict('states/E_karate_xs_'+str(int(100*B)))
    simulation_ts=load_dict('states/E_karate_ts_'+str(int(100*B)))
    
    centralities=list(load_dict('centralities/E_karate_centralities_'+str(int(100*B))))
    if len(centralities) >= 10:
        centralities=centralities[:8] + [centralities[9]]
    
    ranks = {index: np.argsort(centralities[index])[::-1] for index in range(len(centralities))}
    
    colors=['black','green','brown','#3233A6','#A632A2','#D0B0B0','#00CBD8','#FEFF99','#FF0000']
    names=['DC','EC','BC','CC','PR', 'GC','SP', 'LPP', 'PTC']
    
    xs_selected=dict()
    
    # if(k==2):
    #     fig=plt.figure(figsize=(8.5,6))
    #     ax=fig.add_subplot(111)
    # else:
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(111)
    
    x_min=10000
    x_max=0
    y_min=10000
    y_max=0
    ax1=fig.add_axes(positions[k])
    for i in ranks:
        print(names[i])
        x=np.mean([simulation_xs[ranks[i][j]] for j in range(10)],axis=0)
        t=np.mean([simulation_ts[ranks[i][j]] for j in range(10)],axis=0)
        ax1.plot(t[-50:],x[-50:],color=colors[i],linewidth=2.5)
        x_min=np.min([x_min,np.min(t[-50:])])
        x_max=np.max([x_max,np.max(t[-50:])])
        y_min=np.min([y_min,np.min(x[-50:])])
        y_max=np.max([y_max,np.max(x[-50:])])
    
    y_length=y_max-y_min
    
    ax1.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
    bwith=2.5
    ax1.spines['bottom'].set_linewidth(bwith)
    ax1.spines['left'].set_linewidth(bwith)
    ax1.spines['right'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.ticklabel_format(style='sci', scilimits=(-1,1), axis='x')
    ax1.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
    ax1.tick_params(axis='x',labelsize=15)
    ax1.tick_params(axis='y',labelsize=15)
    
    for i in ranks:
        x=np.mean([simulation_xs[ranks[i][j]] for j in range(10)],axis=0)
        t=np.mean([simulation_ts[ranks[i][j]] for j in range(10)],axis=0)
        ax.plot(t,x,color=colors[i],linewidth=2.5,label=names[i])
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    rect=plt.Rectangle(xy=(x_min,y_min-0.2*y_length-100), width=x_max-x_min, height=1.4*y_length+200,facecolor='none',linewidth=1,edgecolor='gray')
    ax.add_patch(rect)
    
    ax.arrow((x_min+x_max)/2,y_min-0.2*y_length-100,-2,-400,width=0.08,head_width=1,head_length=150,facecolor='gray',linewidth=1,edgecolor='gray')
    ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
    bwith=2.5
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    # ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='x')
    ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
    
    # if(k==2):
    #     ax.legend(fontsize=20,loc=2,bbox_to_anchor=(1.1,1))
    
    ax.tick_params(axis='x',labelsize=25)
    ax.tick_params(axis='y',labelsize=25)
    ax.set_xlabel(r'$t$',fontsize=35)
    ax.set_ylabel(r'$\mathcal{E}_i(t)$',fontsize=35)
    ax.set_title(r'$B={}$'.format(B),fontsize=35)
    plt.tight_layout()
    plt.savefig('figures/E_karate_trend_'+str(int(100*B))+'.pdf',dpi=300,bbox_inches='tight')
    plt.show()