# -*- coding: utf-8 -*-

import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
gamma=0.3
eta=0.3
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def fit(x, y, coef = 2):
    x_fit=np.linspace(np.min(x),np.max(x),100)
    coef = np.polyfit(x, y, coef)
    
    f=np.poly1d(coef)
    
    x_fit=np.linspace(np.min(x),np.max(x),10)
    y_fit=f(x_fit)
    return x_fit, y_fit

B=0.2
C=0.1

centralities=list(load_dict('centralities/M_stmarks_centralities_'+str(int(100*B))+'_'+str(int(100*C))))
if len(centralities) >= 10:
    centralities=centralities[:8] + [centralities[9]]

names=['DC','EC','BC','CC','PR', 'GC','SP', 'LPP', 'PTC']
colors = ['#8DDFBF','#328CA0', '#006837']

fig=plt.figure(figsize=(30,30))
for i in range(len(centralities)):
    for j in range(len(centralities)):
        if(i<=j):
            continue
        ax=fig.add_subplot(len(centralities),len(centralities),i*len(centralities)+j+1)
        ax.scatter(centralities[i], centralities[j], s=200, marker = 'o', c='none', edgecolors=colors[1])
        x_fit, y_fit = fit(centralities[i], centralities[j])
        ax.plot(x_fit, y_fit, linewidth=4, color=colors[2])
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))

        ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
        bwith=2.5
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='x')
        ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')

        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel(names[i],fontsize=35)
        plt.ylabel(names[j],fontsize=35)
        plt.tight_layout()
fig.subplots_adjust(wspace=0.6,hspace=0.6)
plt.tight_layout()

plt.savefig('figures/M_stmarks_comparision_'+str(int(100*B))+'_'+str(int(100*C))+'.pdf',dpi=300,bbox_inches='tight')
plt.show()