# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import warnings
import pickle
warnings.filterwarnings('ignore')
from scipy.io import loadmat

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def calculate_edge_value(G,degrees):
    H=G.copy()
    H=nx.DiGraph(H)
    for (u,v) in H.edges:
        H.edges[u,v]['weight_hens']=(float(degrees[v,0])**(theta_J))
        H.edges[u,v]['Cij']=(float(degrees[v,0])**(theta_E)*float(degrees[u,0])**(theta_H))
    return H

def degree_centrality(G,degrees):
    G_length=len(G.nodes)
    for u in G.nodes:
        G.nodes[u]['degree_centrality'] = (float(degrees[u,0])/G_length)
    return G

def eigenvector_centrality(G):
    A=np.asmatrix(nx.to_numpy_array(G))
    vec=np.abs(np.linalg.eigh(A)[1][:,-1])
    for u in G.nodes:
        G.nodes[u]['eigenvector_centrality'] = vec[u,0]
    return G
    
def betweenness_centrality(G):
    G_length=len(G.nodes)
    for u in G.nodes:
        G.nodes[u]['betweenness_centrality']=0
    for u in G.nodes:
        for v in G.nodes:
            if(u==v):
                continue
            else:
                paths=[path for path in nx.all_shortest_paths(G,source=u,target=v)]
                path_length_uv=len([i for i in paths])
                for path in paths:
                    for q in path:
                        G.nodes[q]['betweenness_centrality']+=(1/path_length_uv/((G_length-1)*(G_length-2)))
    return G

def clossness_centrality(G):
    for u in G.nodes:
        path_length = nx.shortest_path_length(G, source=u)
        G.nodes[u]['clossness_centrality'] = 0
        for v in path_length:
            if(u==v):
                continue
            else:
                G.nodes[u]['clossness_centrality'] += (path_length[v]/(len(G.nodes)-1))
        G.nodes[u]['clossness_centrality']=1/G.nodes[u]['clossness_centrality']
    return G

def pagerank_centrality(G):
    H = nx.DiGraph(G)
    pr = nx.pagerank(H, alpha=0.9)
    for u in G.nodes:
        G.nodes[u]['pagerank_centrality']=pr[u]
    return G

def gravity_centrality(G, degrees):
    for u in G.nodes:
        path_length = nx.shortest_path_length(G, source=u)
        G.nodes[u]['gravity_centrality'] = 0
        for v in path_length:
            if(u==v):
                continue
            else:
                G.nodes[u]['gravity_centrality'] += (float(degrees[u,0])*float(degrees[v,0]) / path_length[v]**2)
    return G

def spreading_probability(G, degrees):
    average_degree=np.mean(degrees)
    distances=np.zeros(shape=(len(G.nodes),len(G.nodes)))
    for node in nx.shortest_path_length(G):
        for u in node[1]:
            distances[node[0],u]=node[1][u]
    
    for u in G.nodes:
        ans_sp=0
        to_nodes=np.where(distances[u,:]<=3)[0]
        for v in to_nodes:
            path_numbers=len([path for path in nx.all_shortest_paths(G,source=u,target=v)])
            ans_sp+=(path_numbers*(1/average_degree)**(distances[u,v]))
        G.nodes[u]['spreading_probability']=ans_sp
    return G

def local_propagation_probability(G, degrees):
    w=0.5
    beta_th=np.mean(degrees)/np.mean(np.power(degrees,2))
    gamma=w*beta_th
    distances=np.zeros(shape=(len(G.nodes),len(G.nodes)))
    for node in nx.shortest_path_length(G):
        for u in node[1]:
            distances[node[0],u]=node[1][u]
    
    core_numbers=nx.core_number(G_edge)
    def pro(u,x,i):
        if(i==1):
            return gamma
        else:
            length_i_1_nodes=np.where(distances[u,:]==i-1)[0]
            to_nodes=G[u]
            Gamma_i_1_set=[i for i in to_nodes if i in length_i_1_nodes]
            
            sub_pro=list()
            for y in Gamma_i_1_set:
                sub_pro.append(1-pro(u,y,i-1)*gamma)
            
            return 1-np.prod(sub_pro)
    
    def eta(u,x,i):
        to_nodes=np.where(distances[u,:]==i)[0]
        if(len(to_nodes)==0):
            return 0
        else:
            sub_Gamma=nx.subgraph(G, to_nodes)
            sub_degrees=nx.degree(sub_Gamma)
            return sub_degrees[x]/np.max(sub_degrees)
    
    def ks(x):
        return core_numbers[x]
    
    def CS(u,x,i):
        return pro(u,x,i)*(1+eta(u,x,i))*ks(x)
    
    for u in G.nodes:
        ans_lpp=0
        for i in range(1,4):
            for x in np.where(distances[u,:]==i)[0]:
                ans_lpp+=CS(u,x,i)
        G.nodes[u]['local_propagation_probability']=ans_lpp
    return G

def propagation_time_centrality1(G,H):
    G_length=len(G.nodes)
    for u in G.nodes:
        G.nodes[u]['propagation_time_centrality1']=0
    propagation_time=np.zeros(shape=(G_length,G_length))
    for v in H.nodes:
        for u in H.nodes:
            if(u==v):
                continue
            else:
                # paths=[path for path in nx.all_simple_paths(H, source=v, target=u, cutoff=int(nx.shortest_path_length(H,source=u,target=v)) + 1)]
                paths=[path for path in nx.all_shortest_paths(G,source=v,target=u)]
                
                value_multis=list()
                Cij_multis=list()
                
                for (k,path) in enumerate(paths):
                    value=0
                    Cij_multi=1
                    for i in range(0,len(path)-1):
                        Cij_multi*=H.edges[path[i],path[i+1]]['Cij']
                        value+=H.edges[path[i],path[i+1]]['weight_hens']
                    value_multis.append(value)
                    Cij_multis.append(Cij_multi)
                propagation_time[v,u]=np.sum([value*Cij_multi/np.sum(Cij_multis) for (value,Cij_multi) in zip(value_multis, Cij_multis)])
    for u in G.nodes:
        G.nodes[u]['propagation_time_centrality1']=0
        for v in G.nodes:
            if(u==v):
                continue
            else:
                G.nodes[u]['propagation_time_centrality1']+=(propagation_time[v,u]/(G_length-1))
    
    for u in G.nodes:
        G.nodes[u]['propagation_time_centrality1']=(1/G.nodes[u]['propagation_time_centrality1'])
    return G

def propagation_time_centrality2(G,H):
    G_length=len(G.nodes)
    for u in G.nodes:
        G.nodes[u]['propagation_time_centrality2']=0
    propagation_time=np.zeros(shape=(G_length,G_length))
    for v in H.nodes:
        for u in H.nodes:
            if(u==v):
                continue
            else:
                # paths=[path for path in nx.all_simple_paths(H, source=v, target=u, cutoff=int(nx.shortest_path_length(H,source=u,target=v)) + 1)]
                paths=[path for path in nx.all_shortest_paths(G,source=v,target=u)]
                
                value_multis=list()
                Cij_multis=list()
                
                for (k,path) in enumerate(paths):
                    value=0
                    Cij_multi=1
                    for i in range(0,len(path)-1):
                        Cij_multi*=H.edges[path[i],path[i+1]]['Cij']
                        value+=H.edges[path[i],path[i+1]]['weight_hens']
                    value_multis.append(value)
                    Cij_multis.append(Cij_multi)
                propagation_time[v,u]=np.sum([value*Cij_multi/np.sum(Cij_multis) for (value,Cij_multi) in zip(value_multis, Cij_multis)])
    for u in G.nodes:
        G.nodes[u]['propagation_time_centrality2']=0
        for v in G.nodes:
            if(u==v):
                continue
            else:
                G.nodes[u]['propagation_time_centrality2']+=(propagation_time[u,v]/(G_length-1))
    
    for u in G.nodes:
        G.nodes[u]['propagation_time_centrality2']=(1/G.nodes[u]['propagation_time_centrality2'])
    return G

G_edge=nx.read_weighted_edgelist('../../Networks/eco-stmarks.edges')
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
G_edge=nx.convert_node_labels_to_integers(G_edge)
A_edge=np.asmatrix(nx.to_numpy_array(G_edge))
A_edge=(A_edge>0.1)
G_edge=nx.from_numpy_array(np.asarray(A_edge))
G_edge.remove_edges_from(nx.selfloop_edges(G_edge))
G_edge=nx.subgraph(G_edge, max(nx.connected_components(G_edge)))
G_edge=nx.convert_node_labels_to_integers(G_edge)
A_edge=np.asmatrix(nx.to_numpy_array(G_edge))
degrees=np.sum(A_edge,axis=1)
print(len(G_edge.nodes),len(G_edge.edges),np.mean(degrees),nx.average_clustering(G_edge))

params1=[(B,0.1) for B in np.linspace(0.15,0.25,5)]
params2=[(0.2,C) for C in np.linspace(0.05,0.25,5)]

params=list()
params.extend(params1)
params.extend(params2)

a=1

for (B,C) in params:
    print(B,C)
    theta_J=-1
    theta_E=1-1/a
    theta_H=-2/a
    
    H_edge=calculate_edge_value(G_edge,degrees)
    
    centralities=list()
    G_edge = degree_centrality(G_edge, degrees)
    degree_cent = [G_edge.nodes[u]['degree_centrality'] for u in G_edge.nodes]
    centralities.append(degree_cent)
    
    G_edge = eigenvector_centrality(G_edge)
    eigenvector_cent = [G_edge.nodes[u]['eigenvector_centrality'] for u in G_edge.nodes]
    centralities.append(eigenvector_cent)
    
    G_edge = betweenness_centrality(G_edge)
    betweenness_cent = [G_edge.nodes[u]['betweenness_centrality'] for u in G_edge.nodes]
    centralities.append(betweenness_cent)
    
    G_edge = clossness_centrality(G_edge)
    clossness_cent = [G_edge.nodes[u]['clossness_centrality'] for u in G_edge.nodes]
    centralities.append(clossness_cent)
    
    G_edge = pagerank_centrality(G_edge)
    pagerank_cent = [G_edge.nodes[u]['pagerank_centrality'] for u in G_edge.nodes]
    centralities.append(pagerank_cent)
    
    G_edge = gravity_centrality(G_edge, degrees)
    gravity_cent = [G_edge.nodes[u]['gravity_centrality'] for u in G_edge.nodes]
    centralities.append(gravity_cent)
    
    G_edge = spreading_probability(G_edge, degrees)
    spreading_pro = [G_edge.nodes[u]['spreading_probability'] for u in G_edge.nodes]
    centralities.append(spreading_pro)
    
    G_edge = local_propagation_probability(G_edge, degrees)
    local_propagation_pro = [G_edge.nodes[u]['local_propagation_probability'] for u in G_edge.nodes]
    centralities.append(local_propagation_pro)
    
    G_edge = propagation_time_centrality1(G_edge,H_edge)
    propagation_time_cent1 = [G_edge.nodes[u]['propagation_time_centrality1'] for u in G_edge.nodes]
    centralities.append(propagation_time_cent1)
    
    G_edge = propagation_time_centrality2(G_edge,H_edge)
    propagation_time_cent2 = [G_edge.nodes[u]['propagation_time_centrality2'] for u in G_edge.nodes]
    centralities.append(propagation_time_cent2)
    
    save_dict(centralities, 'centralities/M_stmarks_centralities_'+str(int(100*B))+'_'+str(int(100*C)))