"""
SCAN: A Structural Clustering Algorithm for Networks
As described in http://ualr.edu/nxyuruk/publications/kdd07.pdf
"""

from collections import deque, defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import tqdm

def struct_similarity(vcols, wcols, v_nodeList, w_nodeList):
    """ Compute the similartiy normalized on geometric mean of vertices"""
    # count the similar rows for unioning edges
    v_set = set(vcols)
    w_set = set(wcols)
    union_set = v_set | w_set  # 使用集合的并集操作
    
    max_sim=0
    min_sim=0
    for node in union_set:
        if v_nodeList[node][1] > w_nodeList[node][1]:
            max_sim += v_nodeList[node][1]
            min_sim += w_nodeList[node][1]
        else:
            max_sim += w_nodeList[node][1]
            min_sim += v_nodeList[node][1]
    
    ans = min_sim / max_sim
    return ans

def neighborhood(G, vertex_v, target_v, eps, G_adjList):
    """ Returns the neighbors, as well as all the connected vertices """
    N = deque()
    vcols = vertex_v.tocoo().col
    #check the similarity for each connected vertex
    v_nodeList = G_adjList[target_v]
    for index in vcols:
        wcols = G[index,:].tocoo().col
        w_nodeList = G_adjList[index]
        
        if struct_similarity(vcols, wcols, v_nodeList, w_nodeList)> eps:
            N.append(index)
    return N, vcols

def scan_weight(G, eps =0.7, mu=2.):
    """
    Vertex Structure = sum of row + itself(1)
    Structural Similarity is the geometric mean of the 2Vertex size of structure
    """
    # 转成邻接表
    
    
    c = 0
    v = G.shape[0]
    
    G_adjList = defaultdict(list)
    G_arr = G.toarray()
    for vertex in range(v):
        G_adjList[vertex] = [(node, sim) for node, sim in enumerate(G_arr[vertex])]

    print("adjList constructed")
    
    # All vertices are labeled as unclassified(-1)
    vertex_labels = -np.ones(v, dtype=np.int32)
    # start with a neg core(every new core we incr by 1)
    cluster_id = -1
    for vertex in tqdm.tqdm(range(v)):
        N ,vcols = neighborhood(G, G[vertex,:], vertex, eps, G_adjList)
        # must include vertex itself
        N.appendleft(vertex)
        if len(N) >= mu:
            #print "we have a cluster at: %d ,with length %d " % (vertex, len(N))
            # gen a new cluster id (0 indexed)
            cluster_id +=1
            while N:
                y = N.pop()
                R , ycols = neighborhood(G, G[y,:], y, eps, G_adjList)
                # include itself
                R.appendleft(y)
                # (struct reachable) check core and if y is connected to vertex
                if len(R) >= mu and y in vcols:
                    #print "we have a structure Reachable at: %d ,with length %d " % (y, len(R))
                    while R:
                        r = R.pop()
                        label = vertex_labels[r]
                        # if unclassified or non-member
                        if (label == -1) or (label==0):
                            vertex_labels[r] =  cluster_id
                        # unclassified ??
                        if label == -1:
                            N.appendleft(r)
        else:
            vertex_labels[vertex] = 0
    
    #classify non-members
    # 将-2,-3当做是一个Community（新的类）
    for index in np.where(vertex_labels ==0)[0]:
        ncols= G[index,:].tocoo().col
        if len(ncols) >=2:
            ## mark as a hub
            vertex_labels[index] = -2 
            continue
            
        else:
            ## mark as outlier
            vertex_labels[index] = -3
            continue

    return vertex_labels

if __name__=='__main__':

    # Based on Example from paper
    rows = [0,0,0,0,1,1,1,2,2,2,3,3,3,3,4,4,4,4,
            5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,8,8,8,
            9,9,9,9,10,10,10,10,11,11,11,11,
            12,12,12,12,12,13]
    cols = [1,4,5,6,0,5,2,1,5,3,2,4,5,6,0,3,5,6,
            0,1,2,3,4,4,0,3,7,11,10,6,11,12,8,7,
            12,9,8,12,10,13,9,12,11,6,7,12,10,6,
            7,8,9,10,11,9]
    data = np.ones(len(rows))
    G =csr_matrix((data,(rows,cols)),shape=(14,14))

    #print G.todense()
    #print neighborhood(G, G[0,:],.4 )
    print(scan_weight(G, .7, 2))