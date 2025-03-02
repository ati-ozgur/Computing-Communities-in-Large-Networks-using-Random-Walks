######################################################################
# Computing Communities in Large Networks Using Random Walks
# L42: Assessment 1
# Jan Ondras (jo356), Trinity College
######################################################################
# Walktrap algorithm
######################################################################

import numpy as np
import networkx as nx
from heapq import heappush, heappop
import copy

# Return set of partitions and communities for graph G after t steps of random walk
def walktrap(G, t, add_self_edges=True, verbose=False):

    ##################################################################
    class Community:
        def __init__(self, new_C_id, C1=None, C2=None):
            self.id = new_C_id
            # New community from single vertex
            if C1 == None:
                self.size = 1
                self.P_c = P_t[self.id] # probab vector
                self.adj_coms = {}
                self.vertices = set([self.id])
                self.internal_weight = 0. 
                self.total_weight = self.internal_weight + (len([id for id, x in enumerate(A[self.id]) if x == 1. and id != self.id])/2.) #External edges have 0.5 weight, ignore edge to itself
            # New community by merging 2 older ones
            else:
                self.size = C1.size + C2.size
                self.P_c = (C1.size * C1.P_c + C2.size * C2.P_c) / self.size
                # Merge info about adjacent communities, but remove C1, C2
                self.adj_coms =dict( C1.adj_coms.items() | C2.adj_coms.items()) 

                del self.adj_coms[C1.id]
                del self.adj_coms[C2.id]
                self.vertices = C1.vertices.union(C2.vertices)
                weight_between_C1C2 = 0.
                for v1 in C1.vertices:
                    for id, x in enumerate(A[v1]):
                        if x == 1. and id in C2.vertices:
                            weight_between_C1C2 += 1.
                self.internal_weight = C1.internal_weight + C2.internal_weight + weight_between_C1C2
                self.total_weight = C1.total_weight + C2.total_weight

        def modularity(self):
            return ( self.internal_weight - (self.total_weight*self.total_weight/G_total_weight) ) / G_total_weight
    ##################################################################
    
    # If needed, add self-edges
    if add_self_edges:
        for v in G.nodes:
            G.add_edge(v, v)
    
    #G = nx.convert_node_labels_to_integers(G) # ensure that nodes are represented by integers starting from 0
    N = G.number_of_nodes()
    
    # Build adjacency matrix A
    A = np.array(nx.to_numpy_matrix(G))

    # Build transition matrix P from adjacency matrix
    # and diagonal degree matrix Dx of negative square roots degrees 
    Dx = np.zeros((N,N))
    P = np.zeros((N,N))
    for i, A_row in enumerate(A):
        d_i = np.sum(A_row)
        P[i] = A_row / d_i
        Dx[i,i] = d_i ** (-0.5)

    # Take t steps of random walk
    P_t = np.linalg.matrix_power(P, t)

    # Weight of all the edges excluding self-edges
    G_total_weight = G.number_of_edges() - N

    # Total number of all communities created so far
    community_count = N
    # Dictionary of all communities created so far, indexed by comID
    communities = {}
    for C_id in range(N):
        communities[C_id] = Community(C_id)

    # Minheap to store delta sigmas between communitites: <deltaSigma(C1,C2), C1_id, C2_id>
    min_sigma_heap = []
    for e in G.edges:
        C1_id = e[0]
        C2_id = e[1]
        if C1_id != C2_id:
            # Apply Definition 1 and Theorem 3
            ds = (0.5/N) * np.sum(np.square( np.matmul(Dx,P_t[C1_id]) - np.matmul(Dx,P_t[C2_id]) ))
            heappush(min_sigma_heap, (ds, C1_id, C2_id))
            # Update each community with its adjacent communites
            communities[C1_id].adj_coms[C2_id] = ds
            communities[C2_id].adj_coms[C1_id] = ds  

    # Record delta sigmas of partitions merged at each step
    delta_sigmas = []
    # Store IDs of current communities for each k
    # Partitions is a list of length k that stores IDs of communities for each partitioning
    partitions = [] # at every step active communities are in the last entry of 'partitions'
    # Make first partition, single-vertex communities
    partitions.append(set(np.arange(N)))
    # Calculate modularity Q for this partition
    modularities = [np.sum([communities[C_id].modularity() for C_id in partitions[0]])]
    if verbose:
        print("Partition 0: ", partitions[0])
        print("Q(0) = ", modularities[0])

    for k in range(1, N):
        # Current partition: partitions[k-1]
        # New partition to be created in this iteration: partitions[k]

        # Choose communities C1, C2 to merge, according to minimum delta sigma
        # Need to also check if C1_id and C2_id are communities at the current partition partitions[k-1]
        while not not min_sigma_heap:
            delta_sigma_C1C2, C1_id, C2_id = heappop(min_sigma_heap)
            if C1_id in partitions[k-1] and C2_id in partitions[k-1]:
                break
        # Record delta sigma at this step
        delta_sigmas.append(delta_sigma_C1C2)

        # Merge C1, C2 into C3, assign to it next possible ID, that is C3_ID = totComCnt
        C3_id = community_count
        community_count += 1 # increase for the next one
        communities[C3_id] = Community(C3_id, communities[C1_id], communities[C2_id])

        # Add new partition (k-th)
        partitions.append(copy.deepcopy(partitions[k-1]))
        partitions[k].add(C3_id) # add C3_ID
        partitions[k].remove(C1_id)
        partitions[k].remove(C2_id)

        # Update delta_sigma_heap with entries concerning community C3 and communities adjacent to C1, C2
        # Check all new neighbours of community C3
        for C_id in communities[C3_id].adj_coms.keys():
            # If C is neighbour of both C1 and C2 then we can apply Theorem 4
            if (C_id in communities[C1_id].adj_coms) and (C_id in communities[C2_id].adj_coms):
                delta_sigma_C1C = communities[C1_id].adj_coms[C_id]
                delta_sigma_C2C = communities[C2_id].adj_coms[C_id]
                # Apply Theorem 4 to (C, C3)
                ds = ( (communities[C1_id].size + communities[C_id].size)*delta_sigma_C1C + (communities[C2_id].size + communities[C_id].size)*delta_sigma_C2C - communities[C_id].size*delta_sigma_C1C2 ) / (communities[C3_id].size + communities[C_id].size)

            # Otherwise apply Theorem 3 to (C, C3)
            else:
                ds = np.sum(np.square( np.matmul(Dx,communities[C_id].P_c) - np.matmul(Dx,communities[C3_id].P_c) )) * communities[C_id].size*communities[C3_id].size / ((communities[C_id].size + communities[C3_id].size) * N)

            # Update min_sigma_heap and update delta sigmas between C3 and C
            heappush(min_sigma_heap, (ds ,C3_id , C_id))
            communities[C3_id].adj_coms[C_id] = ds
            communities[C_id].adj_coms[C3_id] = ds  

        # Calculate and store modularity Q for this partition
        modularities.append(np.sum([communities[C_id].modularity() for C_id in partitions[k]]))
            
        if verbose:
            print("Partition ", k, ": ", partitions[k])
            print("\tMerging ", C1_id, " + ", C2_id, " --> ", C3_id)
            print("\tQ(", k, ") = ", modularities[k])
            print("\tdelta_sigma = ", delta_sigmas[k-1])

    return np.array(partitions), communities, np.array(delta_sigmas), np.array(modularities)






