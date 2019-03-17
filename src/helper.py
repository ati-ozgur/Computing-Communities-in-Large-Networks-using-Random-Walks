import networkx as nx
import copy
import time
import numpy as np
from matplotlib import pyplot as plt
from walktrap import walktrap
######################################################################
# Calculate Rand index for partitions P1 and P2, given lists of sets
######################################################################
def calculate_rand_index(P1, P2):
    N = 0
    sum_intersect = 0.
    sum_C1 = 0.
    sum_C2 = np.sum([len(s)**2 for s in P2])
    for s1 in P1:
        N += len(s1)
        sum_C1 += len(s1)**2
        for s2 in P2:
            sum_intersect += len(s1.intersection(s2))**2
    return (N*N*sum_intersect - sum_C1*sum_C2) / ( 0.5*N*N*(sum_C1 + sum_C2) - sum_C1*sum_C2)
######################################################################
# Convert partition representation: set -> list of sets
######################################################################
def partition_set_to_sets(comms, partition):
    list_of_sets = []
    for C_id in partition:
        list_of_sets.append(copy.deepcopy(comms[C_id].vertices))
    return list_of_sets

######################################################################
# Convert partition representation: dictionary -> list of sets
######################################################################
def partition_dict_to_sets(d):
    inverse_dict = {}
    for k, v in d.items():
        if v in inverse_dict:
            inverse_dict[v].add(k)
        else:
            inverse_dict[v] = set([k])
        
    return inverse_dict.values()


######################################################################
# Convert edge list to Networkx Graph
######################################################################
def edge_list_to_graph(edges, verbose=False):
    G = nx.Graph()
    G.add_edges_from(edges)
    if verbose:
        print(G.number_of_edges(), " edges, ", G.number_of_nodes(), " nodes")
    return G

######################################################################
# Prepare partition for color plotting: as dictionary
######################################################################
def partition_to_plot(coms, partition):
    p_dict = {}
    for i, C_id in enumerate(partition):
        for v in coms[C_id].vertices:
            p_dict[v] = i
    return p_dict


######################################################################
# Comparison with Markov Clustering algorithm: return best partition
######################################################################
# import markov_clustering
# external script Python3: appended at the end of this file

######################################################################
# Comparison with Louvain algorithm: return best partition
######################################################################
import community

def compare_with_Louvain(G, add_self_edges=True, verbose=True):
#     # If needed, add self-edges
#     if add_self_edges:
#         for v in G.nodes:
#             G.add_edge(v, v)
    start_time = time.time()
    part = community.best_partition(G)
    Q = community.modularity(part, G)
    if verbose:
        print("Louvain algorithm:")
        print("\tOptimal number of communities: K = ", len(np.unique(part.values())))
        print("\tBest modularity: Q = ", Q)
        print("\tRuntime: ", time.time() - start_time, " seconds")
    return part # dictionary, suitable for plotting




######################################################################
# Experiments on real-graphs: comparison function
######################################################################

def compare_algos(G, K_true):
    
    G = nx.convert_node_labels_to_integers(G)
    pos = nx.spring_layout(G)
    print("Ground truth: ",K_true, " communities")

    # 1.) step
    # Q vs k for various t
    plt.figure()
    list_of_t = list(range(2,9))+[20,50,100]
    #list_of_t = [3,5,7,9,12,20,100]
    for t in list_of_t:
        parts, coms, _, Qs = walktrap(G, t)
        ks = np.arange(len(Qs))
        # Best number of communities
        K = len(Qs) - np.argmax(Qs)
        plt.plot(ks, Qs, label='t = ' + str(t) + ", K = " + str(K))
    plt.xlabel('iteration k')
    plt.ylabel('Modularity Q')
    plt.title('Modularity Q vs iteration')
    plt.legend()
    plt.show()
    # eta vs k for various t
    plt.figure()
    list_of_t = list(range(2,9))+[20,50,100]
    #list_of_t = [3,5,7,9,12,20,100]
    for t in list_of_t:
        parts, coms, ds, _ = walktrap(G, t)
        etas = ds[1:] / ds[0:-1]
        ks = np.arange(len(etas)) + 1
        # Best number of communities
        K = 1 + len(etas) - np.argmax(etas)
        plt.plot(ks, etas, label='t = ' + str(t) + ", K = " + str(K))
    plt.xlabel('iteration k')
    plt.ylabel('$\eta$')
    plt.title('Quality $\eta$ of partition vs iteration')
    plt.legend()
    plt.show()


    # 2.) Comparison

    ################## WT
    t = 2
    start_time = time.time()
    parts, coms, _, Qs = walktrap(G, t) 
    wt_time = time.time() - start_time
    Qmax_index = np.argmax(Qs)
    print("Walktrap ( t =",str(t),") algorithm:")
    print("\tOptimal number of communities: K = ", len(Qs) - Qmax_index)
    print("\tBest modularity: Q = ", Qs[Qmax_index])
    print("\tRuntime: ", wt_time, " seconds")
    my_best_part = partition_to_plot(coms, parts[Qmax_index])
    nx.draw(G, pos, node_color= list(my_best_part.values()))
    plt.show()

    t = 5
    start_time = time.time()
    parts, coms, _, Qs = walktrap(G, t) 
    wt_time = time.time() - start_time
    Qmax_index = np.argmax(Qs)
    print("Walktrap ( t =",str(t),") algorithm:")
    print("\tOptimal number of communities: K = ", len(Qs) - Qmax_index)
    print("\tBest modularity: Q = ", Qs[Qmax_index])
    print("\tRuntime: ", wt_time, " seconds")
    my_best_part = partition_to_plot(coms, parts[Qmax_index])
    nx.draw(G, pos, node_color= list(my_best_part.values()))
    plt.show()

    t = 8
    start_time = time.time()
    parts, coms, _, Qs = walktrap(G, t) 
    wt_time = time.time() - start_time
    Qmax_index = np.argmax(Qs)
    print("Walktrap ( t =",str(t),") algorithm:")
    print("\tOptimal number of communities: K = ", len(Qs) - Qmax_index)
    print("\tBest modularity: Q = ", Qs[Qmax_index])
    print("\tRuntime: ", wt_time, " seconds")
    my_best_part = partition_to_plot(coms, parts[Qmax_index])
    nx.draw(G, pos, node_color= list(my_best_part.values()))
    plt.show()

    ################## LO
    louvain_best_part = compare_with_Louvain(G)
    nx.draw(G, pos, node_color= list(louvain_best_part.values()))
    plt.show()

    ################## MC
    # external script
