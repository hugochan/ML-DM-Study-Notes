#!/usr/bin/python

from copy import deepcopy


# dfs_codes format: [(from_iid, to_iid, from_label, to_label, e_label), ...]
# isomorphism format: {dfs_id: id, }, ps: dfs_id is iid
# graph format: [(from_id, to_id, from_label, to_label, e_label), ...]
# dataset format: [graph, ...]

def gSpan(dfs_codes, dataset, minsup):
    ext_sups = extend_right_most_path(dfs_codes, dataset) # extensions and supports
    for t, sup in ext_sups:
        new_dfs_codes = deepcopy(dfs_codes)
        new_dfs_codes.append(t) # extend the code with extended edge tuple t
        # recursively call gSpan() if code is frequent and canonical
        if sup >= minsup and is_canonical(new_dfs_codes):
            gSpan(new_dfs_codes, dataset, minsup)

def extend_right_most_path(dfs_codes, dataset):
    right_most_path = build_right_most_path(dfs_codes) # build the right most path through the dfs code
    right_most_nodes = [] # nodes on the rightmost path
    for x in right_most_path:
        right_most_nodes.extend(x[:2])
    right_most_nodes = list(set(right_most_nodes))
    right_most_child = right_most_path[0][1] # rightmost child (dfs id)
    exts =[] # set of extensions

    for i in range(len(dataset)):
        graph = dataset[i]
        if not dfs_codes:
            f = list(set([(0, 1) + x[2:] for x in graph])) # forward edges
            exts.extend(zip(f, [i for x in range(len(f))]))
        else:
            isomorphisms = get_sub_graph_isomorphisms(dfs_codes, graph)
            for iso in isomorphisms:
                iso_reverse = dict(zip(iso.values(), iso.keys()))
                neighbors = get_neighbors(iso[right_most_child], graph)
                for x in neighbors:
                    if iso_reverse.has_key(x) and iso_reverse[x] in right_most_nodes:
                        if not edge_in_dfs_graph(right_most_child, iso_reverse[x], dfs_codes):
                            labels = get_labels_by_edge((iso[right_most_child], x), graph)
                            b = (right_most_child, iso_reverse[x], labels[0], labels[1], labels[2]) # backward edges
                            exts.append((b, i))
                for u in right_most_nodes:
                    neighbors = get_neighbors(iso[u], graph)
                    for x in neighbors:
                        if not iso_reverse.has_key(x):
                            labels = get_labels_by_edge((iso[u], x), graph)
                            f = (u, right_most_child + 1, labels[0], labels[1], labels[2]) # forward edges
                            exts.append((f, i))

    # what if an extension occurs in one graph for multiple times?
    ext_sups = []
    extensions = [x[0] for x in exts]
    for s in set(extensions):
        sup = extensions.count(s)
        ext_sups.append((s, sup))
    # tuple sorted order???
    ext_sups = sorted(ext_sups, key=lambda d:d[1], reverse=True)
    return ext_sups

def get_node_dfs_label(iid, dfs_codes):
    for code in dfs_codes:
        if iid == code[0]:
            return code[2]
        elif iid == code[1]:
            return code[3]
    return None

def get_labels_by_edge(edge, graph): # from graph
    for each in graph:
        if edge[0] == each[0] and edge[1] == each[1]:
            return [each[2], each[3], each[4]]
        elif edge[0] == each[1] and edge[1] == each[0]:
            return [each[3], each[2], each[4]]
    return None

def edge_in_dfs_graph(iid, iid2, dfs_codes):
    for code in dfs_codes:
        if iid in code[:2] and iid2 in code[:2]:
            return True
    return False

def get_neighbors(node_id, graph):
    neighbors = []
    for edge in graph:
        if node_id == edge[0]:
            neighbors.append(edge[1])
        elif node_id == edge[1]:
            neighbors.append(edge[0])
    return neighbors


def get_sub_graph_isomorphisms(dfs_codes, graph):
    pass

def is_canonical(dfs_codes):
    pass

def build_right_most_path(dfs_codes):
    """
    Build the right most path through the DFS codes

    """
    path = []
    prev_iid = -1
    for idx, code in reversed(list(enumerate(dfs_codes))):
        if code[0] < code[1] and (len(path) == 0 or prev_iid == code[1]):
            prev_iid = code[0]
            path.append(code)
    #print path
    return path

