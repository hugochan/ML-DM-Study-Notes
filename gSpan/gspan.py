#!/usr/bin/python

from copy import deepcopy

# dfs_codes format: [(from_iid, to_iid, from_label, to_label, e_label), ...]
# isomorphism format: {dfs_id: id, }, ps: dfs_id is iid
# graph format: [(from_id, to_id, from_label, to_label, e_label), ...]
# dataset format: [graph, ...]

def gSpan(dfs_codes, dataset, minsup, patterns):
    if patterns == []:
        patterns.append(dfs_codes)
    ext_sups = extend_right_most_path(dfs_codes, dataset) # extensions and supports
    for t, sup in ext_sups:
        new_dfs_codes = deepcopy(dfs_codes)
        new_dfs_codes.append(t) # extend the code with extended edge tuple t

        # recursively call gSpan() if code is frequent and canonical
        if sup >= minsup and is_canonical(new_dfs_codes):
            patterns.append(new_dfs_codes)
            gSpan(new_dfs_codes, dataset, minsup, patterns)
    return patterns

def extend_right_most_path(dfs_codes, dataset):
    if dfs_codes:
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
            f = list(set([(0, 1) + x[2:] for x in graph] + [(0, 1) + (x[3], x[2], x[4]) for x in graph]))
            exts.extend(zip(f, [i for x in range(len(f))]))
        else:
            isomorphisms = get_subgraph_isomorphisms(dfs_codes, graph)
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

    # number of distinct graph ids that support tuple s
    ext_sups = []
    exts = list(set(exts))
    extensions = [x[0] for x in exts]
    for s in set(extensions):
        sup = extensions.count(s)
        ext_sups.append((s, sup))
    # sorting tuples in dfs order
    ext_sups = sorted(ext_sups, cmp=dfs_code_cmp, key=lambda e:e[0])
    return ext_sups

def get_subgraph_isomorphisms(dfs_codes, graph):
    isomorphisms = []
    nodes = []
    for x in graph:
        nodes.extend([(x[0], x[2]), (x[1], x[3])])
    nodes = list(set(nodes))
    for node_id, node_label in nodes:
        if node_label == get_node_dfs_label(0, dfs_codes):
            isomorphisms.append({0: node_id}) # dfs id -> node id

    for code in dfs_codes:
        tmp_isomorphisms = []
        for iso in isomorphisms:
            iso_reverse = dict(zip(iso.values(), iso.keys()))
            if code[1] > code[0]: # forward edge
                neighbors = get_neighbors(iso[code[0]], graph)
                for x in neighbors:
                    if not iso_reverse.has_key(x):
                        labels = get_labels_by_edge((iso[code[0]], x), graph)
                        if labels[1] == code[3] and labels[2] == code[4]:
                            tmp_iso = deepcopy(iso)
                            tmp_iso[code[1]] = x
                            tmp_isomorphisms.append(tmp_iso)
            else: # backward edge
                if iso[code[1]] in get_neighbors(iso[code[0]], graph):
                    tmp_isomorphisms.append(iso)
        isomorphisms = tmp_isomorphisms
    return isomorphisms

def is_canonical(dfs_codes):
    # graph corresponding to the dfs code
    graph = get_graph_by_dfs_codes(dfs_codes)
    canonical_dfs_code = []
    for i in range(len(dfs_codes)):
        ext_sups = extend_right_most_path(canonical_dfs_code, [graph]) # extensions of canonical_dfs_code
        ext = ext_sups[0][0] # least rightmost edge extension of canonical_dfs_code
        if dfs_code_cmp(ext, dfs_codes[i]) == -1: # ext < dfs_codes[i]
            return False # // canonical_dfs_code is smaller, thus dfs_codes is not canonical
        canonical_dfs_code.append(ext)
    return True

def get_graph_by_dfs_codes(dfs_codes):
    return dfs_codes

def dfs_code_cmp(a, b):
    ret = dfs_edge_cmp(a[:2], b[:2]) # compare edges
    if ret == -1:
        return -1
    elif ret == 0: # compare labels
        if a[2] == b[2] and a[3] == b[3] and a[4] == b[4]:
            return 0
        if a[2] < b[2] or (a[2] == b[2] and a[3] < b[3]) or \
            (a[2] == b[2] and a[3] == b[3] and a[4] < b[4]):
            return -1
        else:
            return 1
    else:
        return 1

def dfs_edge_cmp(e1, e2):
    """
    compare dfs edge order
    """
    if e1[0] < e1[1] and e2[0] < e2[1]: # they are both forward edges
        if e1[1] < e2[1] or (e1[1] == e2[1] and e1[0] > e2[0]):
            return -1 # e1 <e e2
        elif e1[1] == e2[1] and e1[0] == e2[0]:
            return 0 # e1 =e e2
        else:
            return 1 # e1 >e e2
    if e1[0] > e1[1] and e2[0] > e2[1]: # they are both backward edges
        if e1[0] < e2[0] or (e1[0] == e2[0] and e1[1] < e2[1]):
            return -1
        elif e1[0] == e2[0] and e1[1] == e2[1]:
            return 0
        else:
            return 1
    if e1[0] < e1[1] and e2[0] > e2[1]: # e1 is a forward edge and e2 is a backward edge
        if e1[1] <= e2[0]:
            return -1
        else:
            return 1
    if e1[0] > e1[1] and e2[0] < e2[1]: # e1 is a backward edge and e2 is a forward edge
        if e2[1] <= e1[0]:
            return 1
        else:
            return -1

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

def load_data(file_in, sep=' '):
    data = []
    index = -1 # graph index
    try:
        with open(file_in, 'r') as f:
            for i in f:
                tmp = i.rstrip('\r\n').split(sep)
                if tmp[0] == 't': # graph
                    data.append([])
                    node_label = {}
                    index += 1
                elif tmp[0] == 'v': # vertex
                    node_label[tmp[1]] = tmp[2] # id -> label
                elif tmp[0] == 'e': # edge
                    data[index].append((tmp[1], tmp[2], node_label[tmp[1]], node_label[tmp[2]], tmp[3]))
    except Exception, e:
        print e
        exit()
    f.close()
    return data

if __name__ == '__main__':
    import sys
    try:
        in_file = sys.argv[1]
        minsup = int(sys.argv[2])
    except:
        print "ERROR: missing or invalid arguments"
        exit()
    dataset = load_data(in_file)
    patterns = gSpan([], dataset, minsup, patterns=[])
    for i in range(len(patterns)):
        print "pattern %s"% (i + 1)
        if patterns[i] == []:
            print "()"
        for each in patterns[i]:
            print each
        print

