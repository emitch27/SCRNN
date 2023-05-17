import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
import data



#build List of lists of dictionaries representing simplicial complexes (entries of outer list corespond to time bins; entries of inner lists correspond to simplicial dimensions)
def build_all_complexes(full_binary, st_ind_dict, max_simplex_dim):
    '''
    Inputs:
        full_binary:     binarized spike count matrix
        st_ind_dict:     list of dictionaries that assign an index to each simplicial complex (each list entry corresponds to differnt simplicial dimension)
        max_simplex_dim: maximum dimension of simplicial complex

    Returns: List of lists of dictionaries representing simplicial complexes
    '''
    time_bins = full_binary.shape[1]
    Sts = []
    for i in range(time_bins):
        # print(i)
        st_dict = data.create_simplex(full_binary[:,i], max_simplex_dim, st_ind_dict)
        Sts.append(st_dict)

    return Sts



#Build the boundary operators from a list of simplices.
#modification of code from https://github.com/stefaniaebli/simplicial_neural_networks/blob/main/data/s2_6_complex_to_laplacians.py
def build_boundaries(simplices):
    '''
    Inputs:
        simplices: list of dictionaries that assign an index to each simplicial complex (each list entry corresponds to different simplicial dimension)

    Returns: list of incidence matrices saved as sparse coo matrices (each list entry corresponds to different simplicial dimension)
    '''
    boundaries = list()
    for d in range(1, len(simplices)):#Exclude looking at nodes
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in simplices[d].items(): #iteratess through simplex and its index
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1)**i)
                face = simplex.difference({left_out})
                idx_faces.append(simplices[d-1][face])
        assert len(values) == (d+1) * len(simplices[d])
        boundary = coo_matrix((values, (idx_faces, idx_simplices)),
                                     dtype=np.float32,
                                     shape=(len(simplices[d-1]), len(simplices[d])))
        boundaries.append(boundary)
    return boundaries


#Build upper and lower Laplacian operators from the boundary operators.
#modification of code from https://github.com/stefaniaebli/simplicial_neural_networks/blob/main/data/s2_6_complex_to_laplacians.py
def build_ups_downs(boundaries):
    '''
    Inputs:
        boundaries: list of incidence matrices saved as sparse coo matrices (each list entry corresponds to different simplicial dimension)

    Returns: 
        ups:   list of upper laplacians (each list entry corresponds to different simplicial dimension)
        downs: list of lower laplacians (each list entry corresponds to different simplicial dimension)
    '''
    ups = list()
    downs = list()
    up = data.coo2tensor(coo_matrix(boundaries[0] @ boundaries[0].T, dtype=np.float32))
    ups.append(up)
    for d in range(len(boundaries)-1):
        down = boundaries[d].T @ boundaries[d]
        up = boundaries[d+1] @ boundaries[d+1].T
        downs.append(data.coo2tensor(coo_matrix(down, dtype=np.float32)))
        ups.append(data.coo2tensor(coo_matrix(up, dtype=np.float32)))
    down = boundaries[-1].T @ boundaries[-1]
    downs.append(data.coo2tensor(coo_matrix(down, dtype=np.float32)))
    return ups, downs



