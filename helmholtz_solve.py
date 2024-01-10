# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 10:50:48 2023

@author: tomkeen
"""


# Python packages
import matplotlib.pyplot as plt
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sys
from scipy.linalg import eig
from scipy.sparse.linalg import eigsh
# MRG packages
import solutions
from sympy import Matrix, symbols
from scipy.linalg import null_space
from  numpy.linalg import matrix_rank as rank
def add_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid_coords):
    # .. todo:: Modify the lines below to add one node to the mesh
    node_coords=np.vstack([node_coords,nodeid_coords])
    return node_coords, p_elem2nodes, elem2nodes


def add_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid2nodes):
    # .. todo:: Modify the lines below to add one element to the mesh
    elem2nodes = np.vstack([elem2nodes, elemid2nodes])
    
    # Update the p_elem2nodes array
    new_p_elem = p_elem2nodes[-1] + len(elemid2nodes)
    p_elem2nodes = np.append(p_elem2nodes, new_p_elem)
    return node_coords, p_elem2nodes, elem2nodes


def remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, nodeid):
    # .. todo:: Modify the lines below to remove one node to the mesh
    p_node2elems, node2elems=build_node2elems(p_elem2nodes,elem2nodes)
    p_elem_id_st=p_node2elems[nodeid]
    p_elem_id_ed=p_node2elems[nodeid+1]

    elem_ids=node2elems[p_elem_id_st:p_elem_id_ed]
    e2n_idx = []
    for eid in elem_ids:
        start = p_elem2nodes[eid]
        end = p_elem2nodes[eid+1]
        e2n_idx.extend(list(range(start, end)))
    e2n_idx = sorted(list(set(e2n_idx)))
    
    # Delete the elements from elem2nodes that use the node
    elem2nodes = np.delete(elem2nodes, e2n_idx)
    
    sort_elem_ids=sorted(elem_ids)
    
    for i in sort_elem_ids:
        size=p_elem2nodes[i+1]-p_elem2nodes[i]
        p_elem2nodes[i+1:]=p_elem2nodes[i+1:]-size
    p_elem2nodes = np.delete(p_elem2nodes, elem_ids+1)
    node_idx_chg=np.where(elem2nodes>nodeid)
    elem2nodes[node_idx_chg]=elem2nodes[node_idx_chg]-1
    # Remove the node's coordinates
    node_coords = np.delete(node_coords, nodeid, axis=0)
   
        
    return node_coords, p_elem2nodes, elem2nodes


def remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, elemid):
    # .. todo:: Modify the lines below to remove one element to the mesh
    start_id=p_elem2nodes[elemid]
    end_id=p_elem2nodes[elemid+1]
    indices_to_delete = range(start_id, end_id)
    size_node=end_id-start_id
    p_elem2nodes = np.delete(p_elem2nodes,[elemid])
    p_elem2nodes[elemid:]=p_elem2nodes[elemid:]-size_node
    elem2nodes=np.delete(elem2nodes,indices_to_delete)
    
    return node_coords, p_elem2nodes, elem2nodes


def compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes):
    """
    Compute the barycenters for each element.

    Parameters:
    - node_coords: A list of node coordinates. E.g., [(x1,y1), (x2,y2), ...]
    - p_elem2nodes: A list providing the start index in the elem2nodes list for each element.
    - elem2nodes: A list of nodes for all elements. E.g., [node1, node2, ...]

    Returns:
    - barycenters: A list of barycenters for each element as a tuple (x,y).
    """
    
    spacedim=node_coords.shape[1]
    nelems=p_elem2nodes.shape[0]-1
    elem_coords=np.zeros((nelems,spacedim),dtype=np.float64)
    print(p_elem2nodes)
    print(len(node_coords))
    for i in range(0,nelems):
        node_indices=elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        elem_coords[i,:]=np.mean(node_coords[node_indices,:],axis=0)

    # print(node_coords)
    # print(p_elem2nodes)
    return elem_coords

def random_shift_node(boundary_idx,node_coords,amp):
    
    
    mask = np.ones(node_coords.shape[0], dtype=bool)
    mask[boundary_idx] = False
    
    # Generate random shifts
    random_shifts = np.random.uniform(-amp, amp, size=node_coords.shape)
    
    # Apply shifts only to rows specified by the mask
    node_coords[mask] += random_shifts[mask]
    
    return node_coords

def compute_aspect_ratio_of_element(node_coords, p_elem2nodes, elem2nodes):
    # .. todo:: Modify the lines below to compute the quality criteria of all the elements
    
    # assume the aspect ratio is generally defined as the ratio of the largest dimension 
    #or distance between nodes to the smallest dimension or distance. 

    nelems=p_elem2nodes.shape[0]-1
    elem_quality=np.zeros((nelems,1),dtype=np.float64)
    for i in range(0,nelems):
        node_indices=elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        node_coord=node_coords[node_indices,:]

        # node_coord=sort_clockwise(node_coord)

        if len(node_coord)==3:
            diffs = node_coord[:, np.newaxis] - node_coord
            
            distances = np.linalg.norm(diffs, axis=2)
            distances=[distances[0,1],distances[0,2],distances[1,2]]
            
            d_max = np.max(distances)
            d_min = np.min(distances)
            s=np.sum(distances)/2
            K=np.sqrt(s*(s-distances[0])*(s-distances[1])*(s-distances[2]))
            r=K/s

            elem_quality[i,:]=d_max / r *(np.sqrt(3)/6)
        if len(node_coord)==4:
            total = 0
            for j in range(4):
                # Compute the normalized inner product between row i and row (i+1)%4

                v1=node_coord[(j+1)%4,:]-node_coord[j,:]
                v2=node_coord[(j+2)%4,:]-node_coord[(j+1)%4,:]
                nip = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))                
                total += abs(nip)

            elem_quality[i,:]=1-total/4


    return elem_quality


def compute_edge_length_factor_of_element(node_coords, p_elem2nodes, elem2nodes):
    # .. todo:: Modify the lines below to compute the quality criteria of all the elements
    nelems=p_elem2nodes.shape[0]-1
    elem_quality=np.zeros((nelems,1),dtype=np.float64)
    for i in range(0,nelems):
        node_indices=elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        node_coord=node_coords[node_indices,:]
        if len(node_coord)==3:
            diffs = node_coord[:, np.newaxis] - node_coord
            distances = np.linalg.norm(diffs, axis=2)
            distances=[distances[0,1],distances[0,2],distances[1,2]]
            distances.sort()
            
            d_max = np.max(distances)
            d_min = np.min(distances)
            d_mid = distances[1]


            elem_quality[i,:]=d_min/d_mid
    return elem_quality


def compute_pointedness_of_element(node_coords, p_elem2nodes, elem2nodes):
    # .. todo:: Modify the lines below to compute the quality criteria of all the elements
    pass
    # return elem_quality

def build_node2elems(p_elem2nodes,elem2nodes):
# elem2nodes connectivity matrixe2n_ coef  numpy.ones(len(elem2nodes)，dtypernumpy.int64)e2n mtx = scipy.sparse.csr matrix((e2n coef, elem2nodes, p elem2nodes))# node2elems connectivity matrixn2e mtx = e2n mtx.transpose()n2e mtx = n2e mtx.tocsr()
# output
    e2n_coef=  np.ones(len(elem2nodes))
    e2n_mtx = scipy.sparse.csr_matrix((e2n_coef, elem2nodes, p_elem2nodes))
    n2e_mtx = e2n_mtx.transpose()
    n2e_mtx = n2e_mtx.tocsr()
    p_node2elems = n2e_mtx.indptr
    node2elems=n2e_mtx.indices
    return p_node2elems, node2elems


def remove_adjacent_duplicates(input_list):
    if not input_list:
        return []

    result = [input_list[0]]
    for i in range(1, len(input_list)):
        if input_list[i] != input_list[i-1]:
            result.append(input_list[i])

    return result

def fractal_koch_idx(start_x,start_y,size,degree,edge_idx):
    if size<4*(degree-1):
        print("can't generate. change size or degree")
        return
    if degree==1 and edge_idx==0:
        coord=[(start_x,start_y),(start_x+size,start_y),(start_x+size,start_y+size),(start_x+2*size,start_y+size),(start_x+2*size,start_y-size),
               (start_x+3*size,start_y-size),(start_x+3*size,start_y),(start_x+4*size,start_y)]

        return coord
    if degree==1 and edge_idx==1:
        coord=[(start_x,start_y),(start_x,start_y+size),(start_x-size,start_y+size),(start_x-size,start_y+2*size),(start_x+size,start_y+2*size),
               (start_x+size,start_y+3*size),(start_x,start_y+3*size),(start_x,start_y+4*size)]

        return coord
    if edge_idx==0:
        coord_1=fractal_koch_idx(start_x,start_y,int(size/4),degree-1,0)
        coord_2=fractal_koch_idx(start_x+size,start_y,int(size/4),degree-1,1)
        coord_3=fractal_koch_idx(start_x+size,start_y+size,int(size/4),degree-1,0)
        coord_4=fractal_koch_idx(start_x+2*size,start_y,int(size/4),degree-1,1)
        coord_5=fractal_koch_idx(start_x+2*size,start_y-size,int(size/4),degree-1,1)
        coord_6=fractal_koch_idx(start_x+2*size,start_y-size,int(size/4),degree-1,0)
        coord_7=fractal_koch_idx(start_x+3*size,start_y-size,int(size/4),degree-1,1)
        coord_8=fractal_koch_idx(start_x+3*size,start_y,int(size/4),degree-1,0)
    if edge_idx==1:
        coord_1=fractal_koch_idx(start_x,     start_y,int(size/4),degree-1,1)
        coord_2=fractal_koch_idx(start_x-size,     start_y+size,int(size/4),degree-1,0)
        coord_3=fractal_koch_idx(start_x-size,start_y+size,int(size/4),degree-1,1)
        coord_4=fractal_koch_idx(start_x-size,start_y+2*size,int(size/4),degree-1,0)
        coord_5=fractal_koch_idx(start_x,     start_y+2*size,int(size/4),degree-1,0)
        coord_6=fractal_koch_idx(start_x+size,start_y+2*size,int(size/4),degree-1,1)
        coord_7=fractal_koch_idx(start_x,start_y+3*size,int(size/4),degree-1,0)
        coord_8=fractal_koch_idx(start_x,start_y+3*size,int(size/4),degree-1,1)
        
    coord_assemble=coord_1+coord_2+coord_3+coord_4+coord_5+coord_6+coord_7+coord_8


    return coord_assemble
import math
def intersection(ray_origin, ray_direction, segment):
    # Ray in parametric form: P = R_o + tR_d
    R_o = ray_origin
    vec_x,vec_y = ray_direction

    px,py=ray_origin
    
    x1, y1 = segment[0]
    x2, y2 = segment[1]
    if y1==y2 and px >= min(x1, x2) and px <= max(x1, x2) and y1==py:
        return "on boundary"
    if x1==x2 and py >= min(y1, y2) and py <= max(y1, y2) and x1==px:
        return "on boundary"
    # Segment in parametric form: Q = P_1 + u(P_2 - P_1)
    
    k=vec_y/vec_x
    if x1==x2:
        y_int=k*(x1-px)+py
        if y_int >= min(y1, y2) and y_int <= max(y1, y2) and y_int>py:
            return True
        else:
            return False
    if y1==y2:
        x_int=(y1-py)/k+px
        if x_int >= min(x1, x2) and x_int <= max(x1, x2) and x_int>px:
            return True
        else:
            return False
def is_inside(px, py, nodes):
    ray_direction = (math.cos(math.radians(15)), math.sin(math.radians(15)))
    count = 0
    for i in range(len(nodes)):
        segment = (nodes[i], nodes[(i+1)%len(nodes)])
        if intersection((px, py), ray_direction, segment)=="on boundary":

            return "on boundary"
        if intersection((px, py), ray_direction, segment)==True:
            # print(segment)
            # print(intersection((px, py), ray_direction, segment))
            count += 1
    if count % 2 == 1:
        return "inside"
    else:
        return "outside"

def fractal_mesh_quad(nelemsx,start,size,degree): 
    
    if size<4*(degree-1):
        print("can't generate. change size or degree")
        return
    if start+size*4>nelemsx:
        print("can't generate. change size or start")
        return
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 2.0

    nelemsy=2*nelemsx
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    # .. todo:: Modify the line below to call to generate a grid with quadrangles
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = my_set_quadmesh(...)
    # .. note:: If you do not succeed, uncomment the following line to access the solution
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_quadmesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    if degree!=0:
        fract_coords_idx=fractal_koch_idx(start,nelemsx,size,degree,0)
        fract_coords_idx=remove_adjacent_duplicates(fract_coords_idx)
    else:
        fract_coords_idx=[]
    fract_coords_idx.insert(0, (0,nelemsx))
    fract_coords_idx.insert(0, (0,0))
    fract_coords_idx.append((nelemsx,nelemsx))
    fract_coords_idx.append((nelemsx,0))
    fract_coords_idx=remove_adjacent_duplicates(fract_coords_idx)
    dh=1/nelemsx

    node_coords_idx=np.round(node_coords/ dh)
    
    j=0
    for i in range(len(node_coords_idx)):
        print(str(i)+" over "+ str(len(node_coords_idx)))

        if is_inside(int(node_coords_idx[i,0]),int(node_coords_idx[i,1]), fract_coords_idx)=="outside":
            
            real_idx=i-j
            
            try:
                
                node_coords, p_elem2nodes, elem2nodes=remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, real_idx)
                print(node_coords_idx[i])
                j=j+1
            except:
                continue
        
        elif is_inside(int(node_coords_idx[i,0]),int(node_coords_idx[i,1]), fract_coords_idx)=="on boundary":
            try:
                boundary_node_coords=np.vstack([boundary_node_coords,node_coords_idx[i]])
            except:
                boundary_node_coords=node_coords_idx[i]
            
    elem_coords=compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes)
    elem_coords_idx=elem_coords/dh
    j=0
    for i in range(len(elem_coords_idx)):
        print(str(i)+" over "+ str(len(elem_coords_idx)))

        if is_inside(elem_coords_idx[i,0],elem_coords_idx[i,1], fract_coords_idx)=="outside":
            
            real_idx=i-j
            try:
                node_coords, p_elem2nodes, elem2nodes=remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, real_idx)
            except:
                break
            j=j+1
    #get rid of node that not assign to a element
    
    node_idx_with_elem_max = np.max(list(set(elem2nodes)))

    node_coords=node_coords[:node_idx_with_elem_max+1 ]
    # print(len(node_coords))


    
    idx_node_remove=[]
    nn=len(node_coords)
    for i in range(nn):
        if i not in elem2nodes:
            idx_node_remove.append(i)
    
    idx_node_remove = sorted(idx_node_remove, reverse=True)
    for i in idx_node_remove:
        node_coords, p_elem2nodes, elem2nodes=remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, i)
    
    node_coords_idx=np.round(node_coords/ dh)
    mask = np.all(boundary_node_coords[:, np.newaxis] == node_coords_idx, axis=2)
    indices = np.argmax(mask, axis=1)
    return node_coords, p_elem2nodes, elem2nodes,indices

def quad2tri(node_coords, p_elem2nodes, elem2nodes):
    nelems=p_elem2nodes.shape[0]-1
    p_elem2nodes_new=[0]
    elem2nodes_new=[]
    for i in range(0,nelems):
        node_indices=elem2nodes[p_elem2nodes[i]:p_elem2nodes[i+1]]
        idx1=p_elem2nodes[i]
        idx2=p_elem2nodes[i+1]
        if len(node_indices)==4:
            node_diag1=node_indices[1]
            node_diag2=node_indices[3]
            node_indices1=[node_indices[0],node_diag1,node_diag2]
            node_indices2=[node_diag1,node_indices[2],node_diag2]
            elem2nodes_new=elem2nodes_new+node_indices1+node_indices2
            p_elem2nodes_new=p_elem2nodes_new+[(2*i+1)*3]+[(2*i+2)*3]            
    return node_coords, p_elem2nodes_new, elem2nodes_new


def fractal_mesh_tri(nelemsx,start,size,degree): 
    
    if size<4*(degree-1):
        print("can't generate. change size or degree")
        return
    if start+size*4>nelemsx:
        print("can't generate. change size or start")
        return
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 2.0

    nelemsy=2*nelemsx
    nelems = nelemsx * nelemsy
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    # .. todo:: Modify the line below to call to generate a grid with quadrangles
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = my_set_quadmesh(...)
    # .. note:: If you do not succeed, uncomment the following line to access the solution
    node_coords, node_l2g, p_elem2nodes, elem2nodes = solutions._set_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)
    
    if degree!=0:
        fract_coords_idx=fractal_koch_idx(start,nelemsx,size,degree,0)
        fract_coords_idx=remove_adjacent_duplicates(fract_coords_idx)
    else:
        fract_coords_idx=[]
    fract_coords_idx.insert(0, (0,nelemsx))
    fract_coords_idx.insert(0, (0,0))
    fract_coords_idx.append((nelemsx,nelemsx))
    fract_coords_idx.append((nelemsx,0))
    fract_coords_idx=remove_adjacent_duplicates(fract_coords_idx)
    dh=1/nelemsx

    node_coords_idx=np.round(node_coords/ dh)
    
    j=0
    for i in range(len(node_coords_idx)):
        print(str(i)+" over "+ str(len(node_coords_idx)))

        if is_inside(int(node_coords_idx[i,0]),int(node_coords_idx[i,1]), fract_coords_idx)=="outside":
            
            real_idx=i-j
            
            try:
                
                node_coords, p_elem2nodes, elem2nodes=remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, real_idx)
                j=j+1
            except:
                continue
        
        elif is_inside(int(node_coords_idx[i,0]),int(node_coords_idx[i,1]), fract_coords_idx)=="on boundary":
            try:
                boundary_node_coords=np.vstack([boundary_node_coords,node_coords_idx[i]])
            except:
                boundary_node_coords=node_coords_idx[i]
            
    elem_coords=compute_barycenter_of_element(node_coords, p_elem2nodes, elem2nodes)
    elem_coords_idx=elem_coords/dh
    j=0
    for i in range(len(elem_coords_idx)):
        print(str(i)+" over "+ str(len(elem_coords_idx)))

        if is_inside(elem_coords_idx[i,0],elem_coords_idx[i,1], fract_coords_idx)=="outside":
            
            real_idx=i-j
            try:
                node_coords, p_elem2nodes, elem2nodes=remove_elem_to_mesh(node_coords, p_elem2nodes, elem2nodes, real_idx)
            except:
                break
            j=j+1
    #get rid of node that not assign to a element
    
    node_idx_with_elem_max = np.max(list(set(elem2nodes)))

    node_coords=node_coords[:node_idx_with_elem_max+1 ]
    # print(len(node_coords))

    
    nn=len(node_coords)
    
    idx_node_remove=[]
    for i in range(nn):
        if i not in elem2nodes:
            idx_node_remove.append(i)
    
    idx_node_remove = sorted(idx_node_remove, reverse=True)
    for i in idx_node_remove:
        node_coords, p_elem2nodes, elem2nodes=remove_node_to_mesh(node_coords, p_elem2nodes, elem2nodes, i)
    
    node_coords_idx=np.round(node_coords/ dh)
    mask = np.all(boundary_node_coords[:, np.newaxis] == node_coords_idx, axis=2)
    indices = np.argmax(mask, axis=1)

    return node_coords, p_elem2nodes, elem2nodes, indices 
import zsolutions4students as solutions1


# ..todo: Uncomment for displaying limited digits
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def plane_wave(wavenumber, degree, a, plot_type="2d",alignment="right"):
    # Depending on the 'degree', select the appropriate mesh resolution
    if degree > 2:
        if alignment=="right":
            node_coords, p_elem2nodes, elem2nodes, boundary_idx = fractal_mesh_tri(64, 0, 16, degree)
        else:
            node_coords, p_elem2nodes, elem2nodes, boundary_idx = fractal_mesh_quad(64, 0, 16, degree)
            node_coords, p_elem2nodes, elem2nodes=quad2tri(node_coords, p_elem2nodes, elem2nodes)
    else:
        if alignment=="right":
            node_coords, p_elem2nodes, elem2nodes, boundary_idx = fractal_mesh_tri(32, 0, 8, degree)
        else:
            node_coords, p_elem2nodes, elem2nodes, boundary_idx = fractal_mesh_quad(32, 0, 8, degree)
            node_coords, p_elem2nodes, elem2nodes=quad2tri(node_coords, p_elem2nodes, elem2nodes)
    # Determine the number of nodes and elements from the mesh data
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes) - 1

    # Plot the mesh using the provided node and element data
    fig = matplotlib.pyplot.figure(1)
    ax = matplotlib.pyplot.subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    solutions1._plot_mesh(p_elem2nodes, elem2nodes, node_coords, color='orange')
    matplotlib.pyplot.show()

    # Build the mapping from nodes to elements
    p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)

    # Identify the nodes lying on the southern boundary
    nodes_on_south = solutions1._set_square_nodes_boundary_south(node_coords)
    nodes_on_boundary = nodes_on_south

    # Set Dirichlet boundary conditions
    values_at_nodes_on_boundary = np.zeros((nnodes, 1), dtype=np.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = a

    # Prepare finite element matrices and the right-hand side
    f_unassembled = np.zeros((nnodes, 1), dtype=np.complex128)
    coef_k = np.ones((nelems, 1), dtype=np.complex128)
    coef_m = np.ones((nelems, 1), dtype=np.complex128)
    K, M, F = solutions1._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    
    # Formulate the system matrix A
    A = K - wavenumber**2 * M
    B = F
    
    
    
    # Apply the Dirichlet boundary conditions to the system
    A, B = solutions1._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)
    print(np.linalg.matrix_rank(A))
    print(np.linalg.matrix_rank(np.hstack([A, B])))
    
    # Solve the system. If A is singular, use the pseudo-inverse
    if np.linalg.matrix_rank(A)==len(A):
        sol = scipy.linalg.solve(A, B)
        print("not singular")
    else:
        print("singular")
        pseudo_inv_A = scipy.linalg.pinv(A)
        sol = pseudo_inv_A.dot(B)
    
    # Plot the solution (either in 3D or 2D) based on the provided 'plot_type'
    solreal = sol.reshape((sol.shape[0],))
    if plot_type == "3d":
        _ = solutions1._plot_contourf(nelems, p_elem2nodes, elem2nodes, node_coords, np.real(solreal))
    else:
        _ = solutions1._plot_2d_contour(nelems, p_elem2nodes, elem2nodes, node_coords, np.real(solreal))
    return A ,B

def find_eig(degree,alignment="right"):
    if degree > 2:
        if alignment=="right":
            node_coords, p_elem2nodes, elem2nodes, boundary_idx = fractal_mesh_tri(64, 0, 16, degree)
            h=1/64
        else:
            node_coords, p_elem2nodes, elem2nodes, boundary_idx = fractal_mesh_quad(64, 0, 16, degree)
            node_coords, p_elem2nodes, elem2nodes=quad2tri(node_coords, p_elem2nodes, elem2nodes)
            h=1/64
    else:
        if alignment=="right":
            node_coords, p_elem2nodes, elem2nodes, boundary_idx = fractal_mesh_tri(32, 0, 8, degree)
            h=1/32
        else:
            node_coords, p_elem2nodes, elem2nodes, boundary_idx = fractal_mesh_quad(32, 0, 8, degree)
            node_coords, p_elem2nodes, elem2nodes=quad2tri(node_coords, p_elem2nodes, elem2nodes)
            h=1/32
    # Determine the number of nodes and elements from the mesh data
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes) - 1
    # Build the mapping from nodes to elements
    p_node2elems, node2elems = build_node2elems(p_elem2nodes, elem2nodes)

    # Identify the nodes lying on the southern boundary
    nodes_on_south = solutions1._set_square_nodes_boundary_south(node_coords)
    nodes_on_boundary = nodes_on_south



    # Prepare finite element matrices and the right-hand side
    f_unassembled = np.zeros((nnodes, 1), dtype=np.complex128)
    coef_k = np.ones((nelems, 1), dtype=np.complex128)
    coef_m = np.ones((nelems, 1), dtype=np.complex128)
    K, M, F = solutions1._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    K_reduced, M_reduced=solutions1._set_dirichlet_condition_eig(K, M, nodes_on_boundary)
    # Apply the Dirichlet boundary conditions to the system
    # eigenvalues, eigenvectors = eigsh(K, k=20, which='LM', M=M)
    if degree==3:
        eigenvalues, eigenvectors = eigsh(K_reduced, k=50, which='LM', M=M_reduced)
    else:
        eigenvalues, eigenvectors = eig(K_reduced, M_reduced)
    
    sorted_indices = np.argsort(np.real(eigenvalues))
    sorted_eigenvalues = np.real(eigenvalues)[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvectors = np.insert(sorted_eigenvectors,nodes_on_boundary, 0, axis=0)
    return np.sqrt(sorted_eigenvalues ),sorted_eigenvectors,nelems, p_elem2nodes, elem2nodes, node_coords,boundary_idx,h

def plot_eigenmode(mode_number,ks,eigenvectors,nelems, p_elem2nodes, elem2nodes, node_coords):
    _ = solutions1._plot_2d_contour_eig(ks[mode_number],nelems, p_elem2nodes, elem2nodes, node_coords, np.real(eigenvectors[:,mode_number]))

def plot_existance_surface(ks_list, eigenvectors_list, labels=None):
    fig, ax = plt.subplots()
    
    markers = ['o', 's', '^', '*', 'D']  # circle, square, triangle, star, diamond
    
    for i in range(len(ks_list)):
        ks = ks_list[i]
        eigenvectors = eigenvectors_list[i]
        
        power4 = np.real(eigenvectors)**4
        n_nodes = len(eigenvectors)
        existance_surface = 1 / (n_nodes * np.sum(power4,axis=0))
        
        ax.scatter(ks / np.pi, existance_surface, label=(labels[i] if labels else f"Set {i+1}"), s=80-10*i, marker=markers[i % len(markers)])

    ax.set_title("Existance Surface")
    ax.set_xlabel("k/pi")
    ax.set_ylabel("Existance Surface")
    
    if labels:
        ax.legend()
    
    plt.show()

def plot_dissipated_energy(ks_list, eigenvectors_list,boundary_ls,h_ls, labels=None):
    fig, ax = plt.subplots()
    
    markers = ['o', 's', '^', '*', 'D']  # circle, square, triangle, star, diamond
    
    for i in range(len(ks_list)):
        ks = ks_list[i]
        eigenvectors = eigenvectors_list[i]
        n_nodes = len(eigenvectors)
        eigenvectors_norm=eigenvectors*np.sqrt(n_nodes)
        h=h_ls[i]
        boundary_idx=boundary_ls[i]
        power2_b = np.real(eigenvectors_norm[boundary_idx,:])**2
        print(power2_b.shape)
        dissipated_energy =(h * np.sum(power2_b,axis=0))
        print(dissipated_energy.shape)
        ax.scatter(ks / np.pi, dissipated_energy, label=(labels[i] if labels else f"Set {i+1}"), s=80-10*i, marker=markers[i % len(markers)])

    ax.set_title("dissipated energy")
    ax.set_xlabel("k/pi")
    ax.set_ylabel("dissipated energy")
    
    if labels:
        ax.legend()
    
    plt.show()
if __name__ == '__main__':
    ks_list=[]
    eigenvectors_list=[]
    name_ls=[]
    boundary_ls=[]
    h_ls=[]
    for i in range(4):
        ks,eigenvectors,nelems, p_elem2nodes, elem2nodes, node_coords,boundary_idx,h=find_eig(degree=i,alignment="right")
        ks_list.append(ks)
        eigenvectors_list.append(eigenvectors)
        name_ls.append("fractal degree "+str(i))
        boundary_ls.append(boundary_idx)
        h_ls.append(h)
    # plot_dissipated_energy(ks_list, eigenvectors_list,boundary_ls,h_ls, labels=name_ls)
    plot_existance_surface(ks_list, eigenvectors_list,labels=name_ls)
   # plot_eigenmode(20,ks,eigenvectors,nelems, p_elem2nodes, elem2nodes, node_coords)
    # A ,B=plane_wave(ks[408],degree=2,a=2,alignment="right")

 