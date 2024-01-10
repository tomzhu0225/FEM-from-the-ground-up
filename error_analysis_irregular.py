# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:18:39 2023

@author: tomkeen
"""


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


# MRG packages
import solutions
import zsolutions4students as solutions1

def helmholtz_solution_error_shift(wavenumber,nelemsx,amp):

    # -- set equation parameters

    # -- set geometry parameters
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0

    
    nelemsy=nelemsx
    h=1/nelemsx
    # -- generate mesh
    nnodes = (nelemsx + 1) * (nelemsy + 1)
    nelems = nelemsx * nelemsy * 2
    node_coords, p_elem2nodes, elem2nodes, node_l2g = solutions1._set_square_trimesh(xmin, xmax, ymin, ymax, nelemsx, nelemsy)

    # .. todo:: Modify the line below to define a different geometry.
    # p_elem2nodes, elem2nodes, node_coords, node_l2g = ...
    nnodes = node_coords.shape[0]
    nelems = len(p_elem2nodes)-1


    # -- set boundary geometry
    # boundary composed of nodes
    # .. todo:: Modify the lines below to select the ids of the nodes on the boundary of the different geometry.
    nodes_on_north = solutions1._set_square_nodes_boundary_north(node_coords)
    nodes_on_south = solutions1._set_square_nodes_boundary_south(node_coords)
    nodes_on_east = solutions1._set_square_nodes_boundary_east(node_coords)
    nodes_on_west = solutions1._set_square_nodes_boundary_west(node_coords)
    nodes_on_boundary = np.unique(np.concatenate((nodes_on_north, nodes_on_south, nodes_on_east, nodes_on_west)), )
    
    node_coords=random_shift_node(nodes_on_boundary, node_coords, amp*h, seed=42)
    # ..warning: the ids of the nodes on the boundary should be 'global' number.
    # nodes_on_boundary = ...
    # ..warning: for teaching purpose only
    # -- set exact solution
    solexact = np.zeros((nnodes, 1), dtype=np.complex128)
    laplacian_of_solexact = np.zeros((nnodes, 1), dtype=np.complex128)
    for i in range(nnodes):
        x, y, z = node_coords[i, 0], node_coords[i, 1], node_coords[i, 2]
        # set: u(x,y) = e^{ikx}
        solexact[i] = np.exp(complex(0.,1.)*wavenumber*x)
        laplacian_of_solexact[i] = complex(0.,1.)*wavenumber*complex(0.,1.)*wavenumber * solexact[i]
    # ..warning: end

    # -- set dirichlet boundary conditions
    values_at_nodes_on_boundary = np.zeros((nnodes, 1), dtype=np.complex128)
    values_at_nodes_on_boundary[nodes_on_boundary] = solexact[nodes_on_boundary]

    # -- set finite element matrices and right hand side
    f_unassembled = np.zeros((nnodes, 1), dtype=np.complex128)

    # ..warning: for teaching purpose only
    for i in range(nnodes):
        # evaluate: (-\Delta - k^2) u(x,y) = ...
        f_unassembled[i] = - laplacian_of_solexact[i] - (wavenumber ** 2) * solexact[i]
    # ..warning: end

    coef_k = np.ones((nelems, 1), dtype=np.complex128)
    coef_m = np.ones((nelems, 1), dtype=np.complex128)
    K, M, F = solutions1._set_fem_assembly(p_elem2nodes, elem2nodes, node_coords, f_unassembled, coef_k, coef_m)
    A = K - wavenumber**2 * M
    B = F
    # -- apply Dirichlet boundary conditions
    A, B = solutions1._set_dirichlet_condition(nodes_on_boundary, values_at_nodes_on_boundary, A, B)
    # -- solve linear system
    sol = scipy.linalg.solve(A, B)

    # -- plot finite element solution
    solreal = sol.reshape((sol.shape[0], ))

    solexactreal = solexact.reshape((solexact.shape[0], ))

    solerr = solreal - solexactreal

    return  h, np.linalg.norm(solerr)

 

def loglog_power(x, y, xname, yname="Error",isplot=True):
    # Fit a linear equation to the log-transformed data
    coefficients = np.polyfit(np.log(x), np.log(y), 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    print(intercept)
    # Calculate predicted values for R^2 computation
    y_pred = np.exp(intercept) * x**slope
    
    # Compute R^2 value
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    if isplot:
    # Create the log-log plot
        plt.loglog(x, y, 'o', label='Data')
        plt.loglog(x, y_pred, 'r-', label=f'Fit: y = e^{intercept:.2f}x^{slope:.2f}, R^2 = {r2:.2f}')
        
        # Adding title and labels
        plt.title("Log-Log Plot with Linear Fit")
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.legend()
        
        plt.show()
    
    return r2, slope

if __name__ == '__main__':

    elem_ls=np.array([3,5,8,14,20,50])
    ha_ls=1/elem_ls
    kb_ls=np.array([0.1,0.5,1,2,4,8])*np.pi
    ramp_ls=np.array([0.1,0.2,0.3,0.5,0.7,0.8])
    R2_mat=np.zeros([6,6])
    
    # for i in range(len(elem_ls)):

    #     for j in range(len(ramp_ls)):
    #         h, err=helmholtz_solution_error_shift(np.pi,elem_ls[i],ramp_ls[j])
    #         R2_mat[i,j]= err
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # # We're converting scales to log scale using np.log10
    # X, Y = np.meshgrid(ramp_ls, np.log10(ha_ls))
    # Z = np.log10(R2_mat)
    
    # ax.plot_surface(X, Y, Z, cmap='viridis')
    
    # ax.set_xlabel('Relative amplitude')
    # ax.set_ylabel('Log10(h)')
    # ax.set_zlabel('Log10(Error)')
    # ax.set_title('3D Log-Log-Log Plot of Error vs Relative amplitude and h')
    # plt.show()
    
    for i in range(len(kb_ls)):

        for j in range(len(ramp_ls)):
            h, err=helmholtz_solution_error_shift(kb_ls[i],30,ramp_ls[j])
            R2_mat[i,j]= err
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # We're converting scales to log scale using np.log10
    X, Y = np.meshgrid(ramp_ls, np.log10(kb_ls))
    Z = np.log10(R2_mat)
    
    ax.plot_surface(X, Y, Z, cmap='viridis')
    
    ax.set_xlabel('Relative amplitude')
    ax.set_ylabel('Log10(k)')
    ax.set_zlabel('Log10(Error)')
    ax.set_title('3D Log-Log-Log Plot of Error vs Relative amplitude and k')
    plt.show()
    