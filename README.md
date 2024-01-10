
## Description

This repository provides a solution for the Helmholtz equation with a fractal boundary shape, offering options for Dirichlet or Neumann boundary conditions. It is particularly useful for analyzing the effects of fractal shapes on wave localization and their applications in the design of wave-absorbing materials. The code facilitates the construction of Finite Element Method (FEM) solutions, including mesh generation, mesh editing, solving for resonating modes, and ultimately solving the Helmholtz equation.

## Basic Usage

### Main File
- `helmholtz_solve`: This is the primary script to run the solutions.

### Supporting Files
- `solutions`: Handles basic mesh generation.
- `zsolution4students`: Provides functions and methods to formalize the stiffness matrix and related processes.
- Other files are included for analyzing mesh quality and its effects.

### Function Usage
```python
plane_wave(wavenumber, degree, a, plot_type="2d", alignment="right")
```
wavenumber: Sets the 'k' value of the Helmholtz equation.

degree: Determines the order of the Koch fractal.

a: Sets the source intensity.

plot_type: Choose between "2d" or "3d" visualizations.

alignment: Defines the method for dividing a square mesh into triangular segments.

## Further reading
For detailed information on mesh generation and solution processes, please refer to the two PDFs attached in this repository.
