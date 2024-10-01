## Firedrake

Load Firedrake from https://www.firedrakeproject.org/documentation.html either the recommended version or docker version.

## Finite-element exercises (TBC 2023 instructions)

The exercise on FEM modelling via Firedrake is found in: ex23_3wavedynFD.pdf (23-11-2023).

Sample Firedrake programs:
- Poisson equation via weak formulation and minimisation (Yang Lu with Robin Furze)
- Continuous Galerkin FEM (CGn nth-order) shallow-water and Benney-Luke equations (SWE and BLE).
- :new: SWE with local wave-breaking parameterization; see updated Benney-Luke equation in that folder (elegantly combines minimization and dissipative approaches in an elegant way; definition of diffusion function is still incomplete). 
- CGn Variational Boussinesq Model (VBM). TBD.
- :new: DG0-FV/Godunov Linear shallow-water equations. Standing waves (Nbc=1) and waves running up a slope (Nbc=3). See bespoke folder and the code: sweDGFV.py
- Bokhove and Kalogirou 2016: Lecture Notes on the Theory of Water Waves, Edited by: Bridges, Groves and Nicholls, London Mathematical Society Lecture Notes Series 426, 226-259 https://www1.maths.leeds.ac.uk/~obokhove/lms-cup2015.pdf

## Firedrake simulation instructions
See codes' folders.

## Paraview instructions
See codes' folders.

:new: *Warning* (via TH from IT): When other modules have been loaded incompatible libraries then Paraview may not work.
E.g., the "anaconda3" module will cause Qt problems with "paraview". Clean environment by running:

`module purge`

`module load paraview-rhel8/5.11.2`

`paraview`

One should not add any "module load" commands to their .bashrc files to load
modules automatically as this can cause problems with standard system software and
with other modules.  one may need to fix .bashrc files if the purge command
doesn't clear the anaconda3 libraries.

## Shallow-water equations: model, variational principle (continuous/time-discrete)
Potential-flow formulation

Settings nVP=2 in code BenneylukefbVP.py for modified midpoint time-stepping scheme

Settings nVP=0 in code BenneylukefbVP.py for Stormer-Verlet time-stepping scheme

## Benney-Luke equations: model, variational principle (continuous/time-discrete)
Potential flow formulation

Settings nVP=1 in code BenneylukefbVP.py for modified midpoint time-stepping scheme

