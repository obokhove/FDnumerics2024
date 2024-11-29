## Firedrake

Load Firedrake from https://www.firedrakeproject.org/documentation.html either the recommended version or docker version.

## Finite-element exercises (TBC 2024 instructions)

The exercise on FEM modelling via Firedrake is found in: ... TBD.

Sample Firedrake programs:
- Poisson equation via weak formulation and minimisation (Yang Lu with Robin Furze)

## Firedrake simulation instructions
See codes' folders. TBD.

## Paraview instructions TBD
See codes' folders. TBD.

:new: *Warning* (via TH from IT): When other modules have been loaded incompatible libraries then Paraview may not work.
E.g., the "anaconda3" module will cause Qt problems with "paraview". Clean environment by running:

`module purge`

`module load paraview-rhel8/5.11.2`

`paraview`

One should not add any "module load" commands to their .bashrc files to load
modules automatically as this can cause problems with standard system software and
with other modules.  one may need to fix .bashrc files if the purge command
doesn't clear the anaconda3 libraries.

## Paraview visualisation



