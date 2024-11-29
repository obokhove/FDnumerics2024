## Firedrake

Load Firedrake from https://www.firedrakeproject.org/documentation.html either the recommended version or docker version.

## Finite-element exercises (2024 instructions)

The exercise on FEM modelling via Firedrake is found in the FEM folder.

Sample Firedrake programs:
- Poisson equation (codes in named folder) via weak formulation and minimisation (Yang Lu with Robin Furze)
- Load Firedrake environment, go to directory with (downloaded) file
- Type >> python3 Poissons_eq_v2.py

## Paraview instructions

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
Load output file and display.
- Open Paraview
- Go to directory with the output file named "output.pvd"
- Under "File" in menu click open and find that file "output.pvd" (click okay).
- Choose u1 or u2 and then click "Apply"
- Add axis labels, in colorbar set ranges.


