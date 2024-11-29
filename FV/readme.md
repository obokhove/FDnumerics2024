## Finite-volume or discontinuous Galerkin leading-order or Godunov method

OB 29-11-2024: full Firedrake code of lienar shallow-water system for all fluxes, Riemann and standing waves problem, "SWE_DG0.py" added, but it needs modifying to (eta,u)^T

Firedrake (installation and examples)
https://www.firedrakeproject.org/documentation.html

Two download options: follow linux instructions and use chatgpt or use the "dockers image" (link part way donw the documentation page).

The first few questions are theoretical questions on the formulation of the numerical flux and Godunov scheme.

The solution to the final task in the numerical exercise is provided in the provided code "ex24_2FVswecopy.py" (standalone code do not use for the exercises) and sweDFFV.py by using the symmatric flux and a symplectic-Euler semi-implicit time integrator. In that final task you are asked to change the numerical flux to the Riemann flux and either change the time integration to Godunov/Euler foreard scheme or make an adaptation to a symplectic-Euler semi-implicit time integrator (if possible since it may make the scheme implicit). The SE scheme has forward Euler step followed by a backward Euler step.

The given code should give an image but the linux-windows communication hampers that (says Ben).
