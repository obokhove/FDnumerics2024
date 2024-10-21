## Finite-volume or discontinuous Galerkin leading-order or Godunov method

Firedrake (installation and examples)
https://www.firedrakeproject.org/documentation.html

The first few questions are theoretical questions on the formulation of the numerical flux and Godunov scheme.

The solution to the final task in the numerical exercise is provided using the symmatric flux and a symplectic-Euler semi-implicit time integrator. In that final task you are asked to change the numerical flux to the Riemann flux and either change the time integration to Godunov/Euler foreard scheme or make an adaptation to a symplectic-Euler semi-implicit time integrator (if possible since it may make the scheme implicit). The SE scheme has forward Euler step followed by a backward Euler step.
