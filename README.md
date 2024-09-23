


A) **Generalized representation** of linearized PDE by Hopf-Cole transformation (distortion) solution on value function PDE (Monoyios 2004) -

![image](https://github.com/user-attachments/assets/d4a31ae9-e788-4734-9276-f4f6a24ce391)

**Link to mathematic solution:** https://people.maths.ox.ac.uk/monoyios/docs/mm_chapter.pdf





B) 1d indifference price - **analytical approximation** formula (Michael Vellekoop, University of Amsterdam) - 

![image](https://github.com/user-attachments/assets/47d35849-1818-4432-9301-51eca45e2a29)



**Link to numerical scheme:** [https://gohkust-my.sharepoint.com/:b:/g/personal/jkwatra_ust_hk/EViIYCEVwiZKqTSGcqT6nb8BAxIUC1s67PXyM4rV-AWjlw?e=C5jBnC](https://actuaries.org/IAA/Documents/SECTIONS/Sections%20Colloquium%202021/PresentationVellekoopIAAOct2021.pdf)](https://actuaries.org/IAA/Documents/SECTIONS/Sections%20Colloquium%202021/PresentationVellekoopIAAOct2021.pdf)

HJB PDE numerical solution - Hopf-Cole transformation (or distortion power for linearizing the Bergers or reaction-diffusion equations) on the value function (Musiela and Zariphopoulou 2004, Henderson and Hobson)




C) Policy Iteration for HJB (using upwind [for time dim] and backwind Euler [for spatial dim]  PDE scheme). Uses penalised perturbed equations.

![image](https://github.com/user-attachments/assets/efde5361-3cec-46f1-8e0e-fbe7bea6d96e)

i) Numerical Discretization Upwind Finite Difference Scheme (Li and Wang (2009)). Grid schemes used - upwind type in space (problems that involve convection-dominated flows, where the upwind scheme provides more stability than centered schemes) and the backward Euler implicit scheme in time (an unconditionally stable scheme for time-dependent problems, uses iterative solver to handle the implicit nature). 

ii) Uniform time and space discretization for the logarithmic variable x (logS/K). Dirichlet boundary conditions on portfolio value PDE

iii)  Linear system - Solve for the matrix by fast Thomas algorithm with time complexity

iv) option price V is obtained as the difference between certainity equivalents with and without n derivative claims

**Link to algorithm source** - https://www.mdpi.com/1911-8074/14/9/399



D) **Colab link** - https://colab.research.google.com/drive/1caUnjIjXuwxJA_rHz4ys08Zxw8FSM0n0?usp=sharing
