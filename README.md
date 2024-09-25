


**A)** **Generalized representation** of linearized PDE by Hopf-Cole transformation (distortion) solution on value function PDE (Monoyios 2004) -

![image](https://github.com/user-attachments/assets/d4a31ae9-e788-4734-9276-f4f6a24ce391)

**Link to mathematic solution:** https://people.maths.ox.ac.uk/monoyios/docs/mm_chapter.pdf





**B)** 1d Simplified Case for indifference price - **analytical approximation** formula (Michael Vellekoop, University of Amsterdam) - 

![image](https://github.com/user-attachments/assets/47d35849-1818-4432-9301-51eca45e2a29)

![image](https://github.com/user-attachments/assets/f6e4ba40-496f-4d25-97a8-2c057ccc47ec)

![image](https://github.com/user-attachments/assets/8955fdde-982c-485f-9e64-2d7d81b79202)


**Link to numerical scheme:**(https://actuaries.org/IAA/Documents/SECTIONS/Sections%20Colloquium%202021/PresentationVellekoopIAAOct2021.pdf)

Crank Nicolson is better as the explicit scheme (only forward-differencing for the time derivative) requires the time step to be constrained to unacceptably low values in order to ensure stability. 


**C)** Policy Iteration for HJB (using upwind [for time dim] and backwind Euler [for spatial dim]  PDE scheme). Uses penalised perturbed equations.

![image](https://github.com/user-attachments/assets/efde5361-3cec-46f1-8e0e-fbe7bea6d96e)

i) Numerical Discretization Upwind Finite Difference Scheme (Li and Wang (2009)). Grid schemes used - upwind type in space (problems that involve convection-dominated flows, where the upwind scheme provides more stability than centered schemes) and the backward Euler implicit scheme in time (an unconditionally stable scheme for time-dependent problems, uses iterative solver to handle the implicit nature). 

ii) Uniform time and space discretization for the logarithmic variable x (logS/K). Dirichlet boundary conditions on portfolio value PDE

iii)  Linear system - Solve for the matrix by fast Thomas algorithm with time complexity

iv) option price V is obtained as the difference between certainity equivalents with and without n derivative claims

**Link to algorithm source** - https://www.mdpi.com/1911-8074/14/9/399


**D)** n-dimensional Burger's Equation solution Matlab code - 

**Link to code repo: -** https://github.com/LzEfreet/CFD-PIM?tab=readme-ov-file

HJB PDE numerical solution - Hopf-Cole transformation (or distortion power for linearizing the Burgers or reaction-diffusion equations) on the value function (Musiela and Zariphopoulou 2004, Henderson and Hobson)


**E)** Basket - 

(Dzupire 2019) - Assuming 0 correlation between traded asset S (capital market index) and Weather Index I (constructed based on Yi). 
![image](https://github.com/user-attachments/assets/96cbd98b-b427-49e8-8647-2f25781e8e0c)


(Carmona 2004) - Correlation embedded in OLS
![image](https://github.com/user-attachments/assets/2b5435c8-ce04-4aea-b50a-9940365493e2)



**F)** **Colab link** - https://colab.research.google.com/drive/1iEsWgOOY3vK39Unbrobov5RG2dxffYNG?usp=sharing
