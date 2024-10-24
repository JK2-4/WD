
# Index - 

A - Generalized representation

B - Numerical Scheme and approx followed

C - Policy Iteration HJB

D - N-dim Burgers equation (Matlab code)

E - Basket (papers)

F - Variables

G- Colab Jupyter notebook link


## **A)** **Generalized representation** of linearized PDE by Hopf-Cole transformation (distortion) solution on value function PDE (Monoyios 2004) -

![image](https://github.com/user-attachments/assets/d4a31ae9-e788-4734-9276-f4f6a24ce391)

**Link to mathematic solution:** [Monoyios](https://people.maths.ox.ac.uk/monoyios/docs/mm_chapter.pdf)





## **B)** 1d Simplified Case for indifference price - **analytical approximation** formula (Michael Vellekoop, University of Amsterdam) - 

![image](https://github.com/user-attachments/assets/47d35849-1818-4432-9301-51eca45e2a29)

![image](https://github.com/user-attachments/assets/f6e4ba40-496f-4d25-97a8-2c057ccc47ec)

![image](https://github.com/user-attachments/assets/8955fdde-982c-485f-9e64-2d7d81b79202)


**Link to numerical scheme:** 
- [Vellekoop IAAOct2021](https://actuaries.org/IAA/Documents/SECTIONS/Sections%20Colloquium%202021/PresentationVellekoopIAAOct2021.pdf)

Crank Nicolson is better as the explicit scheme (only forward-differencing for the time derivative) requires the time step to be constrained to unacceptably low values in order to ensure stability i.e. Courant-Friedrichs-Lewy (CFL) condition (not suitable for stiff problems). CN scheme averages the explicit and implicit methods.

- Case for implicit scheme and provides empirical OU model parameters for weather simulation [1](https://gohkust-my.sharepoint.com/:b:/g/personal/jkwatra_ust_hk/EUOBQ05vDnhJs6uPWxnPnU0BXZdfkj8Mnj2_F2_mtI85Pg?e=cYAW2T)

- "Also, central difference to the convection term of dominated PDEs produces spurious oscillations. To avoid introducing oscillations, it is necessary to discretize the convection term using a downwind/upwind scheme, which means that the direction of one-sided difference needs to be adjusted adaptively according to the sign of the convection term at each discrete point." [Peng Li 2018](https://www.sciencedirect.com/science/article/pii/S0898122117306880#b13)


## **C)** Policy Iteration for HJB (using upwind [for time dim] and backwind Euler [for spatial dim]  PDE scheme). Uses penalised perturbed equations.

![image](https://github.com/user-attachments/assets/efde5361-3cec-46f1-8e0e-fbe7bea6d96e)

i) Numerical Discretization Upwind Finite Difference Scheme (Li and Wang (2009)). Grid schemes used - upwind type in space (problems that involve convection-dominated flows, where the upwind scheme provides more stability than centered schemes) and the backward Euler implicit scheme in time (an unconditionally stable scheme for time-dependent problems, uses iterative solver to handle the implicit nature). 

ii) Uniform time and space discretization for the logarithmic variable x (logS/K). Dirichlet boundary conditions on portfolio value PDE

iii)  Linear system - Solve for the matrix by fast Thomas algorithm with time complexity

iv) option price V is obtained as the difference between certainity equivalents with and without n derivative claims

**Link to algorithm source** - [MDPI paper](https://www.mdpi.com/1911-8074/14/9/399)


## **D)** n-dimensional Burger's Equation solution Matlab code - 

**Link to code repo: -** [cfd-pim repo](https://github.com/LzEfreet/CFD-PIM?tab=readme-ov-file)

HJB PDE numerical solution - Hopf-Cole transformation (or distortion power for linearizing the Burgers or reaction-diffusion equations) on the value function (Musiela and Zariphopoulou 2004, Henderson and Hobson)


## **E)** Basket - 

- **E.1** (Dzupire 2019) - Assuming 0 correlation between traded asset S (capital market index) and Weather Index I (constructed based on Yi). 
![image](https://github.com/user-attachments/assets/96cbd98b-b427-49e8-8647-2f25781e8e0c)

- **E.2** Code for a [trivariate stochastic yield model paper](https://www.sciencedirect.com/science/article/pii/S2468227623002247) by P. Ngare - [code](https://ars.els-cdn.com/content/image/1-s2.0-S2468227623002247-mmc1.pdf)

- **E.3** (Carmona 2004) - Correlation embedded in OLS
![image](https://github.com/user-attachments/assets/2b5435c8-ce04-4aea-b50a-9940365493e2)

- **E.4** Model Uncertainty

Sparse Regression for non-linear dynamical systems with PDE-FIND algorithm [SINDy documentation](https://github.com/dynamicslab/pysindy) -  Provides a data-driven model discovery for a spatiotemporal system. 

Meshless Physics informed Neural Network ([PINN](https://www.nature.com/articles/s41467-021-26434-1)) and [Deep BSDE (sample code)](https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/DeepBSDE_Solver.ipynb#scrollTo=59xocsR_61C3) - An improved approach to black box NN models by using PINN with DNNs and automatic differentiation to derive the closed-form governing equations.

Implementation with American option pricing under [BS](https://paperswithcode.com/paper/physics-informed-neural-network-for-option)

NVIDIA modulus code [documentation](https://docs.nvidia.com/deeplearning/modulus/release-notes/index.html#id14)

## **F)** Variables 

|Variable |SDE Model |Empirical Parameters| Data|
|---------|---------|---------|---------|
|Rainfall:| [pyraingen](https://www.sciencedirect.com/science/article/pii/S1364815224000458#sec2) for rainfall simulation|  | |
|Hazard Intensity: | [Mockus Equation - hydrological analysis](https://doi.org/10.13031/2013.41082) ; [CLIMADA](https://github.com/CLIMADA-project/climada_python) | | src -  [flood precip compound risk](https://global-flood-database.cloudtostreet.ai/#interactive-map)|
|Wind speed and pressure: | GHYP (Generalized Hyperbolic distribution) | | |
|Temp:| [Alaton](https://rstudio-pubs-static.s3.amazonaws.com/953546_4548bb57d50344ff984963ff47645e2e.html) | | |
|Climate compound risk: | Clim ODE | | |


**Data** - [Real Estate](https://data.gov.hk/en-data/dataset/hk-rvd-tsinfo_rvd-property-market-statistics) - [metadata_dict](https://www.rvd.gov.hk/datagovhk/Data_Dic.pdf)

## **G)** **Colab link** - 

[Google colab - JK](https://colab.research.google.com/drive/1iEsWgOOY3vK39Unbrobov5RG2dxffYNG?usp=sharing)
