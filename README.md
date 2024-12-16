
# [Workflow](https://github.com/JK2-4/WD/blob/1d-PDE/CONCEPT/z_concept%20workflow.pdf) 

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
![image](https://github.com/user-attachments/assets/4da6f75b-21c0-4b7f-88bd-3be587a0f218)

**Link to mathematic solution:** [Monoyios](https://people.maths.ox.ac.uk/monoyios/docs/mm_chapter.pdf)


## **B.1)** 1d Simplified Case (Distorted power solution) for indifference price - **analytical approximation** formula (MC) - 

![image](https://github.com/user-attachments/assets/47d35849-1818-4432-9301-51eca45e2a29)

**Numerical scheme: Michael Vellekoop, University of Amsterdam** 
- [Vellekoop IAAOct2021](https://actuaries.org/IAA/Documents/SECTIONS/Sections%20Colloquium%202021/PresentationVellekoopIAAOct2021.pdf)

Crank Nicolson is better as the explicit scheme (only forward-differencing for the time derivative) requires the time step to be constrained to unacceptably low values in order to ensure stability i.e. Courant-Friedrichs-Lewy (CFL) condition (not suitable for stiff problems). CN scheme averages the explicit and implicit methods.

- Case for implicit scheme and provides empirical OU model parameters for weather simulation [1](https://gohkust-my.sharepoint.com/:b:/g/personal/jkwatra_ust_hk/EUOBQ05vDnhJs6uPWxnPnU0BXZdfkj8Mnj2_F2_mtI85Pg?e=cYAW2T)

- "Also, central difference to the convection term of dominated PDEs produces spurious oscillations. To avoid introducing oscillations, it is necessary to discretize the convection term using a downwind/upwind scheme, which means that the direction of one-sided difference needs to be adjusted adaptively according to the sign of the convection term at each discrete point." [Peng Li 2018](https://www.sciencedirect.com/science/article/pii/S0898122117306880#b13)

## **B.2)** Carmona (2005) (no correlation needed) - Pricing Under historical Measure 

![image](https://github.com/user-attachments/assets/f6e4ba40-496f-4d25-97a8-2c057ccc47ec)

![image](https://github.com/user-attachments/assets/8955fdde-982c-485f-9e64-2d7d81b79202)



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
|Hazard and Compound Intensity: | [Mockus Equation - hydrological analysis](https://doi.org/10.13031/2013.41082) ; [CLIMADA](https://github.com/CLIMADA-project/climada_python) | | src -  [flood precip compound risk](https://global-flood-database.cloudtostreet.ai/#interactive-map)|
|Wind speed and pressure: | GHYP (Generalized Hyperbolic distribution) ; Generalised Extreme Value | params fitted pre 2006 [1](https://jdhconsult.com/index_htm_files/APCWE7%20Extreme%20wind%20speeds%20and%20wind%20load%20factors%20for%20Hong%20Kong.pdf) | |
|Temp:| [Alaton](https://rstudio-pubs-static.s3.amazonaws.com/953546_4548bb57d50344ff984963ff47645e2e.html) | | |
|Climate compound risk: | Clim ODE | |1. [El Nino and El Nina](https://github.com/JK2-4/WD/blob/1d-PDE/RESULTS%20(Other%20papers)/generated/ESMValtool_ClimateModel/%20El%20Ni%C3%B1o%20and%20La%20Ni%C3%B1a%20events/netCDFdata_and_bibtex%20files.zip) 2. [Climate Extreme Index](https://docs.esmvaltool.org/en/latest/recipes/recipe_extreme_index.html) |
|Infrastructure risk - Urban Overheating (HK specific which is a  subtropical high-density city; [Heat stress parametric insurance](https://en.prnasia.com/releases/apac/axa-launches-pioneering-heatwave-parametric-insurance-456281.shtml)) | | Urban Heat Island (UHI), Urban Breeze Intensity (UBI), UHI-wind cross effect (Phenomena when highly dense cities experience assymmetric temperature than other areas. Measures global warming risk.) [Urban morphology in climate models for Hong Kong](https://livrepository.liverpool.ac.uk/3159432/1/UCLIM-2018-Mapping%20the%20local%20climate%20zones%20of%20urban%20areas%20by%20GIS-based%20and%20WUDAPT.pdf)| |
|Marginal risk/Extreme Value Theory (EVT) modeling| | Generalised Pareto (GP) model for upper tail and Gamma for lower tail; Laplace and Gumbel transform on data| station 6001 hourly data 2010-2024 - temp, rel_hum, wind_dir| 
|Spillover risk_cross district **(for optimized weight vector)**|BK and DY model for directional and pairwise volatility spillover indices **(on-hold)** |1 empirical src in fin mkt [application](https://www.nature.com/articles/s41467-023-42925-9#code-availability) 2. method [ref paper](https://www.sciencedirect.com/science/article/abs/pii/S0360544224033218) 3.spatial model ref [pg 36](https://www.math.cmu.edu/users/bachelier/seminar_files/carnegiemellon-feb09-2.pdf)| 1.climate_var_data_[30m resol_hk](https://figshare.com/articles/dataset/Hong_Kong_climate_vegetation_and_topography_rasters/6791276?file=14628722) 2. to aggreg@[3320 spatial units](https://github.com/JK2-4/WD/blob/1d-PDE/DATA_weather/Real%20Estate/big_data/centaline_buildings_ref.csv) |


**Data to cite** - [Real Estate](https://data.gov.hk/en-data/dataset/hk-rvd-tsinfo_rvd-property-market-statistics) - metadata [dict](https://www.rvd.gov.hk/datagovhk/Data_Dic.pdf) | bi-hourly weather [14 years](https://cowin.hku.hk/english/blog.html) | Local climate zone [LCZ](https://lcz-generator.rub.de/factsheets/81c119e7a56e90963d611663e6785b7669aa1473/81c119e7a56e90963d611663e6785b7669aa1473_factsheet.html) and empirical spatial [correl](https://github.com/JK2-4/WD/blob/1d-PDE/DATA_weather/Geospatial/to%20cite_ref_lcz_weight%20schema.jpeg) - [ref](https://www.researchgate.net/figure/a-Dissimilarity-and-b-similarity-metric-for-LCZ-classes-Appendix-A-in-4_fig3_341849288)

**WIP** - [1. HK_districts](https://gohkust-my.sharepoint.com/:w:/g/personal/jkwatra_ust_hk/EQdMTeeAAXpFp8maRr4gYYEBCcaguDTBCdRVDw8yZ5_oNw?e=Vxnm95) | [2. Cross-sectional results to reproduce](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-42925-9/MediaObjects/41467_2023_42925_MOESM1_ESM.pdf) | [3. PPC extreme value stats](https://apc01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2FECSADES%2Fecsades-matlab&data=05%7C02%7Cjkwatra%40ust.hk%7C572c194225874487620708dcf881fe3f%7Cc917f3e2932249269bb3daca730413ca%7C1%7C0%7C638658484439937268%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=ovzfZ1L3548qqgT9sGcHc1S9pHiw1qB0TIoDVa9JMpc%3D&reserved=0) | [4. Climate simulations using ESMValtool](https://docs.esmvaltool.org/en/stable/recipes/recipe_impact.html) | 5. ecotrix - [option index simulation code ref](https://github.com/nhcb/thesis/blob/main/thesis_revised.ipynb) | 6. HJB minimization pricing code [ref](https://cocalc.com/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/4.1%20Option%20pricing%20with%20transaction%20costs.ipynb) | 7. Coupled adv-diff-reac Cauchy Problem for weather basis risk - FEM [code](https://github.com/unifem/fenics-notes/blob/master/notebooks/advection-diffusion-reaction.ipynb)

RESULTS : 
- real estate econometric tests - District/sub-district/Estate level
  [[1]](https://drive.google.com/drive/folders/1DC8c5_DmJmQsNtge0TbPE4R6-ODTvSLr?usp=sharing) ; ARIMA fits [[2]](https://drive.google.com/drive/folders/1sxfWnNBisyYSrpWtmtV-Lo5CekQj60-0?usp=sharing)

Concerns: 
- Hubber/pulse intervention needed in RE data?

HJB with jump difussion [econ_model_codes](https://benjaminmoll.com/codes/)
  

## **G)** **Colab link (to be updated)** - 

[Google colab - JK](https://colab.research.google.com/drive/1iEsWgOOY3vK39Unbrobov5RG2dxffYNG?usp=sharing)
