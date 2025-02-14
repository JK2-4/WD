# Inference of insurance premium pricing from indirect assets 

|Data type|Description|Granularity|variables|src|
|--|-------|-------|--------|---------|
|1|Median Monthly Mortgage Loan repayment (2021) **only cross-sectional data** with **3% of property value** as government rent as per policy| sub-unit group level (1.74k+ units), [TPU](https://www.bing.com/ck/a?!&&p=a3ee3f2e698af25634b375302071eef7cbbf849e7304b1e31081a9274d8657e3JmltdHM9MTczOTQwNDgwMA&ptn=3&ver=2&hsh=4&fclid=30d79aef-a69b-67d5-30ac-8f7fa79366e2&psq=district+coding+system+hong+kong&u=a1aHR0cHM6Ly93d3cucnZkLmdvdi5oay9kb2MvdGMvaGtwcjE1LzA2LnBkZg&ntb=1) level (200+)|mortgage, rent, demographic characteristics|[CSD](https://idds.census2021.gov.hk/app/idds.html#) ; data [1.74k](https://github.com/JK2-4/WD/blob/1d-PDE/DATA_weather/Agents/Insurance/LSUG_data.csv) and [200](https://data-esrihk.opendata.arcgis.com/datasets/esrihk::hong-kong-domestic-households-by-mortgage-payment-and-loan-repayment-by-small-tpu-in-2021/about)|
|0|Annual Underwriting net premium - By insurance type - 2019 to 2023| National level |prem,opex, no. of businesses|CSD| 
|1|Monthly Residential mortgage -  Dec'16 to Dec'24| National level | o/s, loan apps,delinq ratio||
|1|Annual income tax- 2019 to 2023| National level|range-wise amt,no. of payers||
|1| Monthly Residential price index - 1993 to 2023|National level|index||

## 0. Direct premium data available [Link](https://drive.google.com/drive/folders/1PGcvFXZc1SvMfrurqTZV2V3HQs5W-gFA?usp=sharing)

## 1. Mortgage 

Mortgage insurance premium pricing (% of outstanding principal) across LTV ratios. - [Link](https://www.hkmc.com.hk/files/product_file/3/1396/Premium%20Rate%20Sheet_Eng_clean_16102024.pdf)

Government rent roll - payment fixed @ [3%](https://www.rvd.gov.hk/en/our_services/government_rent.html#:~:text=Percentage%20Charge,changes%20in%20the%20rateable%20value.) of rateable value

## 2. Infrastructure Securities 

### 2.1. Retail Infrastructure Bonds

HKSAR Government Retail Infrastructure Bond (December 2024 - June 2025)

Daily accrued interest rate (pricing) per lot - [Data](https://www.hkex.com.hk/-/media/HKEX-Market/Products/Securities/Debt-Securities/Accrued-Interest-Table-for-iBonds/4286-17122024.pdf)

Clean price data

Loan Loss Model: Climate credit valuation adjustment (cva)

[Link](https://www.hkex.com.hk/Products/Securities/Debt-Securities/Market-Information/Accrued-Interest-Table-for-Retail-Green-Bonds?sc_lang=en)

### 2.2. Project and Infra Risk - Tender Pricing 

Construction project risks affect contractors' tender prices. 

Hong Kong Government - Pricing of Tenders [Awarded](https://www1.ets-cs.gov.hk/eppcs_ext/views/index.zul) - Engineering and Architecture (Landslide, water etc)

## 3. Securitised Receivables 

Corporate loan receivables, project loan receivables, consumer loan receivables and property mortgages (residential and commercial). 

ILS 
R/C MBS



