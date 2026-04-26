Cancer Mutation Analysis

--- Project Overview

This project analyzes cancer mutation data to identify patterns across different cancer types


-- Dataset
cBioPortal dataset


-- Tools
Python, Pandas, Matplotlib, Seaborn


-- Status
Work in progress — analysis ongoing - Completed 




Genomic Analysis of Leukemia Survival 


To analyze how genetic mutations and tumor characteristics influence patient survival in Acute Myeloid Leukemia (AML) 
using TCGA data.


Survival Distribution -----------

Survival is highly right-skewed
Majority of patients die within 0–20 months
Few long-term survivors (>60 months)




Survival Status -----------

Deaths significantly exceed survivors




Mutation Load vs Survival -----------

No clear linear relationship in scatter plot
Most patients cluster at low mutation counts
Some high mutation outliers exist




Top Mutated Genes -----------

Top genes identified:

DNMT3A
NPM1
FLT3
TP53
IDH2 / IDH1



Mutation Types -----------

Missense mutations dominate heavily
Followed by splice site and frameshift mutations



High-Impact Mutations -----------

Top high-impact genes:

NPM1 (dominant)
TET2
DNMT3A
RUNX1



TMB vs Survival -----------
results:

Low     --- ~19.47 months
Medium  --- ~18.59 months
High    --- ~16.89 months


Statistical Test:
p-value = 0.467




Mutation Load vs Survival -----------

Low     --- ~19.06 months
Medium  --- ~16.51 months
High    --- ~19.41 months




Correlation Analysis -----------

TMB ↔ Mutation Count → ~1.0 (very strong)
TMB ↔ Survival → ~0.26 (weak)
Mutation Count ↔ Survival → ~0.26 (weak)




Survival Curve Insight -----------

From ECDF:

Dead patients drop early (steep curve)
Survivors extend over longer time




While mutation burden (TMB and mutation count) shows slight trends with survival, statistical testing confirms that 
these factors alone are not sufficient predictors of patient outcomes in AML.
Instead, specific gene mutations (e.g., NPM1, DNMT3A) appear more biologically relevant and may serve as better 
indicators of disease progression.

