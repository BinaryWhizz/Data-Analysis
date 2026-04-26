Cancer Mutation Analysis

--- Project Overview

This project analyzes cancer mutation data to identify patterns across different cancer types


-- Dataset
cBioPortal dataset


-- Tools
Python, Pandas, Matplotlib, Seaborn


-- Status
Work in progress — analysis ongoing.



data_mutations.txt file :  

Genome - The entire set of DNA instructions found in a cell

Core Gene & Identification Columns----
Hugo_Symbol - Standard gene name (HGNC-approved) - grouping mutations by gene, frequency analysis 
Entrez_Gene_Id - Unique gene ID from NCBI database - linking with external biological databases 
Center - Institution or lab that generated the data - data source tracking / bias checking 

Genome Reference & Location----
NCBI_Build - Reference genome version used - coordinates depend on this 
Chromosome - Chromosome number (1–22, X, Y) - chromosomal distribution plots 

Start_Position / End_Position - Exact genomic coordinates of mutation - mutation mapping, visualization

Strand - DNA strand: + → forward, - → reverse - Usually less critical unless doing sequence-level analysis 