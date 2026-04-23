European Organization for Nuclear Research (CERN) Electron Collision Data
Particle collision events with two electrons

Context : 
This dataset contains 100k dielectron events in the invariant mass range 2-110 GeV for use in outreach and education. 
These data were selected for use in education and outreach and contain a subset of the total event information. 
The selection criteria may be different from that used in CMS physics results.

Acknowledgements
Cite as: McCauley, Thomas; (2014). https://opendata.cern.ch/record/304

Goal: Understand particle properties    

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Particle Physics Data Analysis & Invariant Mass Modeling

# Project Summary

This project analyzes particle collision data to study the **invariant mass (M)** distribution and uncover underlying physical phenomena using **data analysis, statistical modeling, and machine learning**.

A key goal is to validate known physics (Z boson peak) and explore whether invariant mass can be predicted from observable features like particle momentum.

---

Problem Statement

In high-energy physics, invariant mass is a crucial quantity used to identify particles produced in collisions.

This project aims to:

* Understand the statistical distribution of invariant mass
* Verify whether the dataset reflects known physical laws
* Analyze relationships between particle momentum and invariant mass
* Evaluate whether machine learning can predict invariant mass

---

Dataset Description

The dataset represents particle collision events.

# Features:

 Feature - Description                       
 ------- - --------------------------------- 
 `pt1`   - Transverse momentum of particle 1 
 `pt2`   - Transverse momentum of particle 2 
 `M`     - Invariant mass of the system      

---

Tech Stack

* Python
* Pandas (data manipulation)
* NumPy (numerical operations)
* Matplotlib & Seaborn (visualization)
* SciPy (Gaussian curve fitting)
* Scikit-learn (machine learning)

---

Project Workflow

Data Loading & Initial Inspection

* Loaded dataset using Pandas
* Checked:

  * Shape of data
  * Data types
  * Summary statistics
* Verified presence of missing values

---

Data Cleaning

* Handled missing values in `M` using mean imputation

Why mean imputation?

* Maintains dataset size
* Suitable when missing values are minimal
* Does not heavily distort distribution

---

Exploratory Data Analysis (EDA)

Distribution of Invariant Mass

* Histogram plotted for `M`
* Observed a **distinct peak around ~90 GeV**

Interpretation:

* Indicates presence of a resonance particle

---

Physics Insight

The observed peak corresponds to the mass of the **Z boson (~91 GeV)**, a well-known particle in particle physics.

---

Scatter Analysis

Plots created:

* `pt1 vs M`
* `pt2 vs M`

Observation:

* No strong linear relationship
* Invariant mass is not dependent on a single particle’s momentum

---

Correlation Analysis

* Correlation matrix computed

Result:

* Weak correlation between:

  * `pt1` and `M`
  * `pt2` and `M`

Interpretation:
Invariant mass depends on **combined energy and momentum**, not individual components.

---

Statistical Modeling — Gaussian Fit

A Gaussian model was fitted to the invariant mass distribution.

Model:

* Normal distribution characterized by:

  * Mean (μ)
  * Standard deviation (σ)

Results:

* μ ≈ 90–91 GeV
* σ represents spread due to detector resolution / natural width

Conclusion:

* The data aligns with expected physical behavior
* Confirms presence of Z boson resonance

---

Key Findings

* Strong peak at ~90 GeV confirms Z boson events
* Invariant mass follows a Gaussian-like distribution
* Weak correlation between individual momentum and invariant mass
* Data analysis aligns with real-world physics principles

---

Limitations

* Limited features (no energy, angles)
* Mean imputation may introduce bias
* Only basic ML model used
* No outlier treatment
* No residual analysis for Gaussian fit

---

Future Improvements

* Add derived features (momentum combinations)
* Include additional physics variables if available
* Use advanced models:

  * Random Forest
  * Gradient Boosting

* Perform:

  * Outlier detection
  * Residual analysis
  * Model comparison

---

Conclusion

This project demonstrates how data analysis and physics knowledge intersect.

It shows that:

* Statistical methods can validate physical theories
* Data visualization can reveal hidden structures

---

This project showcases:

* Strong EDA skills
* Statistical modeling
* Domain understanding (physics)
* Critical thinking about ML limitations

---

# Author

Monugya Borchetia
