# IC PCDA CAPSTONE PROJECT - Harry Bakhshi – Multivariate Time Series Forecasting in Python with ML
My capstone project repository for my Imperial College Professional Certificate in Data Analytics (Nov 2023 - July 2024):

https://harry-b6.github.io/IC_PCDA_capstone_Harry_Bakhshi/

## Background

Time series forecasting is widely utilised across a range of industries, such as in modeling and forecasting of retail, financial, biological, and climate systems (1), as are machine learning approaches (2). Machine learning approaches to time series forecasting problems are an alternative to classical statistical model approaches, such as Winters exponential smoothing and Box-Jenkins ARIMA (3), and SARIMA (4). Machine learning approaches provide the advantage of requiring less "technical" and "sophisticated mathematical" expertise than that which is used in statistical approaches for establishing the model parameters for implementation (4). Furthermore, machine learning approaches do not have to assume the data follow a known distribution as in these parametric methods (4). In parametric approaches, the model parameters for the assumed model are used to forecast the future values of the time series (5), and usually estimated using Maximum Likelihood Estimation (MLE) (6). For an ARMA (= AutoRegressive Moving Average) model assumed, the model parameters would be the constant $c$, all $\varphi$<sub>i</sub>, and all $\phi$<sub>i</sub> (6). An example of an assumed model used in one of these parametric approaches (the Box-Jenkins ARMA approach) would be an "ARMA(2, 1)" model - an AutoRegressive Moving Average (ARMA) model of autoregression order p = 2 and moving average order q = 1 (6), where p and q are the orders used (e.g. determined by corrected Akaike’s Information Criterion (AICc)) (7). Maximum Likelihood Estimation is a method of finding the value of a model parameter that maximises the likelihood of observing a given data set (8). Multivariate/multiple time series (9) versions of these classical and parametric (3)(4)(6), univariate (6) time series forecasting methods also exist, such as VARIMA (Vector ARIMA) (6), VARMA (Vector ARMA) (10), and vector innovations structual time series (VISTS), a multivariate exponential smoothing method (11).

This project implements multivariate Time Series Forecasting by machine learning (regression) in Python on a retail industry panel dataset (converted to multivariate time series problem) with 2 years of data (Jan 2011 - Dec 2013), to forecast 19 weeks ahead, across 28 products found in 76 stores, the metrics:

- sales price
- units sold
- base price
- extent to which product is prominently featured in store (aggregation from counts from dataset)
- extent to which product is on special display in store (aggregation from counts from dataset)

To exemplify this solution, one variable (the sales price of an individual product) was forecast from 873 features containing 135 original variables (from product + metric combinations). These features were produced by first converting the panel data to a multivariate time series, and then by engineering more features using seasonal/non-seasonal Rolling Window aggregations, the EWMA, and temporal and Fourier terms. 4 of the original variables produced from the panel data conversion were found to be non-forecastable by coefficent of variation and residual variability and were dropped before engineering more features. The selection of linear and non-linear approaches used were:

- Linear Regression
- Lasso Regression
- Linear Support Vector Regression (Linear SVR)  
- XGBoost Regression
- LightGBM Regression
- Decision Tree Regression
- Random Forest Regression

The data source:  https://datahack.analyticsvidhya.com/contest/janatahack-demand-forecasting/True/#ProblemStatement

## How to use this project

After cloning the project, to use the adapted source code (adapted from Manu Joseph project - https://github.com/PacktPublishing/Modern-Time-Series-Forecasting-with-Python) in the src folder (used in some of the notebooks - https://github.com/harry-B6/IC_PCDA_capstone_Harry_Bakhshi/tree/main/main/src), simply add the src folder (and panel data conversion module if using) to the PYTHONPATH and for each notebook install extra packages required (commented out at the stage of the project in the Jupyter notebooks).

This project has a GNU GPLv2 license: https://github.com/harry-B6/IC_PCDA_capstone_Harry_Bakhshi/blob/main/LICENSE   

Please read the license in full before first use of the project.   
Please adhere to this license when using this project and reproduce it when required.   

## Challenges and Future Work

The first challenge encountered in this project was how best to work with the panel dataset. I chose to convert the data to a multivariate time series by binning the sales price, base price, units sold values, and featured/display aggregations for each product per timestamp, using justifiable measures selected from measures of central tendency (mean, median, mode) to ensure as much robustness to outliers as possible. Additionally, I created a module for parallel computing to enable parallel conversion of the panel data to a multivariate time series, creating a time series for each metric, product and store combination (= 10640 variables). This worked and was written to be as time- and memory- efficient as possible after much testing and re-examination of the code, implemented with the ability to pause and resume the code. However I chose to create variables only for each product and metric combination due to the time constraints of my capstone project and concerns about the likely forecastability of the majority of the 10640 variables created by the parallel conversion implemention.

In order to reproducibly measure seasonality after my chosen seasonal decomposition method of Fourier Decomposition, I wrote my own implementation using matrixprofile-ts (see https://github.com/target/matrixprofile-ts/tree/master). I created a pipeline for this to perform this process on each product and metric combination. Similarly, I produced an additional pipeline to detect and treat outliers by seasonal/non-seasonal IQR and iForest. After baseline forecasting, forecastability assessments (by C.o.V., Residual Variability, (Residual/) Spectral Entropy, (Modified/) Kaboudan Metric) and making the data weakly stationary for both seasonal and non-seasonal variables (more pipelines), I was not able to automate hyperparameter tuning using GridSearch with my time constraints. This method was lengthy for my target variable in total after performing for each of my 6 models (excluding Linear Regression) and may have been faster using a random search approach.

In the future I would like to experiment with approaches adapted to panel data. Two examples of these approaches are Bayes predictors adapted to panel data forecasting (12) and deep learning methods that allow direct sequence to sequence predictions, such as sequence-to-sequence Regression using Deep Learning (13).

## Acknowledgements

I am grateful to Vikesh Koul (https://github.com/vkoul/vkoul), the Program Leader for my cohort of the Imperial College Professional Certificate in Data Analytics, for his support in my training for my I.C. PCDA certification, and to Manu Joseph for his publication "Modern Time Series Forecasting with Python", which I purchased access to through Perlego (https://www.perlego.com). The supplementary notebooks to the explantations in this book helped me form my own solutions to my project data problem. Both the book and the notebooks helped enrich my understanding of time series forecasting by machine learning using regression.

I am also grateful to my cohort of the I.C. PCDA (Nov. 2023 – July 2024), for their insightful contributions to group work on the course and their unique perspectives shared from many different domains, and to Dr. Alex Ribeiro-Castro, Dr. Fintan Nagle, and Prof. Wolfram Wiesemann, for their time and efforts in contributing to the educational content of the I.C. PCDA programme.

## Bibliography (for README)

(1) - https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0209 - 'Time-series forecasting with deep learning: a survey' - Brian Lim and Stefan Zohren, 2021 - Accessed 26/07/2024   
(2) - https://www.sas.com/en_gb/insights/articles/analytics/applications-of-machine-learning.html - Accessed 23/07/2024   
(3) - https://www.researchgate.net/publication/227612766_An_Empirical_Comparison_of_Machine_Learning_Models_for_Time_Series_Forecasting - "An Empirical Comparison of Machine Learning Models for Time Series Forecasting" - Nesreen K. Ahmed, 2010 - Accessed 23/07/2024; for learning that these classical statistical methods are parametric, (4) and (6) below were used.    
(4) - https://www.sciencedirect.com/science/article/abs/pii/S0020025519300945 - "Evaluation of statistical and machine learning models for time series prediction: Identifying the state-of-the-art and the best conditions for the use of each model" - Parmezan et al., 2019 - Accessed 23/07/2024; for learning that these statistical, parametric methods are classical methods, (3) above and (6) below were used.        
(5) - http://www.jestr.org/downloads/Volume13Issue3/fulltext181332020.pdf - "Parametric versus Non-Parametric Time Series Forecasting Methods: A Review" - Anjali Gautam, Vrijendra Singh, 2020 - Accessed 23/07/2024  
(6) - Machine Learning for Time-Series with Python (Ben Auffarth, 2021); for learning which parameters are estimated in the classical (see (3)), parametric (see (4)) Moving Average models, AutoRegressive models, ARMA and ARIMA models, (7) below was additionally used.    
(7) - https://otexts.com/fpp2/arima-estimation.html - Accessed 25/07/2024  
(8) - Module 13, IC PCDA: 'Introduction and information' - Accessed 29/03/2024   
(9) - https://ricerca.uniba.it/bitstream/11586/174262/7/Using%20Multiple%20Time%20Series%20Analysis%20for%20Geosensor%20Data%20Forecasting_PreprintIRIS.pdf - "Using Multiple Time Series Analysis for Geosensor Data Forecasting" - Pravilovic, S. et al., 2017 - Accessed 26/07/2024; for ascertaining that multiple time series are also known as multivariate time series.   
(10) - https://www.economics-sociology.eu/files/12_Simionescu_1_7.pdf - "The Use of VARMA Models in Forecasting Macroeconomic Indicators' - Mihaela Simionescu, 2013 - Accessed 26/07/2024  
(11) - https://www.sciencedirect.com/science/article/pii/S037722172200354X#bib0005 - "A new taxonomy for vector exponential smoothing and its application to seasonal time series" - Sventunkov, I. et al., 2023 - Accessed 26/07/2024   
(12) - https://www.nber.org/system/files/working_papers/w25102/w25102.pdf - "Forecasting with Dynamic Panel Data Models" - Liu et al., 2018 - Accessed 23/07/2024   
(13) - https://uk.mathworks.com/help/deeplearning/ug/sequence-to-sequence-regression-using-deep-learning.html - Accessed 23/07/2024 

README designed using https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/ - Accessed 23/07/2024   
