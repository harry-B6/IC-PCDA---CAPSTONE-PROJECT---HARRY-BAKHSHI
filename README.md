# IC PCDA CAPSTONE PROJECT - Harry Bakhshi – Time Series Regression in Python with ML
My capstone project repository for my Imperial College Professional Certificate in Data Analytics (Nov 2023 - July 2024):

https://harry-b6.github.io/IC_PCDA_capstone_Harry_Bakhshi/

## Background

Multivariate time series regression is widely utilised across a range of industries, such as in modeling and forecasting of economic, financial, biological, and engineering systems (1), as are machine learning approaches (2). Machine learning approaches to time series regression problems are an alternative to classical statistical model approaches, such as Winters exponential smoothing and Box-Jenkins ARIMA (3), and SARIMA (4). Machine learning approaches provide the advantage of requiring less "technical" and "sophisticated mathematical" expertise than that which is used in statistical approaches for establishing the model parameters for implementation (4). Furthermore, machine learning approaches do not have to assume the data follow a known distribution as in these parametric methods (4). In parametric approaches, the model parameters for the assumed model (e.g. "MA(q)" (Moving Average (MA) model with "order" parameter "q" (5)) are used to forecast the future values of the time series (6) and usually estimated using Maximum Likelihood Estimation (MLE) (5). Maximum likelihood estimation is a method of finding the value of a model parameter that maximises the likelihood of observing a given data set (7).

This project implements Time Series Regression with machine learning in Python on a retail industry panel dataset (converted to multivariate time series problem) with 2 years of data (Jan 2011 - Dec 2013), to forecast 19 weeks ahead, across 28 products found in 76 stores, the metrics:

- sales price
- units sold
- base price
- extent to which product is prominently featured in store (aggregation from counts from dataset)
- extent to which product is on special display in store (aggregation from counts from dataset)

To exemplify this solution, one variable (the sales price of an individual product) was forecast from 873 features containing 135 original variables (from product + metric combinations). These features were produced by first converting the panel data to a multivariate time series, and then by engineering more features using seasonal/non-seasonal Rolling Window aggregations, the EWMA, and temporal and Fourier terms. 4 were found to be non-forecastable by coefficent of variation and residual variability and were dropped. The selection of linear and non-linear approaches used were:

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

In the future I would like to experiment with approaches adapted to panel data. Two examples of these approaches are Bayes predictors adapted to panel data forecasting (8) and deep learning methods that allow direct sequence to sequence predictions, such as sequence-to-sequence Regression using Deep Learning (9).

## Acknowledgements

I am grateful to Vikesh Koul (https://github.com/vkoul/vkoul), the Program Leader for my cohort of the Imperial College Professional Certificate in Data Analytics, for his support in my training for my I.C. PCDA certification, and to Manu Joseph for his publication "Modern Time Series Forecasting with Python", which I purchased access to through Perlego (https://www.perlego.com). The supplementary notebooks to the explantations in this book helped me form my own solutions to my project data problem. Both the book and the notebooks helped enrich my understanding of time series forecasting with regression using machine learning.

I am also grateful to my cohort of the I.C. PCDA (Nov. 2023 – July 2024), for their insightful contributions to group work on the course and their unique perspectives shared from many different domains, and to Dr. Alex Ribeiro-Castro, Dr. Fintan Nagle, and Prof. Wolfram Wiesemann, for their time and efforts in contributing to the educational content of the I.C. PCDA programme.

## Bibliography (for README)

(1) - https://www.mathworks.com/discovery/time-series-regression.html#:~:text=Common%20uses%20of%20time%20series,%2C%20biological%2C%20and%20engineering%20systems.&text=to%20get%20an%20estimate%20of,t)%20to%20the%20design%20matrix. - Accessed 23/07/2024   
(2) - https://www.sas.com/en_gb/insights/articles/analytics/applications-of-machine-learning.html - Accessed 23/07/2024   
(3) - https://d1wqtxts1xzle7.cloudfront.net/47902695/An_Empirical_Comparison_of_Machine_Learn20160808-9481-m871ee-libre.pdf?1470715710=&response-content-disposition=inline%3B+filename%3DAN_EMPIRICAL_COMPARISON_OF_MACHINE_LEARN.pdf&Expires=1721744403&Signature=KA01HnXTdV7LiRqfgD43lWk0~cYRdxFcoVciBkTD~HzJCFd7aWcz6c887RT~QGsa9J6ayRjSJIDJf~WILaqSTV4NSjtNNLSodPD63XM9okYDmgQrHOry1MUJa7YIs~K2~98cXGtpKoybHyhaPl~DJ8BI1BVHlbzkfNnczkVVm9JmI8bZbCX1kJ8Y2b4yG1CaM0ony6u~auylMkSRY~ugCkNq74xneGL3pWt59Mgf6tdQO~CbCiAVko8lJFLH1wfArsdo313XlEuEBIVGp2-y1Ur~Dbmb~TApvonu355D1AoHNoBwcPYIttgy-dh~HW5VBHrqPX4Z7XV2a8EVeL4yWw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA - "An Empirical Comparison of Machine Learning Models for Time Series Forecasting" - Nesreen K. Ahmed, 2010 - Accessed 23/07/2024   
(4) - https://www.sciencedirect.com/science/article/abs/pii/S0020025519300945 - "Evaluation of statistical and machine learning models for time series prediction: Identifying the state-of-the-art and the best conditions for the use of each model" - Parmezan et al., 2019 - Accessed 23/07/2024   
(5) - Machine Learning for Time-Series with Python (Ben Auffarth, 2021)    
(6) - http://www.jestr.org/downloads/Volume13Issue3/fulltext181332020.pdf - "Parametric versus Non-Parametric Time Series Forecasting Methods: A Review" - Anjali Gautam, Vrijendra Singh, 2020 - Accessed 23/07/2024
(7) - Module 13, IC PCDA: 'Introduction and information' - Accessed 29/03/2024   
(8) - https://www.nber.org/system/files/working_papers/w25102/w25102.pdf - "Forecasting with Dynamic Panel Data Models" - Liu et al., 2018 - Accessed 23/07/2024   
(9) - https://uk.mathworks.com/help/deeplearning/ug/sequence-to-sequence-regression-using-deep-learning.html - Accessed 23/07/2024 

README designed using https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/ - Accessed 23/07/2024   
