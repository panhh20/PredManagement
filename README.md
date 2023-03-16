# Predictive Management in Manufacturing using Machine Learning

## Overview

This project addresses the business problem of predicting component failures and answering the questions, "Will a machine fail in the near future due to component problems?" and "What would be the mode of failure?"

The problem is divided into two parts: binary classification and multilabel classification. Machine learning algorithms are used to create a predictive model that learns from machine-functioning simulated data.


## Data Collection & Description
Since real predictive maintenance datasets are generally difficult to obtain due to confidentiality, I obtained a synthetic dataset from the UCI Machine Learning Repository that reflects real predictive maintenance encountered in the industry.

Source:
UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/AI4I%202020%20Predictive%20Maintenance%20Dataset 

Author: Stephan Matzka, School of Engineering - Technology and Life, Hochschule für Technik und Wirtschaft Berlin, 12459 Berlin, Germany, stephan.matzka '@' htw-berlin.de

<b>Data Set Description (UCI Machine Learning Repository):</b>

The dataset consists of 10 000 data points stored as rows with 14 features in columns.

<li> <b>UID</b>: unique identifier ranging from 1 to 10000 </li>
<li> <b>product ID</b>: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number </li>
<li> <b>type</b>: L, M, H for product quality low, medium and high </li>
<li> <b>air temperature [K]</b>: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K </li>
<li> <b>process temperature [K]</b>: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
<li> <b>rotational speed [rpm]</b>: calculated from a power of 2860 W, overlaid with a normally distributed noise </li>
<li> <b>torque [Nm]</b>: torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values </li>
<li> <b>tool wear [min]</b>: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process </li>
<li> <b>'machine failure'</b> label that indicates, whether the machine has failed in this particular datapoint for any of the 5 failure modes:
  <ul>
  <li> <b>tool wear failure (TWF)</b>: the tool will be replaced of fail at a randomly selected tool wear time between 200 and 240 mins (120 times in this dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned). </li>
  <li> <b>heat dissipation failure (HDF)</b>: heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tool’s rotational speed is below 1380 rpm. This is the case for 115 data points. </li>
  <li> <b>power failure (PWF)</b>: the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in this dataset. </li>
  <li> <b>overstrain failure (OSF)</b>: if the product of tool wear and torque exceeds 11,000 Nm/min for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints. </li>
  <li> <b>random failures (RNF)</b>: each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in this dataset. </li>
  </ul>
  </li>

The 'machine failure' label is equals 1 when one or more of the above failure modes is true.

## Exploratory Data Analysis

### Correlation Analysis
The first 2 variables: <b>UID</b> and <b>productID</b> do not seem to have an influence on the machine failure, hence I excluded these variables from the Correlation analysis.

<b>Type</b> is a categorical variable so I created dummy variables for it, and then standardized all variables before calculating the Correlation matrix


<p align="center">
<img width="420" alt="Correlation matrix" src="https://user-images.githubusercontent.com/122824839/225740452-3d19086e-08c8-4a9c-88a9-a70d082cc36b.png">
</p>
<p align="center">
<b>Figure 1</b>. Correlation matrix

</p>

We can see that the Torque [Nm] and Rotational speed [rpm] variables are strongly negatively correlated, while the Process temperature [K] and Air temperature [K] are strongly positively correlated, which might lead to Multicollinearity.
However, this is not surprising because these highly correlated variables were derived based on similar components, including temperature and machine forces. 
Additionally, Torque [Nm] is moderately positively correlated to OSF (overstrain failure) and HDF (heat dissipation failure); hence, it is expected that Rotational speed [rpm] has some negative correlation with these failure modes. This suggests that while a higher Torque may result in overstrain and heat dissipation failure, a higher Rotational speed may mitigate these issues.
Besides, Tool wear [min] is also positively correlated to OSF (overstrain failure) and TWF (tool wear failure), implying that tool wear is likely one of the causes of these two types of failure.

Principal Component Analysis (PCA)
PCA is used for further data exploration instead of doing feature selection since the dataset is not high dimensional. The results of PCA indicated that over 99% of the variance in the dataset can be explained by the first 3 principal components. 
 
<p align="center">
<img width="540" alt="Screen Shot 2023-03-16 at 3 56 14 PM" src="https://user-images.githubusercontent.com/122824839/225741134-b53261bb-5b0b-4542-8dc1-dc0f34b4d3f4.png">
<p align="center">
<b>Figure 2</b>. R PCA summary 
</p>



<p align="center">
<img width="640" alt="PCA plots" src="https://user-images.githubusercontent.com/122824839/225740448-c80dbc6d-ea8f-4a8e-b6f3-def4a10fc681.png">
<p align="center">
<b>Figure 3</b>. Principal Component Loadings
</p>

The bar plot of Principal Components loadings makes it easy to understand what they represent:
<li> PC1 consists primarily of two temperature variables. </li>
<li> PC2 is represented by machine power, which is the combination of Rotational Speed [rpm] and Torque [Nm] values. </li>
<li> PC3 is associated with Tool wear [min]. </li>

### Variable Importance
Furthermore, the variable importances were calculated based on the features’ predictive power in the Random Forest algorithm. Based on the Gini Index, the top 3 important features are the machine power variables (Torque (Nm), Rotational Speed [rpm]) and Tool wear [min].

<p align="center">
<img width="600" alt="Variable importance" src="https://user-images.githubusercontent.com/122824839/225740443-f61a991c-9b69-4564-a730-ec81401741ce.png">

<p align="center">
<b>Figure 4</b>. Variable Importance Plot: Random Forest
</p>


## Machine Learning Models
### Binary Classification
The first question this research seeks to answer is "Will a machine malfunction in the near future?" It's a binary classification with a yes/no answer ("yes" = 1, "no" = 0).

Random Forest and XGBoost are chosen for this problem.

The algorithms were selected primarily for the following reasons: 

(1)	The Random Forest model is a robust method for handling high-variance data since it can reduce the variance significantly by de-correlating and averaging across multiple bagged trees. 

(2)	Random Forest and XGBoost are powerful models capable of capturing more complex and non-linear relationships, which appears to be a good fit for our problem because the relationship between the input and response variables is not simple and may not be linear.

(3)	XGBoost is speed-optimized by incorporating additional approximations and memory-saving tricks. Additionally, XGBoost facilitates custom constraints and regularization to prevent overfitting.

The Training set is 70% of the dataset and the Test set is 30%, which are both randomly sampled without replacement .

#### Random Forest
		 	 	 		
In the Random Forest model, Cross-validation was performed for mtry as a part of hyperparameter tuning. The Random Forest model was repeated many times to test different values of mtry (from 1→ 6 as there are 6 predictors), then selected the optimal mtry that minimizes out-of-bag (OOB) error and refit the model. As a result, the optimal mtry is 5. 

<p align="center">
<img width="450" alt="Optimal mtry" src="https://user-images.githubusercontent.com/122824839/225740439-fb58af3f-b4eb-4b87-929a-669a75c946ab.png">
<p align="center">
<b>Figure 5</b>. Optimal mtry for Random Forest
</p>

I also plotted the OOB Error rate against the number of trees to find the optimal number of trees. The plot shows that after about 250 trees, the error rate begins to flatten. As a result, I chose ntree = 250.

<p align="center"> 
<img width="600" alt="RF OOB error" src="https://user-images.githubusercontent.com/122824839/225740434-7d629f9c-7ed0-4498-b764-7982dc417b70.png">
<p align="center">
<b>Figure 6</b>. Random Forest: OOB Error Rate vs Number of Trees
</p>

#### XGBoost
For the XGBoost Classification model, it’s observed that the optimal hyperparameters were maximum depth of 4 and learning rate of 0.01 using Grid Search.

<p align="center"> 
<img width="800" alt="XGBoost" src="https://user-images.githubusercontent.com/122824839/225741096-41abc0a7-e0e3-45bd-97f2-215095379fb3.png">
<p align="center">
<b>Figure 7</b>. Example of a boosted tree in the XGBoost model (Tree 5)
</p>

### Multi-label Classification

The second question this research seeks to answer is "Which type of machine failure would it be?" It is a multilabel classification problem in which each observation may have many responses.

In this case, different failure modes like tool ware failure, power failure, etc. can occur at the same time, causing machine malfunction. I used the same features as in the binary classification problem above to predict the type of machine failure.

I chose Multivariate Random Forest for this problem since the algorithm can handle observations with multiple outcomes (Xiao & Segal, 2009). Also, Random Forest tends to outperform other classification algorithms for multi-label problems (Wu, Gao & Jiao, 2019).


## Results
### Binary Classification
<p align="center">
<img width="582" alt="Screen Shot 2023-03-16 at 3 57 36 PM" src="https://user-images.githubusercontent.com/122824839/225741138-9e3daa45-6380-4077-8de9-e4e877b3c13c.png">
</p>

Two model evaluation metrics include Accuracy and False Negative Rate, are calculated as follows:
 
where:

TP = Number of True Positives

TN = Number of True Negatives

FP = Number of False Positives

FN = Number of False Negatives


In this case, the Random Forest model appears to perform better. Random Forest has a higher Accuracy and lower False Negative Rate. The False Negative Rate is critical in this business problem because unidentified malfunctioning machines can cause significant production delays and costs.



### Multi-label Classification

Overall, Multivariate Random Forest performed well and achieved a Classification accuracy of greater than 99.9% for each failure mode.

<p align="center">
<img width="395" alt="Screen Shot 2023-03-16 at 3 57 55 PM" src="https://user-images.githubusercontent.com/122824839/225741144-d3740ade-fcdf-4797-8ba3-79099e506e63.png">
<p align="center">
<b>Figure 8</b>. Multi-label Classification Accuracy
</p>

## Conclusion

The findings indicate that XGBoost and Random Forest models perform well for binary classification in general, but Random Forest does slightly better. For multi-label classification problems, Multivariate Random Forest is an effective model.


