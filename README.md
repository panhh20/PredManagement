# Predictive Management in Manufacturing using Machine Learning

## Abstract
Several manufacturing firms incur significant costs as a result of production process delays caused by mechanical issues. There are many potential causes of delays, such as equipment failure or environmental factors. Hence, it is essential to predict these issues in advance so that businesses can prevent them proactively, resulting in substantial cost savings.

This research paper addresses the business problem of predicting component failures and answering the questions, "Will a machine fail in the near future due to component problems?" and "What would be the mode of failure?"

The problem is divided into two parts: binary classification and multilabel classification. Machine learning algorithms are used to create a predictive model that learns from machine-functioning simulated data.


## Data Collection & Description
Since real predictive maintenance datasets are generally difficult to obtain due to confidentiality, I obtained a synthetic dataset from the UCI Machine Learning Repository that reflects real predictive maintenance encountered in the industry.
Source:
UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/AI4I%202020%20Predictive%20Maintenance%20Dataset 
Author: Stephan Matzka, School of Engineering - Technology and Life, Hochschule für Technik und Wirtschaft Berlin, 12459 Berlin, Germany, stephan.matzka '@' htw-berlin.de

Data Set Description (UCI Machine Learning Repository):
The dataset consists of 10 000 data points stored as rows with 14 features in columns.
UID: unique identifier ranging from 1 to 10000
product ID: consisting of a letter L, M, or H for low (50% of all products), medium (30%) and high (20%) as product quality variants and a variant-specific serial number
type: L, M, H for product quality low, medium and high
air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K
process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
rotational speed [rpm]: calculated from a power of 2860 W, overlaid with a normally distributed noise
torque [Nm]: torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values.
tool wear [min]: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. 
'machine failure' label that indicates, whether the machine has failed in this particular datapoint for any of the 5 failure modes:
●	tool wear failure (TWF): the tool will be replaced of fail at a randomly selected tool wear time between 200 and 240 mins (120 times in this dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
●	heat dissipation failure (HDF): heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tool’s rotational speed is below 1380 rpm. This is the case for 115 data points.
●	power failure (PWF): the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in this dataset.
●	overstrain failure (OSF): if the product of tool wear and torque exceeds 11,000 Nm/min for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.
●	random failures (RNF): each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in this dataset.
The 'machine failure' label is equals 1 when one or more of the above failure modes is true.

## Exploratory Data Analysis

### Correlation Analysis
The first 2 variables: UID and productID do not seem to have an influence on the machine failure, hence I excluded these variables from the Correlation analysis.

Type is a categorical variable so I created dummy variables for it, and then standardized all variables before calculating the Correlation matrix.
![image](https://user-images.githubusercontent.com/122824839/225735377-1ca05ca5-5aa4-4b94-910f-41ff699fa6aa.png)

