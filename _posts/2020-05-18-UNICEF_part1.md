---
title: "UNICEF Project - Part 1 (Data Preparation)"
date: 2020-05-18
permalink: /Projects/
tags: [UNICEF, SOWC, Data Science, xlrd]
header:
  image: "/images/Project/unicef.jpg"
excerpt: "UNICEF, SOWC, Data Science, xlrd"
mathjax: "true"
---

# Key Indicators of World's Childeren - Part - 1

***

We are going to go through a dataset from [UNICEF](https://www.unicef.org/). It includes different indicators for different countries around the world. This dataset is a collection of different indicators which contains 13 sheets. Each sheet has a different type of indicators such as *Education, Health, Nutrition, etc*.

**Objectives:**
After reading this post you should take away the following objectives:
- Importing an Excel file format containing different sheets.
- Cleaning the dataset
- Joining different dataframes
- Applying K-means Clustering as an unspervised leaarning method to deal with missing values
- Feature Selection
- Using ordered logistic regression to compute odds ratio as a sanity check for selected features

Since you can become bored working with this dataset, I chunk it to smaller parts. So, let's dive into the first part :)

## Importing Libraries

First, we should load packages and libraries to our work space. `xlrd` is a package for importing excel files. Most of the remainings are from sklearn with which you should highly likely be familiar.


```python
import xlrd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
import restcountries_py.restcountry as rc
from sklearn.svm import SVC
```

## Importing the Dataset

OK, let's import the dataset which is in the `xlsx` file format. We call it `sowc` which stands for *The State of World's Children*.


```python
sowc = xlrd.open_workbook('UNICEF.xlsx')
```

As previously mentioned, it contains different sheets. To see their names, we can use tho following snippet:


```python
[sheet.name for sheet in sowc.sheets()]
```




    ['Basic Indicators',
     'Nutrition',
     'Health',
     'HIV-AIDS',
     'Education',
     'Demographic Indicators',
     'Women',
     'Child Protection',
     'Adolescents',
     'Disparities by Residence',
     'Disparities by Household Wealth',
     'Early Childhood Development',
     'Economic Indicators']



Now, let's read values from these excel sheets and convert them into different dataframes. At the end, we must join all these dataframes into a single, consolidated dataframe. To do so, we have to iterate over rows to read values at each row and keep the values in a list. Then, we can create a dictionary by assigning each item in the list to a specific column. When should we stop the iteration? The trick here is that in all 13 sheets `Zimbabwe` is the last country in the list! If it is not clear enough, see the following codes.

## Sheet 1


```python
sheet1 = sowc.sheet_by_name('Basic Indicators')
```


```python
BasicInd = {}
for i in range(6, sheet1.nrows):

    val = sheet1.row_values(i)

    country = val[1]

    BasicInd[country] = {
        'Under_5_mortality_rank':val[2],
        'Under_5_mortality_rate_1990': val[3],
        'Under_5_mortality_rate_2016': val[4],
        'Under_5_mortality_rate_male': val[5],
        'Under_5_mortality_rate_female': val[6],
        'Under_1_mortality_rate_1990': val[7],
        'Under_1_mortality_rate_2016': val[8],
        'Neonatal_mortality_rate_2016': val[9],
        'Total_population_(thousands)_2016': val[10],
        'Annual_number_of_births_(thousands)_2016': val[11],
        'Annual_number_of_under_5_deaths_(thousands)_2016': val[12],
        'Life_expectancy_at_birth_(years)_2016': val[13],
        'Total_adult_literacy_rate%_2011_2016': val[14],
        'Primary_school_net_enrolment_ratio_2011_2016': val[16],
    }

    if country == 'Zimbabwe':
        break
```

At this point, we create a dataframe from the dictionary created in the above section.


```python
BasicInd = pd.DataFrame(BasicInd).T
```


```python
BasicInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = BasicInd.columns.drop('index')
BasicInd[NumColumns] = BasicInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
BasicInd['Under_5_mortality_diff_1990_2016'] = BasicInd['Under_5_mortality_rate_2016'] -BasicInd['Under_5_mortality_rate_1990']
BasicInd['Under_1_mortality_diff_1990_2016'] = BasicInd['Under_1_mortality_rate_2016'] -BasicInd['Under_1_mortality_rate_1990']
```


```python
BasicInd.drop(columns=['Under_1_mortality_rate_1990', 'Under_5_mortality_rate_1990', 'Under_1_mortality_rate_2016',
                      'Under_5_mortality_rate_2016', 'Under_5_mortality_rate_male',
                       'Under_5_mortality_rate_female'], axis=1, inplace=True)
```


```python
BasicInd.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Under_5_mortality_rank</th>
      <th>Neonatal_mortality_rate_2016</th>
      <th>Total_population_(thousands)_2016</th>
      <th>Annual_number_of_births_(thousands)_2016</th>
      <th>Annual_number_of_under_5_deaths_(thousands)_2016</th>
      <th>Life_expectancy_at_birth_(years)_2016</th>
      <th>Total_adult_literacy_rate%_2011_2016</th>
      <th>Primary_school_net_enrolment_ratio_2011_2016</th>
      <th>Under_5_mortality_diff_1990_2016</th>
      <th>Under_1_mortality_diff_1990_2016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>195.000000</td>
      <td>195.000000</td>
      <td>2.020000e+02</td>
      <td>184.00000</td>
      <td>195.000000</td>
      <td>184.000000</td>
      <td>145.000000</td>
      <td>181.000000</td>
      <td>195.000000</td>
      <td>195.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>95.871795</td>
      <td>13.328205</td>
      <td>3.676863e+04</td>
      <td>763.37919</td>
      <td>28.907692</td>
      <td>71.614109</td>
      <td>80.243046</td>
      <td>89.125296</td>
      <td>-41.702564</td>
      <td>-26.323077</td>
    </tr>
    <tr>
      <td>std</td>
      <td>54.662780</td>
      <td>10.870606</td>
      <td>1.402573e+05</td>
      <td>2392.39872</td>
      <td>102.653565</td>
      <td>7.784424</td>
      <td>21.659742</td>
      <td>12.389407</td>
      <td>44.653578</td>
      <td>24.380642</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>8.010000e-01</td>
      <td>1.56100</td>
      <td>0.000000</td>
      <td>51.835000</td>
      <td>15.456700</td>
      <td>30.938330</td>
      <td>-238.000000</td>
      <td>-121.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>49.000000</td>
      <td>4.000000</td>
      <td>1.320106e+03</td>
      <td>47.21075</td>
      <td>0.000000</td>
      <td>66.440000</td>
      <td>69.425390</td>
      <td>86.854780</td>
      <td>-62.000000</td>
      <td>-38.500000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>98.000000</td>
      <td>10.000000</td>
      <td>7.501282e+03</td>
      <td>164.27200</td>
      <td>2.000000</td>
      <td>73.335000</td>
      <td>91.181360</td>
      <td>93.313070</td>
      <td>-24.000000</td>
      <td>-19.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>142.000000</td>
      <td>21.500000</td>
      <td>2.525010e+04</td>
      <td>632.96350</td>
      <td>18.000000</td>
      <td>77.050750</td>
      <td>97.128750</td>
      <td>96.419100</td>
      <td>-9.000000</td>
      <td>-8.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>192.000000</td>
      <td>46.000000</td>
      <td>1.403500e+06</td>
      <td>25243.76900</td>
      <td>1081.000000</td>
      <td>83.764000</td>
      <td>100.000000</td>
      <td>99.950010</td>
      <td>17.000000</td>
      <td>17.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
BasicInd.isnull().sum()
```




    index                                                0
    Under_5_mortality_rank                               7
    Neonatal_mortality_rate_2016                         7
    Total_population_(thousands)_2016                    0
    Annual_number_of_births_(thousands)_2016            18
    Annual_number_of_under_5_deaths_(thousands)_2016     7
    Life_expectancy_at_birth_(years)_2016               18
    Total_adult_literacy_rate%_2011_2016                57
    Primary_school_net_enrolment_ratio_2011_2016        21
    Under_5_mortality_diff_1990_2016                     7
    Under_1_mortality_diff_1990_2016                     7
    dtype: int64




```python
BasicInd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Under_5_mortality_rank</th>
      <th>Neonatal_mortality_rate_2016</th>
      <th>Total_population_(thousands)_2016</th>
      <th>Annual_number_of_births_(thousands)_2016</th>
      <th>Annual_number_of_under_5_deaths_(thousands)_2016</th>
      <th>Life_expectancy_at_birth_(years)_2016</th>
      <th>Total_adult_literacy_rate%_2011_2016</th>
      <th>Primary_school_net_enrolment_ratio_2011_2016</th>
      <th>Under_5_mortality_diff_1990_2016</th>
      <th>Under_1_mortality_diff_1990_2016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Afghanistan</td>
      <td>25.0</td>
      <td>40.0</td>
      <td>34656.032</td>
      <td>1142.962</td>
      <td>80.0</td>
      <td>63.673</td>
      <td>31.74112</td>
      <td>NaN</td>
      <td>-107.0</td>
      <td>-67.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>114.0</td>
      <td>6.0</td>
      <td>2926.348</td>
      <td>34.750</td>
      <td>0.0</td>
      <td>78.345</td>
      <td>97.24697</td>
      <td>95.51731</td>
      <td>-26.0</td>
      <td>-23.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Algeria</td>
      <td>78.0</td>
      <td>16.0</td>
      <td>40606.052</td>
      <td>949.277</td>
      <td>24.0</td>
      <td>76.078</td>
      <td>75.13605</td>
      <td>97.06215</td>
      <td>-24.0</td>
      <td>-19.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Andorra</td>
      <td>179.0</td>
      <td>1.0</td>
      <td>77.281</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>100.00000</td>
      <td>NaN</td>
      <td>-6.0</td>
      <td>-5.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Angola</td>
      <td>17.0</td>
      <td>29.0</td>
      <td>28813.463</td>
      <td>1180.970</td>
      <td>96.0</td>
      <td>61.547</td>
      <td>66.03011</td>
      <td>84.01231</td>
      <td>-138.0</td>
      <td>-76.0</td>
    </tr>
  </tbody>
</table>
</div>
