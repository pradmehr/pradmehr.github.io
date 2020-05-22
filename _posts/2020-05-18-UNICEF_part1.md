---
layout: "posts"
title: "Key Indicators of World’s Children"
date: 2020-05-18
permalink: /Projects/
tags: [UNICEF, SOWC, Data Science, xlrd]
header:
  image: /images/Project/unicef1.jpg
excerpt: "UNICEF, SOWC, Data Science, xlrd"
mathjax: "true"
---

# Part 1: Data Preparation

***

We are going to go through a dataset from [UNICEF](https://www.unicef.org/). It includes different indicators for different countries around the world. This dataset is a collection of different indicators which contains 13 sheets. Each sheet has a different type of indicators such as *Education, Health, Nutrition, etc*.

**Objectives:**
After reading this post you should take away the following objectives:
- Importing an Excel file format containing different sheets.
- Cleaning the dataset
- Joining different data frames
- Applying K-means Clustering as an unsupervised learning method to deal with missing values
- Feature Selection
- Using ordered logistic regression to compute odds ratio as a sanity check for selected features

Since you can become bored working with this dataset, I chunk it to smaller parts. So, let's dive into the first part :)

## Importing Libraries

First, we should load packages and libraries to our workspace. `xlrd` is a package for importing excel files. Most of the remaining are from sklearn with which you should highly likely be familiar.


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

As previously mentioned, it contains different sheets. To see their names, we can use the following snippet.
*note that:*
- `sheets()` method return all sheets' information. To obtain their name we iterate over them and use `name` attribute.
- I use list comprehension, you can use for loop instead


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



ALso, `sheet_names()` method returns the same results.


```python
sowc.sheet_names()
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



Now, let's read values from these excel sheets and convert them into different data frames. At the end, we must join all these data frames into a single, consolidated data frame. To do so, we have to iterate over rows to read values at each row and keep the values in a list. Then, we can create a dictionary by assigning each item in the list to a specific key (which is supposed to be the future column name!). When should we stop the iteration? The trick here is that in all 13 sheets `Zimbabwe` is the last country in the list! If it is not clear enough, see the following codes.

## Sheet 1

The first sheet is about basic indicators. Let's load its value into a data frame.


```python
sheet1 = sowc.sheet_by_name('Basic Indicators') # load the sheet
```


```python
BasicInd = {}
for i in range(6, sheet1.nrows): # Why 6? becuase the first country 'Afghanistan' starts at row six

    val = sheet1.row_values(i) # Reading all values in i_th row

    country = val[1] # First value is the name of i_th country

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

At this point, we create a data frame from the dictionary that created in the above section. To have countries on the rows, we have to transpose the data frame.


```python
BasicInd = pd.DataFrame(BasicInd).T
```


```python
BasicInd.reset_index(level=0, inplace=True) # To reset index to row numbers
```

The values imported into the data frame are in string format. All, except for Country name, should change to numeric type.


```python
NumColumns = BasicInd.columns.drop('index')
BasicInd[NumColumns] = BasicInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```

We are working with highly correlated dataset. So it is a good habit to replace correlated columns with their meaningful combination.


```python
BasicInd['Under_5_mortality_diff_1990_2016'] = BasicInd['Under_5_mortality_rate_2016'] -BasicInd['Under_5_mortality_rate_1990']
BasicInd['Under_1_mortality_diff_1990_2016'] = BasicInd['Under_1_mortality_rate_2016'] -BasicInd['Under_1_mortality_rate_1990']
```


```python
BasicInd.drop(columns=['Under_1_mortality_rate_1990', 'Under_5_mortality_rate_1990', 'Under_1_mortality_rate_2016',
                      'Under_5_mortality_rate_2016', 'Under_5_mortality_rate_male',
                       'Under_5_mortality_rate_female'], axis=1, inplace=True)
```

By looking at descriptive statistic, we can notice that they follow non normal distribution. We should keep it in our mind.


```python
BasicInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Under_5_mortality_rank</td>
      <td>195.0</td>
      <td>95.871795</td>
      <td>54.662780</td>
      <td>1.00000</td>
      <td>49.00000</td>
      <td>98.00000</td>
      <td>142.00000</td>
      <td>1.920000e+02</td>
    </tr>
    <tr>
      <td>Neonatal_mortality_rate_2016</td>
      <td>195.0</td>
      <td>13.328205</td>
      <td>10.870606</td>
      <td>1.00000</td>
      <td>4.00000</td>
      <td>10.00000</td>
      <td>21.50000</td>
      <td>4.600000e+01</td>
    </tr>
    <tr>
      <td>Total_population_(thousands)_2016</td>
      <td>202.0</td>
      <td>36768.627926</td>
      <td>140257.283611</td>
      <td>0.80100</td>
      <td>1320.10600</td>
      <td>7501.28200</td>
      <td>25250.10275</td>
      <td>1.403500e+06</td>
    </tr>
    <tr>
      <td>Annual_number_of_births_(thousands)_2016</td>
      <td>184.0</td>
      <td>763.379190</td>
      <td>2392.398720</td>
      <td>1.56100</td>
      <td>47.21075</td>
      <td>164.27200</td>
      <td>632.96350</td>
      <td>2.524377e+04</td>
    </tr>
    <tr>
      <td>Annual_number_of_under_5_deaths_(thousands)_2016</td>
      <td>195.0</td>
      <td>28.907692</td>
      <td>102.653565</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>2.00000</td>
      <td>18.00000</td>
      <td>1.081000e+03</td>
    </tr>
    <tr>
      <td>Life_expectancy_at_birth_(years)_2016</td>
      <td>184.0</td>
      <td>71.614109</td>
      <td>7.784424</td>
      <td>51.83500</td>
      <td>66.44000</td>
      <td>73.33500</td>
      <td>77.05075</td>
      <td>8.376400e+01</td>
    </tr>
    <tr>
      <td>Total_adult_literacy_rate%_2011_2016</td>
      <td>145.0</td>
      <td>80.243046</td>
      <td>21.659742</td>
      <td>15.45670</td>
      <td>69.42539</td>
      <td>91.18136</td>
      <td>97.12875</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <td>Primary_school_net_enrolment_ratio_2011_2016</td>
      <td>181.0</td>
      <td>89.125296</td>
      <td>12.389407</td>
      <td>30.93833</td>
      <td>86.85478</td>
      <td>93.31307</td>
      <td>96.41910</td>
      <td>9.995001e+01</td>
    </tr>
    <tr>
      <td>Under_5_mortality_diff_1990_2016</td>
      <td>195.0</td>
      <td>-41.702564</td>
      <td>44.653578</td>
      <td>-238.00000</td>
      <td>-62.00000</td>
      <td>-24.00000</td>
      <td>-9.00000</td>
      <td>1.700000e+01</td>
    </tr>
    <tr>
      <td>Under_1_mortality_diff_1990_2016</td>
      <td>195.0</td>
      <td>-26.323077</td>
      <td>24.380642</td>
      <td>-121.00000</td>
      <td>-38.50000</td>
      <td>-19.00000</td>
      <td>-8.00000</td>
      <td>1.700000e+01</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's see how many missing values are in this data frame. We will deal with missing values after joining data frames.


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



---
We have to do the same thing to all the remaining sheets. Do not panic! The idea is the same.

## Sheet 2


```python
sheet2 = sowc.sheet_by_name('Nutrition')
```


```python
NutInd = {}
for i in range(8, sheet2.nrows):

    val = sheet2.row_values(i)

    country = val[1]

    NutInd[country] = {
        'Low_birthweight_2011_2016':val[2],
        'Early_initiation_of_breastfeeding_2011_2016': val[4],
        'Under_6M_Exclusive_breastfeeding_2011_2016': val[6],
        'Introduction_to_solid_semi_solid_or_soft_foods_6_8_months': val[8],
        'Minimum_acceptable_diet_6_23M_2011_2016': val[10],
        'Breastfeeding_at_age_2_2011_2016': val[12],
        'Stunting_MS_2011_2016': val[14],
        'Overweight_MS_2011_2016': val[16],
        'Wasting_M_2011_2016': val[18],
        'Wasting_MS_2011_2016': val[20],
        'Vitamin_A_supplementation_2015': val[22],
        'Households_consuming_salt_with-iodine_2011_2016': val[24]
    }

    if country == 'Zimbabwe':
        break
```


```python
NutInd = pd.DataFrame(NutInd).T
```


```python
NutInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = NutInd.columns.drop('index')
NutInd[NumColumns] = NutInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
NutInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Low_birthweight_2011_2016</td>
      <td>182.0</td>
      <td>10.468681</td>
      <td>5.393380</td>
      <td>4.200000</td>
      <td>6.900000</td>
      <td>9.55</td>
      <td>11.900000</td>
      <td>34.7</td>
    </tr>
    <tr>
      <td>Early_initiation_of_breastfeeding_2011_2016</td>
      <td>130.0</td>
      <td>51.892308</td>
      <td>18.280042</td>
      <td>12.000000</td>
      <td>40.825000</td>
      <td>51.10</td>
      <td>65.775000</td>
      <td>93.4</td>
    </tr>
    <tr>
      <td>Under_6M_Exclusive_breastfeeding_2011_2016</td>
      <td>132.0</td>
      <td>39.151515</td>
      <td>19.302958</td>
      <td>0.300000</td>
      <td>24.175000</td>
      <td>38.60</td>
      <td>53.125000</td>
      <td>87.3</td>
    </tr>
    <tr>
      <td>Introduction_to_solid_semi_solid_or_soft_foods_6_8_months</td>
      <td>108.0</td>
      <td>71.236994</td>
      <td>18.424580</td>
      <td>15.600000</td>
      <td>60.600000</td>
      <td>75.35</td>
      <td>85.446339</td>
      <td>96.6</td>
    </tr>
    <tr>
      <td>Minimum_acceptable_diet_6_23M_2011_2016</td>
      <td>72.0</td>
      <td>24.064834</td>
      <td>19.706181</td>
      <td>3.000000</td>
      <td>9.675000</td>
      <td>15.45</td>
      <td>35.775000</td>
      <td>77.1</td>
    </tr>
    <tr>
      <td>Breastfeeding_at_age_2_2011_2016</td>
      <td>127.0</td>
      <td>38.966583</td>
      <td>21.735330</td>
      <td>3.900000</td>
      <td>21.300000</td>
      <td>38.00</td>
      <td>53.100000</td>
      <td>88.5</td>
    </tr>
    <tr>
      <td>Stunting_MS_2011_2016</td>
      <td>144.0</td>
      <td>21.577083</td>
      <td>13.439794</td>
      <td>1.300000</td>
      <td>9.375000</td>
      <td>20.65</td>
      <td>31.800000</td>
      <td>55.9</td>
    </tr>
    <tr>
      <td>Overweight_MS_2011_2016</td>
      <td>142.0</td>
      <td>7.104930</td>
      <td>4.984890</td>
      <td>0.000000</td>
      <td>3.525000</td>
      <td>6.20</td>
      <td>9.225000</td>
      <td>26.5</td>
    </tr>
    <tr>
      <td>Wasting_M_2011_2016</td>
      <td>144.0</td>
      <td>5.863889</td>
      <td>4.621062</td>
      <td>0.000000</td>
      <td>2.400000</td>
      <td>4.50</td>
      <td>7.675000</td>
      <td>22.7</td>
    </tr>
    <tr>
      <td>Wasting_MS_2011_2016</td>
      <td>139.0</td>
      <td>1.930216</td>
      <td>1.834597</td>
      <td>0.000000</td>
      <td>0.650000</td>
      <td>1.30</td>
      <td>2.700000</td>
      <td>9.9</td>
    </tr>
    <tr>
      <td>Vitamin_A_supplementation_2015</td>
      <td>60.0</td>
      <td>66.283333</td>
      <td>31.405004</td>
      <td>3.000000</td>
      <td>41.000000</td>
      <td>74.00</td>
      <td>96.000000</td>
      <td>99.0</td>
    </tr>
    <tr>
      <td>Households_consuming_salt_with-iodine_2011_2016</td>
      <td>92.0</td>
      <td>73.698466</td>
      <td>22.223809</td>
      <td>4.433991</td>
      <td>60.500851</td>
      <td>81.40</td>
      <td>90.835657</td>
      <td>99.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
NutInd.isnull().sum()
```




    index                                                          0
    Low_birthweight_2011_2016                                     20
    Early_initiation_of_breastfeeding_2011_2016                   72
    Under_6M_Exclusive_breastfeeding_2011_2016                    70
    Introduction_to_solid_semi_solid_or_soft_foods_6_8_months     94
    Minimum_acceptable_diet_6_23M_2011_2016                      130
    Breastfeeding_at_age_2_2011_2016                              75
    Stunting_MS_2011_2016                                         58
    Overweight_MS_2011_2016                                       60
    Wasting_M_2011_2016                                           58
    Wasting_MS_2011_2016                                          63
    Vitamin_A_supplementation_2015                               142
    Households_consuming_salt_with-iodine_2011_2016              110
    dtype: int64



---

## Sheet 3


```python
sheet3 = sowc.sheet_by_name('Health')
```


```python
HInd = {}
for i in range(8, sheet3.nrows):

    val = sheet3.row_values(i)

    country = val[1]

    HInd[country] = {
        'drinking_water_urban_2015':val[3],
        'drinking_water_rural_2015': val[4],
        'sanitation_services_urban_2015': val[6],
        'sanitation_services_rural_2015': val[7],
        'BCG_2016': val[8],
        'DTP1_2016': val[9],
        'DTP3_2016': val[10],
        'polio3_2016': val[11],
        'MCV1_2016': val[12],
        'MCV2_2016': val[13],
        'HepB3_2016': val[14],
        'Hib3_2016': val[15],
        'rota_2016': val[16],
        'PCV3_2016': val[17],
        'PAB_against_tetanus_2016': val[18],
        'Pneumonia_2011_2016': val[20],
        'Diarrhoea_2011_2016': val[22],
        'Malaria_fever_2011_2016': val[24],
        'Malaria_sleeping_under_ITN_2011_2016': val[26],
        'Malaria_Households_with_at_least_one_ITN_2011_2016': val[28],
    }

    if country == 'Zimbabwe':
        break
```


```python
HInd = pd.DataFrame(HInd).T
```


```python
HInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = HInd.columns.drop('index')
HInd[NumColumns] = HInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
HInd['sanitation_services_rural_to_urban_ratio_2015'] = HInd['sanitation_services_rural_2015'] / HInd['sanitation_services_urban_2015']
HInd['drinking_water_rural_to_urban_ratio_2015'] = HInd['drinking_water_rural_2015'] / HInd['drinking_water_urban_2015']
```


```python
HInd.drop(columns=['sanitation_services_rural_2015', 'sanitation_services_urban_2015',
                       'drinking_water_rural_2015', 'drinking_water_urban_2015'], axis=1, inplace=True)
```


```python
HInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BCG_2016</td>
      <td>160.0</td>
      <td>89.318750</td>
      <td>16.326223</td>
      <td>0.000000</td>
      <td>86.750000</td>
      <td>96.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>DTP1_2016</td>
      <td>195.0</td>
      <td>93.153846</td>
      <td>10.647439</td>
      <td>35.000000</td>
      <td>93.500000</td>
      <td>97.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>DTP3_2016</td>
      <td>195.0</td>
      <td>88.297436</td>
      <td>14.666525</td>
      <td>19.000000</td>
      <td>85.000000</td>
      <td>94.000000</td>
      <td>97.500000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>polio3_2016</td>
      <td>195.0</td>
      <td>87.958974</td>
      <td>14.197007</td>
      <td>20.000000</td>
      <td>83.000000</td>
      <td>94.000000</td>
      <td>97.500000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>MCV1_2016</td>
      <td>195.0</td>
      <td>87.348718</td>
      <td>14.366392</td>
      <td>20.000000</td>
      <td>82.500000</td>
      <td>93.000000</td>
      <td>97.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>MCV2_2016</td>
      <td>195.0</td>
      <td>68.102564</td>
      <td>35.658023</td>
      <td>0.000000</td>
      <td>50.500000</td>
      <td>86.000000</td>
      <td>95.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>HepB3_2016</td>
      <td>195.0</td>
      <td>83.184615</td>
      <td>23.441235</td>
      <td>0.000000</td>
      <td>80.000000</td>
      <td>93.000000</td>
      <td>97.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>Hib3_2016</td>
      <td>195.0</td>
      <td>86.148718</td>
      <td>18.810050</td>
      <td>0.000000</td>
      <td>83.000000</td>
      <td>93.000000</td>
      <td>97.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>rota_2016</td>
      <td>195.0</td>
      <td>34.194872</td>
      <td>41.569256</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>81.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>PCV3_2016</td>
      <td>195.0</td>
      <td>54.882051</td>
      <td>42.349448</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>78.000000</td>
      <td>93.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <td>PAB_against_tetanus_2016</td>
      <td>106.0</td>
      <td>86.792453</td>
      <td>7.981785</td>
      <td>60.000000</td>
      <td>83.000000</td>
      <td>88.500000</td>
      <td>92.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Pneumonia_2011_2016</td>
      <td>120.0</td>
      <td>62.331667</td>
      <td>19.033420</td>
      <td>13.000000</td>
      <td>50.575000</td>
      <td>65.050000</td>
      <td>76.050000</td>
      <td>94.400000</td>
    </tr>
    <tr>
      <td>Diarrhoea_2011_2016</td>
      <td>119.0</td>
      <td>44.021849</td>
      <td>17.140912</td>
      <td>11.100000</td>
      <td>32.700000</td>
      <td>42.600000</td>
      <td>56.100000</td>
      <td>93.600000</td>
    </tr>
    <tr>
      <td>Malaria_fever_2011_2016</td>
      <td>79.0</td>
      <td>59.816456</td>
      <td>13.829914</td>
      <td>22.800000</td>
      <td>50.650000</td>
      <td>61.400000</td>
      <td>69.900000</td>
      <td>92.900000</td>
    </tr>
    <tr>
      <td>Malaria_sleeping_under_ITN_2011_2016</td>
      <td>60.0</td>
      <td>36.193333</td>
      <td>22.805507</td>
      <td>0.200000</td>
      <td>18.550000</td>
      <td>40.250000</td>
      <td>51.850000</td>
      <td>80.600000</td>
    </tr>
    <tr>
      <td>Malaria_Households_with_at_least_one_ITN_2011_2016</td>
      <td>60.0</td>
      <td>50.915000</td>
      <td>26.848210</td>
      <td>1.000000</td>
      <td>29.875000</td>
      <td>59.800000</td>
      <td>69.175000</td>
      <td>93.000000</td>
    </tr>
    <tr>
      <td>sanitation_services_rural_to_urban_ratio_2015</td>
      <td>169.0</td>
      <td>0.773064</td>
      <td>0.273374</td>
      <td>0.088749</td>
      <td>0.613679</td>
      <td>0.882965</td>
      <td>0.999498</td>
      <td>1.124473</td>
    </tr>
    <tr>
      <td>drinking_water_rural_to_urban_ratio_2015</td>
      <td>170.0</td>
      <td>0.829351</td>
      <td>0.210082</td>
      <td>0.083655</td>
      <td>0.670635</td>
      <td>0.902003</td>
      <td>0.999125</td>
      <td>1.407700</td>
    </tr>
  </tbody>
</table>
</div>




```python
HInd.isnull().sum()
```




    index                                                   0
    BCG_2016                                               42
    DTP1_2016                                               7
    DTP3_2016                                               7
    polio3_2016                                             7
    MCV1_2016                                               7
    MCV2_2016                                               7
    HepB3_2016                                              7
    Hib3_2016                                               7
    rota_2016                                               7
    PCV3_2016                                               7
    PAB_against_tetanus_2016                               96
    Pneumonia_2011_2016                                    82
    Diarrhoea_2011_2016                                    83
    Malaria_fever_2011_2016                               123
    Malaria_sleeping_under_ITN_2011_2016                  142
    Malaria_Households_with_at_least_one_ITN_2011_2016    142
    sanitation_services_rural_to_urban_ratio_2015          33
    drinking_water_rural_to_urban_ratio_2015               32
    dtype: int64



---

## Sheet 4


```python
sheet4 = sowc.sheet_by_name('HIV-AIDS')
```


```python
HivInd = {}
for i in range(7, sheet4.nrows):

    val = sheet4.row_values(i)

    country = val[1]

    HivInd[country] = {
        'HIV_incidence_per_thousand_uninfected_population_all_ages_2016':val[2],
        'HIV_incidence_per_thousand_uninfected_population_under_5_2016': val[3],
        'HIV_incidence_per_thousand_uninfected_population_adolescents_2016': val[4],
        'People_living_with_HIV_all_ages_2016':val[5],
        'People_living_with_HIV_under_5_2016': val[6],
        'People_living_with_HIV_adolescents_2016': val[7],
        'New_HIV_infections_all_ages_2016':val[8],
        'New_HIV_infections_under_5_2016': val[9],
        'New_HIV_infections_adolescents_2016': val[10],
        'AIDS_related_deaths_all_ages_2016':val[11],
        'AIDS_related_deaths_under_5_2016': val[12],
        'AIDS_related_deaths_adolescents_2016': val[13],
        'Pregnant_women_reciving_ARVs_for_PMTCT_(%)_2016': val[14],
        'People_living_with_HIV_receiving_ART%_all_ages_2016':val[11],
        'People_living_with_HIV_receiving_ART%_under_5_2016': val[12],
        'People_living_with_HIV_receiving_ART%_adolescents_2016': val[13],
        'Condom_male_2011_2016': val[14],
        'Condom_female_2011_2016': val[16],
        'Adolescents_tested_male_2011_2016': val[18],
        'Adolescents_tested_female_2011_2016': val[20]
    }

    if country == 'Zimbabwe':
        break
```


```python
HivInd = pd.DataFrame(HivInd).T
```


```python
HivInd.reset_index(level=0, inplace=True)
```


```python
for col in HivInd.columns:
    if HivInd[col].dtype == 'object':
        HivInd[col] = np.where(HivInd[col] == '<0.01', 0.005, HivInd[col])
        HivInd[col] = np.where(HivInd[col] == '<0.1', 0.05, HivInd[col])
        HivInd[col] = np.where(HivInd[col] == '<100', 50, HivInd[col])
        HivInd[col] = np.where(HivInd[col] == '<200', 150, HivInd[col])
        HivInd[col] = np.where(HivInd[col] == '<500', 350, HivInd[col])
        HivInd[col] = np.where(HivInd[col] == '<1,000', 750, HivInd[col])
        HivInd[col] = np.where(HivInd[col] == '>95', 97.5, HivInd[col])
```


```python
NumColumns = HivInd.columns.drop('index')
HivInd[NumColumns] = HivInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
HivInd['HIV_incidence_ratio_adolescents_2016'] = HivInd['HIV_incidence_per_thousand_uninfected_population_adolescents_2016'] / HivInd['HIV_incidence_per_thousand_uninfected_population_all_ages_2016']
HivInd['HIV_incidence_ratio_under_5_2016'] = HivInd['HIV_incidence_per_thousand_uninfected_population_under_5_2016'] / HivInd['HIV_incidence_per_thousand_uninfected_population_all_ages_2016']
HivInd['People_living_with_HIV_ratio_adolescents_2016'] = HivInd['People_living_with_HIV_adolescents_2016'] / HivInd['People_living_with_HIV_all_ages_2016']
HivInd['People_living_with_HIV_ratio_under_5_2016'] = HivInd['People_living_with_HIV_under_5_2016'] / HivInd['People_living_with_HIV_all_ages_2016']
HivInd['New_HIV_infections_ratio_adolescents_2016'] = HivInd['New_HIV_infections_adolescents_2016'] / HivInd['New_HIV_infections_all_ages_2016']
HivInd['New_HIV_infections_ratio_under_5_2016'] = HivInd['New_HIV_infections_under_5_2016'] / HivInd['New_HIV_infections_all_ages_2016']
HivInd['AIDS_related_deaths_ratio_adolescents_2016'] = HivInd['AIDS_related_deaths_adolescents_2016'] / HivInd['AIDS_related_deaths_all_ages_2016']
HivInd['AIDS_related_deaths_ratio_under_5_2016'] = HivInd['AIDS_related_deaths_under_5_2016'] / HivInd['AIDS_related_deaths_all_ages_2016']
HivInd['People_living_with_HIV_receiving_ART%ratio_adolescents_2016'] = HivInd['People_living_with_HIV_receiving_ART%_adolescents_2016'] / HivInd['People_living_with_HIV_receiving_ART%_all_ages_2016']
HivInd['People_living_with_HIV_receiving_ART%ratio_under_5_2016'] = HivInd['People_living_with_HIV_receiving_ART%_under_5_2016'] / HivInd['People_living_with_HIV_receiving_ART%_all_ages_2016']
```


```python
HivInd.drop(columns=['HIV_incidence_per_thousand_uninfected_population_adolescents_2016',
                     'HIV_incidence_per_thousand_uninfected_population_under_5_2016',
                     'HIV_incidence_per_thousand_uninfected_population_all_ages_2016',
                     'People_living_with_HIV_adolescents_2016',
                     'People_living_with_HIV_under_5_2016',
                     'People_living_with_HIV_all_ages_2016',
                     'New_HIV_infections_adolescents_2016',
                     'New_HIV_infections_under_5_2016',
                     'New_HIV_infections_all_ages_2016',
                     'AIDS_related_deaths_adolescents_2016',
                     'AIDS_related_deaths_under_5_2016',
                     'AIDS_related_deaths_all_ages_2016',
                     'People_living_with_HIV_receiving_ART%_adolescents_2016',
                     'People_living_with_HIV_receiving_ART%_under_5_2016',
                     'People_living_with_HIV_receiving_ART%_all_ages_2016',                     
                    ], axis=1, inplace=True)
```


```python
HivInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pregnant_women_reciving_ARVs_for_PMTCT_(%)_2016</td>
      <td>106.0</td>
      <td>66.061321</td>
      <td>28.161837</td>
      <td>3.000000</td>
      <td>46.500000</td>
      <td>72.000000</td>
      <td>89.000000</td>
      <td>97.500000</td>
    </tr>
    <tr>
      <td>Condom_male_2011_2016</td>
      <td>106.0</td>
      <td>66.061321</td>
      <td>28.161837</td>
      <td>3.000000</td>
      <td>46.500000</td>
      <td>72.000000</td>
      <td>89.000000</td>
      <td>97.500000</td>
    </tr>
    <tr>
      <td>Condom_female_2011_2016</td>
      <td>102.0</td>
      <td>51.857843</td>
      <td>30.705353</td>
      <td>2.000000</td>
      <td>24.250000</td>
      <td>48.000000</td>
      <td>82.500000</td>
      <td>97.500000</td>
    </tr>
    <tr>
      <td>Adolescents_tested_male_2011_2016</td>
      <td>48.0</td>
      <td>58.159167</td>
      <td>21.090354</td>
      <td>5.200000</td>
      <td>44.725000</td>
      <td>58.550000</td>
      <td>75.135000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Adolescents_tested_female_2011_2016</td>
      <td>47.0</td>
      <td>41.321277</td>
      <td>20.399744</td>
      <td>5.500000</td>
      <td>27.600000</td>
      <td>38.900000</td>
      <td>53.700000</td>
      <td>92.300000</td>
    </tr>
    <tr>
      <td>HIV_incidence_ratio_adolescents_2016</td>
      <td>120.0</td>
      <td>1.510439</td>
      <td>0.703130</td>
      <td>0.350644</td>
      <td>0.780946</td>
      <td>1.747801</td>
      <td>2.071228</td>
      <td>2.751112</td>
    </tr>
    <tr>
      <td>HIV_incidence_ratio_under_5_2016</td>
      <td>120.0</td>
      <td>0.536086</td>
      <td>0.401198</td>
      <td>0.039183</td>
      <td>0.184630</td>
      <td>0.452041</td>
      <td>0.804258</td>
      <td>2.156233</td>
    </tr>
    <tr>
      <td>People_living_with_HIV_ratio_adolescents_2016</td>
      <td>104.0</td>
      <td>0.049304</td>
      <td>0.028400</td>
      <td>0.002000</td>
      <td>0.027295</td>
      <td>0.046349</td>
      <td>0.071429</td>
      <td>0.126316</td>
    </tr>
    <tr>
      <td>People_living_with_HIV_ratio_under_5_2016</td>
      <td>104.0</td>
      <td>0.048289</td>
      <td>0.038783</td>
      <td>0.000357</td>
      <td>0.013409</td>
      <td>0.038315</td>
      <td>0.078393</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <td>New_HIV_infections_ratio_adolescents_2016</td>
      <td>121.0</td>
      <td>0.214936</td>
      <td>0.247288</td>
      <td>0.017241</td>
      <td>0.093750</td>
      <td>0.142857</td>
      <td>0.200000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>New_HIV_infections_ratio_under_5_2016</td>
      <td>105.0</td>
      <td>0.100526</td>
      <td>0.079726</td>
      <td>0.007812</td>
      <td>0.038462</td>
      <td>0.068182</td>
      <td>0.142857</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <td>AIDS_related_deaths_ratio_adolescents_2016</td>
      <td>101.0</td>
      <td>0.119620</td>
      <td>0.175490</td>
      <td>0.003125</td>
      <td>0.038462</td>
      <td>0.066667</td>
      <td>0.142857</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>AIDS_related_deaths_ratio_under_5_2016</td>
      <td>101.0</td>
      <td>0.161024</td>
      <td>0.171056</td>
      <td>0.003125</td>
      <td>0.066667</td>
      <td>0.142857</td>
      <td>0.190909</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>People_living_with_HIV_receiving_ART%ratio_adolescents_2016</td>
      <td>101.0</td>
      <td>0.119620</td>
      <td>0.175490</td>
      <td>0.003125</td>
      <td>0.038462</td>
      <td>0.066667</td>
      <td>0.142857</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>People_living_with_HIV_receiving_ART%ratio_under_5_2016</td>
      <td>101.0</td>
      <td>0.161024</td>
      <td>0.171056</td>
      <td>0.003125</td>
      <td>0.066667</td>
      <td>0.142857</td>
      <td>0.190909</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
HivInd.isnull().sum()
```




    index                                                            0
    Pregnant_women_reciving_ARVs_for_PMTCT_(%)_2016                 96
    Condom_male_2011_2016                                           96
    Condom_female_2011_2016                                        100
    Adolescents_tested_male_2011_2016                              154
    Adolescents_tested_female_2011_2016                            155
    HIV_incidence_ratio_adolescents_2016                            82
    HIV_incidence_ratio_under_5_2016                                82
    People_living_with_HIV_ratio_adolescents_2016                   98
    People_living_with_HIV_ratio_under_5_2016                       98
    New_HIV_infections_ratio_adolescents_2016                       81
    New_HIV_infections_ratio_under_5_2016                           97
    AIDS_related_deaths_ratio_adolescents_2016                     101
    AIDS_related_deaths_ratio_under_5_2016                         101
    People_living_with_HIV_receiving_ART%ratio_adolescents_2016    101
    People_living_with_HIV_receiving_ART%ratio_under_5_2016        101
    dtype: int64



---

## Sheet 5


```python
sheet5 = sowc.sheet_by_name('Education')
```


```python
EduInd = {}
for i in range(8, sheet5.nrows):

    val = sheet5.row_values(i)

    country = val[1]

    EduInd[country] = {
        'Youth_literacy_rate%_male_2011_2016':val[2],
        'Youth_literacy_rate%_female_2011_2016':val[4],        
        'Mobile_phones_2016': val[6],
        'Internet_phones_2016': val[8],
        'PrePrimary_Gross_enrolment_ratio%_male_2011_2016':val[10],
        'PrePrimary_Gross_enrolment_ratio%_female_2011_2016':val[12],
        'Primary_Gross_enrolment_ratio%_male_2011_2016':val[14],
        'Primary_Gross_enrolment_ratio%_female_2011_2016':val[16],
        'Primary_Net_enrolment_ratio%_male_2011_2016':val[18],
        'Primary_Net_enrolment_ratio%_female_2011_2016':val[20],
        'Primary_Net_attendance_ratio%_male_2011_2016':val[22],
        'Primary_Net_attendance_ratio%_female_2011_2016':val[24],        
        'Primary_Out_of_school_rate_of_children_ratio%_male_2011_2016':val[26],
        'Primary_Out_of_school_rate_of_children%_female_2011_2016':val[28],
        'Primary_Survival_rate_to_last_primary_grade%_male_2011_2016':val[30],
        'Primary_Survival_rate_to_last_primary_grade%_female_2011_2016':val[32],        
        'Primary_Net_enrolment_ratio%_male_2011_2016':val[34],
        'Primary_Net_enrolment_ratio%_female_2011_2016':val[36],
        'Low_Secondary_Net_attendance_ratio%_male_2010_2016':val[38],
        'Low_Secondary_Net_attendance_ratio%_female_2010_2016':val[40],        
    }

    if country == 'Zimbabwe':
        break
```


```python
EduInd = pd.DataFrame(EduInd).T
```


```python
EduInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = EduInd.columns.drop('index')
EduInd[NumColumns] = EduInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
EduInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Youth_literacy_rate%_male_2011_2016</td>
      <td>141.0</td>
      <td>89.582256</td>
      <td>14.547541</td>
      <td>34.53362</td>
      <td>85.397480</td>
      <td>97.607900</td>
      <td>99.161730</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Youth_literacy_rate%_female_2011_2016</td>
      <td>142.0</td>
      <td>86.208949</td>
      <td>20.539679</td>
      <td>15.05777</td>
      <td>80.568672</td>
      <td>97.664615</td>
      <td>99.385698</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Mobile_phones_2016</td>
      <td>201.0</td>
      <td>105.562167</td>
      <td>39.637916</td>
      <td>0.00000</td>
      <td>79.648302</td>
      <td>109.189536</td>
      <td>128.037926</td>
      <td>222.985280</td>
    </tr>
    <tr>
      <td>Internet_phones_2016</td>
      <td>200.0</td>
      <td>48.898387</td>
      <td>28.365844</td>
      <td>0.00000</td>
      <td>24.577365</td>
      <td>50.440656</td>
      <td>73.907339</td>
      <td>98.240016</td>
    </tr>
    <tr>
      <td>PrePrimary_Gross_enrolment_ratio%_male_2011_2016</td>
      <td>175.0</td>
      <td>63.722165</td>
      <td>34.906250</td>
      <td>0.84866</td>
      <td>33.829810</td>
      <td>71.332130</td>
      <td>91.511390</td>
      <td>175.000000</td>
    </tr>
    <tr>
      <td>PrePrimary_Gross_enrolment_ratio%_female_2011_2016</td>
      <td>175.0</td>
      <td>63.958206</td>
      <td>34.612292</td>
      <td>0.76166</td>
      <td>33.378840</td>
      <td>72.879130</td>
      <td>92.758630</td>
      <td>160.000000</td>
    </tr>
    <tr>
      <td>Primary_Gross_enrolment_ratio%_male_2011_2016</td>
      <td>182.0</td>
      <td>105.335664</td>
      <td>13.591209</td>
      <td>53.18238</td>
      <td>99.713450</td>
      <td>103.639805</td>
      <td>109.703205</td>
      <td>156.896130</td>
    </tr>
    <tr>
      <td>Primary_Gross_enrolment_ratio%_female_2011_2016</td>
      <td>180.0</td>
      <td>102.639494</td>
      <td>13.804025</td>
      <td>45.93698</td>
      <td>98.365493</td>
      <td>102.194785</td>
      <td>109.113748</td>
      <td>148.873080</td>
    </tr>
    <tr>
      <td>Primary_Net_enrolment_ratio%_male_2011_2016</td>
      <td>143.0</td>
      <td>65.239098</td>
      <td>25.095615</td>
      <td>1.07727</td>
      <td>46.651340</td>
      <td>70.903850</td>
      <td>86.734825</td>
      <td>99.255480</td>
    </tr>
    <tr>
      <td>Primary_Net_enrolment_ratio%_female_2011_2016</td>
      <td>143.0</td>
      <td>67.157282</td>
      <td>25.514661</td>
      <td>0.91088</td>
      <td>48.069535</td>
      <td>74.856320</td>
      <td>87.892040</td>
      <td>99.490580</td>
    </tr>
    <tr>
      <td>Primary_Net_attendance_ratio%_male_2011_2016</td>
      <td>128.0</td>
      <td>86.053080</td>
      <td>15.416122</td>
      <td>23.50000</td>
      <td>78.250000</td>
      <td>92.900000</td>
      <td>97.029092</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Primary_Net_attendance_ratio%_female_2011_2016</td>
      <td>128.0</td>
      <td>85.534929</td>
      <td>16.563485</td>
      <td>19.00000</td>
      <td>77.411694</td>
      <td>92.896668</td>
      <td>97.250000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Primary_Out_of_school_rate_of_children_ratio%_male_2011_2016</td>
      <td>156.0</td>
      <td>9.205586</td>
      <td>11.586475</td>
      <td>0.05000</td>
      <td>2.423760</td>
      <td>5.363560</td>
      <td>10.744140</td>
      <td>64.891730</td>
    </tr>
    <tr>
      <td>Primary_Out_of_school_rate_of_children%_female_2011_2016</td>
      <td>156.0</td>
      <td>9.834123</td>
      <td>12.926495</td>
      <td>0.04998</td>
      <td>1.806412</td>
      <td>4.630665</td>
      <td>12.000275</td>
      <td>73.255400</td>
    </tr>
    <tr>
      <td>Primary_Survival_rate_to_last_primary_grade%_male_2011_2016</td>
      <td>137.0</td>
      <td>83.541847</td>
      <td>17.711928</td>
      <td>21.09233</td>
      <td>74.434990</td>
      <td>91.592150</td>
      <td>97.582830</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Primary_Survival_rate_to_last_primary_grade%_female_2011_2016</td>
      <td>137.0</td>
      <td>84.614415</td>
      <td>17.229271</td>
      <td>21.62765</td>
      <td>77.373430</td>
      <td>93.429970</td>
      <td>97.620580</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Low_Secondary_Net_attendance_ratio%_male_2010_2016</td>
      <td>94.0</td>
      <td>56.014894</td>
      <td>28.609698</td>
      <td>6.20000</td>
      <td>32.200000</td>
      <td>50.550000</td>
      <td>85.625000</td>
      <td>99.400000</td>
    </tr>
    <tr>
      <td>Low_Secondary_Net_attendance_ratio%_female_2010_2016</td>
      <td>94.0</td>
      <td>58.016078</td>
      <td>29.284233</td>
      <td>3.10000</td>
      <td>31.850000</td>
      <td>55.700000</td>
      <td>86.475000</td>
      <td>99.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
EduInd.isnull().sum()
```




    index                                                              0
    Youth_literacy_rate%_male_2011_2016                               61
    Youth_literacy_rate%_female_2011_2016                             60
    Mobile_phones_2016                                                 1
    Internet_phones_2016                                               2
    PrePrimary_Gross_enrolment_ratio%_male_2011_2016                  27
    PrePrimary_Gross_enrolment_ratio%_female_2011_2016                27
    Primary_Gross_enrolment_ratio%_male_2011_2016                     20
    Primary_Gross_enrolment_ratio%_female_2011_2016                   22
    Primary_Net_enrolment_ratio%_male_2011_2016                       59
    Primary_Net_enrolment_ratio%_female_2011_2016                     59
    Primary_Net_attendance_ratio%_male_2011_2016                      74
    Primary_Net_attendance_ratio%_female_2011_2016                    74
    Primary_Out_of_school_rate_of_children_ratio%_male_2011_2016      46
    Primary_Out_of_school_rate_of_children%_female_2011_2016          46
    Primary_Survival_rate_to_last_primary_grade%_male_2011_2016       65
    Primary_Survival_rate_to_last_primary_grade%_female_2011_2016     65
    Low_Secondary_Net_attendance_ratio%_male_2010_2016               108
    Low_Secondary_Net_attendance_ratio%_female_2010_2016             108
    dtype: int64



---

## Sheet 6


```python
sheet6 = sowc.sheet_by_name('Demographic Indicators')
```


```python
DemInd = {}
for i in range(7, sheet6.nrows):

    val = sheet6.row_values(i)

    country = val[1]

    DemInd[country] = {
        'Total_Population_thousand_2016':val[2],
        'Population_thousand_under_18_2016':val[3],
        'Population_thousand_under_5_2016':val[4],
        'Population_annual_growth_rate_1990_2016':val[5],
        'Population_annual_growth_rate_2016_2030':val[6],        
        'Crude_death_rate_1970':val[7],
        'Crude_death_rate_1990':val[8],
        'Crude_death_rate_2016':val[9],
        'Crude_birth_rate_1970':val[10],
        'Crude_birth_rate_1990':val[11],
        'Crude_birth_rate_2016':val[12],
        'Life_expectancy_rate_1970':val[13],
        'Life_expectancy_rate_1990':val[14],
        'Life_expectancy_rate_2016':val[15],
        'Total_fertility_rate_2016':val[16],
        'Urbanized_population_rate_2016':val[17],
        'Urbanized_Population_annual_growth_rate_1990_2016':val[18],
        'Urbanized_Population_annual_growth_rate_2016_2030':val[19],        
    }

    if country == 'Zimbabwe':
        break
```


```python
DemInd = pd.DataFrame(DemInd).T
```


```python
DemInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = DemInd.columns.drop('index')
DemInd[NumColumns] = DemInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
DemInd['Population_ratio_under_18_2016'] = DemInd['Population_thousand_under_18_2016'] / DemInd['Total_Population_thousand_2016']
DemInd['Population_ratio_under_5_2016'] = DemInd['Population_thousand_under_5_2016'] / DemInd['Total_Population_thousand_2016']
DemInd['Crude_death_diff_1990_2016'] = DemInd['Crude_death_rate_2016'] - DemInd['Crude_death_rate_1990']
DemInd['Crude_birth_diff_1990_2016'] = DemInd['Crude_birth_rate_2016'] - DemInd['Crude_birth_rate_1990']
DemInd['Life_expectancy_diff_1990_2016'] = DemInd['Life_expectancy_rate_2016'] - DemInd['Life_expectancy_rate_1990']
```


```python
DemInd.drop(columns=['Total_Population_thousand_2016','Population_ratio_under_18_2016', 'Population_ratio_under_5_2016',
                     'Crude_death_rate_2016', 'Crude_death_rate_1990', 'Crude_death_rate_1970', 'Crude_birth_rate_2016',
                      'Crude_birth_rate_1990', 'Crude_birth_rate_1970', 'Life_expectancy_rate_2016',
                     'Life_expectancy_rate_1990', 'Life_expectancy_rate_1970'], axis=1, inplace=True)
```


```python
DemInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Population_thousand_under_18_2016</td>
      <td>202.0</td>
      <td>11362.177218</td>
      <td>39705.595276</td>
      <td>0.139221</td>
      <td>358.012500</td>
      <td>2160.159500</td>
      <td>9328.680250</td>
      <td>448314.190000</td>
    </tr>
    <tr>
      <td>Population_thousand_under_5_2016</td>
      <td>202.0</td>
      <td>3338.187539</td>
      <td>11061.059200</td>
      <td>0.035478</td>
      <td>106.840000</td>
      <td>639.542500</td>
      <td>2733.807750</td>
      <td>119997.734000</td>
    </tr>
    <tr>
      <td>Population_annual_growth_rate_1990_2016</td>
      <td>202.0</td>
      <td>1.471456</td>
      <td>1.302442</td>
      <td>-2.821764</td>
      <td>0.533569</td>
      <td>1.511982</td>
      <td>2.397635</td>
      <td>6.481664</td>
    </tr>
    <tr>
      <td>Population_annual_growth_rate_2016_2030</td>
      <td>202.0</td>
      <td>1.077905</td>
      <td>0.990575</td>
      <td>-0.859233</td>
      <td>0.308092</td>
      <td>0.957665</td>
      <td>1.846570</td>
      <td>3.759540</td>
    </tr>
    <tr>
      <td>Total_fertility_rate_2016</td>
      <td>184.0</td>
      <td>2.801223</td>
      <td>1.343866</td>
      <td>1.241000</td>
      <td>1.751750</td>
      <td>2.312000</td>
      <td>3.766500</td>
      <td>7.239000</td>
    </tr>
    <tr>
      <td>Urbanized_population_rate_2016</td>
      <td>202.0</td>
      <td>57.539609</td>
      <td>24.129924</td>
      <td>0.000000</td>
      <td>39.419750</td>
      <td>58.741500</td>
      <td>77.368500</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Urbanized_Population_annual_growth_rate_1990_2016</td>
      <td>202.0</td>
      <td>2.145942</td>
      <td>1.818613</td>
      <td>-4.027147</td>
      <td>0.825775</td>
      <td>1.946630</td>
      <td>3.566572</td>
      <td>8.747917</td>
    </tr>
    <tr>
      <td>Urbanized_Population_annual_growth_rate_2016_2030</td>
      <td>202.0</td>
      <td>1.672330</td>
      <td>1.346465</td>
      <td>-0.529898</td>
      <td>0.673539</td>
      <td>1.373605</td>
      <td>2.630008</td>
      <td>5.712731</td>
    </tr>
    <tr>
      <td>Crude_death_diff_1990_2016</td>
      <td>184.0</td>
      <td>-2.336114</td>
      <td>4.088298</td>
      <td>-26.474000</td>
      <td>-3.713750</td>
      <td>-1.317000</td>
      <td>0.132750</td>
      <td>3.913000</td>
    </tr>
    <tr>
      <td>Crude_birth_diff_1990_2016</td>
      <td>184.0</td>
      <td>-8.392750</td>
      <td>4.776916</td>
      <td>-22.793000</td>
      <td>-11.020000</td>
      <td>-8.283000</td>
      <td>-4.602000</td>
      <td>-0.219000</td>
    </tr>
    <tr>
      <td>Life_expectancy_diff_1990_2016</td>
      <td>184.0</td>
      <td>7.236027</td>
      <td>4.598818</td>
      <td>-5.032000</td>
      <td>4.672500</td>
      <td>5.948500</td>
      <td>9.355250</td>
      <td>32.912000</td>
    </tr>
  </tbody>
</table>
</div>




```python
DemInd.isnull().sum()
```




    index                                                 0
    Population_thousand_under_18_2016                     0
    Population_thousand_under_5_2016                      0
    Population_annual_growth_rate_1990_2016               0
    Population_annual_growth_rate_2016_2030               0
    Total_fertility_rate_2016                            18
    Urbanized_population_rate_2016                        0
    Urbanized_Population_annual_growth_rate_1990_2016     0
    Urbanized_Population_annual_growth_rate_2016_2030     0
    Crude_death_diff_1990_2016                           18
    Crude_birth_diff_1990_2016                           18
    Life_expectancy_diff_1990_2016                       18
    dtype: int64



---

## Sheet 7


```python
sheet7 = sowc.sheet_by_name('Women')
```


```python
WomInd = {}
for i in range(7, sheet7.nrows):

    val = sheet7.row_values(i)

    country = val[1]

    WomInd[country] = {
        'Life_expectancy:_females_as_a_%_of_males_2016':val[2],
        'Adult_literacy_rate:_females_as_a_%_of_males_2011_2016':val[3],
        'Enrolment_ratios:_females_as_a_%_of_males_Primary_2011_2016':val[5],
        'Enrolment_ratios:_females_as_a_%_of_males_Secondary_2011_2016':val[6],        
        'Survival_rate_to_the_last_grade_of_primary:_females_as_a_%_of_males_2011_2016':val[7],
        'Demand_for_family_planning_satisfied_with_modern_methods_%_2011_2016':val[8],        
        'Antenatal_care_%_at_least_one_visit_2011_2016':val[10],
        'Antenatal_care_%_at_least_four_visits_2011_2016':val[12],
        'Delivery_Skilled_birth_attendant_2013_2016':val[14],
        'Delivery_Institutional_delivery_2011_2016':val[16],
        'Delivery_C–section_2011_2016':val[18],
        'Newborns_Postnatal_health_check_%_2011_2016':val[20],
        'Mothers_Postnatal_health_check_%_2011_2016':val[22],
        'Reported_Maternal_mortality_ratio_2011_2016':val[24],
        'Adjusted_Maternal_mortality_ratio_2015':val[26],
        'Lifetime_risk_of_maternal_death_(1 in:)_2015':val[27],
    }

    if country == 'Zimbabwe':
        break
```


```python
WomInd = pd.DataFrame(WomInd).T
```


```python
WomInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = WomInd.columns.drop('index')
WomInd[NumColumns] = WomInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
WomInd.drop(['Reported_Maternal_mortality_ratio_2011_2016','Lifetime_risk_of_maternal_death_(1 in:)_2015'],
            axis=1, inplace=True)
```


```python
WomInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Life_expectancy:_females_as_a_%_of_males_2016</td>
      <td>184.0</td>
      <td>107.061973</td>
      <td>3.168680</td>
      <td>100.633694</td>
      <td>104.863657</td>
      <td>106.507876</td>
      <td>108.771489</td>
      <td>119.961262</td>
    </tr>
    <tr>
      <td>Adult_literacy_rate:_females_as_a_%_of_males_2011_2016</td>
      <td>142.0</td>
      <td>88.080845</td>
      <td>16.660248</td>
      <td>38.455000</td>
      <td>80.516000</td>
      <td>95.777000</td>
      <td>99.751750</td>
      <td>125.370000</td>
    </tr>
    <tr>
      <td>Enrolment_ratios:_females_as_a_%_of_males_Primary_2011_2016</td>
      <td>180.0</td>
      <td>97.578119</td>
      <td>6.475816</td>
      <td>63.982681</td>
      <td>96.497276</td>
      <td>99.035529</td>
      <td>100.312528</td>
      <td>111.934511</td>
    </tr>
    <tr>
      <td>Enrolment_ratios:_females_as_a_%_of_males_Secondary_2011_2016</td>
      <td>170.0</td>
      <td>98.465553</td>
      <td>13.303295</td>
      <td>45.736000</td>
      <td>95.434500</td>
      <td>100.186500</td>
      <td>105.400250</td>
      <td>135.846000</td>
    </tr>
    <tr>
      <td>Survival_rate_to_the_last_grade_of_primary:_females_as_a_%_of_males_2011_2016</td>
      <td>137.0</td>
      <td>101.680493</td>
      <td>6.155982</td>
      <td>85.426120</td>
      <td>99.182488</td>
      <td>100.557474</td>
      <td>103.206395</td>
      <td>134.028776</td>
    </tr>
    <tr>
      <td>Demand_for_family_planning_satisfied_with_modern_methods_%_2011_2016</td>
      <td>131.0</td>
      <td>55.884733</td>
      <td>22.257806</td>
      <td>5.600000</td>
      <td>38.400000</td>
      <td>56.400000</td>
      <td>75.050000</td>
      <td>96.600000</td>
    </tr>
    <tr>
      <td>Antenatal_care_%_at_least_one_visit_2011_2016</td>
      <td>165.0</td>
      <td>91.170909</td>
      <td>11.790121</td>
      <td>26.100000</td>
      <td>88.500000</td>
      <td>95.700000</td>
      <td>98.600000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Antenatal_care_%_at_least_four_visits_2011_2016</td>
      <td>147.0</td>
      <td>72.619728</td>
      <td>21.311938</td>
      <td>6.300000</td>
      <td>57.300000</td>
      <td>76.100000</td>
      <td>90.250000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Delivery_Skilled_birth_attendant_2013_2016</td>
      <td>169.0</td>
      <td>84.395858</td>
      <td>21.581952</td>
      <td>9.400000</td>
      <td>77.200000</td>
      <td>96.400000</td>
      <td>99.600000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Delivery_Institutional_delivery_2011_2016</td>
      <td>168.0</td>
      <td>82.759524</td>
      <td>22.370011</td>
      <td>9.400000</td>
      <td>73.000000</td>
      <td>94.250000</td>
      <td>99.100000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Delivery_C–section_2011_2016</td>
      <td>157.0</td>
      <td>18.382166</td>
      <td>13.402753</td>
      <td>0.600000</td>
      <td>6.300000</td>
      <td>16.900000</td>
      <td>26.700000</td>
      <td>58.100000</td>
    </tr>
    <tr>
      <td>Newborns_Postnatal_health_check_%_2011_2016</td>
      <td>85.0</td>
      <td>51.070588</td>
      <td>35.558064</td>
      <td>0.300000</td>
      <td>18.400000</td>
      <td>42.900000</td>
      <td>90.700000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Mothers_Postnatal_health_check_%_2011_2016</td>
      <td>92.0</td>
      <td>65.373913</td>
      <td>27.205982</td>
      <td>1.200000</td>
      <td>45.775000</td>
      <td>71.000000</td>
      <td>88.225000</td>
      <td>99.800000</td>
    </tr>
    <tr>
      <td>Adjusted_Maternal_mortality_ratio_2015</td>
      <td>182.0</td>
      <td>169.549451</td>
      <td>232.733208</td>
      <td>3.000000</td>
      <td>14.250000</td>
      <td>54.000000</td>
      <td>229.000000</td>
      <td>1360.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
WomInd.isnull().sum()
```




    index                                                                              0
    Life_expectancy:_females_as_a_%_of_males_2016                                     18
    Adult_literacy_rate:_females_as_a_%_of_males_2011_2016                            60
    Enrolment_ratios:_females_as_a_%_of_males_Primary_2011_2016                       22
    Enrolment_ratios:_females_as_a_%_of_males_Secondary_2011_2016                     32
    Survival_rate_to_the_last_grade_of_primary:_females_as_a_%_of_males_2011_2016     65
    Demand_for_family_planning_satisfied_with_modern_methods_%_2011_2016              71
    Antenatal_care_%_at_least_one_visit_2011_2016                                     37
    Antenatal_care_%_at_least_four_visits_2011_2016                                   55
    Delivery_Skilled_birth_attendant_2013_2016                                        33
    Delivery_Institutional_delivery_2011_2016                                         34
    Delivery_C–section_2011_2016                                                      45
    Newborns_Postnatal_health_check_%_2011_2016                                      117
    Mothers_Postnatal_health_check_%_2011_2016                                       110
    Adjusted_Maternal_mortality_ratio_2015                                            20
    dtype: int64



---

## Sheet 8


```python
sheet8 = sowc.sheet_by_name('Child Protection')
```


```python
ChildProtInd = {}
for i in range(7, sheet8.nrows):

    val = sheet8.row_values(i)

    country = val[1]

    ChildProtInd[country] = {
        'Child_labour_%_male_2010_2016':val[4],
        'Child_labour_%_female_2010_2016':val[6],
        'married_by_15_2010_2016':val[8],
        'married_by_18_2010_2016':val[10],
        'Birth_registration_%_2010_2016':val[12],
        'Justification_of_wife_beating_%_male_2010_2016':val[20],
        'Justification_of_wife_beating_%_female_2010_2016':val[22],
        'Violent_discipline_%_male_2010_2016':val[26],
        'Violent_discipline_%_female_2010_2016':val[28],        
    }

    if country == 'Zimbabwe':
        break
```


```python
ChildProtInd = pd.DataFrame(ChildProtInd).T
```


```python
ChildProtInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = ChildProtInd.columns.drop('index')
ChildProtInd[NumColumns] = ChildProtInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
ChildProtInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Child_labour_%_male_2010_2016</td>
      <td>111.0</td>
      <td>17.875676</td>
      <td>13.834326</td>
      <td>0.5</td>
      <td>6.300000</td>
      <td>15.30</td>
      <td>26.60</td>
      <td>59.0</td>
    </tr>
    <tr>
      <td>Child_labour_%_female_2010_2016</td>
      <td>111.0</td>
      <td>15.764865</td>
      <td>13.992047</td>
      <td>0.1</td>
      <td>4.250000</td>
      <td>11.20</td>
      <td>24.35</td>
      <td>53.6</td>
    </tr>
    <tr>
      <td>married_by_15_2010_2016</td>
      <td>125.0</td>
      <td>5.618881</td>
      <td>6.091525</td>
      <td>0.0</td>
      <td>1.050169</td>
      <td>3.60</td>
      <td>8.80</td>
      <td>29.7</td>
    </tr>
    <tr>
      <td>married_by_18_2010_2016</td>
      <td>125.0</td>
      <td>23.650400</td>
      <td>15.161633</td>
      <td>1.6</td>
      <td>11.000000</td>
      <td>21.70</td>
      <td>32.60</td>
      <td>76.3</td>
    </tr>
    <tr>
      <td>Birth_registration_%_2010_2016</td>
      <td>169.0</td>
      <td>83.339645</td>
      <td>24.376638</td>
      <td>3.0</td>
      <td>74.800000</td>
      <td>96.10</td>
      <td>100.00</td>
      <td>100.0</td>
    </tr>
    <tr>
      <td>Justification_of_wife_beating_%_male_2010_2016</td>
      <td>74.0</td>
      <td>31.732432</td>
      <td>19.430530</td>
      <td>4.2</td>
      <td>16.625000</td>
      <td>28.35</td>
      <td>44.15</td>
      <td>80.7</td>
    </tr>
    <tr>
      <td>Justification_of_wife_beating_%_female_2010_2016</td>
      <td>115.0</td>
      <td>35.526957</td>
      <td>24.488004</td>
      <td>1.5</td>
      <td>12.450000</td>
      <td>34.00</td>
      <td>55.05</td>
      <td>92.1</td>
    </tr>
    <tr>
      <td>Violent_discipline_%_male_2010_2016</td>
      <td>75.0</td>
      <td>75.461333</td>
      <td>13.650977</td>
      <td>36.8</td>
      <td>68.850000</td>
      <td>78.40</td>
      <td>85.30</td>
      <td>93.9</td>
    </tr>
    <tr>
      <td>Violent_discipline_%_female_2010_2016</td>
      <td>75.0</td>
      <td>72.200000</td>
      <td>15.267135</td>
      <td>34.1</td>
      <td>63.400000</td>
      <td>74.10</td>
      <td>83.90</td>
      <td>93.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
ChildProtInd.isnull().sum()
```




    index                                                 0
    Child_labour_%_male_2010_2016                        91
    Child_labour_%_female_2010_2016                      91
    married_by_15_2010_2016                              77
    married_by_18_2010_2016                              77
    Birth_registration_%_2010_2016                       33
    Justification_of_wife_beating_%_male_2010_2016      128
    Justification_of_wife_beating_%_female_2010_2016     87
    Violent_discipline_%_male_2010_2016                 127
    Violent_discipline_%_female_2010_2016               127
    dtype: int64



---

## Sheet 9


```python
sheet9 = sowc.sheet_by_name('Adolescents')
```


```python
AdolInd = {}
for i in range(7, sheet9.nrows):

    val = sheet9.row_values(i)

    country = val[1]

    AdolInd[country] = {
        'Adolescent_Proportion_of_total_population_%_2016':val[3],
        'Adolescents_currently_married_male_2010_2016':val[4],
        'Adolescents_currently_married_female_2010_2016':val[6],
        'Births_by_age_18_%_2011_2016':val[8],
        'Adolescent_birth_rate_2009_2014':val[10],
        'Justification_of_wife_beating_among_adolescents_%_male_2010_2016':val[12],
        'Justification_of_wife_beating_among_adolescents_%_female_2010_2016':val[14],        
        'Use_of_mass_media_among_adolescents_male_2010_2016':val[16],
        'Use_of_mass_media_among_adolescents_female_2010_2016':val[18],
        'Lower_secondary_school_gross_enrolment_ratio_2011_2016':val[20],
        'Upper_secondary_school_gross_enrolment_ratio_2011_2016':val[21],
        'Comprehensive_knowledge_of_HIV_among_adolescents_male_2011_2016': val[22],
        'Comprehensive_knowledge_of_HIV_among_adolescents_female_2011_2016': val[24]
    }

    if country == 'Zimbabwe':
        break
```


```python
AdolInd = pd.DataFrame(AdolInd).T
```


```python
AdolInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = AdolInd.columns.drop('index')
AdolInd[NumColumns] = AdolInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
AdolInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Adolescent_Proportion_of_total_population_%_2016</td>
      <td>184.0</td>
      <td>16.940290</td>
      <td>4.962188</td>
      <td>7.356964</td>
      <td>12.079707</td>
      <td>17.412905</td>
      <td>21.782824</td>
      <td>24.882812</td>
    </tr>
    <tr>
      <td>Adolescents_currently_married_male_2010_2016</td>
      <td>82.0</td>
      <td>2.703659</td>
      <td>2.836031</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>1.500000</td>
      <td>3.575000</td>
      <td>13.400000</td>
    </tr>
    <tr>
      <td>Adolescents_currently_married_female_2010_2016</td>
      <td>130.0</td>
      <td>14.733385</td>
      <td>10.540851</td>
      <td>0.600000</td>
      <td>7.175000</td>
      <td>13.350000</td>
      <td>19.750000</td>
      <td>61.000000</td>
    </tr>
    <tr>
      <td>Births_by_age_18_%_2011_2016</td>
      <td>116.0</td>
      <td>16.602586</td>
      <td>11.995010</td>
      <td>0.500000</td>
      <td>5.775000</td>
      <td>15.750000</td>
      <td>22.250000</td>
      <td>50.600000</td>
    </tr>
    <tr>
      <td>Adolescent_birth_rate_2009_2014</td>
      <td>199.0</td>
      <td>56.142211</td>
      <td>47.816427</td>
      <td>0.700000</td>
      <td>16.550000</td>
      <td>46.000000</td>
      <td>83.250000</td>
      <td>229.000000</td>
    </tr>
    <tr>
      <td>Justification_of_wife_beating_among_adolescents_%_male_2010_2016</td>
      <td>72.0</td>
      <td>37.601389</td>
      <td>20.855124</td>
      <td>1.600000</td>
      <td>22.175000</td>
      <td>36.500000</td>
      <td>52.525000</td>
      <td>83.100000</td>
    </tr>
    <tr>
      <td>Justification_of_wife_beating_among_adolescents_%_female_2010_2016</td>
      <td>115.0</td>
      <td>36.295652</td>
      <td>24.072717</td>
      <td>1.300000</td>
      <td>13.750000</td>
      <td>35.100000</td>
      <td>54.700000</td>
      <td>89.400000</td>
    </tr>
    <tr>
      <td>Use_of_mass_media_among_adolescents_male_2010_2016</td>
      <td>71.0</td>
      <td>76.432394</td>
      <td>18.413891</td>
      <td>30.000000</td>
      <td>61.250000</td>
      <td>81.300000</td>
      <td>91.750000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Use_of_mass_media_among_adolescents_female_2010_2016</td>
      <td>95.0</td>
      <td>77.510526</td>
      <td>20.451335</td>
      <td>23.200000</td>
      <td>61.850000</td>
      <td>85.300000</td>
      <td>96.150000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>Lower_secondary_school_gross_enrolment_ratio_2011_2016</td>
      <td>184.0</td>
      <td>89.917281</td>
      <td>26.324847</td>
      <td>17.953610</td>
      <td>73.921188</td>
      <td>96.992765</td>
      <td>102.841680</td>
      <td>206.250000</td>
    </tr>
    <tr>
      <td>Upper_secondary_school_gross_enrolment_ratio_2011_2016</td>
      <td>170.0</td>
      <td>77.798477</td>
      <td>36.516499</td>
      <td>5.020860</td>
      <td>52.054832</td>
      <td>81.628365</td>
      <td>99.619465</td>
      <td>194.101990</td>
    </tr>
    <tr>
      <td>Comprehensive_knowledge_of_HIV_among_adolescents_male_2011_2016</td>
      <td>86.0</td>
      <td>29.259302</td>
      <td>14.566395</td>
      <td>2.100000</td>
      <td>20.375000</td>
      <td>28.350000</td>
      <td>39.975000</td>
      <td>66.700000</td>
    </tr>
    <tr>
      <td>Comprehensive_knowledge_of_HIV_among_adolescents_female_2011_2016</td>
      <td>111.0</td>
      <td>26.823423</td>
      <td>15.757716</td>
      <td>0.600000</td>
      <td>16.000000</td>
      <td>23.400000</td>
      <td>39.000000</td>
      <td>65.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
AdolInd.isnull().sum()
```




    index                                                                   0
    Adolescent_Proportion_of_total_population_%_2016                       18
    Adolescents_currently_married_male_2010_2016                          120
    Adolescents_currently_married_female_2010_2016                         72
    Births_by_age_18_%_2011_2016                                           86
    Adolescent_birth_rate_2009_2014                                         3
    Justification_of_wife_beating_among_adolescents_%_male_2010_2016      130
    Justification_of_wife_beating_among_adolescents_%_female_2010_2016     87
    Use_of_mass_media_among_adolescents_male_2010_2016                    131
    Use_of_mass_media_among_adolescents_female_2010_2016                  107
    Lower_secondary_school_gross_enrolment_ratio_2011_2016                 18
    Upper_secondary_school_gross_enrolment_ratio_2011_2016                 32
    Comprehensive_knowledge_of_HIV_among_adolescents_male_2011_2016       116
    Comprehensive_knowledge_of_HIV_among_adolescents_female_2011_2016      91
    dtype: int64



---

## Sheet 10


```python
sheet10 = sowc.sheet_by_name('Disparities by Residence')
```


```python
ResInd = {}
for i in range(6, sheet10.nrows):

    val = sheet10.row_values(i)

    country = val[1]

    ResInd[country] = {
        'Birth_registration_Ratio_of_urban_to_rural_2010_2016':val[6],
        'Skilled_birth_attendant_Ratio_of_urban_to_rural_2011_2016':val[12],
        'Under_5_Stunting_prevalence_Ratio_of_urban_to_rural_2011_2016':val[18],
        'ORS_treatment_Ratio_of_urban_to_rural_2011_2016':val[24],
        'Primary_school_net_attendance_Ratio_of_urban_to_rural_2011_2016':val[30],
        'Comprehensive_knowledge_of_HIV_female_15_24_Ratio_of_urban_to_rural_2011_2016':val[36],
        'Use_of_basic_sanitation_services_Ratio_of_urban_to_rural_2015':val[40],
    }

    if country == 'Zimbabwe':
        break
```


```python
ResInd = pd.DataFrame(ResInd).T
```


```python
ResInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = ResInd.columns.drop('index')
ResInd[NumColumns] = ResInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
ResInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Birth_registration_Ratio_of_urban_to_rural_2010_2016</td>
      <td>115.0</td>
      <td>1.333821</td>
      <td>0.824174</td>
      <td>0.875000</td>
      <td>1.003026</td>
      <td>1.041439</td>
      <td>1.280923</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <td>Skilled_birth_attendant_Ratio_of_urban_to_rural_2011_2016</td>
      <td>126.0</td>
      <td>1.527469</td>
      <td>0.929804</td>
      <td>0.978102</td>
      <td>1.022989</td>
      <td>1.152095</td>
      <td>1.632145</td>
      <td>7.392857</td>
    </tr>
    <tr>
      <td>Under_5_Stunting_prevalence_Ratio_of_urban_to_rural_2011_2016</td>
      <td>113.0</td>
      <td>1.446627</td>
      <td>0.409100</td>
      <td>0.594595</td>
      <td>1.159509</td>
      <td>1.409341</td>
      <td>1.660377</td>
      <td>3.010870</td>
    </tr>
    <tr>
      <td>ORS_treatment_Ratio_of_urban_to_rural_2011_2016</td>
      <td>97.0</td>
      <td>1.188924</td>
      <td>0.349461</td>
      <td>0.606557</td>
      <td>0.981343</td>
      <td>1.105437</td>
      <td>1.282759</td>
      <td>2.952941</td>
    </tr>
    <tr>
      <td>Primary_school_net_attendance_Ratio_of_urban_to_rural_2011_2016</td>
      <td>110.0</td>
      <td>1.147277</td>
      <td>0.311006</td>
      <td>0.976395</td>
      <td>1.002411</td>
      <td>1.024289</td>
      <td>1.167055</td>
      <td>3.446429</td>
    </tr>
    <tr>
      <td>Comprehensive_knowledge_of_HIV_female_15_24_Ratio_of_urban_to_rural_2011_2016</td>
      <td>105.0</td>
      <td>1.788993</td>
      <td>0.883720</td>
      <td>0.738739</td>
      <td>1.253968</td>
      <td>1.525547</td>
      <td>2.016043</td>
      <td>6.666667</td>
    </tr>
    <tr>
      <td>Use_of_basic_sanitation_services_Ratio_of_urban_to_rural_2015</td>
      <td>170.0</td>
      <td>1.685916</td>
      <td>1.359954</td>
      <td>0.000000</td>
      <td>1.000381</td>
      <td>1.132543</td>
      <td>1.627809</td>
      <td>11.267766</td>
    </tr>
  </tbody>
</table>
</div>




```python
ResInd.isnull().sum()
```




    index                                                                              0
    Birth_registration_Ratio_of_urban_to_rural_2010_2016                              87
    Skilled_birth_attendant_Ratio_of_urban_to_rural_2011_2016                         76
    Under_5_Stunting_prevalence_Ratio_of_urban_to_rural_2011_2016                     89
    ORS_treatment_Ratio_of_urban_to_rural_2011_2016                                  105
    Primary_school_net_attendance_Ratio_of_urban_to_rural_2011_2016                   92
    Comprehensive_knowledge_of_HIV_female_15_24_Ratio_of_urban_to_rural_2011_2016     97
    Use_of_basic_sanitation_services_Ratio_of_urban_to_rural_2015                     32
    dtype: int64



---

## Sheet 11


```python
sheet11 = sowc.sheet_by_name('Disparities by Household Wealth')
```


```python
HWInd = {}
for i in range(6, sheet11.nrows):

    val = sheet11.row_values(i)

    country = val[1]

    HWInd[country] = {
        'Birth_registration_Ratio_of_richest_to_poorest_2010_2016':val[6],
        'Skilled_birth_attendant_Ratio_of_richest_to_poorest_2011_2016':val[12],
        'Under_5_Stunting_prevalence_Ratio_of_richest_to_poorest_2011_2016':val[18],
        'ORS_treatment_Ratio_of_richest_to_poorest_2011_2016':val[24],
        'Primary_school_net_attendance_Ratio_of_richest_to_poorest_2011_2016':val[30],
        'Comprehensive_knowledge_of_HIV_female_15_24_Ratio_of_richest_to_poorest_2011_2016':val[36],
        'Comprehensive_knowledge_of_HIV_male_15_24_Ratio_of_richest_to_poorest_2011_2016':val[42]
    }

    if country == 'Zimbabwe':
        break
```


```python
HWInd = pd.DataFrame(HWInd).T
```


```python
HWInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = HWInd.columns.drop('index')
HWInd[NumColumns] = HWInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
HWInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Birth_registration_Ratio_of_richest_to_poorest_2010_2016</td>
      <td>109.0</td>
      <td>1.918860</td>
      <td>2.127195</td>
      <td>0.931522</td>
      <td>1.016277</td>
      <td>1.125891</td>
      <td>1.789579</td>
      <td>14.280000</td>
    </tr>
    <tr>
      <td>Skilled_birth_attendant_Ratio_of_richest_to_poorest_2011_2016</td>
      <td>114.0</td>
      <td>2.575363</td>
      <td>3.326434</td>
      <td>0.939638</td>
      <td>1.061032</td>
      <td>1.381299</td>
      <td>2.613818</td>
      <td>27.100000</td>
    </tr>
    <tr>
      <td>Under_5_Stunting_prevalence_Ratio_of_richest_to_poorest_2011_2016</td>
      <td>109.0</td>
      <td>2.771697</td>
      <td>2.115183</td>
      <td>0.245283</td>
      <td>1.665493</td>
      <td>2.252688</td>
      <td>3.150943</td>
      <td>17.250000</td>
    </tr>
    <tr>
      <td>ORS_treatment_Ratio_of_richest_to_poorest_2011_2016</td>
      <td>73.0</td>
      <td>1.591046</td>
      <td>1.598806</td>
      <td>0.229885</td>
      <td>1.014368</td>
      <td>1.205426</td>
      <td>1.557214</td>
      <td>13.333333</td>
    </tr>
    <tr>
      <td>Primary_school_net_attendance_Ratio_of_richest_to_poorest_2011_2016</td>
      <td>104.0</td>
      <td>1.431720</td>
      <td>1.307555</td>
      <td>0.993631</td>
      <td>1.020400</td>
      <td>1.071623</td>
      <td>1.374258</td>
      <td>13.236842</td>
    </tr>
    <tr>
      <td>Comprehensive_knowledge_of_HIV_female_15_24_Ratio_of_richest_to_poorest_2011_2016</td>
      <td>90.0</td>
      <td>3.373654</td>
      <td>3.262288</td>
      <td>0.771654</td>
      <td>1.616146</td>
      <td>2.256440</td>
      <td>3.395292</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <td>Comprehensive_knowledge_of_HIV_male_15_24_Ratio_of_richest_to_poorest_2011_2016</td>
      <td>51.0</td>
      <td>2.809648</td>
      <td>1.973349</td>
      <td>1.021429</td>
      <td>1.589408</td>
      <td>2.196970</td>
      <td>3.204651</td>
      <td>12.181818</td>
    </tr>
  </tbody>
</table>
</div>




```python
HWInd.isnull().sum()
```




    index                                                                                  0
    Birth_registration_Ratio_of_richest_to_poorest_2010_2016                              93
    Skilled_birth_attendant_Ratio_of_richest_to_poorest_2011_2016                         88
    Under_5_Stunting_prevalence_Ratio_of_richest_to_poorest_2011_2016                     93
    ORS_treatment_Ratio_of_richest_to_poorest_2011_2016                                  129
    Primary_school_net_attendance_Ratio_of_richest_to_poorest_2011_2016                   98
    Comprehensive_knowledge_of_HIV_female_15_24_Ratio_of_richest_to_poorest_2011_2016    112
    Comprehensive_knowledge_of_HIV_male_15_24_Ratio_of_richest_to_poorest_2011_2016      151
    dtype: int64



---

## Sheet 12


```python
sheet12 = sowc.sheet_by_name('Early Childhood Development')
```


```python
ChildDevInd = {}
for i in range(7, sheet12.nrows):

    val = sheet12.row_values(i)

    country = val[1]

    ChildDevInd[country] = {
        'Attendance_in_early_childhood_education_total_2005_2016':val[2],
        'Attendance_in_early_childhood_education_poorest_2005_2016':val[8],
        'Attendance_in_early_childhood_education_richest_2005_2016':val[10],
        'Adult_support_for_learning_total_2005_2016':val[12],
        'Adult_support_for_learning_poorest_2005_2016':val[18],
        'Adult_support_for_learning_richest_2005_2016':val[20],
        "Father's_support_for_learning_2005_2016":val[22],
        "Learning_materials_at_home_Children's_books_poorest_2005_2016":val[26],
        "Learning_materials_at_home_Children's_books_richest_2005_2016":val[28],
        "Learning_materials_at_home_Playthings_poorest_2005_2016":val[30],
        "Learning_materials_at_home_Playthings_richest_2005_2016":val[34],
        'Children_with_inadequate_supervision_total_2005_2016':val[36],
        'Children_with_inadequate_supervision_poorest_2005_2016':val[42],
        'Children_with_inadequate_supervision_richest_2005_2016':val[44]
    }

    if country == 'Zimbabwe':
        break
```


```python
ChildDevInd = pd.DataFrame(ChildDevInd).T
```


```python
ChildDevInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = ChildDevInd.columns.drop('index')
ChildDevInd[NumColumns] = ChildDevInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
ChildDevInd['Attendance_in_early_childhood_education_Ratio_of_richest_to_poorest_2005_2016'] = ChildDevInd['Attendance_in_early_childhood_education_richest_2005_2016']/ChildDevInd['Attendance_in_early_childhood_education_poorest_2005_2016']
```


```python
ChildDevInd['Adult_support_for_learning_Ratio_of_richest_to_poorest_2005_2016'] = ChildDevInd['Adult_support_for_learning_richest_2005_2016']/ChildDevInd['Adult_support_for_learning_poorest_2005_2016']
```


```python
ChildDevInd['Learning_materials_at_home_Playthings_Ratio_of_richest_to_poorest_2005_2016'] = ChildDevInd['Learning_materials_at_home_Playthings_richest_2005_2016']/ChildDevInd['Learning_materials_at_home_Playthings_poorest_2005_2016']
```


```python
ChildDevInd["Learning_materials_at_home_Children's_books_Ratio_of_richest_to_poorest_2005_2016"] = ChildDevInd["Learning_materials_at_home_Children's_books_richest_2005_2016"]/ChildDevInd["Learning_materials_at_home_Children's_books_poorest_2005_2016"]
```


```python
ChildDevInd['Children_with_inadequate_supervision_Ratio_of_richest_to_poorest_2005_2016'] = ChildDevInd['Children_with_inadequate_supervision_richest_2005_2016']/ChildDevInd['Children_with_inadequate_supervision_poorest_2005_2016']
```


```python
ChildDevInd.drop(columns=['Attendance_in_early_childhood_education_richest_2005_2016',
                          'Attendance_in_early_childhood_education_poorest_2005_2016',
                          'Attendance_in_early_childhood_education_total_2005_2016',
                          'Adult_support_for_learning_richest_2005_2016',
                          'Adult_support_for_learning_poorest_2005_2016',
                          'Adult_support_for_learning_total_2005_2016',
                          'Learning_materials_at_home_Playthings_richest_2005_2016',
                          'Learning_materials_at_home_Playthings_poorest_2005_2016',
                          "Learning_materials_at_home_Children's_books_richest_2005_2016",
                          "Learning_materials_at_home_Children's_books_poorest_2005_2016",
                          'Children_with_inadequate_supervision_richest_2005_2016',
                          'Children_with_inadequate_supervision_poorest_2005_2016'], axis=1, inplace=True)
```


```python
ChildDevInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Father's_support_for_learning_2005_2016</td>
      <td>78.0</td>
      <td>33.812564</td>
      <td>24.671337</td>
      <td>0.300000</td>
      <td>10.100000</td>
      <td>28.850000</td>
      <td>53.925000</td>
      <td>84.900000</td>
    </tr>
    <tr>
      <td>Children_with_inadequate_supervision_total_2005_2016</td>
      <td>77.0</td>
      <td>16.032670</td>
      <td>14.959526</td>
      <td>0.800000</td>
      <td>4.955560</td>
      <td>10.500000</td>
      <td>20.700000</td>
      <td>60.700000</td>
    </tr>
    <tr>
      <td>Attendance_in_early_childhood_education_Ratio_of_richest_to_poorest_2005_2016</td>
      <td>66.0</td>
      <td>inf</td>
      <td>NaN</td>
      <td>0.967555</td>
      <td>2.027600</td>
      <td>3.892956</td>
      <td>9.596358</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>Adult_support_for_learning_Ratio_of_richest_to_poorest_2005_2016</td>
      <td>67.0</td>
      <td>1.572239</td>
      <td>0.589444</td>
      <td>0.963260</td>
      <td>1.138836</td>
      <td>1.432137</td>
      <td>1.837242</td>
      <td>3.619632</td>
    </tr>
    <tr>
      <td>Learning_materials_at_home_Playthings_Ratio_of_richest_to_poorest_2005_2016</td>
      <td>64.0</td>
      <td>1.098103</td>
      <td>0.195392</td>
      <td>0.521739</td>
      <td>0.995001</td>
      <td>1.041494</td>
      <td>1.181681</td>
      <td>1.797794</td>
    </tr>
    <tr>
      <td>Learning_materials_at_home_Children's_books_Ratio_of_richest_to_poorest_2005_2016</td>
      <td>65.0</td>
      <td>inf</td>
      <td>NaN</td>
      <td>1.001088</td>
      <td>3.077236</td>
      <td>7.262295</td>
      <td>25.000000</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>Children_with_inadequate_supervision_Ratio_of_richest_to_poorest_2005_2016</td>
      <td>63.0</td>
      <td>0.772650</td>
      <td>0.975625</td>
      <td>0.000000</td>
      <td>0.383849</td>
      <td>0.630485</td>
      <td>0.867302</td>
      <td>7.750000</td>
    </tr>
  </tbody>
</table>
</div>




```python
ChildDevInd.isnull().sum()
```




    index                                                                                  0
    Father's_support_for_learning_2005_2016                                              124
    Children_with_inadequate_supervision_total_2005_2016                                 125
    Attendance_in_early_childhood_education_Ratio_of_richest_to_poorest_2005_2016        136
    Adult_support_for_learning_Ratio_of_richest_to_poorest_2005_2016                     135
    Learning_materials_at_home_Playthings_Ratio_of_richest_to_poorest_2005_2016          138
    Learning_materials_at_home_Children's_books_Ratio_of_richest_to_poorest_2005_2016    137
    Children_with_inadequate_supervision_Ratio_of_richest_to_poorest_2005_2016           139
    dtype: int64



---

## Sheet 13


```python
sheet13 = sowc.sheet_by_name('Economic Indicators')
```


```python
EconInd = {}
for i in range(7, sheet13.nrows):

    val = sheet13.row_values(i)

    country = val[1]

    EconInd[country] = {
        'Population_below_international_poverty_line_2010_2014':val[2],
        'ODA_inflow_as_a_%_of_recipient_GNI_2015':val[7],
        'Share_of_household_income_poorest_2009_2013':val[8],
        'Share_of_household_income_richest_2009_2013':val[10],
    }

    if country == 'Zimbabwe':
        break
```


```python
EconInd = pd.DataFrame(EconInd).T
```


```python
EconInd.reset_index(level=0, inplace=True)
```


```python
NumColumns = EconInd.columns.drop('index')
EconInd[NumColumns] = EconInd[NumColumns].apply(pd.to_numeric, errors='coerce')
```


```python
EconInd['Share_of_household_income_Ratio_of_richest_to_poorest_2009_2013'] = EconInd['Share_of_household_income_richest_2009_2013']/EconInd['Share_of_household_income_poorest_2009_2013']
```


```python
EconInd.drop(columns=['Share_of_household_income_richest_2009_2013',
                      'Share_of_household_income_poorest_2009_2013'], axis=1, inplace=True)
```


```python
EconInd.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Population_below_international_poverty_line_2010_2014</td>
      <td>130.0</td>
      <td>19.413077</td>
      <td>22.575669</td>
      <td>0.000000</td>
      <td>1.325000</td>
      <td>8.200000</td>
      <td>35.125000</td>
      <td>77.800000</td>
    </tr>
    <tr>
      <td>ODA_inflow_as_a_%_of_recipient_GNI_2015</td>
      <td>129.0</td>
      <td>6.219707</td>
      <td>11.062089</td>
      <td>-0.005125</td>
      <td>0.499288</td>
      <td>2.404268</td>
      <td>7.230937</td>
      <td>89.199442</td>
    </tr>
    <tr>
      <td>Share_of_household_income_Ratio_of_richest_to_poorest_2009_2013</td>
      <td>150.0</td>
      <td>3.012061</td>
      <td>1.383625</td>
      <td>1.415323</td>
      <td>2.026402</td>
      <td>2.637952</td>
      <td>3.495688</td>
      <td>9.569444</td>
    </tr>
  </tbody>
</table>
</div>




```python
EconInd.isnull().sum()
```




    index                                                               0
    Population_below_international_poverty_line_2010_2014              72
    ODA_inflow_as_a_%_of_recipient_GNI_2015                            73
    Share_of_household_income_Ratio_of_richest_to_poorest_2009_2013    52
    dtype: int64
