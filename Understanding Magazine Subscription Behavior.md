```python
#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)

from scipy import stats
from scipy.stats import zscore

from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB  # using Gaussian algorithm for Naive Bayes

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC  # using Gaussian Kernel or Radial Basis Function
import sklearn
from sklearn.preprocessing import RobustScaler, StandardScaler
```


```python
#Loading the data set
data = pd.read_excel('Desktop/marketing_campaign.xlsx')
```


```python
#View the data set
data
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
      <th>ID</th>
      <th>Year_Birth</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Dt_Customer</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>AcceptedCmp1</th>
      <th>AcceptedCmp2</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5524</td>
      <td>1957</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>2012-09-04</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2174</td>
      <td>1954</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>2014-03-08</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4141</td>
      <td>1965</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-08-21</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6182</td>
      <td>1984</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>2014-02-10</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5324</td>
      <td>1981</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>2014-01-19</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>10870</td>
      <td>1967</td>
      <td>Graduation</td>
      <td>Married</td>
      <td>61223.0</td>
      <td>0</td>
      <td>1</td>
      <td>2013-06-13</td>
      <td>46</td>
      <td>709</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>4001</td>
      <td>1946</td>
      <td>PhD</td>
      <td>Together</td>
      <td>64014.0</td>
      <td>2</td>
      <td>1</td>
      <td>2014-06-10</td>
      <td>56</td>
      <td>406</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>7270</td>
      <td>1981</td>
      <td>Graduation</td>
      <td>Divorced</td>
      <td>56981.0</td>
      <td>0</td>
      <td>0</td>
      <td>2014-01-25</td>
      <td>91</td>
      <td>908</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>8235</td>
      <td>1956</td>
      <td>Master</td>
      <td>Together</td>
      <td>69245.0</td>
      <td>0</td>
      <td>1</td>
      <td>2014-01-24</td>
      <td>8</td>
      <td>428</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>9405</td>
      <td>1954</td>
      <td>PhD</td>
      <td>Married</td>
      <td>52869.0</td>
      <td>1</td>
      <td>1</td>
      <td>2012-10-15</td>
      <td>40</td>
      <td>84</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2240 rows × 29 columns</p>
</div>




```python
# to understand the type of varaible in the datasets
data.dtypes
```




    ID                       int64
    Year_Birth               int64
    Education               object
    Marital_Status          object
    Income                 float64
    Kidhome                  int64
    Teenhome                 int64
    Dt_Customer             object
    Recency                  int64
    MntWines                 int64
    MntFruits                int64
    MntMeatProducts          int64
    MntFishProducts          int64
    MntSweetProducts         int64
    MntGoldProds             int64
    NumDealsPurchases        int64
    NumWebPurchases          int64
    NumCatalogPurchases      int64
    NumStorePurchases        int64
    NumWebVisitsMonth        int64
    AcceptedCmp3             int64
    AcceptedCmp4             int64
    AcceptedCmp5             int64
    AcceptedCmp1             int64
    AcceptedCmp2             int64
    Complain                 int64
    Z_CostContact            int64
    Z_Revenue                int64
    Response                 int64
    dtype: object




```python
#Information on features 
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2240 entries, 0 to 2239
    Data columns (total 29 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   2240 non-null   int64  
     1   Year_Birth           2240 non-null   int64  
     2   Education            2240 non-null   object 
     3   Marital_Status       2240 non-null   object 
     4   Income               2216 non-null   float64
     5   Kidhome              2240 non-null   int64  
     6   Teenhome             2240 non-null   int64  
     7   Dt_Customer          2240 non-null   object 
     8   Recency              2240 non-null   int64  
     9   MntWines             2240 non-null   int64  
     10  MntFruits            2240 non-null   int64  
     11  MntMeatProducts      2240 non-null   int64  
     12  MntFishProducts      2240 non-null   int64  
     13  MntSweetProducts     2240 non-null   int64  
     14  MntGoldProds         2240 non-null   int64  
     15  NumDealsPurchases    2240 non-null   int64  
     16  NumWebPurchases      2240 non-null   int64  
     17  NumCatalogPurchases  2240 non-null   int64  
     18  NumStorePurchases    2240 non-null   int64  
     19  NumWebVisitsMonth    2240 non-null   int64  
     20  AcceptedCmp3         2240 non-null   int64  
     21  AcceptedCmp4         2240 non-null   int64  
     22  AcceptedCmp5         2240 non-null   int64  
     23  AcceptedCmp1         2240 non-null   int64  
     24  AcceptedCmp2         2240 non-null   int64  
     25  Complain             2240 non-null   int64  
     26  Z_CostContact        2240 non-null   int64  
     27  Z_Revenue            2240 non-null   int64  
     28  Response             2240 non-null   int64  
    dtypes: float64(1), int64(25), object(3)
    memory usage: 507.6+ KB


Income variable has 24 missing values


```python
#To remove the NA values
data = data.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(data))
```

    The total number of data-points after removing the rows with missing values are: 2216



```python
#Let's create a column that shows how many days customers have been registered with the company.
#We need to oldest customer and newest customer.
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)  
#Dates of the newest and oldest recorded customer
print("The newest customer's enrolment date in therecords:",max(dates))
print("The oldest customer's enrolment date in the records:",min(dates))
```

    The newest customer's enrolment date in therecords: 2014-06-29
    The oldest customer's enrolment date in the records: 2012-07-30


    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/3491233832.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])



```python
#Creating a feature ("Customer_For") of the number of days the customers started to shop in the store relative to the last recorded date
#Created a feature "Customer_For"
days = []
d1 = max(dates) #taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")
data.Customer_For
```

    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/4051713353.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Customer_For"] = days
    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/4051713353.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")





    0       57283200000000000
    1        9763200000000000
    2       26956800000000000
    3       12009600000000000
    4       13910400000000000
                  ...        
    2235    32918400000000000
    2236     1641600000000000
    2237    13392000000000000
    2238    13478400000000000
    2239    53740800000000000
    Name: Customer_For, Length: 2216, dtype: int64




```python
print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", data["Education"].value_counts())
```

    Total categories in the feature Marital_Status:
     Married     857
    Together    573
    Single      471
    Divorced    232
    Widow        76
    Alone         3
    Absurd        2
    YOLO          2
    Name: Marital_Status, dtype: int64 
    
    Total categories in the feature Education:
     Graduation    1116
    PhD            481
    Master         365
    2n Cycle       200
    Basic           54
    Name: Education, dtype: int64


# Now, we will generate the following steps to engineer some new features:

Extract the "Age" of a customer by the "Year_Birth" indicating the birth year of the respective person.
Create another feature "Spent" indicating the total amount spent by the customer in various categories over the span of two years.
Create another feature "Living_With" out of "Marital_Status" to extract the living situation of couples.
Create a feature "Children" to indicate total children in a household that is, kids and teenagers.
To get further clarity of household, Creating feature indicating "Family_Size"
Create a feature "Is_Parent" to indicate parenthood status
Lastly, I will create three categories in the "Education" by simplifying its value counts.
Dropping some of the redundant features


```python
#Feature Engineering
#Age of customer today 
data["Age"] = 2021-data["Year_Birth"]

#Total spendings on various items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

#Deriving living situation by marital status"Alone"
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

#Feature for total members in the householde
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

#Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

#Segmenting education levels in three groups
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#Dropping some of the redundant features
to_drop = ["Marital_Status","MntWines", "MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds","Kidhome","Teenhome","Dt_Customer","Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)
```

    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/3565750645.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Age"] = 2021-data["Year_Birth"]
    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/3565750645.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]
    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/3565750645.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/3565750645.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Children"]=data["Kidhome"]+data["Teenhome"]
    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/3565750645.py:15: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]
    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/3565750645.py:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Is_Parent"] = np.where(data.Children> 0, 1, 0)
    /var/folders/17/6mtx8nsj4f715m2700vbq_940000gn/T/ipykernel_2548/3565750645.py:21: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})



```python
data
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
      <th>Education</th>
      <th>Income</th>
      <th>Recency</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>NumCatalogPurchases</th>
      <th>NumStorePurchases</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>...</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
      <th>Customer_For</th>
      <th>Age</th>
      <th>Spent</th>
      <th>Living_With</th>
      <th>Children</th>
      <th>Family_Size</th>
      <th>Is_Parent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Graduate</td>
      <td>58138.0</td>
      <td>58</td>
      <td>3</td>
      <td>8</td>
      <td>10</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>57283200000000000</td>
      <td>64</td>
      <td>1617</td>
      <td>Alone</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Graduate</td>
      <td>46344.0</td>
      <td>38</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>9763200000000000</td>
      <td>67</td>
      <td>27</td>
      <td>Alone</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Graduate</td>
      <td>71613.0</td>
      <td>26</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>26956800000000000</td>
      <td>56</td>
      <td>776</td>
      <td>Partner</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Graduate</td>
      <td>26646.0</td>
      <td>26</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>12009600000000000</td>
      <td>37</td>
      <td>53</td>
      <td>Partner</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Postgraduate</td>
      <td>58293.0</td>
      <td>94</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>13910400000000000</td>
      <td>40</td>
      <td>422</td>
      <td>Partner</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>Graduate</td>
      <td>61223.0</td>
      <td>46</td>
      <td>2</td>
      <td>9</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>32918400000000000</td>
      <td>54</td>
      <td>1341</td>
      <td>Partner</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>Postgraduate</td>
      <td>64014.0</td>
      <td>56</td>
      <td>7</td>
      <td>8</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>1641600000000000</td>
      <td>75</td>
      <td>444</td>
      <td>Partner</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>Graduate</td>
      <td>56981.0</td>
      <td>91</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>13</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>13392000000000000</td>
      <td>40</td>
      <td>1241</td>
      <td>Alone</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>Postgraduate</td>
      <td>69245.0</td>
      <td>8</td>
      <td>2</td>
      <td>6</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
      <td>13478400000000000</td>
      <td>65</td>
      <td>843</td>
      <td>Partner</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>Postgraduate</td>
      <td>52869.0</td>
      <td>40</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>53740800000000000</td>
      <td>67</td>
      <td>172</td>
      <td>Partner</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2216 rows × 24 columns</p>
</div>




```python
#Arrange the target column-Personal Loan to the end of data set
Response = data['Response']
data.drop(labels=['Response'], axis=1, inplace = True)
data.insert(23, 'Response', Response)
data.head()
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
      <th>Education</th>
      <th>Income</th>
      <th>Recency</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>NumCatalogPurchases</th>
      <th>NumStorePurchases</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>...</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Customer_For</th>
      <th>Age</th>
      <th>Spent</th>
      <th>Living_With</th>
      <th>Children</th>
      <th>Family_Size</th>
      <th>Is_Parent</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Graduate</td>
      <td>58138.0</td>
      <td>58</td>
      <td>3</td>
      <td>8</td>
      <td>10</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>57283200000000000</td>
      <td>64</td>
      <td>1617</td>
      <td>Alone</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Graduate</td>
      <td>46344.0</td>
      <td>38</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>9763200000000000</td>
      <td>67</td>
      <td>27</td>
      <td>Alone</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Graduate</td>
      <td>71613.0</td>
      <td>26</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>26956800000000000</td>
      <td>56</td>
      <td>776</td>
      <td>Partner</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Graduate</td>
      <td>26646.0</td>
      <td>26</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>12009600000000000</td>
      <td>37</td>
      <td>53</td>
      <td>Partner</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Postgraduate</td>
      <td>58293.0</td>
      <td>94</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>13910400000000000</td>
      <td>40</td>
      <td>422</td>
      <td>Partner</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
data.describe()
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
      <th>Income</th>
      <th>Recency</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>NumCatalogPurchases</th>
      <th>NumStorePurchases</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>...</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Customer_For</th>
      <th>Age</th>
      <th>Spent</th>
      <th>Children</th>
      <th>Family_Size</th>
      <th>Is_Parent</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>...</td>
      <td>2216.000000</td>
      <td>2216.0</td>
      <td>2216.0</td>
      <td>2.216000e+03</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52247.251354</td>
      <td>49.012635</td>
      <td>2.323556</td>
      <td>4.085289</td>
      <td>2.671029</td>
      <td>5.800993</td>
      <td>5.319043</td>
      <td>0.073556</td>
      <td>0.074007</td>
      <td>0.073105</td>
      <td>...</td>
      <td>0.009477</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>3.054423e+16</td>
      <td>52.179603</td>
      <td>607.075361</td>
      <td>0.947202</td>
      <td>2.592509</td>
      <td>0.714350</td>
      <td>0.150271</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25173.076661</td>
      <td>28.948352</td>
      <td>1.923716</td>
      <td>2.740951</td>
      <td>2.926734</td>
      <td>3.250785</td>
      <td>2.425359</td>
      <td>0.261106</td>
      <td>0.261842</td>
      <td>0.260367</td>
      <td>...</td>
      <td>0.096907</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.749036e+16</td>
      <td>11.985554</td>
      <td>602.900476</td>
      <td>0.749062</td>
      <td>0.905722</td>
      <td>0.451825</td>
      <td>0.357417</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1730.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>0.000000e+00</td>
      <td>25.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35303.000000</td>
      <td>24.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>1.555200e+16</td>
      <td>44.000000</td>
      <td>69.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51381.500000</td>
      <td>49.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>3.071520e+16</td>
      <td>51.000000</td>
      <td>396.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>68522.000000</td>
      <td>74.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>4.570560e+16</td>
      <td>62.000000</td>
      <td>1048.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>666666.000000</td>
      <td>99.000000</td>
      <td>15.000000</td>
      <td>27.000000</td>
      <td>28.000000</td>
      <td>13.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>6.039360e+16</td>
      <td>128.000000</td>
      <td>2525.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 22 columns</p>
</div>




```python
#Plotting following features
To_Plot = [ "Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(data[To_Plot], hue= "Is_Parent",palette= (["#682F2F","#F3AB60"]))
#Taking hue 
plt.show()
```

    Reletive Plot Of Some Selected Features: A Data Subset



    <Figure size 432x288 with 0 Axes>



    
![png](output_15_2.png)
    



```python
#Dropping the outliers by setting a cap on Age and income. 
data = data[(data["Age"]<90)]
data = data[(data["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(data))
```

    The total number of data-points after removing the outliers are: 2212


Correlation among the attributes


```python
sns.pairplot(data.iloc[:,0:]);
```


    
![png](output_18_0.png)
    



```python
data.corr()
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
      <th>Income</th>
      <th>Recency</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>NumCatalogPurchases</th>
      <th>NumStorePurchases</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>...</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Customer_For</th>
      <th>Age</th>
      <th>Spent</th>
      <th>Children</th>
      <th>Family_Size</th>
      <th>Is_Parent</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Income</th>
      <td>1.000000</td>
      <td>0.007965</td>
      <td>-0.108207</td>
      <td>0.459265</td>
      <td>0.696589</td>
      <td>0.631424</td>
      <td>-0.650257</td>
      <td>-0.015152</td>
      <td>0.219633</td>
      <td>0.395569</td>
      <td>...</td>
      <td>-0.027900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.023760</td>
      <td>0.199977</td>
      <td>0.792740</td>
      <td>-0.343529</td>
      <td>-0.286638</td>
      <td>-0.403132</td>
      <td>0.161387</td>
    </tr>
    <tr>
      <th>Recency</th>
      <td>0.007965</td>
      <td>1.000000</td>
      <td>0.002591</td>
      <td>-0.005680</td>
      <td>0.024197</td>
      <td>-0.000460</td>
      <td>-0.018965</td>
      <td>-0.032361</td>
      <td>0.017520</td>
      <td>0.000233</td>
      <td>...</td>
      <td>0.005713</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.025681</td>
      <td>0.015694</td>
      <td>0.020479</td>
      <td>0.018062</td>
      <td>0.014717</td>
      <td>0.002189</td>
      <td>-0.200114</td>
    </tr>
    <tr>
      <th>NumDealsPurchases</th>
      <td>-0.108207</td>
      <td>0.002591</td>
      <td>1.000000</td>
      <td>0.241228</td>
      <td>-0.012015</td>
      <td>0.065635</td>
      <td>0.345623</td>
      <td>-0.023300</td>
      <td>0.015933</td>
      <td>-0.183837</td>
      <td>...</td>
      <td>0.003744</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.217948</td>
      <td>0.066156</td>
      <td>-0.065571</td>
      <td>0.436072</td>
      <td>0.373986</td>
      <td>0.388593</td>
      <td>0.003226</td>
    </tr>
    <tr>
      <th>NumWebPurchases</th>
      <td>0.459265</td>
      <td>-0.005680</td>
      <td>0.241228</td>
      <td>1.000000</td>
      <td>0.386539</td>
      <td>0.515756</td>
      <td>-0.051589</td>
      <td>0.042685</td>
      <td>0.162722</td>
      <td>0.141428</td>
      <td>...</td>
      <td>-0.013524</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.192082</td>
      <td>0.162265</td>
      <td>0.529095</td>
      <td>-0.148938</td>
      <td>-0.121879</td>
      <td>-0.073473</td>
      <td>0.151084</td>
    </tr>
    <tr>
      <th>NumCatalogPurchases</th>
      <td>0.696589</td>
      <td>0.024197</td>
      <td>-0.012015</td>
      <td>0.386539</td>
      <td>1.000000</td>
      <td>0.517887</td>
      <td>-0.522023</td>
      <td>0.104301</td>
      <td>0.140163</td>
      <td>0.321522</td>
      <td>...</td>
      <td>-0.018675</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.097245</td>
      <td>0.125856</td>
      <td>0.780250</td>
      <td>-0.443199</td>
      <td>-0.372319</td>
      <td>-0.452734</td>
      <td>0.219912</td>
    </tr>
    <tr>
      <th>NumStorePurchases</th>
      <td>0.631424</td>
      <td>-0.000460</td>
      <td>0.065635</td>
      <td>0.515756</td>
      <td>0.517887</td>
      <td>1.000000</td>
      <td>-0.433813</td>
      <td>-0.069455</td>
      <td>0.177705</td>
      <td>0.214249</td>
      <td>...</td>
      <td>-0.011947</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.111746</td>
      <td>0.138998</td>
      <td>0.675981</td>
      <td>-0.323823</td>
      <td>-0.265916</td>
      <td>-0.284891</td>
      <td>0.035563</td>
    </tr>
    <tr>
      <th>NumWebVisitsMonth</th>
      <td>-0.650257</td>
      <td>-0.018965</td>
      <td>0.345623</td>
      <td>-0.051589</td>
      <td>-0.522023</td>
      <td>-0.433813</td>
      <td>1.000000</td>
      <td>0.061084</td>
      <td>-0.028969</td>
      <td>-0.276097</td>
      <td>...</td>
      <td>0.020820</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.275673</td>
      <td>-0.120282</td>
      <td>-0.498769</td>
      <td>0.415558</td>
      <td>0.345316</td>
      <td>0.475856</td>
      <td>-0.002625</td>
    </tr>
    <tr>
      <th>AcceptedCmp3</th>
      <td>-0.015152</td>
      <td>-0.032361</td>
      <td>-0.023300</td>
      <td>0.042685</td>
      <td>0.104301</td>
      <td>-0.069455</td>
      <td>0.061084</td>
      <td>1.000000</td>
      <td>-0.079814</td>
      <td>0.080836</td>
      <td>...</td>
      <td>0.009620</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.007935</td>
      <td>-0.061097</td>
      <td>0.053037</td>
      <td>-0.019518</td>
      <td>-0.026126</td>
      <td>-0.005472</td>
      <td>0.253849</td>
    </tr>
    <tr>
      <th>AcceptedCmp4</th>
      <td>0.219633</td>
      <td>0.017520</td>
      <td>0.015933</td>
      <td>0.162722</td>
      <td>0.140163</td>
      <td>0.177705</td>
      <td>-0.028969</td>
      <td>-0.079814</td>
      <td>1.000000</td>
      <td>0.312597</td>
      <td>...</td>
      <td>-0.027030</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.015707</td>
      <td>0.070035</td>
      <td>0.249118</td>
      <td>-0.088427</td>
      <td>-0.076698</td>
      <td>-0.076936</td>
      <td>0.180032</td>
    </tr>
    <tr>
      <th>AcceptedCmp5</th>
      <td>0.395569</td>
      <td>0.000233</td>
      <td>-0.183837</td>
      <td>0.141428</td>
      <td>0.321522</td>
      <td>0.214249</td>
      <td>-0.276097</td>
      <td>0.080836</td>
      <td>0.312597</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.008378</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.004539</td>
      <td>-0.019025</td>
      <td>0.468695</td>
      <td>-0.284635</td>
      <td>-0.225671</td>
      <td>-0.346693</td>
      <td>0.324891</td>
    </tr>
    <tr>
      <th>AcceptedCmp1</th>
      <td>0.327524</td>
      <td>-0.021147</td>
      <td>-0.127586</td>
      <td>0.159100</td>
      <td>0.309130</td>
      <td>0.178462</td>
      <td>-0.195200</td>
      <td>0.095562</td>
      <td>0.242681</td>
      <td>0.409420</td>
      <td>...</td>
      <td>-0.025018</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.040137</td>
      <td>0.011941</td>
      <td>0.381354</td>
      <td>-0.230291</td>
      <td>-0.185711</td>
      <td>-0.279387</td>
      <td>0.297212</td>
    </tr>
    <tr>
      <th>AcceptedCmp2</th>
      <td>0.104036</td>
      <td>-0.001429</td>
      <td>-0.038064</td>
      <td>0.034722</td>
      <td>0.099931</td>
      <td>0.085146</td>
      <td>-0.007483</td>
      <td>0.071649</td>
      <td>0.295015</td>
      <td>0.222918</td>
      <td>...</td>
      <td>-0.011200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.006015</td>
      <td>0.007821</td>
      <td>0.136336</td>
      <td>-0.070037</td>
      <td>-0.059505</td>
      <td>-0.081575</td>
      <td>0.169234</td>
    </tr>
    <tr>
      <th>Complain</th>
      <td>-0.027900</td>
      <td>0.005713</td>
      <td>0.003744</td>
      <td>-0.013524</td>
      <td>-0.018675</td>
      <td>-0.011947</td>
      <td>0.020820</td>
      <td>0.009620</td>
      <td>-0.027030</td>
      <td>-0.008378</td>
      <td>...</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.035852</td>
      <td>0.004602</td>
      <td>-0.034135</td>
      <td>0.032181</td>
      <td>0.027081</td>
      <td>0.018124</td>
      <td>-0.000145</td>
    </tr>
    <tr>
      <th>Z_CostContact</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Z_Revenue</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Customer_For</th>
      <td>-0.023760</td>
      <td>0.025681</td>
      <td>0.217948</td>
      <td>0.192082</td>
      <td>0.097245</td>
      <td>0.111746</td>
      <td>0.275673</td>
      <td>-0.007935</td>
      <td>0.015707</td>
      <td>-0.004539</td>
      <td>...</td>
      <td>0.035852</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.014216</td>
      <td>0.158525</td>
      <td>-0.026131</td>
      <td>-0.027932</td>
      <td>0.001109</td>
      <td>0.196228</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.199977</td>
      <td>0.015694</td>
      <td>0.066156</td>
      <td>0.162265</td>
      <td>0.125856</td>
      <td>0.138998</td>
      <td>-0.120282</td>
      <td>-0.061097</td>
      <td>0.070035</td>
      <td>-0.019025</td>
      <td>...</td>
      <td>0.004602</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.014216</td>
      <td>1.000000</td>
      <td>0.115901</td>
      <td>0.092676</td>
      <td>0.078593</td>
      <td>-0.011841</td>
      <td>-0.020937</td>
    </tr>
    <tr>
      <th>Spent</th>
      <td>0.792740</td>
      <td>0.020479</td>
      <td>-0.065571</td>
      <td>0.529095</td>
      <td>0.780250</td>
      <td>0.675981</td>
      <td>-0.498769</td>
      <td>0.053037</td>
      <td>0.249118</td>
      <td>0.468695</td>
      <td>...</td>
      <td>-0.034135</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.158525</td>
      <td>0.115901</td>
      <td>1.000000</td>
      <td>-0.499931</td>
      <td>-0.424497</td>
      <td>-0.521603</td>
      <td>0.264443</td>
    </tr>
    <tr>
      <th>Children</th>
      <td>-0.343529</td>
      <td>0.018062</td>
      <td>0.436072</td>
      <td>-0.148938</td>
      <td>-0.443199</td>
      <td>-0.323823</td>
      <td>0.415558</td>
      <td>-0.019518</td>
      <td>-0.088427</td>
      <td>-0.284635</td>
      <td>...</td>
      <td>0.032181</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.026131</td>
      <td>0.092676</td>
      <td>-0.499931</td>
      <td>1.000000</td>
      <td>0.849574</td>
      <td>0.799802</td>
      <td>-0.167937</td>
    </tr>
    <tr>
      <th>Family_Size</th>
      <td>-0.286638</td>
      <td>0.014717</td>
      <td>0.373986</td>
      <td>-0.121879</td>
      <td>-0.372319</td>
      <td>-0.265916</td>
      <td>0.345316</td>
      <td>-0.026126</td>
      <td>-0.076698</td>
      <td>-0.225671</td>
      <td>...</td>
      <td>0.027081</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.027932</td>
      <td>0.078593</td>
      <td>-0.424497</td>
      <td>0.849574</td>
      <td>1.000000</td>
      <td>0.692370</td>
      <td>-0.218383</td>
    </tr>
    <tr>
      <th>Is_Parent</th>
      <td>-0.403132</td>
      <td>0.002189</td>
      <td>0.388593</td>
      <td>-0.073473</td>
      <td>-0.452734</td>
      <td>-0.284891</td>
      <td>0.475856</td>
      <td>-0.005472</td>
      <td>-0.076936</td>
      <td>-0.346693</td>
      <td>...</td>
      <td>0.018124</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001109</td>
      <td>-0.011841</td>
      <td>-0.521603</td>
      <td>0.799802</td>
      <td>0.692370</td>
      <td>1.000000</td>
      <td>-0.203885</td>
    </tr>
    <tr>
      <th>Response</th>
      <td>0.161387</td>
      <td>-0.200114</td>
      <td>0.003226</td>
      <td>0.151084</td>
      <td>0.219912</td>
      <td>0.035563</td>
      <td>-0.002625</td>
      <td>0.253849</td>
      <td>0.180032</td>
      <td>0.324891</td>
      <td>...</td>
      <td>-0.000145</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.196228</td>
      <td>-0.020937</td>
      <td>0.264443</td>
      <td>-0.167937</td>
      <td>-0.218383</td>
      <td>-0.203885</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>22 rows × 22 columns</p>
</div>



DATA PREPROCESSING


```python
#Importing requierd libraries for data preprocessing and model building
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
```


```python
#Encoding categorical variables
obj_att = data.dtypes[data.dtypes == 'O'].index.values
le = LabelEncoder()

for i in obj_att:
    data[i] = le.fit_transform(data[i])

data
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
      <th>Education</th>
      <th>Income</th>
      <th>Recency</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>NumCatalogPurchases</th>
      <th>NumStorePurchases</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>...</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Customer_For</th>
      <th>Age</th>
      <th>Spent</th>
      <th>Living_With</th>
      <th>Children</th>
      <th>Family_Size</th>
      <th>Is_Parent</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>58138.0</td>
      <td>58</td>
      <td>3</td>
      <td>8</td>
      <td>10</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>57283200000000000</td>
      <td>64</td>
      <td>1617</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>46344.0</td>
      <td>38</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>9763200000000000</td>
      <td>67</td>
      <td>27</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>71613.0</td>
      <td>26</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>10</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>26956800000000000</td>
      <td>56</td>
      <td>776</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>26646.0</td>
      <td>26</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>12009600000000000</td>
      <td>37</td>
      <td>53</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>58293.0</td>
      <td>94</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>13910400000000000</td>
      <td>40</td>
      <td>422</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>0</td>
      <td>61223.0</td>
      <td>46</td>
      <td>2</td>
      <td>9</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>32918400000000000</td>
      <td>54</td>
      <td>1341</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>1</td>
      <td>64014.0</td>
      <td>56</td>
      <td>7</td>
      <td>8</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>1641600000000000</td>
      <td>75</td>
      <td>444</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>0</td>
      <td>56981.0</td>
      <td>91</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>13</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>13392000000000000</td>
      <td>40</td>
      <td>1241</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>1</td>
      <td>69245.0</td>
      <td>8</td>
      <td>2</td>
      <td>6</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>13478400000000000</td>
      <td>65</td>
      <td>843</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>1</td>
      <td>52869.0</td>
      <td>40</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>11</td>
      <td>53740800000000000</td>
      <td>67</td>
      <td>172</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2212 rows × 24 columns</p>
</div>




```python
X = data.drop(['Response'], axis=1)    # Predictor(Independent) Feature columns
y = data['Response']   
```


```python
# Split X and y into training and test set in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)    # 1 is just any random seed number
```


```python
X_train.head()
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
      <th>Education</th>
      <th>Income</th>
      <th>Recency</th>
      <th>NumDealsPurchases</th>
      <th>NumWebPurchases</th>
      <th>NumCatalogPurchases</th>
      <th>NumStorePurchases</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>...</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Customer_For</th>
      <th>Age</th>
      <th>Spent</th>
      <th>Living_With</th>
      <th>Children</th>
      <th>Family_Size</th>
      <th>Is_Parent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1378</th>
      <td>1</td>
      <td>57420.0</td>
      <td>22</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>31881600000000000</td>
      <td>50</td>
      <td>322</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1027</th>
      <td>2</td>
      <td>15038.0</td>
      <td>93</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>44582400000000000</td>
      <td>34</td>
      <td>80</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>0</td>
      <td>57304.0</td>
      <td>61</td>
      <td>2</td>
      <td>7</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>41040000000000000</td>
      <td>70</td>
      <td>1026</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>641</th>
      <td>1</td>
      <td>76140.0</td>
      <td>57</td>
      <td>1</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>4060800000000000</td>
      <td>73</td>
      <td>1348</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>658</th>
      <td>1</td>
      <td>73059.0</td>
      <td>36</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>26092800000000000</td>
      <td>74</td>
      <td>1095</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_calc(x):

    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

    return(vif)
```

VARIANCE INFLATION FACTOR


```python
vif_calc(X)
```

    /Users/aytajmac/opt/anaconda3/lib/python3.9/site-packages/statsmodels/regression/linear_model.py:1715: RuntimeWarning: divide by zero encountered in double_scalars
      return 1 - self.ssr/self.centered_tss





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
      <th>variables</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Education</td>
      <td>0.929768</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Income</td>
      <td>1.146124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Recency</td>
      <td>0.840756</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NumDealsPurchases</td>
      <td>0.988268</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NumWebPurchases</td>
      <td>1.398741</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NumCatalogPurchases</td>
      <td>2.643914</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NumStorePurchases</td>
      <td>1.851529</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NumWebVisitsMonth</td>
      <td>0.828929</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AcceptedCmp3</td>
      <td>0.990723</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AcceptedCmp4</td>
      <td>1.067871</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AcceptedCmp5</td>
      <td>1.290990</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AcceptedCmp1</td>
      <td>1.184985</td>
    </tr>
    <tr>
      <th>12</th>
      <td>AcceptedCmp2</td>
      <td>1.019209</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Complain</td>
      <td>1.002931</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Z_CostContact</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Z_Revenue</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Customer_For</td>
      <td>1.298169</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Age</td>
      <td>0.507612</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Spent</td>
      <td>2.044609</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Living_With</td>
      <td>0.892283</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Children</td>
      <td>1.159040</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Family_Size</td>
      <td>0.715100</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Is_Parent</td>
      <td>1.048878</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Standardizing the data

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```


```python
#handling imbalanced data in the target variable
from collections import Counter
from sklearn.datasets import make_classification
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot

smote = SMOTE()

X_bal, y_bal = smote.fit_resample(X, y)

print(X_bal.shape)
print(y_bal.shape)
```

    (3758, 23)
    (3758,)


DATA PREPROCESSING


```python
#Logistic regression using statsmodel
import statsmodels.api as sm
log_m = sm.Logit(y_bal, X_bal)
log_f = log_m.fit()
print(log_f.summary())
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.361925
             Iterations: 35
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:               Response   No. Observations:                 3758
    Model:                          Logit   Df Residuals:                     3735
    Method:                           MLE   Df Model:                           22
    Date:                Sun, 12 Jun 2022   Pseudo R-squ.:                  0.4779
    Time:                        22:21:48   Log-Likelihood:                -1360.1
    converged:                      False   LL-Null:                       -2604.8
    Covariance Type:            nonrobust   LLR p-value:                     0.000
    =======================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Education              -0.5236      0.079     -6.588      0.000      -0.679      -0.368
    Income                1.93e-05   4.86e-06      3.974      0.000    9.78e-06    2.88e-05
    Recency                -0.0344      0.002    -17.245      0.000      -0.038      -0.030
    NumDealsPurchases       0.1512      0.034      4.500      0.000       0.085       0.217
    NumWebPurchases         0.0532      0.025      2.091      0.037       0.003       0.103
    NumCatalogPurchases    -0.0233      0.028     -0.824      0.410      -0.079       0.032
    NumStorePurchases      -0.3356      0.024    -13.988      0.000      -0.383      -0.289
    NumWebVisitsMonth       0.1231      0.037      3.307      0.001       0.050       0.196
    AcceptedCmp3            1.5130      0.194      7.784      0.000       1.132       1.894
    AcceptedCmp4            0.2150      0.218      0.987      0.324      -0.212       0.642
    AcceptedCmp5            0.4620      0.219      2.114      0.034       0.034       0.890
    AcceptedCmp1            0.6751      0.220      3.064      0.002       0.243       1.107
    AcceptedCmp2            0.9862      0.518      1.902      0.057      -0.030       2.002
    Complain               -0.7122      0.725     -0.983      0.326      -2.133       0.708
    Z_CostContact          -0.4657        nan        nan        nan         nan         nan
    Z_Revenue              -1.7076        nan        nan        nan         nan         nan
    Customer_For         5.191e-17   3.45e-18     15.047      0.000    4.51e-17    5.87e-17
    Age                    -0.0089      0.005     -1.981      0.048      -0.018   -9.43e-05
    Spent                   0.0012      0.000      7.084      0.000       0.001       0.002
    Living_With           -22.5095   4151.081     -0.005      0.996   -8158.478    8113.459
    Children              -21.1734   4151.081     -0.005      0.996   -8157.142    8114.795
    Family_Size            20.8954   4151.081      0.005      0.996   -8115.073    8156.864
    Is_Parent              -1.6310      0.186     -8.781      0.000      -1.995      -1.267
    =======================================================================================


    /Users/aytajmac/opt/anaconda3/lib/python3.9/site-packages/statsmodels/base/model.py:566: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "



```python
import time
#Logistic regression
start = time.time()
log_reg = LogisticRegression(solver='liblinear', random_state=123)
model = log_reg.fit(X_train, y_train)
y_pred = model.predict(X_test) 
end = time.time()
print('Runtime: ', end-start)
```

    Runtime:  0.016740083694458008



```python
LRM = LogisticRegression(solver='liblinear')
```


```python
#Fitting model to training data set
LRM.fit(X_train, y_train)
```




    LogisticRegression(solver='liblinear')




```python
#Accuracy of the Logistic model
logistic_training_predict = LRM.predict(X_train)

print('Logistic Regression Model In-Sample (Training Set) Accuracy: {0:.4f}'.format(metrics.accuracy_score(y_train, 
                                                                                            logistic_training_predict)))
print('')
```

    Logistic Regression Model In-Sample (Training Set) Accuracy: 0.8966
    



```python
#Accuracy of the Logistic Regression Model with test set
logistic_test_predict = LRM.predict(X_test)

LRM_accuracy = metrics.accuracy_score(y_test, logistic_test_predict)

print('Logistic Regression Model Out-Sample (Test Set) Accuracy: {0:.4f}'.format(LRM_accuracy))
print('')
```

    Logistic Regression Model Out-Sample (Test Set) Accuracy: 0.8765
    



```python
#Confusion Matrix for Actual v/s Predicted values
logistic_cm = metrics.confusion_matrix(y_test, logistic_test_predict, labels=[1,0])
print(logistic_cm)
```

    [[ 42  59]
     [ 23 540]]



```python
logistic_cm_df = pd.DataFrame(logistic_cm, index = [i for i in ["1","0"]], columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize=(7,5))
plt.title('Confusion Matrix for Logistic Regression Model', size=15)
sns.heatmap(logistic_cm_df, annot=True, fmt='g'); # fmt='g' (format) is used to get rid of scientific formats
```


    
![png](output_39_0.png)
    



```python
#Classification Report
print('Logistic Regression Model - Classification Report')
print('')
print(metrics.classification_report(y_test, logistic_test_predict, labels=[1,0]))
```

    Logistic Regression Model - Classification Report
    
                  precision    recall  f1-score   support
    
               1       0.65      0.42      0.51       101
               0       0.90      0.96      0.93       563
    
        accuracy                           0.88       664
       macro avg       0.77      0.69      0.72       664
    weighted avg       0.86      0.88      0.87       664
    



```python
#Create a SVM model
SVCM = SVC(gamma=0.025, C=3)
```

gamma is a measure of influence of a data point. It is inverse of distance of influence.
C is complexity of the model, lower C value creates simple hyperplane surfaces while higher C value creates complex surafce.


```python
#Fitting the model into training data set
SVCM.fit(X_train, y_train)
```




    SVC(C=3, gamma=0.025)




```python
#Performance of the SVC Model
svc_test_predict = SVCM.predict(X_test)

SVCM_accuracy = metrics.accuracy_score(y_test, svc_test_predict)

print('SVC Model Accuracy: {0:.4f}'.format(SVCM_accuracy))
print('')
```

    SVC Model Accuracy: 0.8886
    



```python
#Confusion Matrix of SVM model
svc_cm = metrics.confusion_matrix(y_test, svc_test_predict, labels=[1,0])
print(svc_cm)
```

    [[ 39  62]
     [ 12 551]]



```python
svc_cm_df = pd.DataFrame(svc_cm, index = [i for i in ["1","0"]], columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize=(7,5))
plt.title('Confusion Matrix for SVC Model', size=15)
sns.heatmap(svc_cm_df, annot=True, fmt='g');
```


    
![png](output_46_0.png)
    



```python
#Classification Report
print('SVC Model - Classification Report')
print('')
print(metrics.classification_report(y_test, svc_test_predict, labels=[1,0]))
```

    SVC Model - Classification Report
    
                  precision    recall  f1-score   support
    
               1       0.76      0.39      0.51       101
               0       0.90      0.98      0.94       563
    
        accuracy                           0.89       664
       macro avg       0.83      0.68      0.73       664
    weighted avg       0.88      0.89      0.87       664
    



```python
#Comparisons of the Models
models = ['Logistic Regression','SVM']
model_accuracy_scores = [LRM_accuracy,SVCM_accuracy]
comp_df = pd.DataFrame([model_accuracy_scores], index=['Accuracy'], columns=models)
comp_df
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
      <th>Logistic Regression</th>
      <th>SVM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Accuracy</th>
      <td>0.876506</td>
      <td>0.888554</td>
    </tr>
  </tbody>
</table>
</div>




```python
#FINAL
```
