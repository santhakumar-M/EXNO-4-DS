# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df1=pd.read_csv("/content/bmi.csv")
df2=pd.read_csv("/content/bmi.csv")
df3=pd.read_csv("/content/bmi.csv")
```
```
df.head()
```
![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/3e7d2757-b99b-4945-b0b2-ea3cb0e192ca)

```
df.dropna()
```
![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/c1f77c6d-7df0-423d-b398-0b822677d143)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/3642e1c1-bd8d-4eed-aa65-49f9d3664f4c)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/cafe5fd8-9920-4b20-871f-8f8d4b0ee58c)

```
from sklearn.preprocessing import MinMaxScaler
```
```
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
```
```
df.head(10)
```
![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/7e017bad-f460-4b95-ac17-14aa382ed757)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/7474579a-2e03-43e5-8c99-eeeaa3dfb322)

```
df3=pd.read_csv("/content/bmi.csv")

```
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height',' Weight']]=scaler.fit_transform(df3[['Height','Weight']])
```
```
df3
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/37488a19-329f-4e39-9f90-778095bd2eac)

```
df4=pd.read_csv("/content/bmi.csv")
df4=pd.read_csv("/content/bmi.csv")
```
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/1aafe685-6064-4556-a915-394edf79e564)
```
import pandas as pd
import numpy as np
import seaborn as sns

```
```

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])

```
```
data
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/c3ed3130-a793-4f40-a073-d388de64e632)

```
data.isnull().sum()
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/04df89f6-aef8-4fb8-8775-259cc577ae2a)

```
missing=data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/5a0b4705-6443-4cb7-b370-d3fdb96db1f3)
```
data2=data.dropna(axis=0)
data2
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/55a73bb4-aac1-485a-90ad-326042a2ba75)

```
sal=data['SalStat']
```
```
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/7764dccd-3b5c-458a-8bfb-f22b1686c11a)

```
sal2=data2['SalStat']
```
```
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/47ae55c3-378d-4e8c-b2ea-531931d4bf87)
```
data2

```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/579f3c23-2b9d-407a-9289-41865e8aeac3)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/2751846c-a9e6-46df-a1ac-ae7298bbb1d4)
```
columns_list=list(new_data.columns)
print(columns_list)
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/1d45d9bf-3193-4513-bc9f-6c35cad4e9e3)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/ead8911d-5118-4393-bba4-5bef2b3c6011)

```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/f6f07c62-c76e-4aed-a031-79c89708da50)

```
x=new_data[features].values
print(x)
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/5f0c17cf-3ba0-4546-aeee-f6c571df1d22)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
```
```
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]

}
```
```
df=pd.DataFrame(data)
```
```
df
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/f511a6b4-35bf-44cf-8896-f3c96625e92e)
```
X=df[['Feature1','Feature3']]
y=df[['Target']]
```
```
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/8a38b485-44aa-497e-88a8-a5605ac55af5)

```
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
```
```
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/832abf1d-e6f8-4b2b-a232-fbcc2b013d82)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
```
```
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/37cffe5f-d3a5-4ce5-9acc-e3342c25cea6)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/f1ab3509-670a-433c-bff8-5189ff69f719)
```
chi2, p, _, _=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```

![image](https://github.com/Jeevapriya14/EXNO-4-DS/assets/121003043/5759f4ac-0edd-4662-90cc-8ca68d333372)


# RESULT:
       Thus, Feature selection and Feature Scaling has been used on the given dataset.
