### TITLE: Credit scoring analysis in distributed environment 
### AUTHOR: Pavlo Melnyk
### ADVISORS: dr Mariusz Rafało 
### DEPARTMENT: Econometrics, Statistics and Applied Economics 
### ACADEMIC YEAR: 2019-2020 
 
### LEGEND OF CODE COMMENTS (#) ### 1#: Code actions ### 2#: Optional or additional functions ### 3#: Notes and extra comments about the code 
 
########################################################################################################### 
 
# 1. IMPORTING NECESSARY LIBRARIES AND PACKAGES (THEY NEED TO BE INSTALLED BEFORE ANALYSIS)

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('classic')
%matplotlib inline
import os
import seaborn as sns
import os
import warnings
import sys

import xgboost as xgb

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
from fancyimpute import IterativeImputer as MICE

import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, DateType, FloatType
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, GBTClassifier, LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric

from distutils.version import LooseVersion

from xverse.transformer import WOE
from xverse.ensemble import VotingSelector


# 2. DATA PREPROCESSING AND VISUALIZATION

application = pd.read_csv(r"C:\...\application_train.csv", sep=';', low_memory = False)
 
application["TARGET"].isnull().sum()

application.select_dtypes("object")
application.select_dtypes(["float64",'int64'])

# 2.1 New Spark session creation and Spark SQL preprocessing

spark = SparkSession.builder.appName("mgr").getOrCreate()
sc = SparkContext.getOrCreate()

df = spark.read.format('csv') \
  .option("inferSchema", 'true') \
  .option("header", 'true') \
  .option("sep", ',') \
  .load(r'C:\Users\pm83635\Desktop\praca magisterska\home-credit-default-risk\home-credit-default-risk\df.csv')

df.select('SK_ID_CURR', 'TARGET').head(10)

df.groupBy("TARGET").agg({'AMT_CREDIT':'mean'}).show()

print(df.groupBy("TARGET").agg(avg('AMT_CREDIT'), avg('AMT_INCOME_TOTAL'), avg('AMT_GOODS_PRICE'))

df.printSchema()

df = df.withColumn("DAYS_EMPLOYED", when(col("DAYS_EMPLOYED")==365243, np.nan).otherwise(round(col("DAYS_EMPLOYED")))) #replace wrong data
df = df.withColumn('YEARS_OLD', round(col('DAYS_BIRTH')/(-365),1))
df = df.withColumn("YEARS_EMPLOYED", when(col("DAYS_EMPLOYED")==365243, np.nan).otherwise(round(col("DAYS_EMPLOYED")/(-365),1))) #replace wrong data
df = df.withColumn("ANNUITY_INCOME_PERCENTAGE", round(col('AMT_ANNUITY')/col('AMT_INCOME_TOTAL')))
df = df.withColumn("CREDIT_INCOME_PERCENTAGE", round(col('AMT_CREDIT')/col('AMT_INCOME_TOTAL')))
df = df.withColumn("CREDIT_DURATION", round(col('AMT_CREDIT')/col('AMT_ANNUITY')))
df = df.withColumn("EMP_AGE_PERCENTAGE", round(col('YEARS_EMPLOYED')/col('YEARS_OLD')))
df = df.withColumn("LOG_INCOME_TOTAL", log(col('AMT_INCOME_TOTAL')))
df = df.withColumn("LOG_ANNUITY", log(col('AMT_ANNUITY')))
df = df.withColumn("LOG_CREDIT", log(col('AMT_CREDIT')))
df = df.withColumn("LOG_YEARS_EMPLOYED", log(col('YEARS_EMPLOYED')))

cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'YEARS_OLD', 'CREDIT_DURATION']

stats = df.select(cols).describe().toPandas()
print(stats.to_latex())

#cross table spark
tex1 = df.stat.crosstab("TARGET", "CODE_GENDER")
print(tex1)

df.stat.crosstab("TARGET", "NAME_CONTRACT_TYPE").show()

print(df.stat.crosstab("TARGET", "NAME_CONTRACT_TYPE").toPandas().to_latex())

df.stat.crosstab("TARGET", "FLAG_OWN_REALTY").show()

print(df.stat.crosstab("TARGET", "FLAG_OWN_REALTY").toPandas().to_latex())

#pivot table
tex2 = df.groupBy("TARGET").pivot("NAME_EDUCATION_TYPE").mean('AMT_INCOME_TOTAL').toPandas()

print(tex2.to_latex())

#pivot table - only for few values of categories
statuses = ['Married' ,'Single / not married']
tex3 = df.groupBy("TARGET").pivot("NAME_FAMILY_STATUS", statuses).mean("AMT_INCOME_TOTAL").toPandas()

print(tex3.to_latex())

#SQL 
df_sql = df.createOrReplaceTempView("data")


sql_results = spark.sql("SELECT * FROM data")
sql_results.toPandas()

#top 10 credit amounts

spark.sql("SELECT DISTINCT AMT_CREDIT FROM data ORDER BY AMT_CREDIT DESC LIMIT 5").show()

#how many people took credit bigger than their income
spark.sql("SELECT COUNT(SK_ID_CURR) AS COUNT_CREDIT_MORE_INCOME FROM data WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM data WHERE AMT_CREDIT > AMT_INCOME_TOTAL)").show()

spark.sql("SELECT AMT_INCOME_TOTAL AS CREDIT_AMOUNT FROM data ORDER BY AMT_INCOME_TOTAL DESC LIMIT 10").show()

spark.sql("SELECT CNT_CHILDREN, COUNT(*) AS NUMBER_OF_OBS_DEF FROM data WHERE TARGET = 1 GROUP BY CNT_CHILDREN ORDER BY CNT_CHILDREN ASC").show()

spark.sql("SELECT CNT_CHILDREN, COUNT(*) AS NUMBER_OF_OBS_NODEF FROM data WHERE TARGET = 0 GROUP BY CNT_CHILDREN ORDER BY CNT_CHILDREN ASC").show()

spark.sql("SELECT CNT_CHILDREN, COUNT(*) AS NUMBER_OF_OBS_NODEF FROM data GROUP BY CNT_CHILDREN ORDER BY CNT_CHILDREN ASC").show()

#correlation matrix of chosen columns

def correlation_matrix(df, corr_columns, method='pearson'):
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=columns, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    matrix = Correlation.corr(df_vector, vector_col, method)

    result = matrix.collect()[0]["pearson({})".format(vector_col)].values
    return pd.DataFrame(result.reshape(-1, len(corr_columns)), columns=corr_columns, index=corr_columns)

df_corr = correlation_matrix(df_corr, columns)

df_corr = df_corr.select(cols).toPandas()
plt.figure(figsize=(8,8))
sns.heatmap(df_corr.corr(),
            vmin=-1,
            cmap='coolwarm',
            annot=True);   

# 2.2 Missing values imputation      

df_pandas = df.toPandas()
len(df_pandas.columns)

num_cols = df_pandas.select_dtypes(['float', 'int']).columns
obj_cols = df_pandas.select_dtypes('object').columns
float_cols = df_pandas.select_dtypes('float').columns
int_cols = df_pandas.select_dtypes('int').columns

df_pandas_num = df_pandas.select_dtypes(['float', 'int'])
len(df_pandas_num.columns)

df_pandas_obj = df_pandas.select_dtypes('object')
len(df_pandas_obj.columns)

#MICE IMPUTATION - for numerical

df_num_imputed = MICE().fit_transform(df_pandas_num)

#FFILL AND BFILL IMPUTATION - for object columns

df_obj_imputed = df_pandas_obj.ffill().bfill()

#renaming columns back to original names

df_num_imputed = pd.DataFrame(df_num_imputed)
df_num_imputed.columns = num_cols
len(df_num_imputed)

for col in int_cols:
    df_num_imputed[col] = df_num_imputed[col].astype('int')
df_num_imputed.dtypes

df_obj_imputed = pd.DataFrame(df_obj_imputed)
df_obj_imputed.columns = obj_cols
df_obj_imputed

df_final = pd.concat([df_num_imputed, df_obj_imputed], axis = 1)

df_final.head(100)

# 2.3 Missing values plot

app_null = application.drop(["TARGET"], axis=1).isnull().sum().sort_values(ascending = False)

percent = round(application.drop(["TARGET"], axis=1).isnull().sum()/len(application.drop(["TARGET"], axis=1))*100, ndigits=3)

missing_app = pd.concat([app_null, percent], axis = 1, keys = ["Count of missing values", "Percentage of missing values"], sort = False)

missing_app.head(10)

#missing_app["Percentage of missing values"].drop_duplicates()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 15
fig, ax = plt.subplots(figsize = (9, 5))
plt.plot(missing_app["Percentage of missing values"].drop_duplicates(), linewidth = 4)
plt.grid()
ax.xaxis.grid(color = 'grey', linestyle ='-')
plt.xticks(rotation = 90)
#plt.title('Percentage of missing values for variables')
plt.ylabel('Percents')
plt.show()
#missing_app.head(10)

# 2.4 Balanced data or not - Pie Chart

#data highly imbalanced
not_null_target = application[application["TARGET"].notnull() == True]["TARGET"]
yes = not_null_target[not_null_target == 1]
no = not_null_target[not_null_target == 0]

percent_yes = len(yes)/len(not_null_target)
percent_no = len(no)/len(not_null_target)

#labels = ['Yes', 'No']

colors = ['orange', 'cyan']

percents = [percent_yes, percent_no]

plt.rcParams['font.size'] = 15
fig, ax = plt.subplots(figsize = (8,8))
ax.pie(percents, colors = colors, autopct='%1.2f%%', startangle=70)
# Equal aspect ratio ensures that pie is drawn as a circle  
plt.title("Did client default?")
plt.tight_layout()
plt.legend(['Yes', 'No'])
plt.show();

# 2.5 Credit amount and goods prices density plot / logarithm of credit amount density plot

fig, ax = plt.subplots(figsize = (7,5))
sns.kdeplot(application['AMT_CREDIT'], shade=True, color="cyan")
sns.kdeplot(application['AMT_GOODS_PRICE'], shade=True, color="orangered")
#ax.set_title("Comparison of credit amounts and price of goods", fontsize = 25)
ax.set_xlabel("US Dollars", fontsize = 15)
ax.xaxis.set_major_locator(plt.MultipleLocator(200000))
ax.xaxis.set_minor_locator(plt.MultipleLocator(100000))
ax.set_xlim(0, 3000000)
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show();


# 2.6 Gender split pie chart and bar chart with split over default/not

#pie chart
realy_status = application['CODE_GENDER'].value_counts()
realty = round(realy_status[0]/sum(types_loans), ndigits=2)*100
norealty = round(realy_status[1]/sum(types_loans), ndigits=2)*100
group = [realty, norealty]


fig, ax = plt.subplots(figsize = (7,7))
ax.axis('equal')
mypie, _, _= ax.pie(group, radius=1, autopct = '%1.2f%%', colors = ['cyan', 'orange'])
plt.setp( mypie, width=0.7)
plt.legend(['F', 'M', 'XNA'], loc='best')
#plt.title('Percentage of realty in posession among applicants')
plt.show();

#bar chart

application2 = application[["CODE_GENDER", "TARGET"]].dropna()
bar = application2["CODE_GENDER"].value_counts()

bar_data1 = application2[application2["TARGET"]==1]["CODE_GENDER"].value_counts()
bar_data0 = application2[application2["TARGET"]==0]["CODE_GENDER"].value_counts()

bar1 = round(bar_data1/bar, ndigits=3)*100
bar0 = round(bar_data0/bar, ndigits=3)*100

bar_data = pd.concat([bar0, bar1], axis=0)
bar_data = pd.DataFrame(bar_data.fillna(0)).reset_index()

bar_data


yes_index = bar_data[bar_data['CODE_GENDER']<41].index
bar_data.loc[yes_index, 'Default'] = 'Yes'

no_index = bar_data[bar_data['CODE_GENDER']>41].index
bar_data.loc[no_index, "Default"] = 'No'


g = sns.catplot(x = "index", y = "CODE_GENDER", data=bar_data.sort_values(by = 'CODE_GENDER'),
            kind= 'bar', hue = 'Default', height=6, aspect= 0.9, palette='Set1', legend=False)
g.set_xticklabels(['XNA', "F", 'M'])
axes = g.axes.flatten()
#plt.title('Percentage of defaults by income type')
axes[0].set_xlabel("")
axes[0].set_ylabel("Percentage of paid/default loans")
axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))
axes[0].yaxis.set_minor_locator(plt.MultipleLocator(5))
axes[0].yaxis.grid(color = 'grey', linestyle ='-')
plt.tight_layout()
plt.legend(loc='best', title = 'Default')

# 2.7 Occupation type pie chart and bar chart with split over default/not
#pie chart

types_loans = application['NAME_CONTRACT_TYPE'].value_counts()
status = application['OCCUPATION_TYPE'].value_counts()

group = []
for i in range(len(status)):
    
    group.append(round(status[i]/sum(types_loans), ndigits=2)*100)    

fig, ax = plt.subplots(figsize = (8,8))
ax.axis('equal')
mypie, _, _= ax.pie(group[0:10], radius=1, autopct = '%1.2f%%', 
                    colors = ['red', 'orangered', 'cyan', 'yellow', 'green', 'aquamarine', 'orange', 'lightyellow', 'lightgreen', 'lightblue'])
plt.setp( mypie, width=0.7)
plt.legend([i for i in status.index], loc='upper right', prop={'size':9.2})
#plt.title('Percentage of applicants by family statuses')
plt.show();

#bar chart

application1 = application[["OCCUPATION_TYPE", "TARGET"]].dropna()
bar = application1["OCCUPATION_TYPE"].value_counts()

bar_data1 = application1[application1["TARGET"]==1]["OCCUPATION_TYPE"].value_counts()
bar_data0 = application1[application1["TARGET"]==0]["OCCUPATION_TYPE"].value_counts()

bar1 = round(bar_data1/bar, ndigits=3)*100
bar0 = round(bar_data0/bar, ndigits=3)*100

bar_data = pd.concat([bar0, bar1], axis=0, sort=True)
bar_data = pd.DataFrame(bar_data).reset_index()

yes_index = bar_data[bar_data['OCCUPATION_TYPE']<18].index
bar_data.loc[yes_index, 'Default'] = 'Yes'

no_index = bar_data[bar_data['OCCUPATION_TYPE']>18].index
bar_data.loc[no_index, "Default"] = 'No'

bar_data

g = sns.catplot(x = "index", y = "OCCUPATION_TYPE", data=bar_data.sort_values(by = 'OCCUPATION_TYPE'),
            kind= 'bar', hue = 'Default', height=7, aspect= 1.5, palette='Set1', legend=False)
g.set_xticklabels(rotation=45, horizontalalignment='right')

axes = g.axes.flatten()
#axes[0].set_title('Percentage of defaults by occupation types')
axes[0].set_xlabel("Name of occupation")
axes[0].set_ylabel("Percentage of paid/default loans")

axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))
axes[0].yaxis.set_minor_locator(plt.MultipleLocator(5))
axes[0].yaxis.grid(color = 'grey', linestyle ='-')
plt.tight_layout()
plt.legend(loc = 'best', title = 'Default')

# 2.8 Flag realty possession pie chart and bar chart with split over default/not

realy_status = application['FLAG_OWN_REALTY'].value_counts()
realty = round(realy_status[0]/sum(types_loans), ndigits=2)*100
norealty = round(realy_status[1]/sum(types_loans), ndigits=2)*100
group = [realty, norealty]


fig, ax = plt.subplots(figsize = (7,7))
ax.axis('equal')
mypie, _, _= ax.pie(group, radius=1, autopct = '%1.2f%%', colors = ['cyan', 'orange'])
plt.setp( mypie, width=0.7)
plt.legend(['Realty', 'No realty'], loc='best')
#plt.title('Percentage of realty in posession among applicants')
plt.show();

#bar chart

application2 = application[["FLAG_OWN_REALTY", "TARGET"]].dropna()
bar = application2["FLAG_OWN_REALTY"].value_counts()

bar_data1 = application2[application2["TARGET"]==1]["FLAG_OWN_REALTY"].value_counts()
bar_data0 = application2[application2["TARGET"]==0]["FLAG_OWN_REALTY"].value_counts()

bar1 = round(bar_data1/bar, ndigits=3)*100
bar0 = round(bar_data0/bar, ndigits=3)*100

bar_data = pd.concat([bar0, bar1], axis=0)
bar_data = pd.DataFrame(bar_data.fillna(0)).reset_index()

bar_data


yes_index = bar_data[bar_data['FLAG_OWN_REALTY']<41].index
bar_data.loc[yes_index, 'Default'] = 'Yes'

no_index = bar_data[bar_data['FLAG_OWN_REALTY']>41].index
bar_data.loc[no_index, "Default"] = 'No'


g = sns.catplot(x = "index", y = "FLAG_OWN_REALTY", data=bar_data.sort_values(by = 'FLAG_OWN_REALTY'),
            kind= 'bar', hue = 'Default', height=6, aspect= 0.9, palette='Set1', legend=False)
g.set_xticklabels(['Realty', 'No realty'])
axes = g.axes.flatten()
#plt.title('Percentage of defaults by income type')
axes[0].set_xlabel("")
axes[0].set_ylabel("Percentage of paid/default loans")
axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))
axes[0].yaxis.set_minor_locator(plt.MultipleLocator(5))
axes[0].yaxis.grid(color = 'grey', linestyle ='-')
plt.tight_layout()
plt.legend(loc='best', title = 'Default')

# 2.8 Family status pie chart and bar chart with split over default/not

#pie chart
status = application['NAME_FAMILY_STATUS'].value_counts()

group = []
for i in range(len(status)):
    
    group.append(round(status[i]/sum(types_loans), ndigits=2)*100)    

fig, ax = plt.subplots(figsize = (8,8))
ax.axis('equal')
mypie, _, _= ax.pie(group[0:5], radius=1, autopct = '%1.2f%%', colors = ['orange', 'cyan', 'yellow', 'lightgreen', 'lightblue'])
plt.setp( mypie, width=0.7)
plt.legend([i for i in status.index], loc='best', prop={'size':14}, title = 'Family status')
plt.title('Percentage of applicants by family statuses')
plt.show();

# bar chart

application2 = application[["NAME_FAMILY_STATUS", "TARGET"]].dropna()
bar = application2["NAME_FAMILY_STATUS"].value_counts()

bar_data1 = application2[application2["TARGET"]==1]["NAME_FAMILY_STATUS"].value_counts()
bar_data0 = application2[application2["TARGET"]==0]["NAME_FAMILY_STATUS"].value_counts()

bar1 = round(bar_data1/bar, ndigits=3)*100
bar0 = round(bar_data0/bar, ndigits=3)*100

bar_data = pd.concat([bar0, bar1], axis=0)
bar_data = pd.DataFrame(bar_data.fillna(0)).reset_index()


bar_data

yes_index = bar_data[bar_data['NAME_FAMILY_STATUS']<10].index
bar_data.loc[yes_index, 'Default'] = 'Yes'

no_index = bar_data[bar_data['NAME_FAMILY_STATUS']>10].index
bar_data.loc[no_index, "Default"] = 'No'


g = sns.catplot(x = "index", y = "NAME_FAMILY_STATUS", data=bar_data.sort_values(by = 'NAME_FAMILY_STATUS'),
            kind= 'bar', hue = 'Default', height=7, aspect= 1.1, palette='Set1', legend=False)
g.set_xticklabels(rotation=45, horizontalalignment='right')
axes = g.axes.flatten()
#plt.title('Percentage of defaults by family status')
axes[0].set_xlabel("Family status")
axes[0].set_ylabel("Percentage of paid/default loans")
axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))
axes[0].yaxis.set_minor_locator(plt.MultipleLocator(5))
axes[0].yaxis.grid(color = 'grey', linestyle ='-')
plt.tight_layout()
plt.legend(loc='best', title = 'Default')

# 2.9 Type of income pie chart and bar chart with split over default/not

#pie chart
status = application['NAME_INCOME_TYPE'].value_counts()

group = []
for i in range(len(status)):
    
    group.append(round(status[i]/sum(types_loans), ndigits=2)*100)    

fig, ax = plt.subplots(figsize = (8,8))
ax.axis('equal')
mypie, _, _= ax.pie(group[0:4], radius=1, autopct = '%1.2f%%', colors = ['orange', 'cyan', 'yellow', 'lightgreen'])
plt.setp( mypie, width=0.7)
plt.legend([i for i in status.index], loc='best', prop={'size':13})
#plt.title('Percentage of applicants by income types')
plt.show();

#bar chart

application2 = application[["NAME_INCOME_TYPE", "TARGET"]].dropna()
bar = application2["NAME_INCOME_TYPE"].value_counts()

bar_data1 = application2[application2["TARGET"]==1]["NAME_INCOME_TYPE"].value_counts()
bar_data0 = application2[application2["TARGET"]==0]["NAME_INCOME_TYPE"].value_counts()

bar1 = round(bar_data1/bar, ndigits=3)*100
bar0 = round(bar_data0/bar, ndigits=3)*100

bar_data = pd.concat([bar0, bar1], axis=0)
bar_data = pd.DataFrame(bar_data.fillna(0)).reset_index()

bar_data


yes_index = bar_data[bar_data['NAME_INCOME_TYPE']<41].index
bar_data.loc[yes_index, 'Default'] = 'Yes'

no_index = bar_data[bar_data['NAME_INCOME_TYPE']>41].index
bar_data.loc[no_index, "Default"] = 'No'


g = sns.catplot(x = "index", y = "NAME_INCOME_TYPE", data=bar_data.sort_values(by = 'NAME_INCOME_TYPE'),
            kind= 'bar', hue = 'Default', height=7, aspect= 1, palette='Set1', legend=False)
g.set_xticklabels(rotation=45, horizontalalignment='right')
axes = g.axes.flatten()
#plt.title('Percentage of defaults by income type')
axes[0].set_xlabel("")
axes[0].set_ylabel("Percentage of paid/default loans")
axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))
axes[0].yaxis.set_minor_locator(plt.MultipleLocator(5))
axes[0].yaxis.grid(color = 'grey', linestyle ='-')
plt.tight_layout()
plt.legend(loc='best', title = 'Default')

## 2.10 Types of education pie chart and bar chart with split over default/not

#pie chart

status = application['NAME_EDUCATION_TYPE'].value_counts()

group = []
for i in range(len(status)):
    
    group.append(round(status[i]/sum(types_loans), ndigits=2)*100)    

fig, ax = plt.subplots(figsize = (8,8))
ax.axis('equal')
mypie, _, _= ax.pie(group[0:4], radius=1, autopct = '%1.2f%%', colors = ['orange', 'cyan', 'yellow', 'lightgreen'])
plt.setp( mypie, width=0.7)
plt.legend([i for i in status.index], loc='best', prop={'size':14})
plt.title('Percentage of applicants by education levels')
plt.show();

#bar chart

application2 = application[["NAME_EDUCATION_TYPE", "TARGET"]].dropna()
bar = application2["NAME_EDUCATION_TYPE"].value_counts()

bar_data1 = application2[application2["TARGET"]==1]["NAME_EDUCATION_TYPE"].value_counts()
bar_data0 = application2[application2["TARGET"]==0]["NAME_EDUCATION_TYPE"].value_counts()

bar1 = round(bar_data1/bar, ndigits=3)*100
bar0 = round(bar_data0/bar, ndigits=3)*100

bar_data = pd.concat([bar0, bar1], axis=0)
bar_data = pd.DataFrame(bar_data.fillna(0)).reset_index()

bar_data


yes_index = bar_data[bar_data['NAME_EDUCATION_TYPE']<41].index
bar_data.loc[yes_index, 'Default'] = 'Yes'

no_index = bar_data[bar_data['NAME_EDUCATION_TYPE']>41].index
bar_data.loc[no_index, "Default"] = 'No'


g = sns.catplot(x = "index", y = "NAME_EDUCATION_TYPE", data=bar_data.sort_values(by = 'NAME_EDUCATION_TYPE'),
            kind= 'bar', hue = 'Default', height=7, aspect= 1, palette='Set1', legend=False)
g.set_xticklabels(rotation=45, horizontalalignment='right')
axes = g.axes.flatten()
#plt.title('Percentage of defaults by education')
axes[0].set_xlabel("Name of highest education level accomplished")
axes[0].set_ylabel("Percentage of paid/default loans")
axes[0].yaxis.set_major_locator(plt.MultipleLocator(10))
axes[0].yaxis.set_minor_locator(plt.MultipleLocator(5))
axes[0].yaxis.grid(color = 'grey', linestyle ='-')
plt.tight_layout()
plt.legend(loc='best', title = 'Default')

## 2.10 Credit rating region types pie chart and bar chart with split over default/not

#pie chart

status = application['REGION_RATING_CLIENT'].value_counts()
gr1 = round(status[1]/sum(types_loans), ndigits=2)*100
gr2 = round(status[2]/sum(types_loans), ndigits=2)*100
gr3 = round(status[3]/sum(types_loans), ndigits=2)*100
group = [gr1, gr2, gr3]


fig, ax = plt.subplots(figsize = (8,8))
ax.axis('equal')
mypie, _, _= ax.pie(group, radius=1, autopct = '%1.2f%%', colors = ['orange', 'cyan', 'yellow'])
plt.setp( mypie, width=0.7)
plt.legend(['Group 1', 'Group 2', 'Group 3'], loc='best')
plt.title('Percentage of applicants among scoring groups')
plt.show();

#bar chart

application2 = application[["REGION_RATING_CLIENT", "TARGET"]].dropna()
bar = application2["REGION_RATING_CLIENT"].value_counts()

bar_data1 = application2[application2["TARGET"]==1]["REGION_RATING_CLIENT"].value_counts()
bar_data0 = application2[application2["TARGET"]==0]["REGION_RATING_CLIENT"].value_counts()

bar1 = round(bar_data1/bar, ndigits=3)*100
bar0 = round(bar_data0/bar, ndigits=3)*100

bar_data = pd.concat([bar0, bar1], axis=0)
bar_data = pd.DataFrame(bar_data.fillna(0)).reset_index()

bar_data


yes_index = bar_data[bar_data['REGION_RATING_CLIENT']<41].index
bar_data.loc[yes_index, 'Default'] = 'Yes'

no_index = bar_data[bar_data['REGION_RATING_CLIENT']>41].index
bar_data.loc[no_index, "Default"] = 'No'


g = sns.catplot(x = "index", y = "REGION_RATING_CLIENT", data=bar_data.sort_values(by = 'REGION_RATING_CLIENT'),
            kind= 'bar', hue = 'Default', height=6, aspect= 1, palette='Set1', legend=False)
#g.set_xticklabels(rotation=45, horizontalalignment='right')
axes = g.axes.flatten()
#plt.title('Percentage of defaults by region rating (1,2,3)')
axes[0].set_xlabel("Rating number 1, 2 or 3")
axes[0].set_ylabel("Percentage of paid/default loans")
axes[0].yaxis.set_major_locator(plt.MultipleLocator(5))
axes[0].yaxis.set_minor_locator(plt.MultipleLocator(2.5))
axes[0].yaxis.grid(color = 'grey', linestyle ='-')
plt.tight_layout()
plt.legend(loc='best', title = 'Default')

# 2.11 Years of work distribution plot
fig, ax = plt.subplots(figsize = (10,6))
#plt.hist(application['DAYS_EMPLOYED'])

foo = application['DAYS_EMPLOYED'].value_counts().head() # we have 64648 records with days employed = 265243 - this is wrong data, need to be replaced

application['DAYS_EMPLOYED'] = application['DAYS_EMPLOYED'].replace(365243, np.nan)/(-365)

plt.hist(application['DAYS_EMPLOYED'], bins = 50, color = 'lightgreen')

#ax.set_title("Histogram of years of work of applicants", fontsize = 25)
ax.set_xlabel("Years of work", fontsize = 15)
ax.set_ylabel("Count of applicants", fontsize = 15)
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.grid()
ax.yaxis.grid(linestyle = '--')
plt.xticks(size = 15)
plt.yticks(size = 15)
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show()


# 2.12 Log income distribution

log = np.log(application['AMT_INCOME_TOTAL'])

plt.rcParams['font.size'] = 15
fig, ax = plt.subplots(figsize = (9,5))
plt.hist(application['AMT_INCOME_TOTAL'], bins = 4000, color = 'lightgreen')
#ax.set_title("Credit amount density plot", fontsize = 22)
ax.set_xlabel("US Dollars", fontsize = 15)
# ax.xaxis.set_major_locator(plt.MultipleLocator(200000))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(100000)
ax.set_xlim(0,1000000)
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.show();

# 3. MODELLING

# 3.1 Logistic regression - FROM DATABRICKS

df = sqlContext.sql('select * from df_final2')
df = df.toPandas()
df.head()

df = df.drop(['SK_ID_CURR', 'DAYS_REGISTRATION', 'EXT_SOURCE_2', 'AMT_GOODS_PRICE', 'AMT_ANNUITY'], axis=1) #collinear, not informative and wrongly typed columns
X = df.drop(['TARGET'], axis=1)
y = df['TARGET']

int_cols = ['TARGET', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
       'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21']

for col in int_cols:
    df[col] = df[col].astype('int')
df.dtypes

cols = df.select_dtypes(['float64']).columns
cols = np.asarray(cols)
df = spark.createDataFrame(df)
#cols.remove(["TARGET", 'SK_ID_CURR', 'CNT_CHILDERN', 'AMT_INCOME_TOTAL'])
# import the vector assembler

assembler = VectorAssembler(inputCols=cols,outputCol="features")
# transform method to transform our dataset
raw_data=assembler.transform(df)
raw_data.select("features").show(truncate=False)

#numerical features scaling

from pyspark.ml.feature import StandardScaler
standardscaler=StandardScaler().setInputCol("features").setOutputCol("scaled_features")
data=standardscaler.fit(raw_data).transform(raw_data)
data.select("features","scaled_features").show(10)

#train test split
train, test = data.randomSplit([0.7, 0.3], seed=42)

# adding weight column
data_size=float(train.select("TARGET").count())
num_zeroes=train.select("TARGET").where('TARGET == 1').count()
per_ones=(float(num_zeroes)/float(data_size))*100
num_ones=float(data_size-num_zeroes)
print('The number of ones are {}'.format(num_zeroes))
print('Percentage of ones are {}'.format(per_ones))

balance_ratio= num_ones/data_size
print('balance ratio = {}'.format(balance_ratio))

train=train.withColumn("classWeights", when(train.TARGET == 1,balance_ratio).otherwise(1-balance_ratio))
train.select("classWeights").show(5)

#modelling 

lr = LogisticRegression(labelCol="TARGET", featuresCol="features", weightCol="classWeights", maxIter=5, threshold = 0.617701)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
predict_test.select("TARGET","prediction",'rawPrediction', 'probability').show(100)

model_summary = model.summary

#recall, precision, ROC plots

%matplotlib inline
# Plot the threshold-recall curve
tr = model_summary.recallByThreshold.toPandas()
plt.figure(figsize =(8,6))
plt.plot(tr['threshold'], tr['recall'], color = 'g')
plt.xlabel('Threshold', size = 15)
plt.ylabel('Recall', size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.show()

# Plot the threshold-precision curve
tp = model_summary.precisionByThreshold.toPandas()
plt.figure(figsize =(8,6))
plt.plot(tp['threshold'], tp['precision'], color = 'g')
plt.xlabel('Threshold', size = 15)
plt.ylabel('Precision', size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.show()

# Plot the recall-precision curve
pr = model_summary.pr.toPandas()
plt.figure(figsize =(8,6))
plt.plot(pr['recall'], pr['precision'], color = 'g')
plt.xlabel('Recall', size = 15)
plt.ylabel('Precision', size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.show()


# Plot the threshold-F-Measure curve
fmeasure = model_summary.fMeasureByThreshold.toPandas()
plt.figure(figsize =(8,6))
plt.plot(fm['threshold'], fm['F-Measure'], color = 'g')
plt.xlabel('Threshold', size = 15)
plt.ylabel('F-1 Score', size = 15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.show()

#the best threshold
fmeasure.sort_values(by = 'F-Measure', ascending = False).head(1)

# Create 5-fold CrossValidator
Grid = ParamGridBuilder()\
    .addGrid(lr.aggregationDepth,[2, 7, 15])\
    .addGrid(lr.elasticNetParam,[0.0, 0.4, 1.0])\
    .addGrid(lr.fitIntercept,[False, True])\
    .addGrid(lr.maxIter,[10, 100, 500])\
    .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
    .addGrid(lr.threshold,[0.5, 0.617701, 0.92]) \
    .build()
cv = CrossValidator(estimator=lr, estimatorParamMaps=Grid, evaluator=evaluator, numFolds=5)
# Run cross validations
cvModel = cv.fit(train)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing
predict_train=cvModel.transform(train)
predict_test=cvModel.transform(test)
print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predict_test)))

#model with the best hyperparameters

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="TARGET", featuresCol="features", weightCol="classWeights", maxIter=200, aggregationDepth = 10, threshold = 0.617701, elasticNetParam = 0.8, fitIntercept = False)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
predict_test.select("TARGET","prediction",'rawPrediction', 'probability').show(100)

#WOE and IV applied

woe = WOE()
woe.fit(X, y)
woe_df = woe.woe_df # weight of evidence transformation dataset. This dataset will be used in making bivariate charts as well. 
#woe.iv_df #information value dataset
woe_df.sort_values('Information_Value', ascending = False).head(20)

clf = VotingSelector()
clf.fit(X, y)

clf.feature_importances_

features_voted = np.asarray(clf.feature_votes_.head(25)['Variable_Name'])

df_woe = X[features_voted]
woe = WOE()
woe.fit(df_woe, y)
woe_df = woe.woe_df

output_woe_bins = woe.woe_bins #future transformation
output_mono_bins = woe.mono_custom_binning  #future transformation
clf = WOE(woe_bins=output_woe_bins, mono_custom_binning=output_mono_bins) #output_bins was created earlier
out_X = clf.transform(X)

woe_df = pd.concat([out_X, y], axis = 1)

woedf = spark.createDataFrame(woe_df)
woedf.printSchema()
columnList = [item[0] for item in df.dtypes if item[1].startswith('double') or item[1].startswith('long')]

cols = [item[0] for item in woedf.dtypes if item[1].startswith('double') or item[1].startswith('long')]
cols = np.asarray(cols)
#df = spark.createDataFrame(df)
#cols.remove(["TARGET", 'SK_ID_CURR', 'CNT_CHILDERN', 'AMT_INCOME_TOTAL'])
# Let us import the vector assembler

assembler = VectorAssembler(inputCols=cols,outputCol="features")
# Now let us use the transform method to transform our dataset
raw_data=assembler.transform(woedf)
raw_data.select("features").show(truncate=False)

train, test = raw_data.randomSplit([0.7, 0.3], seed=12345)
train=train.withColumn("classWeights", when(train.TARGET == 1, 0.9193422410666964).otherwise(1-0.9193422410666964))
train.select("classWeights").show(5)

#model for WOE

lr = LogisticRegression(labelCol="TARGET", featuresCol="features", weightCol="classWeights", maxIter=200, aggregationDepth = 10, threshold = 0.617701, elasticNetParam = 0.8, fitIntercept = False)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
predict_test.select("TARGET","prediction",'rawPrediction', 'probability').show(100)

evaluator=BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',labelCol="TARGET")
predict_test.select("TARGET","rawPrediction","prediction","probability").show(5)
print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))

#confusion matrix plot for the best model

data = predict_test.select('TARGET', 'prediction').toPandas()

confusion_matrix = pd.crosstab(data['TARGET'], data['prediction'], rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize = (9,7))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', annot_kws={"size":15})
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel('Predicted', size = 15)
plt.ylabel('Actual', size = 15)
plt.legend(size = 15)
plt.show()


# 3.2 RANDOM FOREST 

#Model 1 - all variables

df = sqlContext.sql('select * from df_final2')
df = df.toPandas()
df.head()

df = df.drop(['SK_ID_CURR', 'DAYS_REGISTRATION', 'EXT_SOURCE_2', 'AMT_GOODS_PRICE', 'AMT_ANNUITY'], axis=1)
X = df.drop(['TARGET'], axis=1)
y = df['TARGET']

int_cols = ['TARGET', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
       'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21']

for col in int_cols:
    df[col] = df[col].astype('int')
df.dtypes

#voted columns to be used for columns chosen by IV (25 best variables)

# voted_cols = ['EXT_SOURCE_3', 'LOG_YEARS_EMPLOYED', 'NAME_EDUCATION_TYPE',
#        'ORGANIZATION_TYPE', 'OCCUPATION_TYPE', 'DAYS_EMPLOYED',
#        'LOG_INCOME_TOTAL', 'CREDIT_DURATION', 'FLOORSMAX_MODE',
#        'AMT_CREDIT', 'DAYS_ID_PUBLISH', 'TOTALAREA_MODE',
#        'DAYS_LAST_PHONE_CHANGE', 'YEARS_BEGINEXPLUATATION_MODE',
#        'YEARS_BEGINEXPLUATATION_MEDI', 'FLOORSMAX_AVG', 'YEARS_EMPLOYED',
#        'YEARS_OLD', 'DAYS_BIRTH', 'NAME_FAMILY_STATUS',
#        'REGION_POPULATION_RELATIVE', 'FLOORSMAX_MEDI',
#        'CREDIT_INCOME_PERCENTAGE', 'FLAG_PHONE',
#        'AMT_REQ_CREDIT_BUREAU_WEEK', 'TARGET']

# df = df[voted_cols]

# num_cols = ['EXT_SOURCE_3', 'LOG_YEARS_EMPLOYED', 'DAYS_EMPLOYED',
#        'LOG_INCOME_TOTAL', 'CREDIT_DURATION', 'FLOORSMAX_MODE', 'AMT_CREDIT',
#        'DAYS_ID_PUBLISH', 'TOTALAREA_MODE', 'DAYS_LAST_PHONE_CHANGE',
#        'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BEGINEXPLUATATION_MEDI',
#        'FLOORSMAX_AVG', 'YEARS_EMPLOYED', 'YEARS_OLD', 'DAYS_BIRTH',
#        'REGION_POPULATION_RELATIVE', 'FLOORSMAX_MEDI',
#        'CREDIT_INCOME_PERCENTAGE', 'FLAG_PHONE', 'AMT_REQ_CREDIT_BUREAU_WEEK']
# cat_cols = ['NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE', 'OCCUPATION_TYPE',
#        'NAME_FAMILY_STATUS']

stages = [] # stages in our Pipeline
for categoricalCol in cat_cols:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
        from pyspark.ml.feature import OneHotEncoderEstimator
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    else:
        from pyspark.ml.feature import OneHotEncoder
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

    # Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="TARGET", outputCol="label")
stages += [label_stringIdx]

# Transform all features into a vector using VectorAssembler
assemblerInputs = [c + "classVec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

df = spark.createDataFrame(df)  
partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(df)
preppedDataDF = pipelineModel.transform(df)

selectedcols = ["label", "features"] + voted_cols
dataset = preppedDataDF.select(selectedcols)
display(dataset)

(trainingData, testData) = preppedDataDF.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# Create an initial RandomForest model
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Train model with Training Data
rfModel = rf.fit(trainingData)

# Evaluate model
evaluator = BinaryClassificationEvaluator()
print(evaluator.evaluate(predictions))
print(evaluator.evaluate(predict_train))

#selected = predictions.select("label", "prediction", "probability").toPandas()
print(selected.head(10).to_latex())

# Cross validation
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6, 10])
             .addGrid(rf.maxBins, [20, 60, 100])
             .addGrid(rf.numTrees, [5, 20, 100])
             .addGrid(rf.impurity, ['gini', 'entropy'])
             .addGrid(rf.subsamplingRate, [0.2, 0.4, 0.7])
             .build())

 # Create 5-fold CrossValidator
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)

#model with chosen hyperparams
rf_cv = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees = 200, maxBins = 20, maxDepth = 10, impurity = 'gini', subsamplingRate = 0.4)

# Train model with Training Data
rfModel = rf_cv.fit(trainingData)

predictions = cvModel.transform(testData)

results = predictions.select(['probability', 'label'])
 
## prepare score-label set
results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)
 
metrics = metric(scoreAndLabels)
print("The ROC score is (numTrees=200): ", metrics.areaUnderROC)

#ROC curve plot

fpr = dict()
tpr = dict()
roc_auc = dict()
 
y_test = [i[1] for i in results_list]
y_score = [i[0] for i in results_list]
 
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
 
%matplotlib inline
plt.figure(figsize = (10,7))
plt.plot(fpr, tpr, label='AUC = 0.797', color = 'g')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size = 15)
plt.ylabel('True Positive Rate', size = 15)
plt.legend(loc="lower right")
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.show()

#confusion matrix


data = predictions.select('TARGET', 'prediction').toPandas()

confusion_matrix = pd.crosstab(data['TARGET'], data['prediction'], rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize = (9,7))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', annot_kws={"size":15})
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel('Predicted', size = 15)
plt.ylabel('Actual', size = 15)
plt.legend(size = 15)
plt.show()

# 3.3 XGBOOST MODEL

df = sqlContext.sql('select * from df_final2')
df = df.toPandas()
df.head()

X = df.drop(['DAYS_REGISTRATION', 'EXT_SOURCE_2'], axis=1)
y = df['TARGET']

#features from xgboost feature importance - to be used for the final model
# features = ['SK_ID_CURR', 'TARGET', 'EXT_SOURCE_3', 'AMT_REQ_CREDIT_BUREAU_DAY', 
#             'YEARS_BEGINEXPLUATATION_MEDI', 'FLOORSMAX_MEDI', 'EMP_AGE_PERCENTAGE', 
#             'DAYS_BIRTH', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'YEARS_BEGINEXPLUATATION_AVG', 
#             'DAYS_LAST_PHONE_CHANGE', 'NAME_EDUCATION_TYPE_Higher education', 'YEARS_OLD', 
#             'AMT_REQ_CREDIT_BUREAU_WEEK', 'DAYS_EMPLOYED', 'YEARS_EMPLOYED', 'LOG_YEARS_EMPLOYED', 
#             'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_INCOME_TOTAL', 'FLOORSMAX_AVG', 'REGION_POPULATION_RELATIVE', 
#             'DAYS_ID_PUBLISH', 'CODE_GENDER_F', 'NAME_EDUCATION_TYPE_Secondary / secondary special', 
#             'AMT_REQ_CREDIT_BUREAU_MON', 'CREDIT_INCOME_PERCENTAGE', 'AMT_CREDIT']

int_cols = ['SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_ID_PUBLISH',
       'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2',
       'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
       'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
       'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
       'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
       'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
       'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
       'FLAG_DOCUMENT_21']

for col in int_cols:
    df[col] = df[col].astype('int')
df.head()

#df = df.drop(['DAYS_REGISTRATION', 'EXT_SOURCE_2'], axis=1)
cat_cols = df.select_dtypes("object").columns
num_cols = df.select_dtypes('float').columns
#df = pd.get_dummies(df, dummy_na=False, columns=cat_cols, dtype=int)
df = df[num_cols]

x = df.values
x = StandardScaler().fit_transform(x)

feat_cols = ['norm_' + i for i in num_cols]
normalised_df = pd.DataFrame(x,columns=feat_cols)
normalised_df.tail()

# Split to train/test
train, test = train_test_split(df, size = 0.3)
train.dtypes

# Split to train/test
train, test = train_test_split(df, size = 0.3)
train.dtypes

dtrain = xgb.DMatrix(train.drop(['TARGET'], axis = 1), label=train["TARGET"])

#cross validation

model = XGBClassifier()
# define grid
weights = [1, 75, 92]
max_depth = [5, 15, 50, 80]
subsample = [0.6, 0.8, 0.9]
n_estimators = [10, 100, 500]
colsample_bytree = [0.2, 0.3, 0.4,]
grow_policy = ['depthwise', 'lossguide']
gamma = [0, 1, 5]
lamda = [0, 1, 5]
alpha = [0, 1, 5]
param_grid = dict(scale_pos_weight=weights, max_depth = max_depth, subsample = subsample, n_estimators = n_estimators, colsample_bytree = colsample_bytree, grow_policy = grow_policy, gamma = gamma, lamda = lamda, alpha = alpha)
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
# execute the grid search
grid_result = grid.fit(X, y)
# report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#set of the best hyperparams
param = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'binary:logistic',
    'max_depth': 4,
    'num_parallel_tree': 10,
    'min_child_weight': 1,
    'nrounds': 200,
    'scale_pos_weight': 92,
    'colsample_bytree': 0.7,
    'grow_policy': 'lossguide',
    'alpha': 1,
    'gamma': 1,
    'lambda': 1  
}

num_round = 10
bst = xgb.train(param, dtrain, num_round)

dtest = xgb.DMatrix(test.drop(['TARGET'], axis = 1))
ypred = bst.predict(dtest)

dtest_train = xgb.DMatrix(train.drop(['TARGET'], axis = 1))
ypred_train = bst.predict(dtest_train)

pre_score_test = roc_auc_score(test["TARGET"],ypred, average='micro')
pre_score_train = roc_auc_score(train["TARGET"],ypred_train, average='micro')
print("xgb_pre_score:", pre_score_test)
print("xgb_pre_score_train:", pre_score_train)

#feature importance
%matplotlib inline

xgb.plot_importance(bst)
plt.rcParams['figure.figsize'] = [23, 15]
plt.show()

#confusion matrix

ypred = np.where(ypred > 0.7327, 1, 0)
from sklearn.metrics import confusion_matrix
confusion_matrix(test['TARGET'], ypred)

#PCA analysis

pca = PCA(n_components=7)
principalComponents = pca.fit_transform(x)

principal_df = pd.DataFrame(data = principalComponents, columns = ["principal component_" + str(i) for i in range(7)])
df = pd.concat([y.astype('int'), principal_df], axis = 1)

#PCA plot

plt.figure()
plt.figure(figsize=(14,8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
#plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = df['TARGET'] == target
    plt.scatter(df.loc[indicesToKeep, 'principal component_1']
               , df.loc[indicesToKeep, 'principal component_2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15}, labels =['Paid', 'Default'])

#Explained variance plot

percentage_var_explained = pca.explained_variance_ratio_  
cum_var_explained=np.cumsum(percentage_var_explained)
#plot PCA spectrum   
plt.figure(2,figsize=(15,8))
plt.clf()  
plt.plot(cum_var_explained,linewidth=2)  
plt.axis('tight')  
plt.grid() 
plt.xlabel('n_components', fontsize = 17) 
plt.ylabel('Cumulative Variance explained',fontsize = 17) 
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()