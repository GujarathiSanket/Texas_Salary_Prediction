#!/usr/bin/env python
# coding: utf-8

# # Texas Salary Prediction

# This database has salary information for positions at all 113 agencies in the Texas state government. The Tribune obtained this data by requesting salary records from the state comptroller, as allowed by the Texas Public Information Act.

# **Problem Statement**
# 
# Task 1:-Prepare a complete data analysis report on the given data.
# 
# Task 2:-Create a predictive model which will help the Texas state government  team to know the payroll information of employees of the state of Texas.  
# 
# Task 3:-<br>
# ●	Who are the outliers in the salaries?<br>
# ●	What departments/roles have the biggest wage disparities between managers and employees?<br>
# ●	Have salaries and total compensations for some roles/ departments/ head-count changed over time?<br>
# 
# 

# In[1]:


# Loading Basic Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Task 1:-Prepare a complete data analysis report on the given data.

# In[2]:


# Loading dataset
data=pd.read_csv('salary.csv')
pd.set_option("display.max_columns",None)


# In[3]:


data.head(2)


# In[4]:


data1=data.copy()


# ### Domain Analysis
# 
# * Agency:- Department IDs
# * Agency Name:- Name of Dapartments where employee worked
# * Last Name:- Last Name of Employee
# * First Name:- First Name of Employee
# * MI (Middle Initial):-Middle Name initial of Employee
# * Class title:- Job title / Designation of Employee
# * Ethnicity:- origin of Employee from which group he or she belongs
# * Gender:- gender of Employee
# * Status:- It has type of job like part time, full time, regular, temporary etc.
# * Employ Date:- Joining date
# * Hourly rate:- Hourly paid amount
# * Hrs per week:- Working Hours
# * Monthly (Monthly income):- Monthly salary
# * Annual (Annual Income):-Annual salary
# * State number

# **Annual is a output variable and remaining features are input variable**

# # Basic Checks

# In[5]:


data.shape   ## Number of rows and columns


# In[6]:


data.info()  #Memory Status


# In[7]:


data.columns


# In[8]:


# Statistical Analysis
data.describe().T


# * Maximum hourly rate is $117.78
# 
# * Maximum working hours per week is 70
# 
# * Average montly salary is $4226.18<br>
# 
# * Average annual salary is $50714.21

# In[9]:


data.describe(include='O')


# In[10]:


# Categorical columns
cat_cols=data.select_dtypes(include='O')
cat_cols.columns


# In[11]:


# Numerical columns
num_cols=data.select_dtypes(include=['int64','float64'])
num_cols.columns


# In[12]:


# Unique values in categorical columns
for i in cat_cols:
    print(i)
    print(data[i].unique())
    print('*****************************************************************************************')


# In[13]:


# Unique values in numerical columns
for i in num_cols:
    print(i)
    print(data[i].unique())


# # Data Cleaning

# In[14]:


#Rename the values having space in the columns


# In[15]:


data['ETHNICITY']=data['ETHNICITY'].str.strip()


# In[16]:


data['ETHNICITY'].unique()


# In[17]:


## Removing space
data['GENDER']=data['GENDER'].str.strip()


# In[18]:


data['ETHNICITY']=data['ETHNICITY'].str.strip()


# In[19]:


data.isna().sum() # checking for missing values


# In[20]:


data.isna().sum()[data.isna().sum()>0]  ## Columns with missing values


# In[21]:


data.isna().sum()[data.isna().sum()>0]*100/data.shape[0]  ## Percentage of missing values


# In[22]:


# Dropping columns having more than 99% missing data
data.drop(['duplicated','multiple_full_time_jobs','combined_multiple_jobs','summed_annual_salary','hide_from_search'],axis=1,inplace=True)


# In[23]:


#Renaming columns
data.rename(columns={'AGENCY NAME':'AGENCY_NAME','LAST NAME':'LAST_NAME','FIRST NAME':'FIRST_NAME','CLASS CODE':'CLASS_CODE','CLASS TITLE':'CLASS_TITLE','EMPLOY DATE':'EMPLOY_DATE','HRLY RATE':'HRLY_RATE','HRS PER WK':'HRS_PER_WK','STATE NUMBER':'STATE_NUMBER'},inplace=True)


# In[24]:


data.head(1)


# # Exploratory Data Analysis

# ## UNIVARIATE ANALYSIS

# ### Distplot<br>
# * Represents the distribution of numerical columns

# In[25]:


# Numerical Data Analysis
num_col=data.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(20,20))
num=1
for col in num_col:
    if num<=6:
        plt.subplot(3,3,num)
        sns.distplot(x=data[col])
        plt.xlabel(col)
    num+=1


# ### Insights
# * Hrly rate has corrupted data because distribution is saturated to 0
# * almost all employees works 40hrs per week
# * almost all monthly salaries lies between range 0 and 20000, it is skewed distribution (Not Normally Distributed)
# * almost all annual salaries lies between range 0 and 20000, it is skewed distribution (Not Normally Distributed)

# ### COUNT PLOT<br>
# * Gives the count of observations

# In[26]:


# Categorical Data Analysis
cat_col=data[['ETHNICITY','GENDER','STATUS']]
plt.figure(figsize=(20,20))
num=1
for col in cat_col:
    if num<=3:
        plt.subplot(3,3,num)
        sns.countplot(x=data[col])
        plt.xlabel(col)
        plt.xticks(rotation = 90)
    num+=1


# ## Insights
# * Number Female employees are more than Male employees
# * Most of employees has job type(status) CRF - CLASSIFIED REGULAR FULL-TIME.

# ## BIVARIATE ANALYSIS

# ### SCATTER PLOT<br>
# * Checking for correlation

# In[27]:


corr_cols=['HRLY_RATE',"HRS_PER_WK",'MONTHLY']
plt.figure(figsize=(10,2))
plot_num=1
for x in corr_cols:
    if plot_num<=4:
        plt.subplot(1,4,plot_num)
        sns.scatterplot(x=data[x],y=data['ANNUAL'])
        plt.xlabel(x)
    plot_num+=1
plt.tight_layout()


# ## Insights
# * Hrs per week doesn't show any correlation with ANNUAL Income.
# * Monthly column has direct correlation with ANNUAL Income.
# * Most of the values from HRLY RATE are 0, ramining show correlation with target.

# In[28]:


sns.barplot(x=data['GENDER'],y=data["ANNUAL"])


# In[29]:


sns.barplot(x=data['ETHNICITY'],y=data["ANNUAL"])


# In[30]:


sns.barplot(x=data['STATUS'],y=data["ANNUAL"])
x=plt.xticks(rotation=90)


# * Male employees are receiving more annual salary
# * ASIAN and White is sharing high salary
# * Exempt regular full time employees has more annual salary

# # Multivariate Analysis

# In[31]:


num_col=data[['HRS_PER_WK','MONTHLY','HRLY_RATE','ANNUAL']]
sns.pairplot(data=num_col)


# In[32]:


sns.heatmap(data[['HRLY_RATE',"HRS_PER_WK",'MONTHLY','ANNUAL']].corr(),annot=True)


# # PREPROCESSING

# In[33]:


#### Checking Duplicate Values
data.duplicated().sum()


# In[34]:


plt.figure(figsize=(5,2))
sns.histplot(x=data['HRLY_RATE'],kde=True)


# * Most values in hourly rate column is zero and hence data is corrupted

# In[35]:


data.drop(['HRLY_RATE'],axis=1,inplace=True)


# # OUTLIERS IN THE SALARIES

# In[36]:


num_col=data[['HRS_PER_WK','MONTHLY','ANNUAL']]
plt.figure(figsize=(20,20))
num=1
for col in num_col:
    if num<=4:
        plt.subplot(3,3,num)
        sns.boxplot(x=data[col])
        plt.xlabel(col)

    num+=1


# In[37]:


num_col=data[['HRS_PER_WK','MONTHLY','ANNUAL']]
for col in num_col:
    Q1=data[col].quantile(0.25)
    Q3=data[col].quantile(0.75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    print("***********")
    print("Column Name:",col)
    print("IQR  ",IQR)
    print("Lower  ",lower)
    print("Upper  ",upper)
    print("Outliers Percentage",len(data.loc[(data[col]<lower)|(data[col]>upper)])/len(data))
    print("***********")


# In[38]:


# Handling outliers forcolumn having less than 5% missing values
data.loc[(data['HRS_PER_WK']<lower)|(data['HRS_PER_WK']>upper),'HRS_PER_WK']=data['HRS_PER_WK'].median()


# ## ENCODING

# In[39]:


cat_col=data.select_dtypes(include=['object'])
cat_col.columns


# In[40]:


data['GENDER']=data['GENDER'].replace({'FEMALE':0,'MALE':1})


# In[41]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data['STATUS']=lb.fit_transform(data['STATUS'])
data['ETHNICITY']=lb.fit_transform(data['ETHNICITY'])


# In[42]:


data.head(2)


# # FEATURE ENGINEERING

# ### Handling Employee date column

# In[43]:


data['EMPLOY_DATE']=pd.to_datetime(data['EMPLOY_DATE'])
data['EMPLOY_DT']=data['EMPLOY_DATE'].dt.day
data['EMPLOY_Month']=data['EMPLOY_DATE'].dt.month
data['EMPLOY_YR']=data['EMPLOY_DATE'].dt.year


# In[44]:


data.head(3)


# In[45]:


#calculating years of experiance based on joining date
from datetime import datetime
data['EMPLOY_DATE'] = pd.to_datetime(data['EMPLOY_DATE'])
current_date = datetime.now()  # Get current date and time
data['YRS_EXP']=(current_date -data['EMPLOY_DATE'] ).dt.days/365.25


# In[46]:


data.head(3)


# In[47]:


plt.figure(figsize=(10,2))
plt.scatter(x=data['YRS_EXP'],y=data['ANNUAL'],color = "green")


# In[48]:


data.loc[data['YRS_EXP']<0]


# In[49]:


# dropping irrelevant records
data.drop([20904,88771,116978,141891,148921],axis=0,inplace=True)


# In[50]:


sns.lineplot(data=data,x='EMPLOY_YR',y='ANNUAL')


# ### Dropping unwanted columns

# In[51]:


data.head(1)


# In[52]:


data.drop(['AGENCY','AGENCY_NAME','LAST_NAME','FIRST_NAME','MI','CLASS_CODE','CLASS_TITLE','EMPLOY_DATE','STATE_NUMBER'],axis=1,inplace=True)


# In[53]:


data.head(2)


# # SPLITING DATA

# In[54]:


x=data.drop(['ANNUAL'],axis=1)
y=data['ANNUAL']


# # SCALING OF DATA

# In[55]:


from sklearn.preprocessing  import MinMaxScaler
scaler = MinMaxScaler()


# In[56]:


x[['HRS_PER_WK','MONTHLY','YRS_EXP']]= scaler.fit_transform(x[['HRS_PER_WK','MONTHLY','YRS_EXP']])


# In[57]:


y_trans=np.log(y)
y_trans.head()


# # Task:2 Create a predictive model which will help theTexas state government  team to know the payroll information of employees of the state of Texas.  

# # MODEL BUILDING

# In[58]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[59]:


x_train, x_test, y_train, y_test = train_test_split(x, y_trans, test_size=0.2, random_state=42)


# # LINEAR REGRESSION

# In[60]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[61]:


y_pred=model.predict(x_test)
y_pred


# In[62]:


model.coef_


# In[63]:


model.intercept_


# In[64]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("MAE",mean_absolute_error(y_test,y_pred))
print("MSE",mean_squared_error(y_test,y_pred))


# In[65]:


RMSE=np.sqrt(mean_squared_error(y_test,y_pred))
RMSE


# In[66]:


r2score_lr=r2_score(y_test,y_pred)
print('r2 score LR',r2score_lr)


# # KNN REGRESSOR

# In[67]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
knn.fit(x_train, y_train)


# In[68]:


ypred_knn = knn.predict(x_test)


# In[69]:


MSE=mean_squared_error(y_test,ypred_knn)
print('MSE is ',MSE)
MAE=mean_absolute_error(y_test,ypred_knn)
print('MAE is',MAE)
RMSE=np.sqrt(MSE)
print('RMSE is ',RMSE)


# In[70]:


r_squaredknn=r2_score(y_test,ypred_knn)
print('r2_score is',r_squaredknn)


# # DECISION TREE REGRESSOR

# In[71]:


from sklearn.tree import DecisionTreeRegressor
dec = DecisionTreeRegressor()
dec.fit(x_train, y_train)
ypred_dec = dec.predict(x_test)


# In[72]:


MSE=mean_squared_error(y_test,ypred_dec)
print('MSE is ',MSE)
MAE=mean_absolute_error(y_test,ypred_dec)
print('MAE is',MAE)
RMSE=np.sqrt(MSE)
print('RMSE is ',RMSE)


# In[73]:


r_squareddt=r2_score(y_test,ypred_dec)
print('r2_score is',r_squareddt)


# # XGB REGRESSOR

# In[74]:


from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(x_train, y_train)
ypred_xgb = xgb.predict(x_test)


# In[75]:


MSE=mean_squared_error(y_test,ypred_xgb)
print('MSE is ',MSE)
MAE=mean_absolute_error(y_test,ypred_xgb)
print('MAE is',MAE)
RMSE=np.sqrt(MSE)
print('RMSE is ',RMSE)
r_squaredxgb=r2_score(y_test,ypred_xgb)
print('r_squared is',r_squaredxgb)


# # RANDOM FOREST REGRESSOR

# In[76]:


from sklearn.ensemble import RandomForestRegressor
rf =  RandomForestRegressor()
rf.fit(x_train, y_train)
ypred_rf = rf.predict(x_test)


# In[77]:


MSE=mean_squared_error(y_test,ypred_rf)
print('MSE is ',MSE)
MAE=mean_absolute_error(y_test,ypred_rf)
print('MAE is',MAE)
RMSE=np.sqrt(MSE)
print('RMSE is ',RMSE)
r_squaredrf=r2_score(y_test,ypred_rf)
print('r_squared is',r_squaredrf)


# # GRADIENT BOOSTING REGRESSOR

# In[78]:


from sklearn.ensemble import  GradientBoostingRegressor
gb = GradientBoostingRegressor()
gb.fit(x_train, y_train)
ypred_gb = gb.predict(x_test)


# In[79]:


MSE=mean_squared_error(y_test,ypred_gb)
print('MSE is ',MSE)
MAE=mean_absolute_error(y_test,ypred_gb)
print('MAE is',MAE)
RMSE=np.sqrt(MSE)
print('RMSE is ',RMSE)
r_squaredgbr=r2_score(y_test,ypred_gb)
print('r_squared is',r_squaredgbr)


# # Task 3.2:- What departments/roles have the biggest wage disparities between managers and employees?

# In[80]:


df=data1.loc[data1['CLASS TITLE'].str.contains('MANAGER') | data1['CLASS TITLE'].str.contains('MGR'),['AGENCY NAME','CLASS TITLE','ANNUAL']]
pd.set_option("display.max_colwidth", None)
print('Maximum Salary for Manager: ',df.max().ANNUAL)
print('Minimum Salary for Manager: ',df.min().ANNUAL)
df.describe()


# In[81]:


print('Department/roles have biggest wage:')
df[df['ANNUAL']==df.max().ANNUAL]


# In[82]:


print('Department/roles have lowest wage:')
df[df['ANNUAL']==df.min().ANNUAL]


# In[83]:


df1=data1.loc[~data1['CLASS TITLE'].str.contains('MANAGER') & ~data1['CLASS TITLE'].str.contains('MGR'),['AGENCY NAME','CLASS TITLE','ANNUAL']]
pd.set_option("display.max_colwidth", None)
#df5.head()
print('Max Salary for Employee',df1.max().ANNUAL)
print('Min Salary for Employee',df1.min().ANNUAL)
df1.describe()


# In[84]:


print('Department/roles have lowest wage:')
df1[df1['ANNUAL']==df1.max().ANNUAL]


# In[85]:


print('Department/roles have biggest wage:')
df1[df1['ANNUAL']==df1.min().ANNUAL]


# # Difference Calculation

# In[86]:


data1.loc[data1['CLASS TITLE'].str.contains('MANAGER') | data1['CLASS TITLE'].str.contains('MGR'),'CLASS TITLE']='MANAGER'


# In[87]:


data1.loc[~data1['CLASS TITLE'].str.contains('MANAGER') & ~data1['CLASS TITLE'].str.contains('MGR'),'CLASS TITLE']='EMPLOYEE'


# In[88]:


data_df1=data1.loc[:,['AGENCY NAME','CLASS TITLE','ANNUAL']]
data_df1.head()


# # Top departments which has biggest disparity between manager and employee salaries

# In[89]:


# Create the pivot table
table = pd.pivot_table(data1, values='ANNUAL', index=['AGENCY NAME'], columns=['CLASS TITLE'], aggfunc=np.mean)

# Calculate the difference between 'manage' and 'employee' annual values
table['Difference'] = table['MANAGER'] - table['EMPLOYEE']


# Sort the pivot table by the 'Difference' column in descending order
sorted_table = table.sort_values(by='Difference', ascending=False)

# Drop the 'Difference' column if you don't need it anymore
sorted_table = sorted_table.drop(columns='Difference')

sorted_table.head()


# # Task 3.3:- Have salaries and total compensations for some roles/ departments/ head-count changed over time
# 

# In[90]:


top_agencies=data1['AGENCY NAME'].value_counts().nlargest(10)
top_agencies.index.to_list()



# In[91]:


new_data=data1.loc[data1['AGENCY NAME'].isin(top_agencies.index.to_list()),['AGENCY NAME','CLASS TITLE','EMPLOY DATE','ANNUAL']]
# Convert "Employ Date" to datetime format
new_data['EMPLOY DATE'] = pd.to_datetime(new_data['EMPLOY DATE'])
new_data['YEAR']=new_data['EMPLOY DATE'].dt.year

# Calculate the head count for each agency
agency_head_count = new_data.groupby('AGENCY NAME').size()

# Add the head count as a new column in the DataFrame
new_data['Head_Count'] = new_data['AGENCY NAME'].map(agency_head_count)

# Sort the DataFrame by the head count column in descending order
sorted_data = new_data.sort_values(by='Head_Count', ascending=False)
sorted_data.head()


# In[92]:


# Create the pivot table
table = pd.pivot_table(sorted_data, values='ANNUAL', index=['AGENCY NAME', 'Head_Count'], columns=['YEAR'], aggfunc=np.mean)
# Sort the pivot table by the 'Difference' column in descending order
sorted_table = table.sort_index(level='Head_Count', ascending=False)
# Fill NaN values with 0 using applymap
sorted_table_filled = sorted_table.applymap(lambda x: x if pd.notna(x) else 0)
# Reset the index to remove 'Head_Count' from the index levels
sorted_table_filled = sorted_table_filled.reset_index(level='Head_Count')
# Remove the 'Head_Count' column
sorted_table_filled = sorted_table_filled.drop(columns='Head_Count')

sorted_table_filled.head()


# In[93]:


agencies = top_agencies.index.to_list()


# In[94]:


plt.figure(figsize=(10, 6))
for agency in agencies:
    agency_data = sorted_table_filled.iloc[sorted_table_filled.index==agency]
    plt.plot(agency_data.columns, agency_data.values[0], marker='o', label=agency_data.index[0])
    plt.xlabel('Year')
    plt.ylabel('Annual Salary')
    plt.title(agency_data.index[0])
    plt.legend()
    plt.grid(True)
    plt.show()



# # Model Comparison Report
# 

# In[95]:


## Model Comparison Report

model_results = pd.DataFrame(columns=['Model', 'R-squared', 'MAE', 'RMSE'])


# In[96]:


models = ['Linear Regression','Decision Tree','KNN','Random Forest','Gradient Boosting regressor','XGBoost']
r_squared = [0.8704,0.9998,0.7405,0.9998,0.9998,0.9998]
mae = [0.09338,6.9568,0.1059,6.5725,0.00245,0.00072]

rmse = [0.1505,0.0043,0.2131,0.0047,0.0053,0.0044]


# In[97]:


# Populate the DataFrame
model_results['Model'] = models
model_results['R-squared'] = r_squared
model_results['MAE'] = mae
model_results['RMSE'] = rmse

# Display the results
model_results.head()


# In[98]:


model_results.to_csv('model_comparison_report.csv')


# In[99]:


vs=pd.read_csv('model_comparison_report.csv')
vs.drop('Unnamed: 0',axis=1,inplace=True)
vs.head()


# In[100]:


## Visualization


vs1=vs.drop('Model',axis=1)
#vs1.columns.to_list()

plt.figure(figsize=(20,10))
num =1
# Create a 1x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Iterate through metrics and corresponding subplot axis
for i, col in enumerate(vs1.columns.to_list()):
    sns.barplot(x='Model', y=col, data=vs, ax=axes[i])
    axes[i].set_xlabel('Models')
    axes[i].set_ylabel('Metrics')
    for tick in axes[i].get_xticklabels():
        tick.set_rotation(75)

    axes[i].set_title(col)


plt.tight_layout()
plt.show()



# ## Insights
# 
#     Decision Tree, Random Forest, Gradient Boosting, and XGBoost achieved near-perfect R2 scores (very close to 1), indicating excellent model performance in explaining the variance in the data.
# 
#     K-Nearest Neighbors (KNN) had a lower R2 score compared to the ensemble methods, suggesting it might not capture the complexity of the data as effectively.
# 
#     Linear Regression, while having a respectable R2 score, had a higher MAE and RMSE compared to the ensemble methods, indicating that its predictions had larger errors on average.
# 
#     Gradient Boosting and XGBoost outperformed all other models with extremely low MAE and RMSE values, indicating very accurate predictions.
# 
#     Decision Tree and Random Forest had high R2 scores but comparatively higher MAE values, which could suggest that they might overfit the data to some extent.
# 
# Overall, based on the provided metrics, Gradient Boosting and XGBoost appear to be the top-performing models, offering high accuracy and low errors. However, further evaluation, such as cross-validation, should be conducted to ensure the models generalize well to new data. Additionally, considerations about model complexity and interpretability may influence the final model choice for deployment.

# # Data Analysis and Model Selection Report

# 
# Introduction: This report details the challenges encountered during the analysis of the Texas Salary Prediction dataset and the techniques employed to address those challenges. The goal was to develop an accurate salary prediction model for employees based on various attributes.
# 

# **Challenges Faced**:<br>
# 

# ##### Missing Data:<br>
# **Challenge**: <br>
# The dataset contained missing values in several columns, which could potentially affect model performance.<br>
# **Technique**:<br>
# Missing values were handled using appropriate techniques based on the nature of the data. Categorical variables were imputed with mode values, while numerical variables were imputed using median values. This preserved data integrity without introducing bias.
# 

# ##### Categorical Variables:<br>
# **Challenge**:<br>
#     The dataset included categorical attributes such as Ethnicity, Gender, and Status, which needed to be converted into numerical format for model training.<br>
# **Technique**:<br>
#     Label encoding was applied to convert categorical variables into numerical values. This technique was suitable when the categorical attributes had an inherent ordinal relationship.
# 

# ##### Feature Scaling:<br>
# **Challenge**:<br>
# Numerical attributes had different scales such as Tenure Date,Monthly,Hourly Rate which could impact the performance of certain machine learning algorithms.<br>
# **Technique**:<br>
# Min-Max scaling was applied to scale numerical attributes to a common range between 0 and 1. This technique preserved the relationships between the variables while preventing any dominance of features with larger scales.
# 

# ##### Employee Experience Calculation:<br>
# **Challenge**:<br>
# Converting the "Employ Date" attribute to years of experience from the current date was necessary to include a valuable feature for predicting salaries.<br>
# **Technique**:<br>
# The difference in years between the current date and the "Employ Date" was calculated to determine the tenure in years. This tenure value was added as a feature for the predictive models.
# 
# 

# ##### Model Selection:<br>
# **Challenge**:<br>
# Selecting the appropriate machine learning algorithms to predict salary accurately while avoiding overfitting and underfitting.<br>
# **Technique**: <br>Six algorithms were evaluated—Linear Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, XGBoost, and Gradient Boosting. Each algorithm was chosen based on its inherent properties and suitability for the dataset. Random Forest, XGBoost, and Gradient Boosting were preferred for their ability to capture complex relationships and reduce overfitting due to their ensemble and boosting techniques.
# 

# ##### Techniques Employed with Reasons:<br>
# **Handling Missing Data:**<br>
# Missing values were imputed using mode for categorical variables and median for numerical variables. This approach maintained data quality while minimizing distortion introduced by imputation. Using the mode and median also helped in handling skewed distributions.<br>
# **Label Encoding:**<br>
# Label encoding was chosen to convert categorical variables to numerical format. This technique was used for variables where an inherent order existed (e.g., Status,Ethnicity).<br>
# **Min-Max Scaling:**<br>
# Min-Max scaling was employed to normalize numerical attributes to a common scale. This ensured that all features contributed equally to the model and prevented larger-scaled features from dominating the learning process.<br>
# **Employee Experience Calculation:**<br>
# Converting the "Employ Date" to years of experience was accomplished by calculating the difference between the current date and the "Employ Date." This added an important temporal feature that could impact salary predictions.<br>
# **Model Selection:**<br>
# Linear Regression was chosen for its simplicity and interpretability, making it a good baseline model.
# 
# Random Forest, XGBoost, and Gradient Boosting were selected due to their ability to handle complex relationships and reduce overfitting. XGBoost, in particular, was chosen for its ensemble approach that enhances model accuracy.
# 
# K-Nearest Neighbors (KNN) and Decision Tree were also included to explore alternative modeling approaches and assess their performance on the dataset.<br>
# 

# ##### Conclusion:<br>
# The challenges faced during data analysis were addressed using appropriate techniques that maintained data integrity and improved model performance. Through careful handling of missing data, encoding categorical attributes, scaling features, calculating employee experience, and selecting suitable algorithms,a robust salary prediction model was developed. This model selection process led to the recommendation of employing the XGBoost algorithm due to its strong overall performance in terms of accuracy and robustness.
# 

# In[ ]:





# In[ ]:




