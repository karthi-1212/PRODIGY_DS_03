#!/usr/bin/env python
# coding: utf-8

# <h1>TASK 3</h1>
# 
# Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository
# 
# 

# 
#   <h1>Bank Marketing Campaign Data</h1>
#   <p>This data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit.</p>
#   
#   <table>
#     <tr>
#       <th>Variable Name</th>
#       <th>Role</th>
#       <th>Type</th>
#       <th>Description</th>
#     </tr>
#     <tr>
#       <td>age</td>
#       <td>Feature</td>
#       <td>Integer</td>
#       <td>Client's age in years</td>
#     </tr>
#     <tr>
#       <td>job</td>
#       <td>Feature</td>
#       <td>Categorical</td>
#       <td>Occupation (type of job)</td>
#     </tr>
#     <tr>
#       <td>marital</td>
#       <td>Feature</td>
#       <td>Categorical</td>
#       <td>Marital Status (married, single, divorced, unknown)</td>
#     </tr>
#     <tr>
#       <td>education</td>
#       <td>Feature</td>
#       <td>Categorical</td>
#       <td>Education Level (basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown)</td>
#     </tr>
#     <tr>
#       <td>default</td>
#       <td>Feature</td>
#       <td>Binary</td>
#       <td>Has credit in default?</td>
#     </tr>
#     <tr>
#       <td>balance</td>
#       <td>Feature</td>
#       <td>Integer</td>
#       <td>Average yearly balance (euros)</td>
#     </tr>
#     <tr>
#       <td>housing</td>
#       <td>Feature</td>
#       <td>Binary</td>
#       <td>Has housing loan?</td>
#     </tr>
#     <tr>
#       <td>loan</td>
#       <td>Feature</td>
#       <td>Binary</td>
#       <td>Has personal loan?</td>
#     </tr>
#     <tr>
#       <td>contact</td>
#       <td>Feature</td>
#       <td>Categorical</td>
#       <td>Contact communication type (cellular, telephone)</td>
#     </tr>
#     <tr>
#       <td>day_of_week</td>
#       <td>Feature</td>
#       <td>Date</td>
#       <td>Last contact day of the week</td>
#     </tr>
#     <tr>
#       <td>month</td>
#       <td>Feature</td>
#       <td>Date</td>
#       <td>Last contact month of year (jan, feb, mar, ..., nov, dec)</td>
#     </tr>
#     <tr>
#       <td>duration</td>
#       <td>Feature</td>
#       <td>Integer</td>
#       <td>Last contact duration in seconds (important for benchmark purposes only)</td>
#     </tr>
#     <tr>
#       <td>campaign</td>
#       <td>Feature</td>
#       <td>Integer</td>
#       <td>Number of contacts performed during this campaign (includes last contact)</td>
#     </tr>
#     <tr>
#       <td>pdays</td>
#       <td>Feature</td>
#       <td>Integer</td>
#       <td>Number of days since last contact from previous campaign (-1 means not previously contacted)</td>
#     </tr>
#     <tr>
#       <td>previous</td>
#       <td>Feature</td>
#       <td>Integer</td>
#       <td>Number of contacts performed before this campaign</td>
#     </tr>
#     <tr>
#       <td>poutcome</td>
#       <td>Feature</td>
#       <td>Categorical</td>
#       <td>Outcome of the previous marketing campaign (failure, nonexistent, success)</td>
#     </tr>
#     <tr>
#       <td>y</td>
#       <td>Target</td>
#       <td>Binary</td>
#       <td>Subscribed to term deposit (yes/no)</td>
#     </tr>
#   </table>
# </body>
# </html>
# 

# <h2>Importing the necessary libraries</h2>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2>Reading the dataset</h2>

# In[2]:


url = 'https://github.com/karthi-1212/PRODIGY_DS_03/raw/main/modified_bank_data.csv'
df= pd.read_csv(url)
df.head(5)


# In[3]:


df.rename(columns={'y':'subscribed_deposit'}, inplace=True)


# In[4]:


df.info()


# <h3>Cheking for null or dupliacte values & dropping them </h3>

# In[5]:


df.isnull().sum().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


df.duplicated().sum()


# <h3>Visualization(Num cols) - Histogram and bar grpah </h2>

# In[9]:


df_obj= df.select_dtypes(include='object').columns
df_num= df.select_dtypes(exclude='object').columns

for feature in df_num:
    sns.histplot(x=feature,data=df,bins=25,kde=True,color='#5f366e')
    plt.show()


# In[10]:


for feature in df_obj:
    plt.figure(figsize=(8,3))
    plt.title(f"Count plot of {feature}")
    sns.countplot(x=feature,data=df,palette='viridis')
    plt.xticks(rotation=40)
    plt.show()


# <h3>Observations & Insights:</h3>
#   <table align="left">
#     <tr>
#       <th>Category</th>
#       <th>Observation</th>
#     </tr>
#     <tr>
#       <td>Job</td>
#       <td>Most clients are admins.</td>
#     </tr>
#     <tr>
#       <td>Marital Status</td>
#       <td>Most clients are married.</td>
#     </tr>
#     <tr>
#       <td>Education</td>
#       <td>Most clients have a university degree.</td>
#     </tr>
#     <tr>
#       <td>Credit Default</td>
#       <td>Most clients have no default.</td>
#     </tr>
#     <tr>
#       <td>Housing Loan</td>
#       <td>Most clients have a housing loan.</td>
#     </tr>
#     <tr>
#       <td>Personal Loan</td>
#       <td>Most clients do not have a personal loan.</td>
#     </tr>
#     <tr>
#       <td>Contact Method</td>
#       <td>Cellular is the preferred method.</td>
#     </tr>
#     <tr>
#       <td>Contact Month</td>
#       <td>May is the most frequent month.</td>
#     </tr>
#     <tr>
#       <td>Contact Day</td>
#       <td>Thursday is the most common day.</td>
#     </tr>
#     <tr>
#       <td>Previous Marketing Outcome</td>
#       <td>"Nonexistent" is the most frequent outcome.</td>
#     </tr>
#     <tr>
#       <td>Target</td>
#       <td>Most clients haven't subscribed to a term deposit.</td>
#     </tr>
#   </table>
# </body>

# <h2>checking for outliers & Treatment for it</h2>

# In[11]:


df.plot(kind='box', subplots=True, layout=(5,2), figsize=(10,30))
plt.show()


# In[12]:


columns = ['age', 'campaign', 'duration']

for column in columns:
    q1 = np.percentile(df[column], 25)
    q3 = np.percentile(df[column], 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Filter the DataFrame for the current column
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
df.plot(kind='box', subplots=True, layout=(5,2), figsize=(10,30))
plt.show()


# <h3>Checking for correlation using heatmap</h3>

# In[13]:


# Select only the numerical columns
numerical_df = df.select_dtypes(include=[np.number])

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[14]:


high_corr_cols = ['emp.var.rate','euribor3m','nr.employed']
# copy the original dataframe

df1=df.copy()

# Removing high correlated columns from the dataset
df1.drop(high_corr_cols, inplace=True, axis=1)
df1.columns


# <h3>Conversion of categorical columns into numerical columns using label encoder.<h3>

# In[15]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_encoded = df1.apply(le.fit_transform)
df_encoded


# <h3>Checking the target variable</h3>

# In[16]:


df_encoded['subscribed_deposit'].value_counts(normalize=True)*100


# In[17]:


# Independent Var
x=df_encoded.iloc[:,:-1]
# Target var
y=df_encoded.iloc[:,-1]


# In[18]:


x.shape;y.shape


# <h3>Splitting the dataset into train and test datasets <h3>

# In[19]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# <h3>Decision Tree classifier</h3>

# In[20]:


from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=10)
dc.fit(x_train,y_train)


# <B>Please rerun this cell in a Jupyter environment to display the HTML representation or trust the notebook. On GitHub, the HTML representation may not render correctly; consider accessing the page through nbviewer.org for proper display.</B>

# In[21]:


#Evaluating Training and Testing Accuracy

print("Training accuracy:",dc.score(x_train,y_train))
print("Testing accuracy:",dc.score(x_test,y_test))


# In[22]:


y_pred=dc.predict(x_test)


# In[23]:


#Evaluating Prediction Accuracy
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(accuracy_score(y_test,y_pred))


# In[24]:


print(confusion_matrix(y_test,y_pred))


# In[25]:


print(classification_report(y_test,y_pred))


# <h3>Plot Decision Tree</h3>

# In[26]:


from sklearn.tree import plot_tree
feature_names=df.columns.tolist()
plt.figure(figsize=(40,20))
class_names=["class_0","class_1"]
plot_tree(dc, feature_names=feature_names, class_names=class_names, filled=True,fontsize=12)
plt.show()


# In[27]:


#Decision Tree classifier using 'entropy' criterion
dc1=DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_split=10)
dc1.fit(x_train,y_train)


# <B>Please rerun this cell in a Jupyter environment to display the HTML representation or trust the notebook. On GitHub, the HTML representation may not render correctly; consider accessing the page through nbviewer.org for proper display.</B>

# In[28]:


#Evaluating Training and Testing Accuracy
print("Training accuracy:",dc1.score(x_train,y_train))
print("Testing accuracy:",dc1.score(x_test,y_test))


# In[29]:


y1_pred=dc1.predict(x_test)


# In[30]:


#Evaluating Prediction Accuracy
print(accuracy_score(y_test,y1_pred))


# In[31]:


print(confusion_matrix(y_test,y1_pred))


# In[32]:


print(classification_report(y_test,y1_pred))


# In[33]:


cn=['no','yes']
fn=x_train.columns
plt.figure(figsize=(40,20))
plot_tree(dc1, feature_names=fn.tolist(), class_names=cn, filled=True,fontsize=12)
plt.show()


# <h1>Conclusion</h1>

#   <h2>Decision Tree Criteria: Gini vs. Entropy</h2>
#   <h3>Both Gini and Entropy Criteria Deliver Strong Results</h3>
#   <p>When comparing Gini impurity and entropy as criteria for building decision trees, both methods achieve impressive accuracy:</p>
#   <ul>
#     <li>Training accuracy: Around 93.6% for both</li>
#     <li>Testing accuracy:</li>
#       <ul>
#         <li>Gini: Around 93.3%</li>
#         <li>Entropy: Around 93.2%</li>
#       </ul>
#   </ul>
#   <h3>Gini for Identifying True Positives</h3>
#   <p>While the difference in testing accuracy is minimal, Gini shows a slight advantage. It excels at correctly identifying positive instances (represented by class 1). This means Gini might be preferable when accurately predicting positive cases is crucial.</p>
#   <h3>Entropy for Reducing False Positives</h3>
#   <p>Entropy results in fewer false positives (incorrectly predicted positives) but comes at the cost of potentially missing some true positives (false negatives). This suggests using Entropy if minimizing false positives is a higher priority.</p>
#   <h3>Choosing the Right Criterion</h3>
#   <p>The best choice between Gini and Entropy depends on the specific application and its priorities. If accurately identifying true positives is paramount, Gini might be better. If minimizing false positives is the main concern, Entropy could be a better fit.</p>
# 
# 
