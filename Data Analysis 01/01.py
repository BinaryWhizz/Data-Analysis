# Load Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
data = pd.read_excel('Strikers_performance.xlsx')
data.head()

# Data Cleaning
## Missing values
data.isnull().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')  
data[['Movement off the Ball', 
      'Big Game Performance', 
      'Penalty Success Rate']] = imputer.fit_transform(data[['Movement off the Ball', 
                                                             'Big Game Performance', 
                                                             'Penalty Success Rate']])
data.isnull().sum()

## Data Types
data.dtypes
variables = ['Goals Scored', 'Assists', 
             'Shots on Target', 
             'Movement off the Ball', 
             'Hold-up Play', 
             'Aerial Duels Won', 
             'Defensive Contribution', 
             'Big Game Performance', 
             'Impact on Team Performance', 
             'Off-field Conduct']

for var in variables:
    data[var] = data[var].astype('int')
    
data.dtypes
data.head()
# Exploratory Data Analysis
### Perform descriptive analysis
round(data.describe(), 2)
### Perform percentage analysis
freq_Footedness = data['Footedness'].value_counts()
perc_Footedness = freq_Footedness/len(data['Footedness'])*100
perc_Footedness
plt.figure(figsize=(12, 6))
perc_Footedness.plot(kind='pie', autopct='%1.2f%%')
plt.title('Percentage of strikers by their footedness')
plt.ylabel('')
plt.show()
### Which nationality strikers have the highest average number of goals scored?
goals_by_nationality = data.groupby('Nationality')['Goals Scored'].mean().sort_values(ascending=False)
round(goals_by_nationality)
### What is the average conversion rate for players based on their footedness?
conversion_rate_by_footedness = data.groupby('Footedness')['Conversion Rate'].mean()
conversion_rate_by_footedness
### What is the distribution of players' footedness across different nationalities?
footedness_by_nationality = pd.crosstab(data['Nationality'], data['Footedness'])
footedness_by_nationality
plt.figure(figsize=(12, 6))
sns.countplot(x='Nationality', hue='Footedness', data=data)
plt.title('Tistribution of players footedness across different nationalities')
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.show()
### Create a correlation matrix with a heatmap
num_variables = data.select_dtypes(include = ['number']).columns

correl_matrix = round(data[num_variables].corr(), 3)
correl_matrix
plt.figure(figsize=(18, 10))
sns.heatmap(correl_matrix, annot=True)
plt.title('Heatmap of Correlation Matrix')
plt.show()
# Statistical Test
### Find whether there is any significant difference in consistency rates among strikers from various nationality
# Normality test
stat, p_value = shapiro(data['Consistency'])
print('P value: ', round(p_value, 3))
# Filtering data
Spain = data.query('Nationality == "Spain"')['Consistency']
France = data.query('Nationality == "France"')['Consistency']
Germany = data.query('Nationality == "Germany"')['Consistency']
Brazil = data.query('Nationality == "Brazil"')['Consistency']
England = data.query('Nationality == "England"')['Consistency']
# Levene test for statistics
stats, p_value = levene(Spain, France, Germany, Brazil, England)
print("P value: ", round(p_value, 3))
# One way ANOVA

Test_stat, p_value = f_oneway(Spain, France, Germany, Brazil, England)
print("P value: ", round(p_value, 2))
### Check if there is any significant correlation between strikers' Hold-up play and consistency rate
# Normality test
stat, p_value = shapiro(data['Hold-up Play'])
print('P value: ', round(p_value, 3))
# Linearity test
plt.figure(figsize = (10, 6))
sns.regplot(x = 'Hold-up Play', y = 'Consistency', data = data)
plt.title('Linearity between Hold-up Play and Consistency')
plt.xlabel('Hold-up Play')
plt.ylabel('Consistency')
plt.show()
# Pearson correlation
HU_play = data['Hold-up Play']
Consistency = data['Consistency']

corr, p_value = pearsonr(HU_play, Consistency)
print("Correlation coefficient: ", round(corr, 3))
print("P value: ", round(p_value, 3))
### Check if strikers' hold-up play significantly influences their consistency rate
x = data['Hold-up Play']
y = data['Consistency']

x_constant = sm.add_constant(x)
model = sm.OLS(y, x_constant).fit()

print(model.summary())
# Feature Engineering
### Create a new feature - Total contribution score
data['Total contribution score'] = (data['Goals Scored'] + data['Assists'] + data['Shots on Target'] + data['Dribbling Success'] + data['Aerial Duels Won'] + data['Defensive Contribution'] + data['Big Game Performance'] + data['Consistency'])
data.head()
### Encode the Footedness and marital status by LabelEncoder
encoder = LabelEncoder()
data['Footedness'] = encoder.fit_transform(data['Footedness'])
data['Marital Status'] = encoder.fit_transform(data['Marital Status'])
data.head()
### Create the dummies for Nationality and add with the data
dummies = pd.get_dummies(data['Nationality'])
processed_df = pd.concat([data, dummies], axis = 1)
processed_df = processed_df.drop('Nationality', axis = 1)
processed_df.head()
# Cluster Analysis
### Perform KMeans clsutering 
# Selecting features
x = processed_df.drop('Striker_ID', axis = 1)

# Calculating WCSS score
wcss = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(x)
    wcss_score = kmeans.inertia_
    wcss.append(wcss_score)
# Plotting elbow chart
plt.figure(figsize = (12, 6))
plt.plot(range(1, 15), wcss, marker = 'o')
plt.title('Elbow methods')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()
# Building KMeans with k = 2
final_km = KMeans(n_clusters = 2)
final_km.fit(x)

# Generating labels
labels = final_km.labels_
labels
# Adding labels
processed_df['Clusters'] = labels
processed_df.head()
# Checking clusters
round(processed_df.groupby('Clusters')['Total contribution score'].mean(), 2)
# Assigning meaningfull names
mapping = {0:'Best strikers', 1:'Regular strikers'}
processed_df['Strikers types'] = processed_df['Clusters'].map(mapping)
# Deleting the Clusters variable
processed_df = processed_df.drop('Clusters', axis = 1)
processed_df.head()
# Data Preprocessing for ML
### New feature mapping
mapping = {'Best strikers':1, 'Regular strikers': 0}
processed_df['Strikers types'] = processed_df['Strikers types'].map(mapping)
processed_df.head()
### Selecting features
x = processed_df.drop(['Striker_ID', 'Strikers types'], axis = 1)
y = processed_df['Strikers types']
### Scaling features
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)
scaled_x
### Train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size = 0.2, random_state = 42)
# Predictive Classification Analytics
### Build a logistic regression machine learning model to predict strikers type
# Model training
lgr_model = LogisticRegression()
lgr_model.fit(x_train, y_train)

#Prediction
y_lgr_pred = lgr_model.predict(x_test)

# Evaluation
accuracy_lgr = accuracy_score(y_test, y_lgr_pred)
print(accuracy_lgr*100,'%')
# Creating confusion matrix
conf_matrix_lgr = confusion_matrix(y_test, y_lgr_pred)

# Plotting confusion matrix
plt.figure(figsize = (12, 6))
sns.heatmap(conf_matrix_lgr, annot = True, fmt = "d", cmap = "Blues")
plt.title('Confusion Matrix for LGR model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
## Thank you!