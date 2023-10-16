# To-predict-the-health-insurance-premium
A health insurance premium is the amount – typically billed monthly – that policyholders pay for health coverage. Policyholders must pay their premiums each month regardless of whether they visit a doctor or use any other healthcare service![image](https://github.com/Jaanu93/To-predict-the-health-insurance-premium/assets/95671214/0785e678-2a2c-42a8-b665-a4cce271116a)

 Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
%matplotlib inline
Import dataset
insurance = pd.read_csv("C:\\Users\\jaanu\\Downloads\\insurance.csv")
insurance
# Data Exploration
insurance.head()
insurance.tail()
insurance.columns
insurance.dtypes
insurance.describe()
# Data Cleaning
insurance.dropna()
Check for NaN values in the dataset
insurance.isnull().sum()
insurance
# Feature Engineering
Binning
bins = [0.0, 18.5, 24.9, 29.9, 60]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
insurance['bmi_cat'] = pd.cut(insurance['bmi'], bins, labels=labels)
smoker_data = pd.get_dummies(insurance,columns=['smoker'])
smoker_data 
# Hypothesis test 1
H0: Premium charges decrease for the person who smokes

HA: Premium charges increase for the person who smokes

To examine whether the charges increase or not for the person who smokes
# One-Way ANOVA Test using statsmodels module
F, p = stats.f_oneway(smoker_data['smoker_no'],smoker_data['smoker_yes'])
# Seeing if the overall model is significant
print('F-Statistic=%.3f, p=%.3f' % (F, p))
# One-Way ANOVA Test using OLS Model 
As we know in regression, we can regress against each input variable and check its influence over the Target variable. 
So, we’ll follow the same approach, the approach we follow in Linear Regression.
model = ols('expenses ~ smoker', insurance).fit()
model.summary()
# Welch's Test
smoker_yes = insurance[(insurance['smoker'] == 'yes')]
smoker_no = insurance[(insurance['smoker'] == 'no')]
stats.ttest_ind(smoker_yes['expenses'], smoker_no['expenses'], equal_var = False)
Based on the results from anova and t-test
The p value is 0 which is less than significance level =0.05,sufficient evidence exists to reject the null hypothesis H0,in favour of the alternate hypothesis,HA
We sugggest that there is statistically significant relationship exists between two variables
# Hypothesis test 2
H0: Premium charges decrease with the increase in a person’s BMI

HA: Premium charges increase with the increase in a person’s BMI

To examine whether the charges increase with increase in a person’s BMI or not.
# One-Way ANOVA Test using statsmodels module

bmi_data = pd.get_dummies(insurance,columns=['bmi_cat'])
bmi_data 
F, p = stats.f_oneway(bmi_data['bmi_cat_Underweight'],bmi_data['bmi_cat_Normal'],bmi_data['bmi_cat_Overweight'],bmi_data['bmi_cat_Obese'])
# Seeing if the overall model is significant
print('F-Statistic=%.3f, p=%.3f' % (F, p))
# One-Way ANOVA Test using OLS Model 
model_age= ols('expenses ~ bmi', insurance).fit()
model_age.summary()
Based on the results the p value is 0 which is less than significance level =0.05,sufficient evidence exists to reject the null hypothesis H0,in favour of the alternate hypothesis,HA We sugggest that there is statistically significant relationship exists between two variables
# Hypothesis test 3
H0: Premium charges decrease with the increase in a person’s Age

HA: Premium charges increase with the increase in a person’s Age

To examine whether the charges increase with increase in the age of the person or not.
# One-Way ANOVA Test using OLS Model 
model = ols('expenses ~ age', insurance).fit()
model.summary()
Based on the results the p value is 0 which is less than significance level =0.05,sufficient evidence exists to reject the null hypothesis H0,in favour of the alternate hypothesis,HA We sugggest that there is statistically significant relationship exists between two variables
# Categorical to Numerical Encoding
bins = [0.0, 18.5, 24.9, 29.9, 60]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
insurance['bmi_cat'] = pd.cut(insurance['bmi'], bins, labels=labels)
insurance1 = insurance.copy()
le = LabelEncoder()

insurance1['children'] = le.fit_transform(insurance1['children'])
insurance1['smoker'] = le.fit_transform(insurance1['smoker'])
ohe = OneHotEncoder() 

insurance1['sex'] = ohe.fit_transform(insurance1[['sex']]).toarray()

regional_area = pd.DataFrame(ohe.fit_transform(insurance1[['region']]).toarray(), columns = ['NE', 'NW', 'SE', 'SW'])
bmi_bins = pd.DataFrame(ohe.fit_transform(insurance1[['bmi_cat']]).toarray(), 
                        columns= ['Normal', 'Obese', 'Overweight', 'Underweight'])
insurance1 = pd.concat([ insurance1.iloc[:,:2], insurance1.iloc[:,7:8], bmi_bins, insurance1.iloc[:,3:6], regional_area, insurance1.iloc[:,-2:-1]], axis=1)
insurance1
insurance2 = insurance1.corr()
insurance2 
Data Visualisation
Age Vs Expenses
sns.scatterplot(x=insurance['age'], y = insurance['expenses'])
plt.xlabel('age')
plt.ylabel('Expenses')
plt.title('age vs Expenses')
plt.show()
Children Vs Expenses
sns.scatterplot(x=insurance['children'], y = insurance['expenses'])
plt.xlabel('children')
plt.ylabel('Expenses')
plt.title('children vs Expenses')
plt.show()
Sex Vs Expenses
sns.scatterplot(x=insurance['sex'], y = insurance['expenses'])
plt.xlabel('sex')
plt.ylabel('Expenses')
plt.title('sex vs Expenses')
plt.show()
Bmi vs expenses
sns.scatterplot(x=insurance['bmi'], y = insurance['expenses'])
plt.xlabel('bmi')
plt.ylabel('Expenses')
plt.title('bmi vs Expenses')
plt.show()
Smoker Rate by region
insurance.groupby(['region','smoker']).size().unstack().plot(kind='bar',stacked=False)
plt.show()
# Plotting the heatmap for correlation
Correlation plot between all the variables
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(insurance1.corr(), annot=True, annot_kws={'size':10},linewidths=0.30)
plt.show()
Correlation plot between numerical variables
ax = sns.heatmap(insurance.corr(), annot=True)
# Data Visualization
import matplotlib.pyplot as plt
def numerical(num):
    sns.scatterplot(x=insurance[num], y = insurance['expenses'])
    plt.xlabel(num)
    plt.ylabel('Expenses')
    plt.title('{n} vs Expenses'.format(n = num))
    plt.show()

 def categorical(depen_var):
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12,8))
        fig.suptitle('{c} vs {n}'.format(c = depen_var.upper(), n = 'Expenses'))
        
        sns.boxplot(ax=axes[0], x = insurance[depen_var], y = insurance['expenses'])
        axes[0].set_title('Boxplot')

        sns.barplot(ax=axes[1],  x = insurance[depen_var], y = insurance['expenses'])
        axes[1].set_title('Bar Chart')

        plt.show()
for feature in insurance.columns:
    if feature == 'expenses':
        continue
    elif feature in ['age', 'bmi']:
        numerical(feature)
    else:
        categorical(feature)
insurance1.drop(columns = ['region', 'bmi_cat'], inplace=True)
# Data Standardization
scaler = RobustScaler()
insurance1[['age', 'children', 'expenses']] = scaler.fit_transform(insurance1[['age', 'children', 'expenses']])
# Model Building
y_data = insurance1['expenses']
x_data = insurance1.iloc[:,:12]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15)
# Multiple Linear Regression

np.random.seed(321)

#regression object
ml_reg = LinearRegression()

#training the model
ml_reg.fit(x_train, y_train)

#predict values
pred_ml_reg = ml_reg.predict(x_test)
# Polynomial Regression

ply_reg = PolynomialFeatures(degree=2)
x_ply_data  = ply_reg.fit_transform(x_data)
x_ply_train, x_ply_test = train_test_split(x_ply_data, test_size = 0.15)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_ply_train,y_train)
 
pred_ply = lin_reg2.predict(x_ply_test)
# Evaluation Metrics

def evaluate(estimator, pred_val):
    if estimator == lin_reg2:
        x = PolynomialFeatures(degree=2).fit_transform(x_data)
    else:
        x = x_data
    r2 = r2_score(y_test, pred_val)
    rmse = np.sqrt(mean_squared_error(y_test, pred_val))
    score = cross_val_score(estimator, x, y_data, cv=4).mean()
    return r2, rmse, score
#Linear Regression
ml_reg_r2, ml_reg_rmse, ml_reg_cvscore = evaluate(ml_reg, pred_ml_reg)
print(ml_reg_r2, ml_reg_rmse, ml_reg_cvscore)
#Polynomial Regression
ply_r2, ply_rmse, ply_cvscore = evaluate(lin_reg2, pred_ply)
print(ply_r2, ply_rmse, ply_cvscore)
# Comparing Model perfomances

r2_scores = [ml_reg_r2, ply_r2]
rmse_scores = [ml_reg_rmse, ply_rmse]
models_comparison = pd.DataFrame([r2_scores, rmse_scores], 
                                 columns = ['Multiple Linear Regression', 'Polynomial Regression'],
                                 index = ['R2 score', 'RMSE score'])
models_comparison
