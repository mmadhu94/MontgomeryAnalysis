#!/usr/bin/env python
# coding: utf-8

# # Motor Vehicle Crash Data:
# ## A Causal and Predictive Analysis
# #### by Madhukar Mantravadi

# In[773]:


# imports
from CHAID import Tree
from datetime import datetime
import numpy as np
import pandas as pd
pd.options.display.max_columns = 100
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import plotly
plotly.tools.set_credentials_file(username='mmadhu94', api_key='mRx4CKJkIDFteX89rFwQ')


# ## EDA

# *Reading the csv files into pandas dataframe form*

# In[582]:


df_IncidentData = pd.read_csv('Crash_Reporting_-_Incidents_Data.csv')


# In[2]:


df_DriversData = pd.read_csv('Crash_Reporting_-_Drivers_Data.csv')


# In[19]:


df_DriversData.info()


# In[559]:


df_IncidentData.info()


# *It can be seen above that there are certain features that do not contain many non-null values (i.e. Related Non-Motorists in the Driver Data, Municipality in the Non Motorist Data, and Non Motorist Substance Abuse in the Incedents Dataset). These features may not be a good choice for analysis because the data is so unbalanced.*

# In[494]:


df_DriversData.head()


# In[621]:


df_IncidentData.head()


# In[608]:


'''I chose the step here to bin the lat/long by such that 
it will divide the area that the accidents took place into 1 sq mi incriments'''

step = 1/70
to_bin = lambda x: np.floor(x / step) * step
df_IncidentData["latbin"] = df_IncidentData.Latitude.map(to_bin)
df_IncidentData["lonbin"] = df_IncidentData.Longitude.map(to_bin)


# In[585]:


#Cleaning up the dataframe columns that are relevant
df_DriversData['Injury Severity'] = df_DriversData['Injury Severity'].str.replace('INJURY','')
df_IncidentData['Second Harmful Event'] = df_IncidentData['Second Harmful Event'].replace(np.NaN,'NO 2ND EVENT')


# In[620]:


#Engineering new features to utilize time and 'Second Harmful Event' data
df_IncidentData['Second Event Existing'] = [0 if i == 'NO 2ND EVENT' else 1 
                                            for i in df_IncidentData['Second Harmful Event']]
df_IncidentData['Hour'] = df_IncidentData['Crash Date/Time'].dt.hour


# In[639]:


#Generating dummy column features based on the types of harmful events in the 'First Harmful Event' feature
df_IncDummy = pd.concat([df_IncidentData.copy(),
                         pd.get_dummies(df_IncidentData['First Harmful Event'],prefix = 'Harm')], axis = 1)


# In[643]:


#Dropping the original column for redundancy
df_IncDummy.drop('First Harmful Event', axis=1, inplace = True)


# In[644]:


df_IncDummy


# *Converting the date/time data to datetime form*

# In[419]:


#Making a function to convert to datetime utilizing the apply method on a pandas dataframe
def timestripper(string):
    return datetime.strptime(string, '%m/%d/%Y %I:%M:%S %p')


# In[614]:


df_DriversData['Crash Date/Time'] = df_DriversData['Crash Date/Time'].apply(timestripper)


# In[615]:


df_IncidentData['Crash Date/Time'] = df_IncidentData['Crash Date/Time'].apply(timestripper)


# *The following is code for several plots looking at frequency of accidents anddifferent incriments of time*

# In[550]:


plt.title('Frequency of Accidents (# of accidents) vs Year and Month (year,month)',fontdict={'fontsize': 30,
        'fontweight' : 'bold'})
df_DriversData['Crash Date/Time'].groupby([df_DriversData['Crash Date/Time'].dt.year, 
                                           df_DriversData['Crash Date/Time'].dt.month]).count().plot(figsize = (24,10), 
                                                                                                     fontsize = 20, kind="bar", 
                                                                                                     width = 0.9, color = 'blue')
plt.savefig('FreqVsYrMnth.png')


# In[551]:


plt.title('Frequency of Accidents (# of accidents) vs Month',fontdict={'fontsize': 30,
        'fontweight' : 'bold'})
df_DriversData['Crash Date/Time'].groupby([df_DriversData['Crash Date/Time'].dt.month]).count().plot(figsize = (24,10), 
                                                                                                     fontsize = 20, kind="bar", 
                                                                                                     width = 0.9, color = 'magenta')
plt.savefig('FreqVsMnth.png')


# In[552]:


plt.title('Frequency of Accidents (# of accidents) vs Year',fontdict={'fontsize': 30,
        'fontweight' : 'bold'})
df_DriversData['Crash Date/Time'].groupby([df_DriversData['Crash Date/Time'].dt.year]).count().plot(figsize = (24,10), 
                                                                                                     fontsize = 20, kind="bar", 
                                                                                                     width = 0.9, color = 'cyan')
plt.savefig('FreqVsYr.png')


# In[553]:


df_DriversData['Crash Date/Time'].groupby([df_DriversData['Crash Date/Time'].dt.hour]).count().plot(figsize = (24,10), 
                                                                                                     fontsize = 20, kind="bar", 
                                                                                                     width = 0.9, color = ['g' if i not in (7,8,9,15,16,17,18) else 'r' for i in range(24)])
plt.savefig('FreqVsHr.png')


# In the lots above looking at the datetime information we can tell a few things:
#  - Looking at the year and month plot, there seems to be some cyclical nature to the frequency of accidents, with the later months of the year (OCT-DEC) seem to have a higher frequency of accidents.
#  - Looking at the month plot and year plot as seperated data, however doesn't show much of a correlation. For the year especially it seems to be almost equal for every year except 2019, which is only small because we are currrently in the 1st month of the year only.
#  - The hour plot shows something that with knowledge of work culture makes sense; The peak accident times are from 7-9 am and 3-6 pm. This makes sense since those are the times of most peoples work commutes and when there is usually more traffic. 

# *Plotly code for interactive 2d histograms of Hr vs Injury Severity and Hr vs Vehicle Damage Extent*

# In[543]:


layout = go.Layout(margin = dict(b = 50, l = 50), 
                   yaxis = dict(
                                title='Injury Severity',
                                tickmode='array',
                                automargin=True,
                                titlefont=dict(size=20)
                                ),
                   xaxis = dict(
                                title = 'Hour of Day (24h)',
                                tickvals = list(range(24)),
                                titlefont=dict(size=20)
                                ))


# In[554]:


x = df_DriversData['Crash Date/Time'].dt.hour
y = df_DriversData['Injury Severity']
trace = [go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'Viridis',
        contours = dict(
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                color = 'white'
            )
        ),
        hoverlabel = dict(
            bgcolor = 'white',
            bordercolor = 'black',
            font = dict(
                family = 'Raleway',
                color = 'black'
            )
        )
)]

fig = go.Figure(data = trace, layout = layout)
py.iplot(fig, filename = "Histogram2d_Crash_Injury")


# *From the plot generated above it can be seen that most accidents fall within either the "Possible Injury" or "No Apparent Injury". Looking at the trend within the more serious injuries, there are small increases there during rush hour times as well. However, the sample size of those types of injuries are much smaller and thus there is not really enough evidence to definitively say that rush hours result in higher amounts of severe crashes.* 

# In[556]:


layout = go.Layout(margin = dict(b = 50, l = 50), 
                   yaxis = dict(
                                title='Vehicle Damage Extent',
                                tickmode='array',
                                automargin=True,
                                titlefont=dict(size=20)
                                ),
                   xaxis = dict(
                                title = 'Hour of Day (24h)',
                                tickvals = list(range(24)),
                                titlefont=dict(size=20)
                                ))


# In[557]:


x = df_DriversData['Crash Date/Time'].dt.hour
y = df_DriversData['Vehicle Damage Extent']
trace = [go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'Viridis',
        contours = dict(
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                color = 'white'
            )
        ),
        hoverlabel = dict(
            bgcolor = 'white',
            bordercolor = 'black',
            font = dict(
                family = 'Raleway',
                color = 'black'
            )
        )
)]

fig = go.Figure(data = trace, layout = layout)
py.iplot(fig, filename = "Histogram2d_Crash_Damage")


# *It can be seen above that most crashes result in either "Disabling", "Superficial", or "Functional" extent to the damage. It is even clearer here that there are more crashes overall during rush hour commute times. There is a large spike in the "No Damage" category during rush hour times as well.*

# **The code above can be run for interactive 2d histograms. A static image of the plots is available in the report accompanying this code.**

# *It could be said with more certainty, however that more crashes happen during rush hour. I will continue to look at the effects of different features, namely whether it's Light or Dark outside and how that change causes change within the severity of crashes.*

# In[573]:


#Looking at the amounts of unique values in each dataframe
df_DriversData.nunique().sort_values()


# In[572]:


df_IncidentData.nunique().sort_values()


# *Using a feature with too many categories/distinct values may not be good for a causal analysis as well because there are too many different factors in that kind of analysis. It is possible, but for the purpose of this analysis the focus will be placed on the top few features in the above series (which have the lowest count of unique values/observations)*

# ## Causal Hypothesis and Testing

# ### Looking at the data the more specific/detailed features such as location and street name have *far* more categories than some of the other more general ones such as wheather the driver is at fault for the accident or not. Because of the amount of unique values, I decided to look into the effect of time of day via the 'Light' feature on Injury Severity and Vehicle Damage Extent

# ### Hypothesis: The light will effect the Injury Severity and Vehicle Damage Extent and that the darker it is, the higher the ratio of severe damage to temperate damage.

#  $H_o =$ **The light (time of day) doesn't have a relationship with the Injury Severity and Vehicle Damage Extent** 
#  <br>
#  $H_1 =$ **The light (time of day) *does* have a relationship with the Injury Severity and Vehicle Damage Extent**

# *To do this experiment, first I will conduct a $\chi^2$ test to determine which hypothesis we will reject. This will not find the actual relationship but provide evidence that there is grounds for one.*

# In[333]:


light_drops = list(df_DriversData['Light'].value_counts().index[-5:])


# In[334]:


vehicle_damage_drops = list(df_DriversData['Vehicle Damage Extent'].value_counts().index[-2:])


# *Below we are defining the dataframe we are going to use and cleaning up a bit. I decided to remove some of the smaller categories in the data that had magnitudes smaller in their value counts.
# <br>
# I created masks to filter the datafram based on the values I wanted to cut and then also dropped rows with NaN.*

# In[366]:


df_CHAID = df_DriversData[['Injury Severity','Vehicle Damage Extent', 'Light']]
df_CHAID.dropna(inplace = True)

mask_light = df_CHAID[df_CHAID['Light'].isin(light_drops)].index
mask_light = list(mask_light)
df_CHAID.drop(mask_light, inplace=True)

mask_vdamage = df_CHAID[df_CHAID['Vehicle Damage Extent'].isin(vehicle_damage_drops)].index
mask_vdamage = list(mask_vdamage)
df_CHAID.drop(mask_vdamage, inplace = True)


# ***Here I'm using the CHAID (Chi-squared Automatic Interaction Detector) algorithm to do the test. I am using an existing library called CHAID that allows me to generate the values easily.***
# 
#  - The CHAID Algorithm checks the statistical significance via the $\chi^2$ test and outputs the counts of all categories within each of the features of the variable. It does this for every variable and generates every crosstab it can that has statistical significance. 
#  
#  - If a feature is shown to not be significant, it will try to combine it with another non-significant feature and run the test again. This can lead to misleading or useless results for certain features. In the following analysis I have taken steps to try and avoid this but it is something to be weary of. One should not just follow a model's results blindly.

# In[348]:


#defining Variables to use in the CHAID function
indep_var = ['Light']
dep_var1 = 'Injury Severity'
dep_var2 = 'Vehicle Damage Extent'

#Building the tree
tree1 = Tree.from_pandas_df(df_CHAID, dict(zip(indep_var, ['nominal'] * 3)), dep_var1, max_depth=1)
tree2 = Tree.from_pandas_df(df_CHAID, dict(zip(indep_var, ['nominal'] * 3)), dep_var2, max_depth=1)


# In[368]:


#Converting the tree to an object to view it
tree1.to_tree()
tree2.to_tree()


# In[350]:


#Printing out the different trees to see the results in raw form
tree1.print_tree()


# In[351]:


tree2.print_tree()


# The **p-value** for *Injury Severity* $= \mathbf{8.8399e-08}$
# <br>
# The **p-value** for *Vehicle Damage Extent* $= \mathbf{1.4651e-171}$
# <br>
# Both of these values are $\mathbf{<0.05 \therefore}$ statistically significant.

# *As we can see above both of the cases have a **p value < 0.05** and are therefore statistically significant. We can also see this by the fact that the $\chi$ score is relatively far from 0 (especially in the case of vehicle damage).* 
# 
# *We are part way into proving our hypothesis true or false but within the context of our $\chi^2$ test, we can reject the null hypothesis and accept the alternate hypothesis that the Light (time of day) has a relationship with Vehicle Damage Extent and Injury Severity.*

# *Below I have calculated the ratio of worse categories to more temperate ones. I assumed that for Injury that "Fatal Injury" and "Suspected Serious Injury" were the worst case categories. For Vehicle Damage Extent, I assumed "Destroyed" and "Disabling" were the worst case.
# The ratio's below are a simple calculation showing that our original hypothesis has grounds to be correct based on the data. However, correlation is not causation, so we cannot assume that this proves our causal hypothesis.*

# In[359]:


#The ratios are the # of worst-case to # of other cases
dark_injury_ratio = (10+181)/(16525+1987+1692)


# In[361]:


daylight_injury_ratio = (13+490)/(45478+5898+4410)


# In[362]:


dark_ratio/daylight_ratio


# This ratio shows a ~5% higher amount of severe injury during the dark than the daylight.

# In[363]:


dark_vdamage_ratio = (255+1050+1050+7143)/(562+83+532+4517+584+4417)


# In[364]:


daylight_vdamage_ratio = (1709+19894)/(16375+2305+15396)


# In[365]:


dark_vdamage_ratio/daylight_vdamage_ratio


# This ratio shows ~40% more severe damage cases in the dark than during the daylight.

# ---

# ---

# ## Predictive Hypothesis and Testing

# ### Hypothesis:
# #### For a predictive hypothesis and model, an analysis of the time data, location, First Harmful Event, and Second Harmful Event. 
# 

# #### Can the existence of a 2nd harmful event be predicted based on the 1st Event, the time, and the location?

# *For this problem I chose multiple models to see which method will work better. The specific models were chosen because they can deal with unbalanced data and also have penalties in place so they can try to avoid falling to either end of the bias-variance spectrum.*

# In[829]:


df_IncidentData['First Harmful Event'].value_counts()


# In[709]:


df_IncDummy


# In[764]:


#Splitting the data into training and test sets
X = df_IncDummy[df_IncDummy.columns[31:]]
y = list(df_IncDummy['Second Event Existing'])

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)


# In[690]:


#setting seed for reproducibility
seed = 42


# In[796]:


# prepare models to choose from, we will tune the one with best score afterwards
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', 
                                        max_iter=500, 
                                        n_jobs=3,
                                        random_state = 42)))
models.append(('SGD', SGDClassifier(max_iter=np.ceil(10**6 / len(y)), 
                                    early_stopping=True, 
                                    n_iter_no_change= 10, 
                                    n_jobs=3,
                                    random_state = 42)))
models.append(('KNN', KNeighborsClassifier(n_jobs=3)))
models.append(('BRF', BalancedRandomForestClassifier(n_jobs=3,
                                                     random_state = 42)))
models.append(('SVM', SVC(gamma='auto', 
                          verbose = True,
                          random_state = 42)))
print('model array created')
# evaluate each model in turn
results_r = []
results_a = []
names = []
scoring = ['recall','accuracy']
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    #cross validating with 10 folds for both recall and accuracy because both are important
    cv_results_r = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring[0], n_jobs=3)
    cv_results_a = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring[1], n_jobs=3)
    results_r.append(cv_results_r)
    results_a.append(cv_results_a)
    names.append(name)
    #Printing results every time a model is finished scoring and displaying the scores and standard deviations
    print(name,'accuracy>', cv_results_a.mean(), cv_results_a.std(),'\n','recall>',cv_results_r.mean(), cv_results_r.std())


# In[830]:


#Box plot of accuracy scores between algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison (accuracy scored)')
ax = fig.add_subplot(111)
plt.boxplot(results_a)
ax.set_xticklabels(names)
plt.savefig('AlgCompAcc.png')
plt.show()


# In[831]:


#Box plot of recall scores between algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison (recall scored)')
ax = fig.add_subplot(111)
plt.boxplot(results_r)
ax.set_xticklabels(names)
plt.savefig('AlgCompRec.png')
plt.show()


# *We can see by the results of the CV tests for each model that the Support Vector Machine and Logistic Regression performed the best in terms of accuracy. However, if we look at bothe recall and accuracy, Random Forest and SGD performed best. SVMs are complex and take a longer time to tune and the SGD here was subject to overfitting (evidenced by the high variance/high standard deviation. So either Logistic Regression or Random Forest would be best. The purpose of the question behind this is inspired by the idea that if authorities were able to better predict whether there was collateral damage and then respond in like with increased assistance on sight etc. In this system False Negatives (situations where the incident did have collateral but the model predicts it doesn't) would be more important and therefor recall should be the main score we should try and aim to maximize.*

# In[799]:


#Preparing the chosen model for hyperperameter tuning
lr = BalancedRandomForestClassifier(n_jobs=3,
                                    random_state = 42)


# In[814]:


#These parameters were chosen because other parameters would not make much of a difference in the score
param_grid = {'criterion':('gini','entropy'),
              'max_features': ('auto','log2')}


# In[815]:


#Initializing the cross validated gridsearch over the different parameters
gs = GridSearchCV(estimator = lr, param_grid = param_grid, 
                  scoring = {'accuracy', 'recall'},
                  n_jobs = 3, 
                  cv = 10,
                  refit='recall',
                  verbose = 2)


# In[816]:


#Fitting the gridsearch
gs.fit(X_train,y_train)


# In[817]:


#taking a look at the best paramaters by score
gs.best_params_


# In[819]:


#Setting the output of the gridsearch to a 'final_pred' variable to verify the strength of the model
final_pred = gs.best_estimator_


# In[820]:


#Entering test data
y_pred = final_pred.predict(X_test)


# In[821]:


#Generating Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


# In[822]:


confusion_matrix(y_test,y_pred).ravel()


# In[826]:


df_cm = pd.DataFrame()
df_cm['Predicted Positive'] = [fp,tp]
df_cm['Predicted Negative'] = [tn,fn]
df_cm.index = ['True Negative','True Positive']


# In[827]:


df_cm


# The confusion matrix above proves our hyothesis somehwat wrong. Although the model does predict the possibility of a 2nd harmful event occurring, it does so with fairly low accuracy. The model has a large amount of false positives because recall was chosen as the metric to focus on. But even so, there are still a lot of false negatives, which makes the model not useful. The point of the model is to try and predict when authorities need to employ extra resources to a crash based on the time and place because of the existence of a secondary harmful event. This model would not perform well in the real world and needs more features in training to be useful.

# In[ ]:




