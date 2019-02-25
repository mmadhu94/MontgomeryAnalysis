'''Motor Vehicle Crash Data:
   A Causal and Predictive Analysis
        by Madhukar Mantravadi'''

# imports
from CHAID import Tree
from datetime import datetime
import numpy as np
import pandas as pd
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



#Reading the csv files into pandas dataframe form

df_IncidentData = pd.read_csv('Crash_Reporting_-_Incidents_Data.csv')
df_DriversData = pd.read_csv('Crash_Reporting_-_Drivers_Data.csv')


'''Cleaning and Preprocessing Data'''

#I chose the step here to bin the lat/long by such that it will divide the area that the accidents took place into 1 sq mi incriments
step = 1/70
to_bin = lambda x: np.floor(x / step) * step
df_IncidentData["latbin"] = df_IncidentData.Latitude.map(to_bin)
df_IncidentData["lonbin"] = df_IncidentData.Longitude.map(to_bin)


#Cleaning up the dataframe columns that are relevant
df_DriversData['Injury Severity'] = df_DriversData['Injury Severity'].str.replace('INJURY','')
df_IncidentData['Second Harmful Event'] = df_IncidentData['Second Harmful Event'].replace(np.NaN,'NO 2ND EVENT')


#Engineering new features to utilize time and 'Second Harmful Event' data
df_IncidentData['Second Event Existing'] = [0 if i == 'NO 2ND EVENT' else 1
                                            for i in df_IncidentData['Second Harmful Event']]
df_IncidentData['Hour'] = df_IncidentData['Crash Date/Time'].dt.hour


#Generating dummy column features based on the types of harmful events in the 'First Harmful Event' feature
df_IncDummy = pd.concat([df_IncidentData.copy(),
                         pd.get_dummies(df_IncidentData['First Harmful Event'],prefix = 'Harm')], axis = 1)


#Dropping the original column for redundancy
df_IncDummy.drop('First Harmful Event', axis=1, inplace = True)


#Converting the date/time string data to datetime form using an element-wise apply and a custom function
def timestripper(string):
    return datetime.strptime(string, '%m/%d/%Y %I:%M:%S %p')

df_DriversData['Crash Date/Time'] = df_DriversData['Crash Date/Time'].apply(timestripper)
df_IncidentData['Crash Date/Time'] = df_IncidentData['Crash Date/Time'].apply(timestripper)


#The following is code for several plots looking at frequency of accidents anddifferent incriments of time

#plot for year and month
plt.title('Frequency of Accidents (# of accidents) vs Year and Month (year,month)',fontdict={'fontsize': 30,
        'fontweight' : 'bold'})
df_DriversData['Crash Date/Time'].groupby([df_DriversData['Crash Date/Time'].dt.year,
                                           df_DriversData['Crash Date/Time'].dt.month]).count().plot(figsize = (24,10),
                                                                                                     fontsize = 20, kind="bar",
                                                                                                     width = 0.9, color = 'blue')
plt.savefig('FreqVsYrMnth.png')

#plot for month
plt.title('Frequency of Accidents (# of accidents) vs Month',fontdict={'fontsize': 30,
        'fontweight' : 'bold'})
df_DriversData['Crash Date/Time'].groupby([df_DriversData['Crash Date/Time'].dt.month]).count().plot(figsize = (24,10),
                                                                                                     fontsize = 20, kind="bar",
                                                                                                     width = 0.9, color = 'magenta')
plt.savefig('FreqVsMnth.png')

#Plot for year
plt.title('Frequency of Accidents (# of accidents) vs Year',fontdict={'fontsize': 30,
        'fontweight' : 'bold'})
df_DriversData['Crash Date/Time'].groupby([df_DriversData['Crash Date/Time'].dt.year]).count().plot(figsize = (24,10),
                                                                                                     fontsize = 20, kind="bar",
                                                                                                     width = 0.9, color = 'cyan')
plt.savefig('FreqVsYr.png')

#Plot for hour
df_DriversData['Crash Date/Time'].groupby([df_DriversData['Crash Date/Time'].dt.hour]).count().plot(figsize = (24,10),
                                                                                                     fontsize = 20, kind="bar",
                                                                                                     width = 0.9, color = ['g' if i not in (7,8,9,15,16,17,18) else 'r' for i in range(24)])
plt.savefig('FreqVsHr.png')



#Plotly code for interactive 2d histograms of Hr vs Injury Severity and Hr vs Vehicle Damage Extent

#HR vs Injury
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



#HR vs Damage
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

'''Causal Hypothesis Test and Model'''

#Masks of values to drop to minimize small categories that are magnitudes smaller than the top few classes
light_drops = list(df_DriversData['Light'].value_counts().index[-5:])
vehicle_damage_drops = list(df_DriversData['Vehicle Damage Extent'].value_counts().index[-2:])


#Defining a slice of the original dataframe to use for the test containing only the relevant features
df_CHAID = df_DriversData[['Injury Severity','Vehicle Damage Extent', 'Light']]
df_CHAID.dropna(inplace = True)


#Dropping the values using the maskes created above
mask_light = df_CHAID[df_CHAID['Light'].isin(light_drops)].index
mask_light = list(mask_light)
df_CHAID.drop(mask_light, inplace=True)


mask_vdamage = df_CHAID[df_CHAID['Vehicle Damage Extent'].isin(vehicle_damage_drops)].index
mask_vdamage = list(mask_vdamage)
df_CHAID.drop(mask_vdamage, inplace = True)


#defining Variables to use in the CHAID function
indep_var = ['Light']
dep_var1 = 'Injury Severity'
dep_var2 = 'Vehicle Damage Extent'


#Building the tree
tree1 = Tree.from_pandas_df(df_CHAID, dict(zip(indep_var, ['nominal'] * 3)), dep_var1, max_depth=1)
tree2 = Tree.from_pandas_df(df_CHAID, dict(zip(indep_var, ['nominal'] * 3)), dep_var2, max_depth=1)


#Converting the tree to an object to view it
tree1.to_tree()
tree2.to_tree()


#Printing out the different trees to see the results in raw form
tree1.print_tree()
tree2.print_tree()


#The ratios are the # of worst-case to # of other cases
dark_injury_ratio = (10+181)/(16525+1987+1692)
daylight_injury_ratio = (13+490)/(45478+5898+4410)
dark_ratio/daylight_ratio
# This ratio shows a ~5% higher amount of severe injury during the dark than the daylight.
dark_vdamage_ratio = (255+1050+1050+7143)/(562+83+532+4517+584+4417)
daylight_vdamage_ratio = (1709+19894)/(16375+2305+15396)
dark_vdamage_ratio/daylight_vdamage_ratio
# This ratio shows ~40% more severe damage cases in the dark than during the daylight.



'''Predictive Hypothesis and Model'''


#Splitting the data into training and test sets
X = df_IncDummy[df_IncDummy.columns[31:]]
y = list(df_IncDummy['Second Event Existing'])

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)


#setting seed for reproducibility
seed = 42


# prepare models to choose from, we will tune the one with best score afterwards
models = []
models.append(('LR', LogisticRegression(solver='lbfgs',
                                        max_iter=500,
                                        n_jobs=3,
                                        random_state = seed)))
models.append(('SGD', SGDClassifier(max_iter=np.ceil(10**6 / len(y)),
                                    early_stopping=True,
                                    n_iter_no_change= 10,
                                    n_jobs=3,
                                    random_state = seed)))
models.append(('KNN', KNeighborsClassifier(n_jobs=3)))
models.append(('BRF', BalancedRandomForestClassifier(n_jobs=3,
                                                     random_state = seed)))
models.append(('SVM', SVC(gamma='auto',
                          verbose = True,
                          random_state = seed)))
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


#Box plot of accuracy scores between algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison (accuracy scored)')
ax = fig.add_subplot(111)
plt.boxplot(results_a)
ax.set_xticklabels(names)
plt.savefig('AlgCompAcc.png')
plt.show()


#Box plot of recall scores between algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison (recall scored)')
ax = fig.add_subplot(111)
plt.boxplot(results_r)
ax.set_xticklabels(names)
plt.savefig('AlgCompRec.png')
plt.show()


#Preparing the chosen model for hyperperameter tuning
lr = BalancedRandomForestClassifier(n_jobs=3,
                                    random_state = 42)


#These parameters were chosen because other parameters would not make much of a difference in the score
param_grid = {'criterion':('gini','entropy'),
              'max_features': ('auto','log2')}


#Initializing the cross validated gridsearch over the different parameters
gs = GridSearchCV(estimator = lr, param_grid = param_grid,
                  scoring = {'accuracy', 'recall'},
                  n_jobs = 3,
                  cv = 10,
                  refit='recall',
                  verbose = 2)


#Fitting the gridsearch
gs.fit(X_train,y_train)


#taking a look at the best paramaters by score
print(gs.best_params_)


#Setting the output of the gridsearch to a 'final_pred' variable to verify the strength of the model
final_pred = gs.best_estimator_


#Entering test data
y_pred = final_pred.predict(X_test)


#Generating Confusion Matrix to evaluate model
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()


#Putting confusion matrix into a dataframe for easier viewing
df_cm = pd.DataFrame()
df_cm['Predicted Positive'] = [fp,tp]
df_cm['Predicted Negative'] = [tn,fn]
df_cm.index = ['True Negative','True Positive']


#Printing Confusion Matrix
print(df_cm)
