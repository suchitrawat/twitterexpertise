import pandas as pd
import os
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import normalized_mutual_info_score
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

df = pd.read_csv('/course/a2/twitter/RPI_Expertise_2016_Features.csv')
df.to_csv('data.csv')



""" during the PreProcessing parts we starting from remove all the unnamed columns of the datasets and then
extract the columns we are working on, then remove all the zeros (keep the data uniform for the model calculation)
string values of 'none' and nan values """


df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
comma_model = df[['utype', 'commas', 'colon','exmark','quesmark', 'semi','periods']].copy()

# removing all columns with zeros values
comma_model = comma_model[~(comma_model[comma_model.columns[1:]] ==0).any(axis = 1)]
#print(comma_model.head(50), end= "\n\n")

# remove the string values of 'none'
slang_sentiment_model = df[['utype', 'slang','sentiment']].copy()
slang_sentiment_model = slang_sentiment_model[(slang_sentiment_model.astype(str).applymap(lambda x : x.lower()) != 'none')]
#slang_sentiment_model = slang_sentiment_model[(slang_sentiment_model.astype(str).applymap(lambda x : x.lower()) != 'nan')]

slang_sentiment_model['slang'] = slang_sentiment_model['slang'].astype(float)
slang_sentiment_model['sentiment'] = slang_sentiment_model['sentiment'].astype(float)

slang_sentiment_model = slang_sentiment_model[(slang_sentiment_model[['slang', 'sentiment']]!=0).all(axis = 1)]

# drop all the nan values
slang_sentiment_model.dropna(axis = 'rows', inplace = True)

#print(slang_sentiment_model.head(50), end = "\n\n")



"""# KNN Nearest neighbour starting from training the data split and use the trained data pattern
to predict the testing data"""

# Test the slang_sentiment_model first
print(len(slang_sentiment_model))

# Divide the data into dependent and independent variables and training the test data splits

x1 = slang_sentiment_model.drop(['utype'], axis = 'columns')
y1 = slang_sentiment_model.utype

x1_train,x1_test,y1_train, y1_test =  train_test_split(x1, y1, test_size= 0.2, random_state= 1)

# choose number of neighbors (the neighbors value is lower than the neighbors of commas model because we only test against two)
knn = KNeighborsClassifier(n_neighbors=5)
# fit into the model and makes prediction
knn.fit(x1_train,y1_train)
y1_pred = knn.predict(x1_test)
accuarcy_rate = metrics.accuracy_score(y1_test, y1_pred)

#(UNCOMMENT TO CHECK ACCURACY FOR THE SLANG SENTIMENT MODEL)print(accuarcy_rate, end = "the accuarcy_rate of slang sentiment model")


# use confusion matrix to visualize it

cm = confusion_matrix(y1_test, y1_pred)
plt.figure(figsize = (7,5))
sn.heatmap(cm, annot = True)
plt.xlabel("Predict")
plt.ylabel("Actual")
plt.savefig('slang_sentiment_confusion matrix')


comma_model.dropna(axis = 'rows', inplace = True)

""" Testing Normalised Mutual Information index to usertype from all the variables we are testing """

# 3 bin equal frequency discresation
equal_width = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')

# binning
comma_model['binned_commas'] = equal_width.fit_transform(comma_model[['commas']]).astype(float)
comma_model['binned_colon'] = equal_width.fit_transform(comma_model[['colon']]).astype(float)
comma_model['binned_exmark'] = equal_width.fit_transform(comma_model[['exmark']]).astype(float)
comma_model['binned_quesmark'] = equal_width.fit_transform(comma_model[['quesmark']]).astype(float)
comma_model['binned_semi'] = equal_width.fit_transform(comma_model[['semi']]).astype(float)
comma_model['binned_periods'] = equal_width.fit_transform(comma_model[['periods']]).astype(float)

slang_sentiment_model['binned_slang'] = equal_width.fit_transform(slang_sentiment_model[['slang']]).astype(float)
slang_sentiment_model['binned_sentiment'] = equal_width.fit_transform(slang_sentiment_model[['sentiment']]).astype(float)

# calculate NMI
nmi_commas_utype = normalized_mutual_info_score(comma_model['utype'], comma_model['binned_commas'], average_method='min')
nmi_colon_utype = normalized_mutual_info_score(comma_model['utype'], comma_model['binned_colon'], average_method='min')
nmi_exmark_utype = normalized_mutual_info_score(comma_model['utype'], comma_model['binned_exmark'], average_method='min')
nmi_quesmark_utype = normalized_mutual_info_score(comma_model['utype'], comma_model['binned_quesmark'], average_method='min')
nmi_semi_utype = normalized_mutual_info_score(comma_model['utype'], comma_model['binned_semi'], average_method='min')
nmi_periods_utype = normalized_mutual_info_score(comma_model['utype'], comma_model['binned_periods'], average_method='min')

""" UNCOMMENT TO CHECK THE NMI VALUES
print('\n')
print('NMI between usertype and commas: ', nmi_commas_utype)
print('NMI between usertype and colon: ', nmi_colon_utype)
print('NMI between usertype and exclaimation mark: ', nmi_exmark_utype)
print('NMI between usertype and question mark: ', nmi_quesmark_utype)
print('NMI between usertype and semi-colon: ', nmi_semi_utype)
print('NMI between usertype and periods: ', nmi_periods_utype)
"""

nmi_slang_utype = normalized_mutual_info_score(slang_sentiment_model['utype'], slang_sentiment_model['binned_slang'], average_method='min')
nmi_sentiment_utype = normalized_mutual_info_score(slang_sentiment_model['utype'], slang_sentiment_model['binned_sentiment'], average_method='min')

print('NMI between usertype and slang words: ', nmi_slang_utype)
print('NMI between usertype and sentiment words: ', nmi_sentiment_utype, end= "\n\n")

# all nmi of commas, exmark, quesmark, periods < 0.1, very low correlation
# 0.1 < nmi of colon, semi, slang, sentiment < 0.3 so relatively correlated

nmi_values = [nmi_commas_utype,nmi_colon_utype,nmi_exmark_utype,nmi_quesmark_utype,nmi_semi_utype,nmi_periods_utype,nmi_slang_utype,nmi_sentiment_utype]
rounded_nmi = [round(no, 3) for no in nmi_values]

#annotate = ['nmi_commas_utype','nmi_colon_utype','nmi_exmark_utype','nmi_quesmark_utype','nmi_semi_utype','nmi_periods_utype','nmi_slang_utype','nmi_sentiment_utype']


# plot graph and visualize the NMI Values of each variables

x_axis = np.arange(len(nmi_values))
fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(9)
p1 = ax.bar(x_axis, rounded_nmi, color = 'r')

ax.tick_params(axis='x', colors='red')
ax.set_xticks(x_axis, labels =['commas','colon','exmark','quesmark','semicolon','periods','slang','sentiment'])
#ax.set_yticks(x_axis, np.arange(0,0.3,0.05))
ax.set_ylabel('NMI Values')

ax.bar_label(p1, label_type = 'center')

plt.savefig("NMI graph2.png")
#

# starts working on comma model to test the KNN 
x2 = comma_model.drop(['utype'], axis = 'columns')
y2 = comma_model.utype

# train the dataset into two parts (80%, 20%)
x2_train, x2_test,y2_train,y2_test = train_test_split(x2,y2,test_size=0.2, random_state=1)


# set the neighbour and predict the testing values

knn1 = KNeighborsClassifier(n_neighbors=6)
knn1.fit(x2_train,y2_train)
y2_pred = knn1.predict(x2_test)

# (UNCOMMENT TO CHECK ACCURACY FOR COMMAS MODEL) print(knn1.score(x2_test,y2_test))


# visualize into confusion matrix

cm1 = confusion_matrix(y2_test,y2_pred)

plt.figure(figsize= (7,5))

sn.heatmap(cm1, annot = True)

plt.xlabel('Predict')
plt.ylabel('Actual')
plt.savefig('comma_model_confusion_matrix')



# Reduce the model to four variables that has highest Mutual information values and repeat the process of model1 and model2

new = df[['utype', 'commas', 'colon','exmark','quesmark', 'semi','periods', 'slang', 'sentiment']].copy()
new = new.drop(['commas', 'exmark','quesmark', 'periods'], axis = 'columns')
new = new[(new.astype(str).applymap(lambda x : x.lower()) != 'none')]
new = new[(new.astype(str).applymap(lambda x : x.lower()) != 'nan')]

new['slang'] = new['slang'].astype(float)
new['sentiment'] = new['sentiment'].astype(float)
new['semi'] = new['semi'].astype(float)
new['colon'] = new['colon'].astype(float)

new.dropna(axis = 'rows', inplace = True)

x3 = new.drop(['utype'], axis = 'columns')
y3 = new.utype
x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3, test_size=0.18, random_state=1)

knn3 = KNeighborsClassifier(n_neighbors=6)
knn3.fit(x3_train,y3_train)
  # (UNCOMMENT TO CHECK THE ACCURACY FOR THE FINAL MODEL) print('the accuarcy rate for the new model:  ' +  str(knn3.score(x3_test,y3_test)))
y3_pred = knn3.predict(x3_test)
cm2 = confusion_matrix(y3_test, y3_pred)
# annotate1 = ['random','between 10 - 3msgs', 'between100-10msgs','expert','friend','mentioned','morethan100msgs']

plt.figure(figsize= (7,6))
sn.heatmap(cm2, annot = True)
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.savefig('new model')

    









