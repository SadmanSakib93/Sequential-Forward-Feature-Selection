# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:54:12 2019

@author: Sadman Sakib
"""

import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ***** Reading the csv file  ***** 
dfm = pd.read_csv('q3_input.csv', delimiter=',')
# ***** Reading headers ***** 
df_headersName=pd.read_csv('q3_input.csv', nrows=1).columns.tolist()
# ***** class variable ***** 
targetName= 'heating_fuel_solar'
oldPerf= 0
# ***** classifier object ***** 
forest = RandomForestClassifier(n_estimators=100,criterion="entropy")
fclss=[]
new_fclss=[]
# ***** Setting Variables for Sequencial feature selection (SFS) ***** 
sfsFeaturesSelected=['tile_count']
newPerf = oldPerf + 1
totalFolds=5
# ***** Function for making 5 Stratified fold with same proportion of class variables ***** 
def runSCV(trainX,testX):
    # ***** Initializing local Variables ***** 
    allFoldX={"0": [], "1": [], "2": [], "3": [], "4": [] }
    allFoldY={"0": [], "1": [], "2": [], "3": [], "4": [] }
    ftable=testX.value_counts()
    for sn in (range((len(testX.unique())))):#loop for differet possible class variable values
        # ***** Generating frequency table for each class variable ***** 
        featureVal= ftable.index.values
        # ***** Finding the indices of the unique class variable values ***** 
    for featureValIndx in range(len(featureVal)):
        current_class= featureVal[featureValIndx]
        inx= testX.index[testX==current_class].tolist()
        fclss=[{0 : current_class, 1 : inx}]
        new_fclss.append(fclss)
    foldX_All=dict.fromkeys(["0", "1", "2", "3", "4"])
    # ***** making proportional splits ***** 
    for i in range(len(featureVal)):
        inx_append=new_fclss[i][0][1]        
        for item in range(len(inx_append)):
            X_list= trainX.iloc[inx_append[item]]
            Y_list= testX.iloc[inx_append[item]]            
            allFoldX[str(item%totalFolds)].append(X_list)
            allFoldY[str(item%totalFolds)].append(Y_list)
            valX=allFoldX[str(item%5)]            
            foldX_All[str(item%totalFolds)]=pd.concat([valX[i] for i in range(len(valX))],axis=1).T                
    # ***** creating combinations for 5 fols ***** 
    foldsAllFive=[foldX_All["0"], foldX_All["1"], foldX_All["2"], foldX_All["3"]]
    fold_Xtrain= pd.concat(foldsAllFive)
    fold_Xtest=np.concatenate((allFoldY["0"], allFoldY["1"], allFoldY["2"], allFoldY["3"]), axis=0)
# ***** Returning Splits for training & testing ***** 
    return fold_Xtrain, fold_Xtest, foldX_All["4"], allFoldY["4"]
def fiveFoldPerf(performance,trainX,testX):
    dx, lx, dy, ly= runSCV(trainX,testX)
    forest.fit(dx, lx)
    prediction = forest.predict(dy)
    #calculating Mean Accuracy
    newPerf_temp= 100-(np.square(np.subtract(np.array(ly), prediction)).mean())
    performance.append(newPerf_temp)
    return performance
print("Running ...")
# ***** loop to incrementally add features ***** 
for i in range(1,len(df_headersName)-1):
    # ***** keep selecting eature if accuracy improves *****  
    performance=[]
    # ***** check to see if performance is increasing ***** 
    if (newPerf>oldPerf):
        # ***** Adding feature ***** 
        sfsFeaturesSelected.append(df_headersName[i])
        oldPerf= newPerf
    trainX=dfm[sfsFeaturesSelected]
    testX=dfm[targetName]
    # ***** Running 5 fold S-CV ***** 
    for iterations in range(0,totalFolds): 
        fiveFoldPerf(performance,trainX,testX)
    performance=np.array(performance)
    newPerf=performance.mean()
# ***** PRINTING IMPORTANT FEATURES ON CONSOLE ***** 
def displayResults(selectedAttrb, acc):       
    print("Selected Features Using SFS:")
    for featureIndx in range(len(selectedAttrb)):
            print(str(featureIndx+1)+". "+selectedAttrb[featureIndx]) 
    print('Accuracy using this feature set: ', acc,'%') 
displayResults(sfsFeaturesSelected, newPerf) 




