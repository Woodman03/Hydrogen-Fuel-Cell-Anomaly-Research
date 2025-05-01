# NASA Hydrogen Fuel Cell Anomaly Research (Supervised Machine Learning)
# Dillon Wood - Arkansas Tech University - B.S. Electrical Engineering 2025

#Required Libraries
import pandas as pd
import numpy as np
import researchpy as rp
import scipy.stats as stats
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
import shap
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

from numpy.random import seed
from numpy.random import randn
from numpy.random import normal
from scipy.stats import ttest_ind
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import roc_curve, auc
    #Data Preprocessing

from sklearn.ensemble import AdaBoostClassifier



def data_spliting(X,y, test_size = 0.2, smote=True):
    
    if smote:
        #Balancing the data
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)

    #Test train split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size, random_state = 101)


    return X_train, X_test, y_train, y_test

#Confusion Matrix Calculation
def conf_matrix_cal(y_test, y_pred):
  # Creating the confusion matrix
  cm = metrics.confusion_matrix(y_test, y_pred)
  # Assigning columns names
  cm_df = pd.DataFrame(cm, 
            columns = ['Predicted Negative', 'Predicted Positive'],
            index = ['Actual Negative', 'Actual Positive'])
  # Showing the confusion matrix
  kappa = cohen_kappa_score(y_test, y_pred)
  print(cm_df)
  return cm, kappa 


#Creating Confusion Matrix
def confusion_mxtrix (conf_matrix):

    results = []
# save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print('-'*50)
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Precision: {round(conf_precision,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')
    
# ----------------- Logistic Regression -----------------

#10 by 10 Fold Logistic Regression
def logistic_regression_10x10(X,y):
    model = LogisticRegression(solver='saga', tol=0.0001, max_iter=800,random_state=0,C = 2.4110535042865755, penalty= 'l1')
    cv = RepeatedKFold(n_splits=10, random_state=1, n_repeats=10)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    print("Mean Accuracy: ",np.mean(scores))
    scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    print("Mean ROC_AUC: ",np.mean(scores))
    scores = cross_val_score(model, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
    print("Mean F1 score: ",np.mean(scores))
    scores = cross_val_score(model, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
    print("Mean balanced_accuracy: ",np.mean(scores))
    #Logistic Regression
def logistic_regression(X_train, X_test, y_train, y_test):
        model_Logistic = LogisticRegression(solver='saga', tol=0.0001, max_iter=800,random_state=0,C = 2.4110535042865755, penalty= 'l1')
        model_Logistic.fit(X_train,y_train)
        y_pred = model_Logistic.predict(X_test)
        print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
        print("area under curve (auc): ", metrics.roc_auc_score(y_test, y_pred))
        cm,kappa = conf_matrix_cal(y_test,y_pred)
        confusion_mxtrix (cm)
        print('Kappa: ', kappa)
        y_train_pred = model_Logistic.predict(X_train)   
        
        train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
        test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)
        
        roc_auc_train = auc(train_fpr, train_tpr)
        roc_auc_test = auc(test_fpr, test_tpr)
        plt.grid()
        
        plt.plot(train_fpr, train_tpr, label=' AUC TRAIN = %0.3f' % roc_auc_train)
        plt.plot(test_fpr, test_tpr, label=" AUC TEST = %0.3f" % roc_auc_test)
        plt.plot([0,1],[0,1],'g--')
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("AUC(ROC curve)")
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        plt.show()

# ----------------- Random Forest -----------------

def random_forest_classifier_10x10(X, y):
        model_RF = RandomForestClassifier(max_depth=7, random_state=42,criterion='gini',n_estimators=10,min_samples_leaf=1, min_samples_split=3)
        cv = RepeatedKFold(n_splits=10, random_state=42, n_repeats=10)
        scores1 = cross_val_score(model_RF, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        print("Mean Accuracy: ",np.mean(scores1))
        scores2 = cross_val_score(model_RF, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
        print("Mean ROC_AUC: ",np.mean(scores2))
        F1_scores = cross_val_score(model_RF, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
        print("Mean F1 score: ",np.mean(F1_scores))
        scores4 = cross_val_score(model_RF, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
        print("Mean balanced_accuracy: ",np.mean(scores4))
        #Random Forest Classifier
def random_forest_classifier(X_train, X_test, y_train, y_test):
            model_RF = RandomForestClassifier(max_depth=10, random_state=1,n_estimators=10,min_samples_leaf=1, min_samples_split=2)
            model_RF.fit(X_train,y_train)
            y_pred = model_RF.predict(X_test)
            
            print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
            print("area under curve (auc): ", metrics.roc_auc_score(y_test, y_pred))
            cm,kappa = conf_matrix_cal(y_test,y_pred)
            confusion_mxtrix (cm)
            print('Kappa: ', kappa)
            y_train_pred = model_RF.predict(X_train)   
            
            train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
            test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)
            
            roc_auc_train = auc(train_fpr, train_tpr)
            roc_auc_test = auc(test_fpr, test_tpr)
            plt.grid()
            
            plt.plot(train_fpr, train_tpr, label=' AUC TRAIN = %0.3f' % roc_auc_train)
            plt.plot(test_fpr, test_tpr, label=" AUC TEST = %0.3f" % roc_auc_test)
            plt.plot([0,1],[0,1],'g--')
            plt.legend()
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("AUC(ROC curve)")
            plt.grid(color='black', linestyle='-', linewidth=0.5)
            plt.show()
            
# ----------------- DA Classifier -----------------

def DA_classifier_10x10(X_train, y_train):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
        model_DA = QuadraticDiscriminantAnalysis()
        cv = RepeatedKFold(n_splits=10, random_state=1, n_repeats=10)
        scores = cross_val_score(model_DA, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        
        print("Mean Accuracy: ",np.mean(scores))
        
def DA_classifier(X_train, X_test, y_train, y_test):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
    model_DA = QuadraticDiscriminantAnalysis()
    model_DA.fit(X_train, y_train)
    y_pred = model_DA.predict(X_test)
    
    print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
    print("area under curve (auc): ", metrics.roc_auc_score(y_test, y_pred))
    cm,kappa = conf_matrix_cal(y_test,y_pred)
    confusion_mxtrix (cm)
    print('Kappa: ', kappa)
    y_train_pred = model_DA.predict(X_train)   
    
    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)
    
    roc_auc_train = auc(train_fpr, train_tpr)
    roc_auc_test = auc(test_fpr, test_tpr)
    plt.grid()
    
    plt.plot(train_fpr, train_tpr, label=' AUC TRAIN = %0.3f' % roc_auc_train)
    plt.plot(test_fpr, test_tpr, label=" AUC TEST = %0.3f" % roc_auc_test)
    plt.plot([0,1],[0,1],'g--')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC(ROC curve)")
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.show()
#Fit the QDA model


# ----------------- SVM -----------------

        #10 by 10 Fold Support Vector Machine
def SVM_10x10(X_train, y_train):
            model_svm = SVC(random_state=0, kernel= 'rbf', gamma = 'scale', C=300)
            cv = RepeatedKFold(n_splits=10, random_state=1, n_repeats=1)
            scores = cross_val_score(model_svm, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
            print("Mean Accuracy: ",np.mean(scores))
            scores = cross_val_score(model_svm, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
            print("Mean ROC_AUC: ",np.mean(scores))
            scores = cross_val_score(model_svm, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
            print("Mean F1 score: ",np.mean(scores))
            scores = cross_val_score(model_svm, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
            print("Mean balanced_accuracy: ",np.mean(scores))

        #Support Vector Machine
def SVM(X_train, X_test, y_train, y_test):
            model_svm = SVC(random_state=0, kernel= 'rbf', gamma = 'scale', C=300,probability=True)
            model_svm.fit(X_train,y_train)
            y_pred = model_svm.predict(X_test)
            print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
            print("area under curve (auc): ", metrics.roc_auc_score(y_test, y_pred))
            cm,kappa = conf_matrix_cal(y_test,y_pred)
            confusion_mxtrix (cm)
            print('Kappa: ', kappa)
            y_train_pred = model_svm.predict(X_train)   
            
            train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
            test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)
            roc_auc_train = auc(train_fpr, train_tpr)
            roc_auc_test = auc(test_fpr, test_tpr)
            plt.grid()
           
            plt.plot(train_fpr, train_tpr, label=' AUC TRAIN = %0.3f' % roc_auc_train)
            plt.plot(test_fpr, test_tpr, label=" AUC TEST = %0.3f" % roc_auc_test)
            plt.plot([0,1],[0,1],'g--')
            plt.legend()
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("AUC(ROC curve)")
            plt.grid(color='black', linestyle='-', linewidth=0.5)
            plt.show()
            
# ----------------- Tree -----------------

def tree_10x10(X_train, y_train):
            from sklearn.tree import DecisionTreeClassifier
            model_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            cv = RepeatedKFold(n_splits=10, random_state=1, n_repeats=1)
            scores = cross_val_score(model_tree, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
            print("Mean Accuracy: ",np.mean(scores))
            scores = cross_val_score(model_tree, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
            print("Mean ROC_AUC: ",np.mean(scores))
            scores = cross_val_score(model_tree, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
            print("Mean F1 score: ",np.mean(scores))
            scores = cross_val_score(model_tree, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
            print("Mean balanced_accuracy: ",np.mean(scores))
            
        #Support Vector Machine
def tree(X_train, X_test, y_train, y_test):
            from sklearn.tree import DecisionTreeClassifier        
            model_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            model_tree.fit(X_train,y_train)
            y_pred = model_tree.predict(X_test)
            print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
            print("area under curve (auc): ", metrics.roc_auc_score(y_test, y_pred))
            cm,kappa = conf_matrix_cal(y_test,y_pred)
            confusion_mxtrix (cm)
            print('Kappa: ', kappa)
            y_train_pred = model_tree.predict(X_train)   
            
            train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
            test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)
            roc_auc_train = auc(train_fpr, train_tpr)
            roc_auc_test = auc(test_fpr, test_tpr)
            plt.grid()
           
            plt.plot(train_fpr, train_tpr, label=' AUC TRAIN = %0.3f' % roc_auc_train)
            plt.plot(test_fpr, test_tpr, label=" AUC TEST = %0.3f" % roc_auc_test)
            plt.plot([0,1],[0,1],'g--')
            plt.legend()
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("AUC(ROC curve)")
            plt.grid(color='black', linestyle='-', linewidth=0.5)
            plt.show()  
            
# ----------------- IDK -----------------

def Adaboost(X_train, X_test, y_train, y_test):
            #from sklearn.ensemble import AdaBoostClassifier  
            model_ada = DecisionTreeClassifier(random_state = 42)
            model_ada.fit(X_train,y_train)
            y_pred = model_ada.predict(X_test)
            print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
            print("area under curve (auc): ", metrics.roc_auc_score(y_test, y_pred))
            cm,kappa = conf_matrix_cal(y_test,y_pred)
            confusion_mxtrix (cm)
            print('Kappa: ', kappa)
            y_train_pred = model_ada.predict(X_train)   
            
            train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
            test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)
            roc_auc_train = auc(train_fpr, train_tpr)
            roc_auc_test = auc(test_fpr, test_tpr)
            plt.grid()
           
            plt.plot(train_fpr, train_tpr, label=' AUC TRAIN = %0.3f' % roc_auc_train)
            plt.plot(test_fpr, test_tpr, label=" AUC TEST = %0.3f" % roc_auc_test)
            plt.plot([0,1],[0,1],'g--')
            plt.legend()
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("AUC(ROC curve)")
            plt.grid(color='black', linestyle='-', linewidth=0.5)
            plt.show()   
def Adaboost_ptimization(X_train, y_train, X_test, y_test):
  
  from sklearn.ensemble import AdaBoostClassifier
  model9 = AdaBoostClassifier()
  model9.fit(X_train, y_train)
  print('[9]Adaboost Classifier Training Accuracy:', model9.score(X_train, y_train))
  y_pred9 = model9.predict(X_test)
  cm_AdaBoost = confusion_matrix(y_test, y_pred9)
  print('[9]AdaBoost Classifier Testing Accuracy:', model9.score(X_test,y_test))
  cm,kappa = conf_matrix_cal(y_test,y_pred)
  confusion_mxtrix (cm)
  print('Kappa: ', kappa)
  weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)
  for nx in range(50, 1050, 50):
     model_ada = AdaBoostClassifier(base_estimator=weak_learner,n_estimators=nx, learning_rate=1.0, algorithm='SAMME.R', random_state=42)
     model_ada.fit(X_train,y_train)
     print('Training Accuracy:', model_ada.score(X_train, y_train),'Testing Accuracy:', model_ada.score(X_test,y_test), 'n_neighbor:', nx) 
     #print(' Testing Accuracy:', model_knn.score(X_test,y_test)) 
  error_uniform = []
  error_distance = []   
  k_range = range(50, 1050, 50)
  for nx in k_range:
    
    clf1 = AdaBoostClassifier(base_estimator=weak_learner,n_estimators=nx, learning_rate=1.0, algorithm='SAMME', random_state=42)
    clf1.fit(X_train, y_train)
    predictions1 = clf1.predict(X_test)
    error_uniform.append(1 - accuracy_score(y_test, predictions1))
    
    clf2 = AdaBoostClassifier(base_estimator=weak_learner,n_estimators=nx, learning_rate=1.0, algorithm='SAMME.R', random_state=42)
    clf2.fit(X_train, y_train)
    predictions2 = clf2.predict(X_test)
    error_distance.append(1 - accuracy_score(y_test, predictions2))   
  import plotly.graph_objects as go
  from sklearn.datasets import make_blobs
        
  error_df = pd.DataFrame((zip(k_range, error_uniform, error_distance)),
               columns =['n_estimators', 'Error_SAMME', 'Error_SAMME.R'])
    
  error_fig = go.Figure()

  # error_plots=[go.Scatter(x=error_df['n_estimators'], y=error_df['Error_SAMME'], name='Error_SAMME',line=dict(color='firebrick', width=3)),
  #               go.Scatter(x=error_df['n_estimators'], y=error_df['Error_SAMME.R'], name='Error_SAMME.R', line=dict(color='royalblue', width=3))]
    
  # error_fig = go.Figure(data=error_plots)
  # error_fig.update_layout(yaxis_range=[0.04,0.09])

  # error_fig.update_layout(height=400, width=900, title_text='<b>Error SAMME vs SAMME.R', title_x=0.5,
  #                   font_size=14, template='plotly_dark')

  # error_fig.show()
  plt.grid()
  
  plt.plot(error_df['n_estimators'], error_df['Error_SAMME'], label='SAMME')
  plt.plot(error_df['n_estimators'], error_df['Error_SAMME.R'], label='SAMME.R')
  
  
  plt.legend()
  plt.xlabel("No of estimators")
  plt.ylabel("Error rate")
  plt.title("Error")
  plt.grid(color='black', linestyle='-', linewidth=0.5)
  plt.show()
def AdaBoost_10x10(X_train, y_train):
          model_ada = AdaBoostClassifier()
          cv = RepeatedKFold(n_splits=10, random_state=1, n_repeats=1)
          scores1 = cross_val_score(model_ada, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
          scores2 = cross_val_score(model_ada, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
          #print("Mean ROC_AUC: ",np.mean(scores2))
          scores3 = cross_val_score(model_ada, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
          scores4 = cross_val_score(model_knn,  X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
          print("Mean Accuracy: ",np.mean(scores1),"Mean ROC_AUC: ",np.mean(scores2),"Mean F1 score: ",np.mean(scores3),"Mean balanced_accuracy: ",np.mean(scores4))

    
def KNN_ptimization(X_train, y_train, X_test, y_test):
  for nx in range(2, 30, 1):
     model_knn = KNeighborsClassifier(n_neighbors=nx, weights='distance')
     model_knn.fit(X_train,y_train)
     print('Training Accuracy:', model_knn.score(X_train, y_train),'Testing Accuracy:', model_knn.score(X_test,y_test), 'n_neighbor:', nx) 
     #print(' Testing Accuracy:', model_knn.score(X_test,y_test)) 
  error_uniform = []
  error_distance = []   
  k_range = range(1, 28)
  for k in k_range:
    
    clf1 = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    clf1.fit(X_train, y_train)
    predictions1 = clf1.predict(X_test)
    error_uniform.append(1 - accuracy_score(y_test, predictions1))
    
    clf2 = KNeighborsClassifier(n_neighbors=k, weights='distance')
    clf2.fit(X_train, y_train)
    predictions2 = clf2.predict(X_test)
    error_distance.append(1 - accuracy_score(y_test, predictions2))   
  import plotly.graph_objects as go
  from sklearn.datasets import make_blobs
        
  error_df = pd.DataFrame((zip(k_range, error_uniform, error_distance)),
               columns =['K', 'Error_Uniform', 'Error_Distance'])
    
  error_fig = go.Figure()

  error_plots=[go.Scatter(x=error_df['K'], y=error_df['Error_Uniform'], name='Error_Uniform',line=dict(color='firebrick', width=3)),
                go.Scatter(x=error_df['K'], y=error_df['Error_Distance'], name='Error_Distance', line=dict(color='royalblue', width=3))]
    
  error_fig = go.Figure(data=error_plots)
  error_fig.update_layout(yaxis_range=[0.04,0.09])

  error_fig.update_layout(height=400, width=900, title_text='<b>Error Uniform vs Error Distance', title_x=0.5,
                    font_size=14, template='plotly_dark')

  error_fig.show()
  plt.grid()
  
  plt.plot(error_df['K'], error_df['Error_Uniform'], label='Error Uniform')
  plt.plot(error_df['K'], error_df['Error_Distance'], label='Error Distance')
  
  
  plt.legend()
  plt.xlabel("No of neighbors")
  plt.ylabel("Error rate")
  plt.title("Error")
  plt.grid(color='black', linestyle='-', linewidth=0.5)
  plt.show()
  
def KNN(X_train, X_test, y_train, y_test):
          model_knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
          model_knn.fit(X_train,y_train)
          y_pred = model_knn.predict(X_test)
          for i in range (len(y_pred)):
              if y_pred[i] < 0.5:
                  y_pred[i] = 0
              else:
                  y_pred[i] = 1
          print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
          print("area under curve (auc): ", metrics.roc_auc_score(y_test, y_pred))
          cm,kappa = conf_matrix_cal(y_test,y_pred)
          confusion_mxtrix (cm)
          print('Kappa: ', kappa)
          y_train_pred = model_knn.predict(X_train)   
          print("Training Accuracy: ",metrics.accuracy_score(y_train,y_train_pred))
          train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
          test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)
          roc_auc_train = auc(train_fpr, train_tpr)
          roc_auc_test = auc(test_fpr, test_tpr)
          plt.grid()
          
          plt.plot(train_fpr, train_tpr, label=' AUC TRAIN = %0.3f' % roc_auc_train)
          plt.plot(test_fpr, test_tpr, label=" AUC TEST = %0.3f" % roc_auc_test)
          plt.plot([0,1],[0,1],'g--')
          plt.legend()
          plt.xlabel("False Positive Rate")
          plt.ylabel("True Positive Rate")
          plt.title("AUC(ROC curve)")
          plt.grid(color='black', linestyle='-', linewidth=0.5)
          plt.show()

def KNN_10x10(X_train, y_train):
          model_knn = KNeighborsClassifier(n_neighbors=4,weights='distance')
          cv = RepeatedKFold(n_splits=10, random_state=1, n_repeats=1)
          scores1 = cross_val_score(model_knn, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
          scores2 = cross_val_score(model_knn, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
          #print("Mean ROC_AUC: ",np.mean(scores2))
          scores3 = cross_val_score(model_knn, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
          scores4 = cross_val_score(model_knn,  X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
          print("Mean Accuracy: ",np.mean(scores1),"Mean ROC_AUC: ",np.mean(scores2),"Mean F1 score: ",np.mean(scores3),"Mean balanced_accuracy: ",np.mean(scores4))

def NB_10x10(X_train, y_train):
          model_nb = GaussianNB()
          cv = RepeatedKFold(n_splits=10, random_state=1, n_repeats=1)
          scores = cross_val_score(model_nb, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
          print("Mean Accuracy: ",np.mean(scores))

def Naive_Bayes(X_train, X_test, y_train, y_test):
    #Calling the Class
    naive_bayes = GaussianNB()
    #Fitting the data to the classifier
    naive_bayes.fit(X_train , y_train)
    #Predict on test data
    y_pred = naive_bayes.predict(X_test)
    for i in range (len(y_pred)):
       if y_pred[i] < 0.5:
           y_pred[i] = 0
       else:
           y_pred[i] = 1
    print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
    print("area under curve (auc): ", metrics.roc_auc_score(y_test, y_pred))
    cm,kappa = conf_matrix_cal(y_test,y_pred)
    confusion_mxtrix (cm)
    print('Kappa: ', kappa)  
    y_train_pred = naive_bayes.predict(X_train)   
    
    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)
    roc_auc_train = auc(train_fpr, train_tpr)
    roc_auc_test = auc(test_fpr, test_tpr)
    plt.grid()
    
    plt.plot(train_fpr, train_tpr, label=' AUC TRAIN = %0.3f' % roc_auc_train)
    plt.plot(test_fpr, test_tpr, label=" AUC TEST = %0.3f" % roc_auc_test)
    
    plt.plot([0,1],[0,1],'g--')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC(ROC curve)")
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.show()     


#SHAP value calculation
def shap_value(X_train, X_test, y_train,y_test):
            model_RF = RandomForestRegressor(max_depth=7,n_estimators=10,min_samples_leaf=1, min_samples_split=3)#KNeighborsRegressor(n_neighbors=4)
            model_RF.fit(X_train,y_train)
            y_pred = model_RF.predict(X_test)

            for i in range (len(y_pred)):
                if y_pred[i] < 0.5:
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1

            print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
            import shap
            explainer = shap.TreeExplainer(model_RF)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test)
            shap.summary_plot(shap_values, X_test, plot_type="bar")
            for name in X_test.columns:
                shap.dependence_plot(name, shap_values, X_test)
    
    
# Feature Importance Analysis (using Random Forest as an example)
def plot_feature_importance(X_train, y_train, feature_names):
    """
    Trains a Random Forest model and plots the feature importances as a bar graph.

    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training labels.
        feature_names (list): List of feature names.
    """
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import seaborn as sns

    model_RF = RandomForestClassifier(n_estimators=100, random_state=1)
    model_RF.fit(X_train, y_train)

    importances = model_RF.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance for Anomaly Detection')
    plt.show()
        
    
    #main function
if __name__ == "__main__":

                  
            df = pd.read_csv('nasabattery_with_anomaliesTest_new2.csv')
            df = df.dropna(axis=1)

            # Convert columns to numeric, handling errors
            for column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')

            # Drop columns that could not be converted to numeric
            df = df.dropna(axis=1, how='all') #drops columns where all values are NaN.
            df = df.dropna(axis=0) #drops rows where any value is NaN.

            # Normalization
            for column in df.columns:
                df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

            y = df["anomaly"]
            sns.countplot(y, label="Count")
            plt.show()

            X = df[["capacity", "temperature_measured", "measured_power", "load_power", "power_difference", "current_load", "voltage_load",
                    "average_power", "temperature_difference", "load_resistance" ]]
            print(y.value_counts())
                        
            import re
            oversample = SMOTE()
            X, y = oversample.fit_resample(X, y)
            sns.countplot(y,label="Count")
            
            y.value_counts() #data is blanced after applying smote
            
            # from sklearn.preprocessing import StandardScaler
            # scaler=StandardScaler()
            # scaler.fit(X)
            # scaler.transform(X)
            #smote
            test_size=0.2
            #Test train split
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)
            
            from sklearn.decomposition import PCA
            import mpl_toolkits.mplot3d  # Import for 3D plotting
            # Apply SMOTE for balancing the data
            
            # Apply PCA with 3 components
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X)

            # Create a 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot with color coding for anomalies
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', s=50)

            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')
            ax.set_title('3D PCA of Battery Data with Anomalies')

            plt.show()
                
            #from sklearn.model_selection import train_test_split
           # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 17)
          
           # Feature Importance Plot
            plot_feature_importance(X_train, y_train, X.columns) # X.columns contains the feature names.
            
            from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import classification_report
            from sklearn.linear_model import LogisticRegression
            log = LogisticRegression(random_state = 1)
            log.fit(X_train, y_train)
            #Using SVC linear
            from sklearn.svm import SVC
            svc_lin = SVC(kernel = 'linear', random_state = 1)
            svc_lin.fit(X_train, y_train)
            #Using SVC rbf
            from sklearn.svm import SVC
            svc_rbf = SVC(kernel = 'rbf', random_state = 1)
            svc_rbf.fit(X_train, y_train)
            #Using DecisionTreeClassifier 
            from sklearn.tree import DecisionTreeClassifier
            tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
            tree.fit(X_train, y_train)
            #Using Random Forest Classifier
            from sklearn.ensemble import RandomForestClassifier
            fruits = ["log2", "sqrt", "auto"]
            RF = RandomForestClassifier(n_estimators =100, random_state = 1,max_depth=None)
            RF.fit(X_train, y_train)  
            RF_y_pred = RF.predict(X_test)
            accuracy = accuracy_score(RF_y_pred,y_test)
            cm = confusion_matrix(y_test, RF_y_pred)
            print('Train Accuracy: ', RF.score(X_train, y_train), 'Test Accuracy: ', accuracy)
                #print(classification_report(y_test,RF_y_pred ))
                
            from sklearn.neighbors import KNeighborsClassifier
            model_knn = KNeighborsClassifier(n_neighbors=10)
            model_knn.fit(X_train,y_train)
            y_pred = model_knn.predict(X_test)
                
            
            #print model accuracy on the training data.
            print('[0]Logistic Regression Training Accuracy:', log.score(X_train, y_train))
            print('[1]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, y_train))
            print('[2]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, y_train))
            print('[3]Decision Tree Classifier Training Accuracy:', tree.score(X_train, y_train))
            print('[4]Random Forest Classifier Training Accuracy:', RF.score(X_train, y_train))
            print('[5]KNN Classifier Training Accuracy:', model_knn.score(X_train, y_train)) 
          
          
            print('\n\n[0]Logistic Regression Testing Accuracy:', log.score(X_test,y_test))
            print('[1]Support Vector Machine (Linear Classifier) Testing Accuracy:', svc_lin.score(X_test,y_test))
            print('[2]Support Vector Machine (RBF Classifier) Testing Accuracy:', svc_rbf.score(X_test,y_test))
            print('[3]Decision Tree Classifier Testing Accuracy:', tree.score(X_test,y_test))
            print('[4]Random Forest Classifier Testing Accuracy:', RF.score(X_test,y_test))
            print('[5]KNN Classifier Testing Accuracy:', model_knn.score(X_test,y_test))


# Principle Componets analysis
#
from sklearn import decomposition
import mpl_toolkits.mplot3d 
from sklearn.decomposition import PCA#, loadings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
plt.cla()
pca = PCA(n_components=10)
X_r = pca.fit(X).transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
#X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components

# Exploring our PCA Data
pcs=PCA()
pca.fit(X)
#X_r = pca.fit(X).transform(X)
expl_var = pca.explained_variance_ratio_
df_expl_var = pd.DataFrame(
    data=zip(range(1, len(expl_var) + 1), expl_var, expl_var.cumsum()), 
    columns=['PCA', 'Explained Variance (%)', 'Total Explained Variance (%)']
    ).set_index('PCA').mul(100).round(1)
print(df_expl_var)

#Plotting our explained variance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,8))
ax.bar(x=df_expl_var.index, height=df_expl_var['Explained Variance (%)'], label='Explained Variance', width=0.9, color='#AAD8D3')
ax.plot(df_expl_var['Total Explained Variance (%)'], label='Total Explained Variance', marker='o', c='#37B6BD')

plt.ylim(0, 100)
plt.ylabel('Explained Variance (%)')
plt.xlabel('PCA')
plt.grid(True, axis='y')
plt.title('Understanding Explained Variance in PCA')
plt.legend()
# # Printing Loading Scores
print(pca.components_)

# Plotting a Heatmap of Our Loadings
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,8))

ax = sns.heatmap(
    pca.components_,
    cmap='coolwarm',
    yticklabels=[f'PCA{x}' for x in range(1,pca.n_components_+1)],
    xticklabels=(X.columns),
    linewidths=1,
    annot=True,
    fmt=',.2f',
    cbar_kws={"shrink": 0.8, "orientation": 'horizontal'}
    )
