###################################################################################################################################
# This script is used for hyperparameter tunning of the Random Forest model described in the Part 3: Model Training and Evaluation.
# To run this script, please enter two non-negative integers seperated by a space in the "Argument" box above the screen.
# The first argument indicates the number of important tokens that we selected for each answer class. 
# The second argument indicates the number of trees to be constructed in the Random Forest model.
###################################################################################################################################


import pandas as pd
import numpy as np
import os, sys, warnings
from azureml.logging import get_azureml_logger
from sklearn.ensemble import RandomForestClassifier
sys.path.append("")
from modules.feature_extractor import (tokensToIds, countMatrix, priorProbabilityAnswer, posterioriProb, 
                                       feature_selection, featureWeights, wordProbabilityInAnswer, 
                                       wordProbabilityNotinAnswer, normalizeTF, softmax)
from naive_bayes import (rank)
warnings.filterwarnings("ignore")

run_logger = get_azureml_logger()
run_logger.log('amlrealworld.QnA-matching.random-forest','true')

#########################################
# User Defined Functions
#########################################

# train one-vs-rest classifier using NB scores as features.
def ovrClassifier(trainLabels, x_wTrain, x_wTest, NBWeights, clf, ratio):
    uniqueLabel = np.unique(trainLabels)
    dummyLabels = pd.get_dummies(trainLabels)
    numTest = x_wTest.shape[1]
    Y_test_prob = np.zeros(shape=(numTest, len(uniqueLabel)))

    for i in range(len(uniqueLabel)):
        X_train_all, Y_train_all = x_wTrain.T * NBWeights[:, i], dummyLabels.iloc[:, i]
        X_test = x_wTest.T * NBWeights[:, i]
        
        # with sample selection.
        if ratio is not None:
            # ratio = # of Negative/# of Positive
            posIdx = np.where(Y_train_all == 1)[0]
            negIdx = np.random.choice(np.where(Y_train_all == 0)[0], ratio*len(posIdx))
            allIdx = np.concatenate([posIdx, negIdx])
            X_train, Y_train = X_train_all[allIdx], Y_train_all.iloc[allIdx]
        else: # without sample selection.
            X_train, Y_train = X_train_all, Y_train_all
            
        clf.fit(X_train, Y_train)
        if hasattr(clf, "decision_function"):
            Y_test_prob[:, i] = clf.decision_function(X_test)
        else:
            Y_test_prob[:, i] = clf.predict_proba(X_test)[:, 1]

    return softmax(Y_test_prob)


#########################################
# Main Function
#########################################

def main():

    #########################################
    # Accept One Argument as Input
    #########################################
    
    try:
        topN = int(sys.argv[1])
        n_estimators = int(sys.argv[2])
    except IndexError:
        print("This script takes two arguments. Please enter valid non-negative integer numbers.\n")
        raise


    #########################################
    # Access trainQ and testQ from Part 2
    #########################################

    workfolder = os.environ.get('AZUREML_NATIVE_SHARE_DIRECTORY')

    # paths to trainQ and testQ.
    trainQ_path = os.path.join(workfolder, 'trainQ_part2')
    testQ_path = os.path.join(workfolder, 'testQ_part2')

    # load the training and test data.
    trainQ = pd.read_csv(trainQ_path, sep='\t', index_col='Id', encoding='latin1')
    testQ = pd.read_csv(testQ_path, sep='\t', index_col='Id', encoding='latin1')


    #########################################
    # Extract Features
    #########################################
    
    token2IdHashInit = tokensToIds(trainQ['Tokens'], featureHash=None)

    # get unique answerId in ascending order
    uniqueAnswerId = list(np.unique(trainQ['AnswerId']))

    # calculate the count matrix of all training questions.
    N_wAInit = countMatrix(trainQ, token2IdHashInit, 'AnswerId', uniqueAnswerId)

    P_A = priorProbabilityAnswer(trainQ['AnswerId'], uniqueAnswerId)
    P_Aw = posterioriProb(N_wAInit, P_A, uniqueAnswerId)

    # select top N important tokens per answer class.
    featureHash = feature_selection(P_Aw, token2IdHashInit, topN=topN)
    token2IdHash = tokensToIds(trainQ['Tokens'], featureHash=featureHash)

    N_wA = countMatrix(trainQ, token2IdHash, 'AnswerId', uniqueAnswerId)

    alpha = 0.0001
    P_w = featureWeights(N_wA, alpha)

    beta = 0.0001
    P_wA = wordProbabilityInAnswer(N_wA, P_w, beta)
    P_wNotA = wordProbabilityNotinAnswer(N_wA, P_w, beta)   

    x_wTrain = normalizeTF(trainQ, token2IdHash)
    x_wTest = normalizeTF(testQ, token2IdHash)


    #########################################
    # Train Naive Bayes Classifier
    #########################################

    NBWeights = np.log(P_wA / P_wNotA)
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', random_state=1)


    #########################################
    # Predict Probabilities on Test
    #########################################

    Y_test_prob = ovrClassifier(trainQ["AnswerId"], x_wTrain, x_wTest, NBWeights, clf, ratio=3)


    #########################################
    # Evaluate Model Performance
    #########################################
    # We use two evaluation matrices (Average Rank and Top 3 Percentage) to test our model performance. 
    # The Average Rank can be interpreted as in average at which position we can find the correct answer among all available answers for a given question.
    # The Top 3 Percentage can be interpreted as how many percentage of the new questions that we can find their correct answers in the first 3 choices.
    # sort the similarity scores in descending order and map them to the corresponding AnswerId in Answer set

    testQ = rank(testQ, Y_test_prob, uniqueAnswerId)

    AR = np.floor(testQ['Rank'].mean())
    top3 = round(len(testQ.query('Rank <= 3'))/len(testQ), 3)

    print('Top %d important tokens selected per Class.' %topN)  
    print('# of trees in the Random Forest: ' + str(n_estimators))
    print('Average of rank: ' + str(AR))
    print('Percentage of questions find answers in the first 3 choices: ' + str(top3))


    #########################################
    # Log Parameters and Performance
    #########################################

    # initialize the logger.
    run_logger = get_azureml_logger() 

    # log performance.
    run_logger.log("Top N Tokens Selected", topN)
    run_logger.log("Number of Trees", n_estimators)
    run_logger.log("Average Rank", AR)
    run_logger.log("Top 3 Percentage", top3)


if __name__ == "__main__":
    main()
    print("\nRun is complete!")