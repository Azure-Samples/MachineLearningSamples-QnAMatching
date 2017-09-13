###################################################################################################################################
# This script is used for hyperparameter tunning of the Naive Bayes model described in the Part 3: Model Training and Evaluation.
# To run this script, please enter a non-negative integer in the "Argument" box above the screen.
# This argument indicates the number of important tokens that we selected for each answer class. 
# The total number of tokens that are selected for all classes construct the whole feature space.
###################################################################################################################################

import pandas as pd
import numpy as np
import os, warnings
from azureml.logging import get_azureml_logger
from modules.feature_extractor import (tokensToIds, countMatrix, priorProbabilityAnswer, posterioriProb, 
                               feature_selection, featureWeights, wordProbabilityInAnswer, 
                               wordProbabilityNotinAnswer, normalizeTF, softmax)
warnings.filterwarnings("ignore")



#########################################
# User Defined Functions
#########################################

# get the rank of answerIds for a given question. 
def rank(frame, scores, uniqueAnswerId):
    frame['SortedAnswers'] = list(np.array(uniqueAnswerId)[np.argsort(-scores, axis=1)])
    
    rankList = []
    for i in range(len(frame)):
        rankList.append(np.where(frame['SortedAnswers'].iloc[i] == frame['AnswerId'].iloc[i])[0][0] + 1)
    frame['Rank'] = rankList
    
    return frame


#########################################
# Main Function
#########################################

def main():

    #########################################
    # Accept One Argument as Input
    #########################################

    try:
        topN = int(sys.argv[1])
    except IndexError:
        print("This script takes one argument. Please enter a valid non-negative integer number.\n")
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


    #########################################
    # Train Naive Bayes Classifier
    #########################################

    NBWeights = np.log(P_wA / P_wNotA)


    #########################################
    # Predict Probabilities on Test
    #########################################

    beta_A = 0
    x_wTest = normalizeTF(testQ, token2IdHash)
    Y_test_prob = softmax(-beta_A + np.dot(x_wTest.T, NBWeights))


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
    print('Average of rank: ' + str(AR))
    print('Percentage of questions find answers in the first 3 choices: ' + str(top3))


    #########################################
    # Log Parameters and Performance
    #########################################

    # initialize the logger
    run_logger = get_azureml_logger()

    # log performance.
    run_logger.log("Top N Tokens Selected", topN)
    run_logger.log("Average Rank", AR)
    run_logger.log("Top 3 Percentage", top3)



if __name__ == "__main__":
    main()
    print("\nRun is complete!")
