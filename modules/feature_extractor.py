################################################
# User Defined Functions for Feature Extraction
################################################

import pandas as pd
import numpy as np
from azureml.logging import get_azureml_logger

run_logger = get_azureml_logger()
run_logger.log('amlrealworld.QnA-matching.feature-extractor','true')


# get Token to ID mapping: {Token: tokenId}
def tokensToIds(tokens, featureHash):
    token2IdHash = {}
    for i in range(len(tokens)):
        tokenList = tokens.iloc[i].split(',')
        if featureHash is None:
            for t in tokenList:
                if t not in token2IdHash.keys():
                    token2IdHash[t] = len(token2IdHash)
        else:
            for t in tokenList:
                if t not in token2IdHash.keys() and t in list(featureHash.keys()):
                    token2IdHash[t] = len(token2IdHash)
            
    return token2IdHash

# create a matrix to store the token frequency.
def countMatrix(frame, token2IdHash, labelColumnName=None, uniqueLabel=None):
    # create am empty matrix with the shape of:
    # num_row = num of unique tokens
    # num_column = num of unique answerIds (N_wA) or num of questions in testQ (tfMatrix)
    # rowIdx = token2IdHash.values()
    # colIdx = index of uniqueClass (N_wA) or index of questions in testQ (tfMatrix)
    num_row = len(token2IdHash)
    if uniqueLabel is not None:  # get N_wA
        num_column = len(uniqueLabel)
    else:
        num_column = len(frame)
    countMatrix = np.zeros(shape=(num_row, num_column))

    # loop through each question in the frame to fill in the countMatrix with corresponding counts
    for i in range(len(frame)):
        tokens = frame['Tokens'].iloc[i].split(',')
        if uniqueLabel is not None:   # get N_wA
            label = frame[labelColumnName].iloc[i]
            colIdx = uniqueLabel.index(label)
        else:
            colIdx = i
            
        for t in tokens:
            if t in token2IdHash.keys():
                rowIdx = token2IdHash[t]
                countMatrix[rowIdx, colIdx] += 1

    return countMatrix

# calculate the prior probability of each answer class P(A): [P_A1, P_A2, ...]
def priorProbabilityAnswer(answerIds, uniqueLabel): 
    P_A = []
    # convert a pandas series to a list
    answerIds = list(answerIds)
    
    for id in uniqueLabel:
        P_A.append(answerIds.count(id)/len(answerIds))
    return np.array(P_A)

# calculate the conditional probability of each answer class by giving a token P(A|w).
def posterioriProb(N_wAInit, P_A, uniqueLabel):
    # N_A is the total number of answers
    N_A = len(uniqueLabel)
    # N_w is the total number of times w appears over all documents 
    # rowSum of count matrix (N_wAInit)
    N_wInit = np.sum(N_wAInit, axis = 1)
    # P(A|w) = (N_w|A + N_A * P(A))/(N_w + N_A)
    N = N_wAInit + N_A * P_A
    D = N_wInit + N_A
    P_Aw = np.divide(N.T, D).T    
    
    return P_Aw

# select the top N tokens w which maximize P(A|w) for each A.
# get FeatureHash: {token: 1}
def feature_selection(P_Aw, token2IdHashInit, topN):
    featureHash = {}
    # for each answer A, sort tokens w by P(A|w)
    sortedIdxMatrix = np.argsort(P_Aw, axis=0)[::-1]
    # select top N tokens for each answer A
    topMatrix = sortedIdxMatrix[0:topN, :]
    # for each token w in topMatrix, add w to FeatureHash if it has not already been included
    topTokenIdList = np.reshape(topMatrix, topMatrix.shape[0] * topMatrix.shape[1])
    # get ID to Token mapping: {tokenId: Token}
    Id2TokenHashInit = {y:x for x, y in token2IdHashInit.items()}
    
    for tokenId in topTokenIdList:
        token = Id2TokenHashInit[tokenId]
        if token not in featureHash.keys():
            featureHash[token] = 1
    return featureHash

# calculate the weight for each feature. 
def featureWeights(N_wA, alpha):
    # N_w is the total number of times w appears over all documents 
    # rowSum of count matrix (N_wA)
    N_w = np.sum(N_wA, axis = 1)
    # N_W is the total count of all words
    N_W = np.sum(N_wA)
    # N_V is the count of unique words in the vocabulary
    N_V = N_wA.shape[0]
    # P(w) = (N_w + 1*alpha) / (N_W +N_V*alpha)
    N2 = N_w + 1 * alpha
    D2 = N_W + alpha * N_V
    P_w = N2/D2

    return P_w

# calculate the conditional probability of each token within an answer class P(A|w).
def wordProbabilityInAnswer(N_wA, P_w, beta):
    # N_V is the count of unique words in the vocabulary
    N_V = N_wA.shape[0]
    # N_WA is the total count of all words in questions on answer A 
    # colSum of count matrix (N_wA)
    N_WA = np.sum(N_wA, axis=0)
    # P(w|A) = (N_w|A + beta N_V P(w))/(N_W|A + beta * N_V)
    N = (N_wA.T + beta * N_V * P_w).T
    D = N_WA + beta * N_V
    P_wA = N / D
    
    return P_wA

# calculate the conditional probability of each token not within an answer class P(notA|w).
def wordProbabilityNotinAnswer(N_wA, P_w, beta):
    # N_V is the count of unique words in the vocabulary
    N_V = N_wA.shape[0]
    # N_wNotA is the count of w over all documents but not on answer A
    # N_wNotA = N_w - N_wA
    N_w = np.sum(N_wA, axis = 1)
    N_wNotA = (N_w - N_wA.T).T
    # N_WNotA is the count of all words over all documents but not on answer A
    # N_WNotA = N_W - N_WA
    N_W = np.sum(N_wA)
    N_WA = np.sum(N_wA, axis=0)
    N_WNotA = N_W - N_WA
    # P(w|NotA) = (N_w|NotA + beta * N_V * P(w))/(N_W|NotA + beta * N_V)
    N = (N_wNotA.T + beta * N_V * P_w).T
    D = N_WNotA + beta * N_V
    P_wNotA = N / D
    
    return P_wNotA

# calculate the normalized Term Frequency.
def normalizeTF(frame, token2IdHash):
    
    N_wQ = countMatrix(frame, token2IdHash)
    N_WQ = np.sum(N_wQ, axis=0)
    
    # find the index where N_WQ is zero
    zeroIdx = np.where(N_WQ == 0)[0]
    
    # if N_WQ is zero, then the x_w for that particular question would be zero.
    # for a simple calculation, we convert the N_WQ to 1 in those cases so the demoninator is not zero. 
    if len(zeroIdx) > 0:
        N_WQ[zeroIdx] = 1
    
    # x_w = P_wd = count(w)/sum(count(i in V))
    x_w = N_wQ / N_WQ
    
    return x_w

# calculate the Inverse Document Frequency.
def getIDF(N_wQ):
    # N is total number of documents in the corpus
    # N_V is the number of tokens in the vocabulary
    N_V, N = N_wQ.shape
    # D is the number of documents where the token w appears
    D = np.zeros(shape=(0, N_V))
    for i in range(N_V):
        D = np.append(D, len(np.nonzero(N_wQ[i, ])[0]))
    return np.log(N/D)

# create a softmax function.
def softmax(scores2D):
    # input: scores from different models
    # row: test example
    # column: label
    return np.exp(scores2D)/np.sum(np.exp(scores2D), axis=1)[:, None]