##############################################
# User Defined Functions for Phrase Learning
##############################################

import pandas as pd
import numpy as np
import re, nltk, time, gc, math


def CleanAndSplitText(frame):
    
    global EMPTY, SPACE, NLTK_PUNKT_EN, SENTENCE_BREAKER
    EMPTY = ''
    SPACE = ' '
    nltk.download("punkt")
    NLTK_PUNKT_EN = 'tokenizers/punkt/english.pickle'
    SENTENCE_BREAKER = nltk.data.load(NLTK_PUNKT_EN)

    textDataOut = [] 

    # This regular expression is for punctuation that we wish to clean out
    # We also will split sentences into smaller phrase like units using this expression
    rePhraseBreaks = re.compile("[\"\!\?\)\]\}\,\:\;\*\-]*\s+\([0-9]+\)\s+[\(\[\{\"\*\-]*"                         
                                "|[\"\!\?\)\]\}\,\:\;\*\-]+\s+[\(\[\{\"\*\-]*" 
                                "|\.\.+"       # ..
                                "|\s*\-\-+\s*" # --
                                "|\s+\-\s+"    # -  
                                "|\:\:+"       # ::
                                "|\s+[\/\(\[\{\"\-\*]+\s*"  
                                "|[\,!\?\"\)\(\]\[\}\{\:\;\*](?=[a-zA-Z])"
                                "|[\"\!\?\)\]\}\,\:\;]+[\.]*$"
                             )
    
    # Regex for underbars
    regexUnderbar = re.compile('_|_+')
    
    # Regex for space
    regexSpace = re.compile(' +')
 
    # Regex for sentence final period
    regexPeriod = re.compile("\.$")
    
    # Regex for parentheses
    regexParentheses = re.compile("\(\$?")
    
    # Regex for equal sign
    regexEqual = re.compile("=")

    # Iterate through each document and do:
    #    (1) Split documents into sections based on section headers and remove section headers
    #    (2) Split the sections into sentences using NLTK sentence tokenizer
    #    (3) Further split sentences into phrasal units based on punctuation and remove punctuation
    #    (4) Remove sentence final periods when not part of a abbreviation 

    for i in range(0,len(frame)):
        
        # Extract one document from frame
        docID = frame.index.values[i]
        docText = frame['Text'].iloc[i] 

        # Set counter for output line count for this document
        lineIndex=0

        sentences = SENTENCE_BREAKER.tokenize(docText)
        
        for sentence in sentences:

            # Split each sentence into phrase level chunks based on punctuation
            textSegs = rePhraseBreaks.split(sentence)
            numSegs = len(textSegs)

            for j in range(0,numSegs):
                if len(textSegs[j])>0:
                    # Convert underbars to spaces 
                    # Underbars are reserved for building the compound word phrases                   
                    textSegs[j] = regexUnderbar.sub(" ",textSegs[j])
                    
                    # Split out the words so we can specially handle the last word
                    words = regexSpace.split(textSegs[j])
                    
                    # Remove parentheses and equal signs
                    words = [regexEqual.sub("", regexParentheses.sub("", w)) for w in words]
                    
                    phraseOut = ""
                    last = len(words) -1
                    for i in range(0, last):
                        phraseOut += words[i] + " "
                    # If the last word ends in a period then remove the period
                    lastWord = regexPeriod.sub("", words[last])
                    # If the last word is an abbreviation like "U.S."
                    # then add the word final perios back on
                    if "\." in lastWord:
                        lastWord += "."
                    phraseOut += lastWord    

                    textDataOut.append([docID,lineIndex,phraseOut, phraseOut.lower()])
                    lineIndex += 1
                        
    # Convert to pandas frame 
    frameOut = pd.DataFrame(textDataOut, columns=['DocID','DocLine','CleanedText', 'LowercaseText'])                      
    
    return frameOut

# count the number of occurances of all 2-gram, 3-ngram, and 4-gram word sequences.
def ComputeNgramStats(textData,functionwordHash,blacklistHash):
    
    # Create an array to store the total count of all ngrams up to 4-grams
    # Array element 0 is unused, element 1 is unigrams, element 2 is bigrams, etc.
    ngramCounts = [0]*5;
       
    # Create a list of structures to tabulate ngram count statistics
    # Array element 0 is the array of total ngram counts,
    # Array element 1 is a hash table of individual unigram counts
    # Array element 2 is a hash table of individual bigram counts
    # Array element 3 is a hash table of individual trigram counts
    # Array element 4 is a hash table of individual 4-gram counts
    ngramStats = [ngramCounts, {}, {}, {}, {}]
          
    # Create a regular expression for assessing validity of words
    # for phrase modeling. The expression says words in phrases
    # must either:
    # (1) contain an alphabetic character, or 
    # (2) be the single charcater '&', or
    # (3) be a one or two digit number
    reWordIsValid = re.compile('[A-Za-z]|^&$|^\d\d?$')
    
    # Go through the text data line by line collecting count statistics
    # for all valid n-grams that could appear in a potential phrase
    numLines = len(textData)
    for i in range(0, numLines):

        # Split the text line into an array of words
        wordArray = textData[i].split()
        numWords = len(wordArray)
        
        # Create an array marking each word as valid or invalid
        validArray = [];
        for word in wordArray:
            validArray.append(reWordIsValid.match(word) != None)        
            
        # Tabulate total raw ngrams for this line into counts for each ngram bin
        # The total ngrams counts include the counts of all ngrams including those
        # that we won't consider as parts of phrases
        for j in range(1,5):
            if j<=numWords:
                ngramCounts[j] += numWords - j + 1 
        
        # Collect counts for viable phrase ngrams and left context sub-phrases
        for j in range(0,numWords):
            word = wordArray[j]

            # Only bother counting the ngrams that start with a valid content word
            # i.e., valids words not in the function word list or the black list
            if ( ( word not in functionwordHash ) and ( word not in blacklistHash ) and validArray[j] ):

                # Initialize ngram string with first content word and add it to unigram counts
                ngramSeq = word 
                if ngramSeq in ngramStats[1]:
                    ngramStats[1][ngramSeq] += 1
                else:
                    ngramStats[1][ngramSeq] = 1

                # Count valid ngrams from bigrams up to 4-grams
                stop = 0
                k = 1
                while (k<4) and (j+k<numWords) and not stop:
                    n = k + 1
                    nextNgramWord = wordArray[j+k]
                    # Only count ngrams with valid words not in the blacklist
                    if ( validArray[j+k] and nextNgramWord not in blacklistHash ):
                        ngramSeq += " " + nextNgramWord
                        if ngramSeq in ngramStats[n]:
                            ngramStats[n][ngramSeq] += 1
                        else:
                            ngramStats[n][ngramSeq] = 1 
                        k += 1
                        if nextNgramWord not in functionwordHash:
                            # Stop counting new ngrams after second content word in 
                            # ngram is reached and ngram is a viable full phrase
                            stop = 1
                    else:
                        stop = 1
    return ngramStats

# rank potential phrases by the Weighted Pointwise Mutual Information of their constituent words
def RankNgrams(ngramStats,functionwordHash,minCount):
    # Create a hash table to store weighted pointwise mutual 
    # information scores for each viable phrase
    ngramWPMIHash = {}
        
    # Go through each of the ngram tables and compute the phrase scores
    # for the viable phrases
    for n in range(2,5):
        i = n-1
        for ngram in ngramStats[n].keys():
            ngramCount = ngramStats[n][ngram]
            if ngramCount >= minCount:
                wordArray = ngram.split()
                # If the final word in the ngram is not a function word then
                # the ngram is a valid phrase candidate we want to score
                if wordArray[i] not in functionwordHash: 
                    leftNgram = wordArray[0]
                    for j in range(1,i):
                        leftNgram += ' ' + wordArray[j]
                    rightWord = wordArray[i]
                    
                    # Compute the weighted pointwise mutual information (WPMI) for the phrase
                    probNgram = float(ngramStats[n][ngram])/float(ngramStats[0][n])
                    probLeftNgram = float(ngramStats[n-1][leftNgram])/float(ngramStats[0][n-1])
                    probRightWord = float(ngramStats[1][rightWord])/float(ngramStats[0][1])
                    WPMI = probNgram * math.log(probNgram/(probLeftNgram*probRightWord));

                    # Add the phrase into the list of scored phrases only if WMPI is positive
                    if WPMI > 0:
                        ngramWPMIHash[ngram] = WPMI  
    
    # Create a sorted list of the phrase candidates
    rankedNgrams = sorted(ngramWPMIHash, key=ngramWPMIHash.__getitem__, reverse=True)

    # Force a memory clean-up
    ngramWPMIHash = None
    gc.collect()

    return rankedNgrams

# apply the phrase rewrites to training data.
def ApplyPhraseRewrites(rankedNgrams,textData,learnedPhrases,                 
                        maxPhrasesToAdd,maxPhraseLength,verbose):
    
    if len(rankedNgrams) == 0:
        return
    
    # This function will consider at most maxRewrite 
    # new phrases to be added into the learned phrase 
    # list as specified by the calling fuinction
    maxRewrite=maxPhrasesToAdd

    # If the remaining number of proposed ngram phrases is less 
    # than the max allowed, then reset maxRewrite to the size of 
    # the proposed ngram phrases list
    numNgrams = len(rankedNgrams)
    if numNgrams < maxRewrite:
        maxRewrite = numNgrams
    
    # Create empty hash tables to keep track of phrase overlap conflicts
    leftConflictHash = {}
    rightConflictHash = {}
    
    # Create an empty hash table collecting the set of rewrite rules
    # to be applied during this iteration of phrase learning
    ngramRewriteHash = {}
    
    # Precompile the regex for finding spaces in ngram phrases
    regexSpace = re.compile(' ')

    # Initialize some bookkeeping variables
    numLines = len(textData)
    numPhrasesAdded = 0
    numConsidered = 0
    lastSkippedNgram = ""
    lastAddedNgram = ""
  
    # Collect list up to maxRewrite ngram phrase rewrites
    stop = False
    index = 0
    while not stop:

        # Get the next phrase to consider adding to the phrase list
        inputNgram = rankedNgrams[index]

        # Create the output compound word version of the phrase
        # The extra space is added to make the regex rewrite easier
        outputNgram = " " + regexSpace.sub("_",inputNgram)

        # Count the total number of words in the proposed phrase
        numWords = len(outputNgram.split("_"))

        # Only add phrases that don't exceed the max phrase length
        if (numWords <= maxPhraseLength):
    
            # Keep count of phrases considered for inclusion during this iteration
            numConsidered += 1

            # Extract the left and right words in the phrase to use
            # in checks for phrase overlap conflicts
            ngramArray = inputNgram.split()
            leftWord = ngramArray[0]
            rightWord = ngramArray[len(ngramArray)-1]

            # Skip any ngram phrases that conflict with earlier phrases added
            # These ngram phrases will be reconsidered in the next iteration
            if (leftWord in leftConflictHash) or (rightWord in rightConflictHash): 
                if verbose: 
                    print ("(%d) Skipping (context conflict): %s" % (numConsidered,inputNgram))
                lastSkippedNgram = inputNgram
                
            # If no conflict exists then add this phrase into the list of phrase rewrites     
            else: 
                if verbose:
                    print ("(%d) Adding: %s" % (numConsidered,inputNgram))
                ngramRewriteHash[" " + inputNgram] = outputNgram
                learnedPhrases.append(inputNgram) 
                lastAddedNgram = inputNgram
                numPhrasesAdded += 1
            
            # Keep track of all context words that might conflict with upcoming
            # propose phrases (even when phrases are skipped instead of added)
            leftConflictHash[rightWord] = 1
            rightConflictHash[leftWord] = 1

            # Stop when we've considered the maximum number of phrases per iteration
            if ( numConsidered >= maxRewrite ):
                stop = True
            
        # Increment to next phrase
        index += 1
    
        # Stop if we've reached the end of the ranked ngram list
        if index >= len(rankedNgrams):
            stop = True

    # Now do the phrase rewrites over the entire set of text data
    if numPhrasesAdded == 1:
        # If only one phrase to add use a single regex rule to do this phrase rewrite        
        inputNgram = " " + lastAddedNgram
        outputNgram = ngramRewriteHash[inputNgram]
        regexNgram = re.compile (r'%s(?= )' % re.escape(inputNgram)) 
        # Apply the regex over the full data set
        for j in range(0,numLines):
            textData[j] = regexNgram.sub(outputNgram, textData[j])
    elif numPhrasesAdded > 1:
        # Compile a single regex rule from the collected set of phrase rewrites for this iteration
        ngramRegex = re.compile(r'%s(?= )' % "|".join(map(re.escape, ngramRewriteHash.keys())))
        # Apply the regex over the full data set
        for i in range(0,len(textData)):
            # The regex substituion looks up the output string rewrite  
            # in the hash table for each matched input phrase regex
            textData[i] = ngramRegex.sub(lambda mo: ngramRewriteHash[mo.string[mo.start():mo.end()]], textData[i]) 
      
    return

# run the full iterative phrase learning process.
def ApplyPhraseLearning(textData, learnedPhrases, maxNumPhrases=200, maxPhraseLength=7, maxPhrasesPerIter=50, 
    minCount=5, functionwordHash={}, blacklistHash={}, verbose=False):
    
    stop = 0
    iterNum = 0
    
    # Start timing the process
    functionStartTime = time.clock()
    
    numPhrasesLearned = len(learnedPhrases)
    print ("Start phrase learning with %d phrases of %d phrases learned" % (numPhrasesLearned,maxNumPhrases))

    while not stop:
        iterNum += 1
                
        # Start timing this iteration
        startTime = time.clock()
 
        # Collect ngram stats
        ngramStats = ComputeNgramStats(textData,functionwordHash,blacklistHash)

        # Rank ngrams
        rankedNgrams = RankNgrams(ngramStats,functionwordHash,minCount)
        
        # Incorporate top ranked phrases into phrase list
        # and rewrite the text to use these phrases
        maxPhrasesToAdd = maxNumPhrases - numPhrasesLearned
        if maxPhrasesToAdd > maxPhrasesPerIter:
            maxPhrasesToAdd = maxPhrasesPerIter
        ApplyPhraseRewrites(rankedNgrams,textData,learnedPhrases,maxPhrasesToAdd,maxPhraseLength,verbose)
        numPhrasesAdded = len(learnedPhrases) - numPhrasesLearned

        # Garbage collect
        ngramStats = None
        rankedNgrams = None
        gc.collect();
               
        elapsedTime = time.clock() - startTime

        numPhrasesLearned = len(learnedPhrases)
        print ("Iteration %d: Added %d new phrases in %.2f seconds (Learned %d of max %d)" % 
               (iterNum,numPhrasesAdded,elapsedTime,numPhrasesLearned,maxNumPhrases))
        
        if numPhrasesAdded >= maxPhrasesToAdd or numPhrasesAdded == 0:
            stop = 1
        
    # Remove the space padding at the start and end of each line
    regexSpacePadding = re.compile('^ +| +$')
    for i in range(0,len(textData)):
        textData[i] = regexSpacePadding.sub("",textData[i])
    
    gc.collect()
 
    elapsedTime = time.clock() - functionStartTime
    elapsedTimeHours = elapsedTime/3600.0;
    print ("*** Phrase learning completed in %.2f hours ***" % elapsedTimeHours) 

    return

# apply the learned phrases to test data.
def ApplyPhraseRewritesInPlace(textFrame, textColumnName, phraseRules):
        
    # Get text data column from frame
    textData = textFrame[textColumnName]
    numLines = len(textData)
    
    # initial a list to store output text
    textOutput = [None] * numLines
    
    # Add leading and trailing spaces to make regex matching easier
    for i in range(0,numLines):
        textOutput[i] = " " + textData[i] + " "  

    # Make sure we have phrase to add
    numPhraseRules = len(phraseRules)
    if numPhraseRules == 0: 
        print ("Warning: phrase rule lise is empty - no phrases being applied to text data")
        return

    # Precompile the regex for finding spaces in ngram phrases
    regexSpace = re.compile(' ')
   
    # Initialize some bookkeeping variables

    # Iterate through full set of phrases to find sets of 
    # non-conflicting phrases that can be apply simultaneously
    index = 0
    outerStop = False
    while not outerStop:
       
        # Create empty hash tables to keep track of phrase overlap conflicts
        leftConflictHash = {}
        rightConflictHash = {}
        prevConflictHash = {}
    
        # Create an empty hash table collecting the next set of rewrite rules
        # to be applied during this iteration of phrase rewriting
        phraseRewriteHash = {}
    
        # Progress through phrases until the next conflicting phrase is found
        innerStop = 0
        numPhrasesAdded = 0
        while not innerStop:
        
            # Get the next phrase to consider adding to the phrase list
            nextPhrase = phraseRules[index]            
            
            # Extract the left and right sides of the phrase to use
            # in checks for phrase overlap conflicts
            ngramArray = nextPhrase.split()
            leftWord = ngramArray[0]
            rightWord = ngramArray[len(ngramArray)-1] 

            # Stop if we reach any phrases that conflicts with earlier phrases in this iteration
            # These ngram phrases will be reconsidered in the next iteration
            if ((leftWord in leftConflictHash) or (rightWord in rightConflictHash) 
                or (leftWord in prevConflictHash) or (rightWord in prevConflictHash)): 
                innerStop = True
                
            # If no conflict exists then add this phrase into the list of phrase rewrites     
            else: 
                # Create the output compound word version of the phrase
                                
                outputPhrase = regexSpace.sub("_",nextPhrase);
                
                # Keep track of all context words that might conflict with upcoming
                # propose phrases (even when phrases are skipped instead of added)
                leftConflictHash[rightWord] = 1
                rightConflictHash[leftWord] = 1
                prevConflictHash[outputPhrase] = 1           
                
                # Add extra space to input an output versions of the current phrase 
                # to make the regex rewrite easier
                outputPhrase = " " + outputPhrase
                lastAddedPhrase = " " + nextPhrase
                
                # Add the phrase to the rewrite hash
                phraseRewriteHash[lastAddedPhrase] = outputPhrase
                  
                # Increment to next phrase
                index += 1
                numPhrasesAdded  += 1
    
                # Stop if we've reached the end of the phrases list
                if index >= numPhraseRules:
                    innerStop = True
                    outerStop = True
                    
        # Now do the phrase rewrites over the entire set of text data
        if numPhrasesAdded == 1:
        
            # If only one phrase to add use a single regex rule to do this phrase rewrite        
            outputPhrase = phraseRewriteHash[lastAddedPhrase]
            regexPhrase = re.compile (r'%s(?= )' % re.escape(lastAddedPhrase)) 
        
            # Apply the regex over the full data set
            for j in range(0,numLines):
                textOutput[j] = regexPhrase.sub(outputPhrase, textOutput[j])
       
        elif numPhrasesAdded > 1:
            # Compile a single regex rule from the collected set of phrase rewrites for this iteration
            regexPhrase = re.compile(r'%s(?= )' % "|".join(map(re.escape, phraseRewriteHash.keys())))
            
            # Apply the regex over the full data set
            for i in range(0,numLines):
                # The regex substituion looks up the output string rewrite  
                # in the hash table for each matched input phrase regex
                textOutput[i] = regexPhrase.sub(lambda mo: phraseRewriteHash[mo.string[mo.start():mo.end()]], textOutput[i]) 
    
    # Remove the space padding at the start and end of each line
    regexSpacePadding = re.compile('^ +| +$')
    for i in range(0,len(textOutput)):
        textOutput[i] = regexSpacePadding.sub("",textOutput[i])
    
    return textOutput

# reconstruct the full processed text and put it back into a new data frame.
def ReconstituteDocsFromChunks(textData, idColumnName, textColumnName):
    dataOut = []
    
    currentDoc = "";
    currentDocID = "";
    
    for i in range(0,len(textData)):
        textChunk = textData[textColumnName][i]
        docID = textData[idColumnName][i]
        if docID != currentDocID:
            if currentDocID != "":
                dataOut.append(currentDoc)
            currentDoc = textChunk
            currentDocID = docID
        else:
            currentDoc += " " + textChunk
    dataOut.append(currentDoc)
    
    return dataOut

# create the Vocabulary with some filtering criteria.
def CreateVocabForTopicModeling(textData,stopwordHash):

    print ("Counting words")
    numDocs = len(textData) 
    globalWordCountHash = {} 
    globalDocCountHash = {} 
    for textLine in textData:
        docWordCountHash = {}
        for word in textLine.split():
            if word in globalWordCountHash:
                globalWordCountHash[word] += 1
            else:
                globalWordCountHash[word] = 1
            if word not in docWordCountHash: 
                docWordCountHash[word] = 1
                if word in globalDocCountHash:
                    globalDocCountHash[word] += 1
                else:
                    globalDocCountHash[word] = 1

    minWordCount = 5;
    minDocCount = 2;
    maxDocFreq = .25;
    vocabCount = 0;
    vocabHash = {}

    excStopword = 0
    excNonalphabetic = 0
    excMinwordcount = 0
    excNotindochash = 0
    excMindoccount = 0
    excMaxdocfreq =0

    print ("Building vocab")
    for word in globalWordCountHash.keys():
        # Test vocabulary exclusion criteria for each word
        if ( word in stopwordHash ):
            excStopword += 1
        elif ( not re.search(r'[a-zA-Z]', word, 0) ):
            excNonalphabetic += 1
        elif ( globalWordCountHash[word] < minWordCount ):
            excMinwordcount += 1
        elif ( word not in globalDocCountHash ):
            print ("Warning: Word '%s' not in doc count hash") % (word)
            excNotindochash += 1
        elif ( globalDocCountHash[word] < minDocCount ):
            excMindoccount += 1
        elif ( float(globalDocCountHash[word])/float(numDocs) > maxDocFreq ):
            excMaxdocfreq += 1
        else:
            # Add word to vocab
            vocabHash[word]= globalWordCountHash[word];
            vocabCount += 1 
    print ("Excluded %d stop words" % (excStopword))       
    print ("Excluded %d non-alphabetic words" % (excNonalphabetic))  
    print ("Excluded %d words below word count threshold" % (excMinwordcount)) 
    print ("Excluded %d words below doc count threshold" % (excMindoccount))
    print ("Excluded %d words above max doc frequency" % (excMaxdocfreq)) 
    print ("Final Vocab Size: %d words" % vocabCount)
            
    return vocabHash