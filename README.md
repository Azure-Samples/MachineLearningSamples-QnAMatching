# QnA Matching


## Link to the Microsoft DOCS site

The detailed documentation for this Q & A matching example includes the step-by-step walk-through:
[https://docs.microsoft.com/azure/machine-learning/preview/scenario-qna-matching](https://docs.microsoft.com/azure/machine-learning/preview/scenario-qna-matching)


## Link to the Gallery GitHub repository

The public GitHub repository for this Q & A matching example contains all the code samples:
[https://github.com/Azure/MachineLearningSamples-QnAMatching](https://github.com/Azure/MachineLearningSamples-QnAMatching)


## Overview

This example addresses the problem of mapping user questions to pre-existing Question & Answer (Q&A) pairs as is typically provided in a list of Frequently Asked Questions (that is, a FAQ) or in the Q&A pairs present on websites like [Stack Overflow](https://stackoverflow.com/). There are many approaches to match a question to its correct answer, such as finding the answer that is the most similar to the question. However, in this example open ended questions are matched to previously asked questions by assuming that each answer in the FAQ can answer multiple semantically equivalent questions.

The key steps required to deliver this solution are as follows:

1. Clean and process text data.
2. Learn informative phrases, which are multi-word sequences that provide more information when viewed in sequence than when treated independently.
3. Extract features from text data.
4. Train text classification models and evaluate model performance.


## Key components needed to run this example

1. An [Azure account](https://azure.microsoft.com/free/) (free trials are available).
2. An installed copy of Azure Machine Learning Workbench with a workspace created.
3. This example could be run on any compute context. However, it is recommended to run it on a multi-core machine with at least of 16-GB memory and 5-GB disk space.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (for example, label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

