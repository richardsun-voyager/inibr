# Implicit N-grams Induced by Recurrence

In this work, we present a study that shows there actually exist some explainable components that reside within the hidden states, which are reminiscent of the classical n-grams features. We evaluated such extracted explainable features from trained RNNs on downstream sentiment analysis tasks and found they could be used to model interesting linguistic phenomena such as negation and intensification. Furthermore, we examined the efficacy of using such n-gram components alone as encoders on tasks such as sentiment analysis and language modeling, revealing they could be playing important roles in contributing to the overall performance of RNNs.

To run the code, please open the corresponding jupyter notebook file. 
- For sentiment analysis, please launch and run the file "RNN_classification.ipynb".
- For relation classification, please launch and run the file "RNN_relation_classification(semeval2010-8).ipynb".
- For NER, please launch and run the file "RNN_NER_CRF_IOBES.ipynb" (Please download the dataset from https://www.ldc.upenn.edu/).
- For the n-gram analysis, please launch and run the file "RNN_ngram_phenomena_analysis.ipynb". 

