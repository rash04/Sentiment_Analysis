"""
COMP 551 (Applied Machine Learning) Assignment 3 Question 1
"Sentiment Classification" - Dataset Generation
Name: RASHIK HABIB
McGill University
Date: 17th February, 2018
"""

import numpy as np
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer

dataset = "yelp"

"""----------------------------FEATURE EXTRACTION----------------------------"""
#The steps performed below were done for both IMDB and yelp datasets by changing the variable 'dataset'
# in order to generate the datasets in the format required for the assignment

#remove HTML tags, punctuations and use lower case
file_input_train = open("hwk3_datasets/" +str(dataset) + "-train.txt", "r")
train_stripped = []
for line in file_input_train.readlines():
    line = line.replace('<br />', ' ')
    for ch in punctuation:
        line = line.replace(ch, '')
    train_stripped.append(line.lower())
file_input_train.close()

file_input_valid = open("hwk3_datasets/" +str(dataset) + "-valid.txt", "r")
valid_stripped = []
for line in file_input_valid.readlines():
    line = line.replace('<br />', ' ')
    for ch in punctuation:
        line = line.replace(ch, '')
    valid_stripped.append(line.lower())
file_input_valid.close()

file_input_test = open("hwk3_datasets/" +str(dataset) + "-test.txt", "r")
test_stripped = []
for line in file_input_test.readlines():
    line = line.replace('<br />', ' ')
    for ch in punctuation:
        line = line.replace(ch, '')
    test_stripped.append(line.lower())
file_input_test.close()

#vectorize the entire training vocabulary, including a max of 10000 unique words
#which are not common english articles (like 'the', 'and', etc.)
vectorizer = CountVectorizer(stop_words='english', max_features=10000, encoding='utf-8', strip_accents='ascii', decode_error='ignore')
sparse_mat = vectorizer.fit_transform(train_stripped)
features = vectorizer.vocabulary_
frequencies = sparse_mat.sum(axis=0)

# Output to a file called vocab.txt
file_output = open("hwk3_datasets/" +str(dataset) + "_vocab.txt", "w")
for k, v in features.items():
    file_output.write(k + str('\t') + str(v) + str('\t') + str(frequencies[0,v]) + str('\n'))
file_output.close()

#convert the train/test/valid files into vector representations and output to respective text files
file_output = open("hwk3_datasets/" +str(dataset) + "-train-submit.txt", "w")
for review in train_stripped:
    words = review[:-2].split()
    i = 0;
    for word in words:
        i = i + 1        
        if word == len(words):    #avoid writing a space if it is the last word
            if word in features:
                file_output.write(str(features[word]))
        else:
            if word in features:
                file_output.write(str(features[word]) + ' ')
    file_output.write('\t' + review[-2] + '\n')    # add class label
file_output.close()    

file_output = open("hwk3_datasets/" +str(dataset) + "-valid-submit.txt", "w")
for review in valid_stripped:
    words = review[:-2].split()
    i = 0;
    for word in words:
        i = i + 1
        if word == len(words):    #avoid writing a space if it is the last word
            if word in features:
                file_output.write(str(features[word]))
        else:
            if word in features:
                file_output.write(str(features[word]) + ' ')
    file_output.write('\t' + review[-2] + '\n')    # add class label
file_output.close()

file_output = open("hwk3_datasets/" +str(dataset) + "-test-submit.txt", "w")
for review in test_stripped:
    words = review[:-2].split()
    i = 0;
    for word in words:
        i = i + 1 
        if i == len(words):    #avoid writing a space if it is the last word
            if word in features:
                file_output.write(str(features[word]))
        else:
            if word in features:
                file_output.write(str(features[word]) + ' ')
    file_output.write('\t' + review[-2] + '\n')    # add class label
file_output.close()