import pandas as pd
import numpy as np

def preprocessingData(data):
    train_max_len = 0
    bag_of_words = {} #create a bag of words and stores their frequency
    for text in data['text']:
        words = text.split(" ")
        for word in words:
            if word not in bag_of_words or word.isnumeric == False:
                bag_of_words[word] = 1
            else:
                bag_of_words[word] += 1

    counter = 0
    indexes = []
    New_blog = pd.Series([])

    for text in data['text']:
        words = text.split(" ")
        sentence = " " #will be constructed along the way
        sentenceLength = 0
        for word in words:
            if bag_of_words[word] > 3 and bag_of_words[word] < 6000: #keep words which are not too frequent or almost not frequent
                sentence = sentence + " " + word #add the accepted word to the sentence
                sentenceLength += 1
        if sentenceLength > train_max_len:
            train_max_len = sentenceLength
        if (sentence == " "): #check if we ended up with no sentence to avoid lack of data
            indexes.append(counter) #keep the index of where this happened to later delete the row
            New_blog[counter] = sentence #build the panda series of blogs
        else:
            New_blog[counter] = sentence
        counter += 1

    train = data.assign(blogs=New_blog.values)
    for idx in indexes:
        train.drop([train.index[idx]], axis=0, inplace=True) #removes any rows which now ended up without a blog as no words of the blog are accepted by the preprocessing method

    train.to_csv("modifiedSet.csv", index=False, encoding='utf8')#add the processed data to a new csv file

def changeLabelsToNumerical(train,test):
    # Change the labels into binary values 0, or 1 to pass into the classifiers which require numerical input.  Do it for both test labels and train labels
    labels = np.array(train['gender'])
    train_labels = []
    for label in labels:
        if label == 'male':
            train_labels.append(0)
        else:
            train_labels.append(1)

    test_labels = []
    labels2 = np.array(test['gender'])
    for label in labels2:
        if label == 'male':
            test_labels.append(0)
        else:
            test_labels.append(1)

    return train_labels, test_labels