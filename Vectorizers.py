from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import Classifiers as c
import numpy as np
import yuleK

def Count_Vectorizer(train, test): #creates a CountVectorizer for all classifiers except the NN
    counter = CountVectorizer(stop_words="english")
    train_features = counter.fit_transform(
        train['text'])  # pass training set to count vectoriser and apply fit transform
    test_features = counter.transform(test['text'])
    return train_features, test_features

def Count_Vectorizer_For_NN(train, test):
    counter = CountVectorizer(stop_words="english", max_features=7000) #creates a CountVectorizer specifically to the NN
    train_features = counter.fit_transform(train['text']).toarray()
    test_features = counter.transform(test['text'])
    return train_features, test_features

def Tfidf_Vectorizer(train, test): #creates a TFIDF Vect for all classifiers except the NN
    tfidf_vec = TfidfVectorizer(stop_words='english')
    train_features = tfidf_vec.fit_transform(train['text'])
    test_features = tfidf_vec.transform(test['text'])
    return train_features, test_features

def Tfidf_Vectorizer_for_NN(train, test): #creates a TFIDF Vect specifically to the NN
    tfidf_vec = TfidfVectorizer(stop_words='english', max_features=7000) #2709
    train_features = tfidf_vec.fit_transform(train['text']).toarray()
    test_features = tfidf_vec.transform(test['text'])
    return train_features, test_features


#extracts character-level features
def characterFeatureVec(data):
    character_feature_vector = []
    for text in data['text']: #loop through all blogs
        no_Of_chars = len(text)

        if no_Of_chars > 0:
            #at each iteration initialize the counts
            no_of_digits=0
            no_of_letters = 0
            no_of_upper_cases = 0
            no_of_white_spaces = 0

            for c in text: #loop through each character
                if c.isalpha(): #if letter
                    no_of_letters += 1
                    if c.isupper(): #if uppercase
                        no_of_upper_cases += 1
                elif c.isdigit(): #if digit
                    no_of_digits += 1
                elif c.isspace(): #if white space
                    no_of_white_spaces += 1
            no_of_special_chars = no_Of_chars - no_of_letters - no_of_white_spaces - no_of_digits #These are the characters which are left

            #normalize them
            norm_no_of_letters = no_of_letters / no_Of_chars
            norm_no_of_upper_cases = no_of_upper_cases / no_Of_chars
            norm_no_of_white_spaces = no_of_white_spaces/no_Of_chars
            norm_no_of_digits = no_of_digits/no_Of_chars
            norm_no_of_special_chars = no_of_special_chars/no_Of_chars

            #build feature vector
            character_feature_vector.append([no_Of_chars,norm_no_of_letters, norm_no_of_special_chars, norm_no_of_digits,
                                             norm_no_of_upper_cases, norm_no_of_white_spaces])
        else:
            character_feature_vector.append([0.0, 0.0 , 0.0, 0.0, 0.0, 0.0])
    np_character_feature_vector = np.array(character_feature_vector) #turn array into numpy array of arrays

    return np_character_feature_vector

def wordFeatureVec(data):
    word_feature_vector = []

    bag_of_words = {} #create a bag of words to be able to analyse the words' frequency and identify unique ones
    for text in data['text']:
        words = text.split(" ")
        for word in words:
            if word not in bag_of_words or word.isnumeric == False:
                bag_of_words[word] = 1
            else:
                bag_of_words[word] += 1

    for text in data['text']:
        words = text.split(" ")
        no_Of_words = len(words)

        if no_Of_words > 0:
            #initialize counters
            word_lens = 0
            no_of_long_words = 0
            no_of_short_words = 0
            no_of_unique_words = 0

            for word in words:
                word_lens = word_lens + len(word) #accumulate teh word lengths to then average them out
                if len(word) >= 6: #words longer or equal to 6 letters
                    no_of_long_words +=1
                if len(word) <=3 : #words less than or equal to 3 letters
                    no_of_short_words+=1
                if bag_of_words[word]==1: #check for unique words (frequency one)
                    no_of_unique_words += 1

            try:
                lexicalRichness = yuleK.yule(text) #perform Yule K richness
            except:
                lexicalRichness = 0

            #normalize the counts
            average_word_len = word_lens/no_Of_words
            vocab_rich = no_of_unique_words/no_Of_words
            norm_no_of_long_words = no_of_long_words/no_Of_words
            norm_no_of_short_words = no_of_short_words/no_Of_words

            #create feature vector and add it to the previous to slowly create a matrix
            word_feature_vector.append([no_Of_words, average_word_len, vocab_rich, norm_no_of_long_words, norm_no_of_short_words, lexicalRichness])
        else:
            word_feature_vector.append([0.0, 0.0 , 0.0, 0.0, 0.0, 0.0])

    np_character_feature_vector = np.array(word_feature_vector) #change array into numpy of arrays

    return np_character_feature_vector

#Call the classifiers and passing the Counter Vectorizer results
def classifyWithCounterVec(count_train_features, count_test_features, train_labels, test_labels, count_train_features_NN, count_test_features_NN):
    print("\n\n---- Using Counter Vectorizer ----")
    print("\n\n----Naive Bayes Classifier ----")
    c.NaiveBayes(count_train_features, count_test_features, train_labels, test_labels)
    print("\n\n----Logistic Regression----")
    c.Logistic_Regression(count_train_features, count_test_features, train_labels, test_labels)
    print("\n\n----Random Forest----")
    c.RandomForest(count_train_features, train_labels, count_test_features, test_labels)
    print("\n\n----SVM----")
    c.SVM(count_train_features, train_labels, count_test_features, test_labels)
    print("\n\n----Feed-Forward Neural Network----")
    c.NeuralNet(count_train_features_NN, train_labels, count_test_features_NN, test_labels)

#Call the classifiers and passing the TFIDF Vectorizer results
def classifyWithTFidfVec(tfidf_train_features, tfidf_test_features, train_labels, test_labels, tfidf_train_features_NN, tfidf_test_features_NN):
    print("\n\n---- Using TFidf Vectorizer ----")
    print("\n\n----Naive Bayes Classifier ----")
    c.NaiveBayes(tfidf_train_features, tfidf_test_features, train_labels, test_labels)
    print("\n\n----Logistic Regression----")
    c.Logistic_Regression(tfidf_train_features, tfidf_test_features, train_labels, test_labels)
    print("\n\n----Random Forest----")
    c.RandomForest(tfidf_train_features, train_labels, tfidf_test_features, test_labels)
    print("\n\n----SVM----")
    c.SVM(tfidf_train_features, train_labels, tfidf_test_features, test_labels)
    print("\n\n----Feed-Forward Neural Network----")
    c.NeuralNet(tfidf_train_features_NN, train_labels, tfidf_test_features_NN, test_labels)

#call the classifiers and pass the word-level feature vectors or the character-level feature vectors
def classifyWithWordorCharVec(train_features, test_features, train_labels, test_labels):
    print("\n\n---- Using Word or Character level features ----")
    print("\n\n----Naive Bayes Classifier ----")
    c.NaiveBayes(train_features, test_features, train_labels, test_labels)
    print("\n\n----Logistic Regression----")
    c.Logistic_Regression(train_features, test_features, train_labels, test_labels)
    print("\n\n----Random Forest----")
    c.RandomForest(train_features, train_labels, test_features, test_labels)
    #print("\n\n----SVM----") #commented out as it can be really time consuming
    #c.SVM(train_features, train_labels, test_features, test_labels)
    print("\n\n----Feed-Forward Neural Network----")
    c.NeuralNetForWordorCharFeatures(train_features, train_labels, test_features, test_labels)