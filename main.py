from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import Vectorizers as v
import DataPreprocessing as d
import pandas as pd
import seaborn as sns

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def main():
    #uncomment this if you would like to open the original csv file and create a smaller dataset.  This is commented as it has already been done and doesn t have to be run each time
    '''
    df = pd.read_csv('blogtext.csv')
    text = df.iloc[0:1500,:]
    text.to_csv("smallerDataSet.csv", index=False, encoding='utf8')
    '''

    #uncomment the below if you would like to preprocess the data. This is commented as it has already been done and doesn t have to be run each time as the results are saved in modifiedSet.csv
    '''
    data = pd.read_csv('smallerDataSet.csv', encoding = "utf")
    d.preprocessingData(data)
    '''

    df = pd.read_csv('modifiedSet.csv', encoding="utf")
    df = shuffle(shuffle(shuffle(df)))
    train, test = train_test_split(df, test_size=0.25, random_state=42)

    train_labels, test_labels = d.changeLabelsToNumerical(train, test) #get binary labels male = 0 female =1

    sns.countplot(train['gender'])  # show a bar chart of the frequency of each class in our dataset
    plt.title("Bar chart of frequencies of each class")
    plt.show()

    #Let the user choose what features he wants to use and then call the functions needed according to the choice.
    choice = input("Choose the type of Vectorizer:\n1. Count Vectorizer\n2. TFidf Vectorizer\n3.Character-level feature vector\n"
                   "4.Word-level feature vector\n 5. all\nEnter number of choice(1/2/3/4/5): ")
    if choice == '1':
        count_train_features, count_test_features = v.Count_Vectorizer(train, test)
        count_train_features_NN, count_test_features_NN = v.Count_Vectorizer_For_NN(train, test)  # for the neural network
        v.classifyWithCounterVec(count_train_features, count_test_features, train_labels, test_labels,
                               count_train_features_NN, count_test_features_NN)
    elif choice == '2':
        tfidf_train_features, tfidf_test_features = v.Tfidf_Vectorizer(train, test)
        tfidf_train_features_NN, tfidf_test_features_NN = v.Tfidf_Vectorizer_for_NN(train, test)  # for neural net
        v.classifyWithTFidfVec(tfidf_train_features, tfidf_test_features, train_labels, test_labels,
                             tfidf_train_features_NN, tfidf_test_features_NN)
    elif choice == '3':
        char_feature_vec_train = v.characterFeatureVec(train)
        char_feature_vec_test = v.characterFeatureVec(test)
        v.classifyWithWordorCharVec(char_feature_vec_train, char_feature_vec_test, train_labels, test_labels)
    elif choice == '4':
        word_feature_vec_train = v.wordFeatureVec(train)
        word_feature_vec_test = v.wordFeatureVec(test)
        v.classifyWithWordorCharVec(word_feature_vec_train, word_feature_vec_test, train_labels, test_labels)
    elif choice == '5':
        count_train_features, count_test_features = v.Count_Vectorizer(train, test)
        count_train_features_NN, count_test_features_NN = v.Count_Vectorizer_For_NN(train, test) #for the neural network

        tfidf_train_features, tfidf_test_features = v.Tfidf_Vectorizer(train, test)
        tfidf_train_features_NN, tfidf_test_features_NN = v.Tfidf_Vectorizer_for_NN(train, test) #for neural net

        char_feature_vec_train = v.characterFeatureVec(train)
        char_feature_vec_test = v.characterFeatureVec(test)

        word_feature_vec_train = v.wordFeatureVec(train)
        word_feature_vec_test = v.wordFeatureVec(test)

        v.classifyWithCounterVec(count_train_features, count_test_features, train_labels, test_labels,
                               count_train_features_NN, count_test_features_NN)
        v.classifyWithTFidfVec(tfidf_train_features, tfidf_test_features, train_labels, test_labels,
                             tfidf_train_features_NN, tfidf_test_features_NN)

        v.classifyWithWordorCharVec(char_feature_vec_train, char_feature_vec_test, train_labels, test_labels)
        v.classifyWithWordorCharVec(word_feature_vec_train, word_feature_vec_test, train_labels, test_labels)

    input("Press enter to quit: ")


main()

