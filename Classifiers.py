from keras import models
from keras import layers
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#This function output the accuracy, classification report and the confusion matrix of every classifier.
def outputResults(test_labels, predictions, classifier):
    print(classifier, " Accuracy: ", metrics.accuracy_score(test_labels, predictions) * 100)
    print("\n\n", classifier," Classification Report: \n", metrics.classification_report(test_labels, predictions))
    print("\n\n",classifier," Confusion Matrix:\n", metrics.confusion_matrix(test_labels, predictions))


def NaiveBayes(train_features, test_features, train, test):
    # -------NAIVE BAYES CLASSIFIER-----
    naiveBayes = MultinomialNB().fit(train_features, train)
    NB_predictions = naiveBayes.predict(test_features)
    # print('{} => {} -{}'.format(test['text'].iloc[100], NB_predictions[100],test['gender'].iloc[100])) #prints an example
    outputResults(test, NB_predictions, classifier = 'NB')


def Logistic_Regression(train_features, test_features, train, test):
    #------LOGISTICmREGRESSION - ------
    logistic_regression = LogisticRegression().fit(train_features, train)
    LR_predictions = logistic_regression.predict(test_features)
    outputResults(test, LR_predictions, classifier = 'LR')


def RandomForest(train_features, train_labels, test_features, test_labels):
    # -----RANDOM FOREST CLASSIFIER-----
    random_forest = RandomForestClassifier(n_estimators=100)
    rf = random_forest.fit(train_features, train_labels)
    RF_predictions = rf.predict(test_features)
    outputResults(test_labels, RF_predictions, classifier='RF')

def SVM(train_features, train_labels, test_features, test_labels):
    # -----SVM------
    kernels = ['poly', 'linear', 'rbf']  # an array of the kernels the experiments will be carried out on
    #gamma_values = [0.1, 0.01, 10]  # an array of the gamma values the experiments will be carried out on
    #c_values = [0.1, 0.01, 10, 100]  # an array of the regularization values the experiments will be carried out on
    gamma_values = [0.01,0.1,0.01]
    c_values = [0.01,0.1,10]

    indx=0
    for kernelType in kernels:  # for each kernel
        #for gammaValue in gamma_values:  # for each gamma value
            #for cValue in c_values:  # for each regularization value
                gammaValue = gamma_values[indx]
                cValue = c_values[indx]
                # print the parameters being used for the user to comprehend what the program is currently computing
                print('SVM Classifier with gamma =', gammaValue, ', Kernel = ', kernelType, ', C= ', cValue)
                # set the parameters to the actual classifier
                classifier = SVC(gamma=gammaValue, kernel=kernelType, random_state=0, C=cValue)
                classifier.fit(train_features, train_labels)  # start training
                SVM_predictions = classifier.predict(test_features)
                outputResults(test_labels, SVM_predictions, classifier='SVM')


def NeuralNet(train_features, train_labels, test_features, test_labels):
    model = models.Sequential()
    #model.add(layers.Dropout(0.2, input_shape=(5000,)))
    model.add(layers.Dense(64, activation='relu', input_shape=(7000,)))  # there are 10000 inputs going into 16 neurons
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))  # this is the second layer with 16 other neurons
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))  # output

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(train_features, train_labels, batch_size=32, epochs=40, verbose=1)

    pred = model.predict_classes(test_features)
    outputResults(test_labels, pred, classifier='NN')

def NeuralNetForWordorCharFeatures(train_features, train_labels, test_features, test_labels): #this is different from the above as the input shape is now 6 and the hyperparameters are also modified to suite better the feature set
    model = models.Sequential()
    #model.add(layers.Dropout(0.2, input_shape=(5000,)))
    model.add(layers.Dense(16, activation='relu', input_shape=(6,)))  # there are 10000 inputs going into 16 neurons
    model.add(layers.Dense(16, activation='relu'))  # this is the second layer with 16 other neurons
    model.add(layers.Dense(1, activation='sigmoid'))  # output

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(train_features, train_labels, batch_size=8, epochs=100, verbose=1)

    pred = model.predict_classes(test_features)
    outputResults(test_labels, pred, classifier='NN')

