## Python files:
main.py contains the main() function which is the runner. 
Classifiers.py contains all the functions that implement different classifiers, passing the required parameters each time.
DataPreprocessing.py performs data preprocessing.  This does not have to be used as the preprocessed data can be found in "modifiedSet.csv" under the "blogs" column.
Vectorizers.py contains functions that perform feature extraction (Count vector, TF-IDF, word features and character features)
yuleK.py contains the algorithm of yule K richness.

## CSV files:
* The large corpus is not provided as it was only used to extract a smaller one.  The link to the large corpus is provided as a link in the documentation.
* smallerDataSet.csv  contains the smaller data set extracted from the original. 
* modifiedSet.csv is the same as the smallerDataSet.csv but has an extra column containing the preprocessed data.

## Running the project:
* A cmd file called RunMe.cmd is provided.
* Double click it to start running the program in your command prompt.
* You are provided with a choice when running the program, to choose the feature set you would like to use.  All the classifiers are then applied on your choice.
* Please note that at first it might take a little while to display anything due to its processing of data.

## Requirements:
* keras
* scikit-learn
* matplotlib
* seaborn
* pandas
* numpy
* nltk
