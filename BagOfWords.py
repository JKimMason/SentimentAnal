import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import nltk

if __name__ == '__name__':
	# Read data
	train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data',
		'labeledTrainData.tsv'), header=0, \
					delimiter="\t", quoting=3)
	test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data',
		'testData.tsv'), header=0, delimiter="\t", \
					quoting=3)
	print ("First review:")
	print (train["review"][0])
	raw_input("Press Enter to continue...")

	# Clean training data
	print ("Download text data sets")
	nltk.download()
	clean_train_reviews = []

	print ("Cleaning and parsing training set movie reviews...")
	print
	for i in range(0, len(train['review'])):
		clean_train_reviews.append(" ".join(
			KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

	# Create bag of words
	print("Creating bag of words...")
	pritn
	vectorizer = CountVectorizer(analyzer = "word", 	\
								tokenizer = None,	\
								preprocessor = None,	\
								stop_words = None,	\
								max_features = 5000)
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	train_data_features = train_data_features.toarray()

	# Train classifier
	print ("Training random forest ")
	print ("May take awhile~...")
	forest = RandomForestCalssifier(n_estimators = 100)
	forest = forest.fit(train_data_features, train["sentiment"])
	clean_test_reviews = []

	# Format testing data
	print ("Cleaning and parsing the test set movie reviews...\n")
	for i in range(0, len(test["review"])):
		clean_test_reviews.append(" ".join(
			KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

	test_data_features = vectorizer.transform(clean_test_reviews)
	test_data_features = test_data_features.toarray()


	# Predict reviews in testing data
	print ("Predicting test labels...\n")
	result = forest.predict(test_data_features)
	output = pd.DataFrame( data={"id":test["id"], "sentiment":result})
	# Use pandas to write to output file
	output.to_csv(os.path.join(os.path.dirname(__file__),
		'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
	print ("Wrote results to Bag_of_Words-model.csv")
