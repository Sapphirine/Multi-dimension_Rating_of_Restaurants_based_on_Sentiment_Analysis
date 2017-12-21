# Multi-dimentsion Rating of Restaurant Based on Sentiment Analysis

This code implements a novel aspect extraction methods based on syntax tree and syntactic rules are used for obtaining aspect-opinion pairs. We also developed an aspect classification algorithm based on similarity measured by dice coefficient. And to capture the inner relationship between words and sentences and to effectively use data from previous time steps, we implemented recurrent neural network to predict sentiment of user-generated review, thus giving the rating of attributes based on the polarity. 

## Requirements

This code is written in Python 3.5, and depends on having TensorFlow installed.

The bare minimum you should need to do to get everything running, assuming you have Python, is

```Bash
sudo pip install --upgrade tensorflow
sudo pip install numpy
sudo pip install nltk
```

## Using it

First, download the SemEval training dataset and specify the file path in `aspect_extractor.py`

```python
trainfile = "/Users/cee/Downloads/AspectBasedSentimentAnalysis/datasets/Restaurants_Train_v2.xml"
testfile = "testReview.xml"
```

Next, you should download twitter training dataset from website, then download test data in the same folder. You need to first specify the file path in `sentiment_analysis.py`:

```python
train_data = json.load(open('tweets_data/trainTweets_preprocessed.json', 'r'))
test_data = json.load(open('tweets_data/testTweets_preprocessed.json', 'r'))
```

After preparing the data, you can run `aspect_extractor.py` to extract aspect and categorize the extract aspetct term. The genererated prediction file will be stored in `AspectBasedSA--test.xml`

Then you can run  `sentiment_analysis.py` to predict the sentiment.

## Contact us

If you encounter any problem, just contact any of us:

__Zixi Huang__: zh2313@columbia.edu

__Zekun Gong__: zg2273@columbia.edu

__Shilin Hu__: sh3659@columbia.edu
