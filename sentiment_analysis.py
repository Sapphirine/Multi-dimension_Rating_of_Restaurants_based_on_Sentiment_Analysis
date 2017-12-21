import re
from __future__ import print_function
import tensorflow as tf
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import pickle
import nltk
import string
#nltk.download()  
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


def tokenize(raw_docs):
    tokenized_sents = sent_tokenize(raw_docs)
    tokenized_words = [word_tokenize(sentence) for sentence in tokenized_sents]
    regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

    tokenized_words_no_punctuation = []

    for review in tokenized_words:
        new_review = []
        for token in review:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)
    
        tokenized_words_no_punctuation.append(new_review)

    while [] in tokenized_words_no_punctuation:
        tokenized_words_no_punctuation.remove([])
    return tokenized_words_no_punctuation


def linear(input_, output_size, name, init_bias=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        init = tf.truncated_normal([shape[-1], output_size], mean=0.0, stddev=1.0 / shape[-1]**0.5)
        W = tf.get_variable("weight", initializer=init)
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(name):
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(input_, W) + b


def embedding(input_, vocab_size, output_size, name):
    """
    1. Define an embedding matrix
    2. return both the lookup results and the embedding matrix.
    """
    with tf.variable_scope(name, reuse=None):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, output_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, input_)
    return embed, embeddings



def transform(string, vocabulary):
    #data_num = len(string)
    ret = []
    v = vocabulary.tolist()
    for data in string:
        trans = []
        for word in data:
            if word in v:
                trans.append(v.index(word))
            if len(trans) > 20:
                trans = trans[0:20]
        ret.append(trans)    
    ret = np.array(ret)
    final = []
    for t in ret:
        t = np.pad(t,(0,20-len(t)),'constant')
        t = np.reshape(t,(-1,20))
        final = np.append(final,t)
    final = np.reshape(final,(-1,20))
    
    return final


def probability(dataset):
    # input: a = [['i','like','your','mother'],['i','like','apple','and','pen']]
    with open("./tweets_data/vocabulary.pkl", "rb") as f:
        vocabulary = pickle.load(f)

    # load our data and separate it into tweets and labels
    train_data = json.load(open('tweets_data/trainTweets_preprocessed.json', 'r'))
    train_data = list(map(lambda row:(np.array(row[0],dtype=np.int32),str(row[1])),train_data))
    train_tweets = np.array([t[0] for t in train_data])
    train_labels = np.array([int(t[1]) for t in train_data])

    test_data = json.load(open('tweets_data/testTweets_preprocessed.json', 'r'))
    test_data = list(map(lambda row:(np.array(row[0],dtype=np.int32),str(row[1])),test_data))
    test_tweets = np.array([t[0] for t in test_data]) # get the data
    test_labels = np.array([int(t[1]) for t in test_data]) #label

    print("size of original train set: {}".format(len(train_tweets)))
    print("size of original test set: {}".format(len(test_tweets)))

    # only select first 1000 test sample for test
    test_tweets = test_tweets[:1000]
    test_labels = test_labels[:1000]

    print("*"*100)
    print("size of train set: {}, #positive: {}, #negative: {}".format(len(train_tweets), np.sum(train_labels), len(train_tweets)-np.sum(train_labels)))
    print("size of test set: {}, #positive: {}, #negative: {}".format(len(test_tweets), np.sum(test_labels), len(test_tweets)-np.sum(test_labels)))

    
    #tensorflow model
    tweet_size = 20
    hidden_size = 100
    vocab_size = 7597
    batch_size = 64

    
    tf.reset_default_graph()

    # make placeholders for data we'll feed in
    tweets = tf.placeholder(tf.int32, [None, tweet_size])
    labels = tf.placeholder(tf.float32, [None])

    embed, embeddings = embedding(tweets, vocab_size, 100, 'embedding')

    # define the lstm cell
    lstm_cell = tf.contrib.rnn.LSTMCell(hidden_size)

    # define the op that runs the LSTM, across time, on the data
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, embed, dtype=tf.float32)

    # define that our final sentiment logit is a linear function of the final state of the LSTM
    sentiment = linear(final_state[-1], 1, name="output")

    # define cross entropy/sigmoid loss function
    sentiment = tf.squeeze(sentiment, [1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=sentiment, labels=labels)
    loss = tf.reduce_mean(loss)

    # compute accuracy
    prob = tf.nn.sigmoid(sentiment)
    prediction = tf.to_float(tf.greater_equal(prob, 0.5))
    acc = tf.to_float(tf.equal(prediction, labels))
    acc = tf.reduce_mean(acc)
    
    # define optimizer
    trainer = tf.train.AdamOptimizer()
    gradients = trainer.compute_gradients(loss)
    gradients_clipped = [(tf.clip_by_value(t[0],-1,1),t[1]) for t in gradients]
    optimizer = trainer.apply_gradients(gradients_clipped)
    
  
    num_steps = 1000
    num_train = 60000
    best_acc = 0
    batch_size = 60

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(num_steps):
            # feed training batch
            batch_tweets = train_tweets[(step * batch_size): ((1 + step) * batch_size)]
            batch_labels = train_labels[step * batch_size: (1 + step) * batch_size]

           
            _, train_loss, train_acc =  sess.run([optimizer, loss, acc], feed_dict={tweets: batch_tweets, labels: batch_labels})#pass
    
            if (step % 50 == 0):
                epc = step/50
                print("epoch {} ".format(epc + 1))
                #s1 = sess.run(sentiment,feed_dict={tweets: test_tweets})
                #s.append(s1)
                test_acc = sess.run(acc, feed_dict={tweets: test_tweets, labels: test_labels})
            
                if test_acc > best_acc:
                    best_acc = test_acc
                    embed_table = embeddings.eval()
                print('training loss:{} test_accuracy: {}%'.format(train_loss, test_acc*100))
        #trans = transform(dataset, vocabulary)
#        print(trans)
        proba = {}
        for i in range(len(dataset)):
            trans = transform(dataset[i], vocabulary)
            proba[i] = sess.run(prob,feed_dict={tweets: trans})           
#        proba = sess.run(prob,feed_dict={tweets: trans})
    return proba


if __name__ == "__main__":
    #read json file
    json_file = './rrdata.json'
    Handle = open(json_file,'r')
    Buff = Handle.readlines()
    data_dict = []
    for line in Buff:
        data_dict.append(json.loads(line))
    rr_data = data_dict[0]
    score = {}
    index = 0
    r = 0

    process = {}
    for i in range(len(a)):
        process[i] = tokenize(a[i])
    process = np.array(process)
    p = probability(process)
    #choose 50000 reviews to test
    while index < 50000:
        l = len(rr_data[r]['review'])
        #print("len=%d"%(l))
        score[rr_data[r]['id']] = []
        for i in range(l):
            score[rr_data[r]['id']].append(p[index])
            index += 1
            if index >= 50000:
                break
        r += 1

    print(score) # score is the final score of aspects of restaurants

