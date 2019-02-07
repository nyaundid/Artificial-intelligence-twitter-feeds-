# define args
run_env = 'windows'
drop_user = True
simplify_http = True
remove_stop_words = False
apply_porter_stemmer = True
padding = 'post'
verbose = 1
n_samples = 150000

if run_env == 'colaboratory':
    #!pip install gensim
    from google.colab import drive
    drive.mount('/content/drive')
    wrk_dr = '/content/drive/My Drive/Colab Notebooks/'
else:
    wrk_dr = 'C:/Users/toblon/StThomas/764 Artificial Intelligence/project/'

data_dir = wrk_dr + 'data/'

import ast
import datetime
from gensim.models import Word2Vec
import io
from keras import backend as K
from keras import regularizers
from keras.layers import LSTM, GRU, GRUCell, Dense, Flatten, TimeDistributed, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import porter
#from nltk.tokenize import word_tokenize
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import string
from textblob import TextBlob

# load raw data
start = datetime.datetime.now()
twcs = pd.read_csv(data_dir + 'twcs.zip')[['author_id','text']]
print('upload time: ' + str(datetime.datetime.now() - start))

# setup train and test size
if run_env == 'windows':
    val_and_test_prop = 0.07
    val_and_test_size = int(n_samples * val_and_test_prop)
    n_samples += val_and_test_size * 2
    twcs = twcs.sample(n=n_samples, random_state=345)
print(twcs.head())

# label tweet as one of 109 corp accounts, or non-corp acct (92 classes) 40% of data is other class
# consolidate non-corporate accounts
start = datetime.datetime.now()
twcs['author_id'] = twcs['author_id']\
  .apply(lambda x: x if x.replace('_','')\
  .replace('O2','O').isalpha() else 'non-corporate')
print('consolidate author time: ' + str(datetime.datetime.now() - start))

# avoid not having class representitives in train and validate sets
if run_env == 'windows':
    n_tweets_by_author = twcs['author_id'].value_counts()
    twcs = twcs[twcs['author_id'].isin(n_tweets_by_author.index[n_tweets_by_author >= 500])]
    min_tweets = 3
    print(str(len(n_tweets_by_author[n_tweets_by_author >= min_tweets])) + ' included classes:')
    print(n_tweets_by_author[n_tweets_by_author >= min_tweets])
    val_and_test_size = int(len(twcs) * val_and_test_prop)

# sns.barplot(n_tweets_by_author.values[:50], n_tweets_by_author.index[:50], orient='h')

# cleanse text
# remove @____ text

drop_user_pattern = re.compile("(@[A-Za-z0-9]+)")
http_pattern = re.compile('http[^\s]+')
translator = str.maketrans('', '', string.punctuation)
porter_stemmer = porter.PorterStemmer()

def word_cleanse(word):
    # porter stem & remove punctuation
    word = porter_stemmer.stem(word.translate(translator))

    # simplify web addresses
    if simplify_http:
        return re.sub(pattern=http_pattern, repl='http', string=word)
    else:
        return word

def text_cleanse(tweet_text):
    """
    cleanse text column of the tweet
    :param tweet_text: string of words (tweet)
    :return: list of cleansed words in tweet text
    """
    # replace @username tags with marker
    if drop_user:
        tweet_text = ' '.join(re.sub(drop_user_pattern, " ", tweet_text).split())\
            .lower()\
            .replace('  ', ' ')\
            .split(' ')
    else:
        tweet_text = tweet_text.lower() \
            .replace('  ', ' ') \
            .split(' ')

    # remove stop words (decided not to do this because removes important words like et (eastern time))
    if remove_stop_words:
        tweet_text = [word for word in tweet_text.split(' ')
                      if (word not in stopwords.words('english'))
                      & (len(word)>2) & (len(word)<15)]

    # apply porter stemmer & remove punctuation
    if apply_porter_stemmer:
        tweet_text = [word_cleanse(word=word) for word in tweet_text]

    return tweet_text

start = datetime.datetime.now()
twcs['text'] = twcs['text'].apply(lambda x: text_cleanse(tweet_text=x))
print('cleanse time: ' + str(datetime.datetime.now() - start))

if run_env == 'colaboratory':
    twcs.to_csv(data_dir + 'twcs_cleansed.gzip', compression='gzip', index=False)

twcs.head()

if run_env == 'colaboratory':
    start = datetime.datetime.now()
    twcs = pd.read_csv(data_dir + 'twcs_cleansed.gzip', compression='gzip')
    print('read data time:' + str(datetime.datetime.now() - start))

    start = datetime.datetime.now()
    twcs['text'] = twcs['text'].apply(ast.literal_eval)
    print('string to list time:' + str(datetime.datetime.now() - start))

start = datetime.datetime.now()
x_train, x_val, y_train, y_val = \
    train_test_split(twcs['text'], twcs['author_id'],
                     test_size=val_and_test_size,
                     stratify=twcs['author_id'],
                     random_state=3135,
                     shuffle=True)

x_train, x_test, y_train, y_test = \
    train_test_split(x_train, y_train,
                     test_size=val_and_test_size,
                     stratify=y_train,
                     random_state=3135,
                     shuffle=True)

x_train.reset_index(inplace=True, drop=True)
x_val.reset_index(inplace=True, drop=True)
x_test.reset_index(inplace=True, drop=True)
print('split time: ' + str(datetime.datetime.now() - start))

# reformat y to one-hot encoding for Keras cat target
y_train = [TextBlob(' '.join(x)).sentiment[0] for x in x_train]
y_val = [TextBlob(' '.join(x)).sentiment[0] for x in x_val]
y_test = [TextBlob(' '.join(x)).sentiment[0] for x in x_test]

def to_unique_words(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for sent in seq:
       for item in sent:
           marker = idfun(item)
           if marker in seen: continue
           seen[marker] = 1
           result.append(item)
   return result


unique_words = to_unique_words(x_train)
vocab_size = len(unique_words)

# convert x to sequence data
sent_len = x_train.apply(len)
max_sent_len = np.max(sent_len)
print('\nmax_steps: ' + str(max_sent_len))


def to_token(x, vocab_size, max_len, padding):
    x = x.apply(lambda s: ' '.join(s))
    x = [one_hot(w, vocab_size) for w in x]
    return pad_sequences(x, maxlen=max_len, padding=padding)

start = datetime.datetime.now()
x_train = to_token(x=x_train, vocab_size=vocab_size, max_len=max_sent_len, padding=padding)
x_val = to_token(x=x_val, vocab_size=vocab_size, max_len=max_sent_len, padding=padding)
x_test = to_token(x=x_test, vocab_size=vocab_size, max_len=max_sent_len, padding=padding)
print('to_token time:' + str(datetime.datetime.now() - start))

def plot_training_results(metric, history, nn, x_test, y_test, file=None):
    test_loss = np.sqrt(nn.evaluate(x=x_test, y=y_test))
    test_result = test_loss if metric == 'loss' else test_acc

    plt.figure(figsize=(8*1.5, 6*1.5))
    plt.plot(history.history[metric], label='train')
    plt.plot(history.history['val_' + metric], label='val')
    plt.xlabel('epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title(metric.title() + ' by Epoch    |    Test RMSE: ' + str(round(test_result, 3)))

    if file is not None:
        plt.savefig(file, transparent=True)
    else:
        plt.show()

    plt.close()

# y = y_test
# x=x_test
# nn=model
def plot_confusion_matrix(x, y, nn, file=None):
    y = np.argmax(y, axis=1)
    y_hat = nn.predict(x=x)
    y_hat = np.argmax(y_hat, axis=1)

    f1 = np.round(f1_score(y, y_hat, average='micro'), 3)

    conf_matrix = np.log(1 + confusion_matrix(y_true=y, y_pred=y_hat))
    # conf_matrix = confusion_matrix(y_true=y, y_pred=y_hat)

    plt.figure(figsize=(8 * 1.5, 6 * 1.5))
    sns.heatmap(conf_matrix, center=np.median(conf_matrix))
    plt.title('Confusion Matrix (log of 1 + count)    |    F1: ' + str(f1))

    if file is not None:
        plt.savefig(file, transparent=True)
    else:
        plt.show()

    plt.close()

# plot_confusion_matrix(x=x_test, y=y_test, nn=model)

# # define fully connected model
# fc_model = Sequential()
# fc_model.add(Embedding(vocab_size, 500, input_length=max_sent_len))
# fc_model.add(Flatten())
# fc_model.add(Dense(250, activation='relu'))
# fc_model.add(Dropout(.6))
# fc_model.add(Dense(100, activation='relu'))
# fc_model.add(Dropout(.4))
# fc_model.add(Dense(50, activation='relu'))
# fc_model.add(Dense(y_val.shape[1], activation='softmax'))
# # compile the model
# fc_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# # summarize the model
# print(fc_model.summary())
# # fit the model
# start = datetime.datetime.now()
# history = fc_model.fit(x_train, y_train, epochs=15, verbose=verbose, validation_data=[x_val, y_val], batch_size=256)
# print('fc train time: ' + str(datetime.datetime.now() - start))
#
# plot_training_results(metric='acc', history=history, nn=fc_model, x_test=x_test, y_test=y_test,
#                       file=data_dir + 'fc_acc.png')
# plot_training_results(metric='loss', history=history, nn=fc_model, x_test=x_test, y_test=y_test,
#                       file=data_dir + 'fc_loss.png')
# plot_confusion_matrix(x=x_test, y=y_test, nn=fc_model, file=data_dir + 'fc_confusion.png')
#
# fc_model.save(data_dir + 'fc_model.h5')
#
# t = np.bincount(np.argmax(y_test, axis=1)) / np.sum(np.argmax(y_test, axis=1))


# define word embedding model
# we_model = Sequential()
# we_model.add(Embedding(vocab_size, 500, input_length=max_sent_len))
# we_model.add(Flatten())
#
# we_model.add(Dense(y_val.shape[1], activation='softmax'))
# # compile the model
# we_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# # summarize the model
# print(we_model.summary())
# # fit the model
# start = datetime.datetime.now()
# history = we_model.fit(x_train, y_train, epochs=15, verbose=verbose, validation_data=[x_val, y_val], batch_size=256)
# print('we train time: ' + str(datetime.datetime.now() - start))
#
# plot_training_results(metric='acc', history=history, nn=we_model, x_test=x_test, y_test=y_test,
#                       file=data_dir + 'we_acc.png')
# plot_training_results(metric='loss', history=history, nn=we_model, x_test=x_test, y_test=y_test,
#                       file=data_dir + 'we_loss.png')
# plot_confusion_matrix(x=x_test, y=y_test, nn=we_model, file=data_dir + 'we_confusion.png')
#
# we_model.save(data_dir + 'we_model.h5')

# # define sequential model
# batch_size = 2**9
# embed_size = 500
# epochs = 4
# lstm_model = Sequential()
# lstm_model.add(Embedding(vocab_size, embed_size, input_length=max_sent_len))
# # lstm_model.add(LSTM(200, return_sequences=True, stateful=False))
# lstm_model.add(LSTM(500, return_sequences=False, stateful=False))
# # lstm_model.add(Dropout(.5))
# lstm_model.add(Dense(200))
# # lstm_model.add(Dropout(.25))
# lstm_model.add(Dense(100))
# # lstm_model.add(Dropout(.25))
# lstm_model.add(Dense(100))
# lstm_model.add(Dense(y_val.shape[1], activation='softmax'))
# # compile the model
# lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# # summarize the model
# print(lstm_model.summary())
# # fit the model
# loss = []
# val_loss = []
# acc = []
# val_acc = []
#
# train_minutes = 1
# start = datetime.datetime.now()
# while datetime.datetime.now() - start < datetime.timedelta(minutes=train_minutes):
#     history = lstm_model.fit(x=x_train, y=y_train, validation_data=[x_val, y_val],
#                              epochs=epochs, batch_size=batch_size, shuffle=True, verbose=verbose)
#
#     loss.append(history.history['loss'])
#     val_loss.append(history.history['val_loss'])
#     acc.append(history.history['acc'])
#     val_acc.append(history.history['val_acc'])
#
# history.history['loss'] = np.array(loss).ravel()
# history.history['val_loss'] = np.array(val_loss).ravel()
# history.history['acc'] = np.array(acc).ravel()
# history.history['val_acc'] = np.array(val_acc).ravel()
#
# print(history.history['loss'])
# print(history.history['val_loss'])
#
# plot_training_results(metric='loss', history=history, nn=lstm_model, x_test=x_test, y_test=y_test,
#                       file=data_dir + 'lstm_loss.png')
# plot_training_results(metric='acc', history=history, nn=lstm_model, x_test=x_test, y_test=y_test,
#                       file=data_dir + 'lstm_acc_loss.png')
# plot_confusion_matrix(x=x_test, y=y_test, nn=lstm_model, file=data_dir + 'lstm_confusion.png')


# define text blob
# define fully connected model
sentiment_model = Sequential()
sentiment_model.add(Embedding(vocab_size, 500, input_length=max_sent_len))
sentiment_model.add(Flatten())
sentiment_model.add(Dense(250, activation='relu'))
sentiment_model.add(Dropout(.6))
sentiment_model.add(Dense(100, activation='relu'))
sentiment_model.add(Dropout(.4))
sentiment_model.add(Dense(50, activation='relu'))
sentiment_model.add(Dense(1))
# compile the model
sentiment_model.compile(optimizer='adam', loss='mean_squared_error')
# summarize the model
print(sentiment_model.summary())
# fit the model
start = datetime.datetime.now()
history = sentiment_model.fit(x_train, y_train, epochs=20, verbose=verbose, validation_data=[x_val, y_val], batch_size=256)
print('fc train time: ' + str(datetime.datetime.now() - start))

plot_training_results(metric='loss', history=history, nn=sentiment_model, x_test=x_test, y_test=y_test,
                      file=data_dir + 'sentiment_loss.png')

sentiment_model.save(data_dir + 'sentiment_model.h5')

def plot_residuals(x, y, nn, file=None):
    y_hat = nn.predict(x=x).ravel()

    resids = y - y_hat

    rmse = np.sqrt(mean_squared_error(y, y_hat))

    random_sample = np.random.randint(0, len(y), 5000)

    y_sample = []
    resids_sample = []

    for i in range(len(y)):
        y_sample.append(y[i])
        resids_sample.append(resids[i])

    plt.figure(figsize=(8 * 1.5, 6 * 1.5))
    plt.scatter(y_sample, resids_sample, s=4, alpha=.1)
    plt.xlabel('Actual Sentiment Given by TextBlob')
    plt.ylabel('residual (actual - predicted sentiment)')
    plt.title('Residuals Plot    |    RMSE:' + str(rmse))

    if file is not None:
        plt.savefig(file, transparent=True)
    else:
        plt.show()

    plt.close()


plot_residuals(x=x_test, y=y_test, nn=sentiment_model, file=data_dir + 'sentiment_residuals.png')