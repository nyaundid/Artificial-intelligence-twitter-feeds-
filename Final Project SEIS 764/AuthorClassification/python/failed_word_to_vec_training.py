# define args
run_env = 'windows'
drop_user = False
simplify_http = False
remove_stop_words = False
apply_porter_stemmer = True

if run_env == 'colaboratory':
    #!pip install gensim
    from google.colab import drive
    drive.mount('/content/drive')
    wrk_dr = '/content/drive/My Drive/Colab Notebooks/'
else:
    wrk_dr = 'C:/Users/toblon/StThomas/764 Artificial Intelligence/project/'

data_dir = wrk_dr + 'data/'

# import libraries
# from google.colab import files
import ast
import datetime
from gensim.models import Word2Vec
import io
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.optimizers import Adam
# from keras.layers import Flatten
from keras.layers import LSTM, GRU, GRUCell
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import porter
#from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import string

# load raw data
start = datetime.datetime.now()
twcs = pd.read_csv(data_dir + 'twcs.zip')[['author_id','text']]
print('upload time: ' + str(datetime.datetime.now() - start))
if run_env == 'windows': twcs = twcs[:1000]
print(twcs.head())

# label tweet as one of 109 corp accounts, or non-corp acct (92 classes) 40% of data is other class
# consolidate non-corporate accounts
twcs['author_id'] = twcs['author_id']\
  .apply(lambda x: x if x.replace('_','')\
  .replace('O2','O').isalpha() else 'non-corporate')

# avoid not having class representitives in train and validate sets
if run_env == 'windows':
    n_tweets_by_author = twcs['author_id'].value_counts()
    twcs = twcs[twcs['author_id'].isin(n_tweets_by_author.index[n_tweets_by_author >= 3])]

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

val_and_test_size = int(len(twcs) * .07)

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

# reformat y to one-hot encoding for kears cat target
y_encoder = LabelBinarizer().fit(y_val.values)
y_train = y_encoder.transform(y_train)
y_val = y_encoder.transform(y_val)
y_test = y_encoder.transform(y_test)

# train word embeddings
def make_word_embeddings(epochs, w2v_size, w2v_file, data_dir):
    try:
        w2v = Word2Vec.load(data_dir + w2v_file)
        print('loaded pre-trained model from \n' + data_dir + w2v_file)
    except FileNotFoundError:
        print('creating new model')
        w2v = Word2Vec(x_train.tolist(), w2v_size=w2v_size, window=5, min_count=50,
                       workers=8, sg=0, negative=5)

    print('training for %i epochs' % epochs)
    start = datetime.datetime.now()
    w2v.train(x_train.tolist(), total_examples=len(x_train), epochs=epochs)
    print('word2vec train time: ' + str(datetime.datetime.now() - start))

    w2v.save(data_dir + w2v_file)
    print('saved model to ' + data_dir + w2v_file)

epochs = 2
w2v_size = 500
w2v_file = 'w2v.model_' + str(w2v_size)

if run_env == 'colaboratory':
    while True:
        make_word_embeddings(epochs=epochs, w2v_size=w2v_size, w2v_file=w2v_file, data_dir=data_dir)
else:
    for i in range(1):
        make_word_embeddings(epochs=epochs, w2v_size=w2v_size, w2v_file=w2v_file, data_dir=data_dir)

# load file
w2v = Word2Vec.load(data_dir + w2v_file)

# test output
w2v['are']
len(w2v.wv.vocab)

# convert x to sequence data
obs_steps = x_train.apply(len)
max_steps = np.max(obs_steps)
print('\nmax_steps: ' + str(max_steps))


def to_time_steps(x, n_steps, w2v, w2v_size):
    """
    convert list of words to timeseries word embeddings

    :param x: list of words
    :param n_steps: int number of steps in output
    :param w2v: word to vect object containing embeddings
    :param w2v_size: embedding size
    :return: sequence for timeseries of single tweet.
    """
    x = [w for w in x if w in w2v.wv.vocab]
    if x == []: return np.zeros((n_steps, w2v_size))

    sent_len = len(x)
    pad_len = n_steps - sent_len
    padding = np.array([np.repeat(0.0, w2v_size) for p in range(pad_len)]).reshape((pad_len, w2v_size))
    steps = np.array([w2v.wv[w] for w in x])
    return np.vstack((padding, steps))

# setup timeseries data
x_train_seq = np.zeros((len(x_train), max_steps, w2v_size))
x_val_seq = np.zeros((len(x_val), max_steps, w2v_size))
x_test_seq = np.zeros((len(x_test), max_steps, w2v_size))

for i in x_train.index:
    x_train_seq[i] = to_time_steps(x_train[i], n_steps=max_steps, w2v=w2v, w2v_size=w2v_size)

for i in x_val.index:
    x_val_seq[i] = to_time_steps(x_val[i], n_steps=max_steps, w2v=w2v, w2v_size=w2v_size)

for i in x_test.index:
    x_test_seq[i] = to_time_steps(x_test[i], n_steps=max_steps, w2v=w2v, w2v_size=w2v_size)


# setup an RNN class to avoid annoying coding while exploring hyper parameters
class RnnNetwork:
    # an M:1 classification network

    def __init__(self, rnn_units=[1, 1], dense_units=[4, 4], n_classes=2,
                 lr=0.001, decay=0.0025,
                 l2_rnn_kernel=0.001, l2_rnn_activity=0.001, l2_fc=0.001,
                 clipnorm=1.0, epochs=3, batch_size=2 ** 10,
                 stateful=False, shuffle=True,
                 verbose=1):

        # hyper parameters
        self.lr = lr
        self.decay = decay
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.dense_units = dense_units
        self.n_classes = n_classes
        self.l2_rnn_kernel = l2_rnn_kernel
        self.l2_rnn_activity = l2_rnn_activity
        self.l2_fc = l2_fc
        self.clipnorm = clipnorm
        self.epochs = epochs
        self.batch_size = batch_size
        self.stateful = stateful
        self.shuffle = shuffle

        # placeholder for model
        self.model = None
        self.verbose = verbose

        # place to store model history output
        self.History = None
        self.loss = None
        self.val_loss = None

    def build_model(self, x_train):
        """
        Build the LSTM regression model

        :param x_train: sequance data set of shape (n, T, F)
        :return: none, sets self.model
        """
        # setup params
        n_steps = x_train.shape[1]

        n_feats = x_train.shape[2]

        # define model
        model = Sequential()

        # setup regularizers
        l2_rnn_kernel = regularizers.l2(self.l2_rnn_kernel)
        l2_rnn_activity = regularizers.l2(self.l2_rnn_activity)
        l2_fc = regularizers.l2(self.l2_fc)

        # return the activation values at each timestep as a sequence
        for i in range(len(self.rnn_units)):
            if i == len(self.rnn_units):
                model.add(LSTM(self.rnn_units[i],
                               activation='tanh',
                               input_shape=(None, n_feats),
                               batch_input_shape=(self.batch_size, n_steps, n_feats),
                               return_sequences=False,
                               stateful=self.stateful,
                               kernel_regularizer=l2_rnn_kernel,
                               activity_regularizer=l2_rnn_activity))
            else:
                model.add(LSTM(self.rnn_units[i],
                               activation='tanh',
                               input_shape=(None, n_feats),
                               batch_input_shape=(self.batch_size, n_steps, n_feats),
                               return_sequences=True,
                               stateful=self.stateful,
                               kernel_regularizer=l2_rnn_kernel,
                               activity_regularizer=l2_rnn_activity))

        if self.dense_units is not None:
            # apply a single set of dense layers to each timestep to produce a sequence
            for i in range(len(self.dense_units)):
                model.add(Dense(self.dense_units[i], activation='relu', kernel_regularizer=l2_fc))

            # add output layer
            model.add(Dense(self.n_classes, activation='softmax'))
        else:
            model.add(Dense(self.n_classes, activation='softmax'))

        adam_optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999,
                              epsilon=None, decay=self.decay, amsgrad=False,
                              clipnorm=self.clipnorm)

        model.compile(loss='mean_squared_error', optimizer=adam_optimizer)

        self.model = model

    def fit_model(self, x_train, y_train, x_val, y_val):
        """
        train the model

        :param x_train: numpy array (n, t, f)
        :param y_train: numpy array (n, t)
        :param x_val: numpy array (n, t, f)
        :param y_val: numpy array (n, t)
        :return: none, fits self.model
        """
        start_time = datetime.datetime.now()

        if self.stateful:
            self.loss = []
            self.val_loss = []

            for i in range(self.epochs):
                self.History = self.model.fit(x=x_train, y=y_train,
                                              validation_data=[x_val, y_val],
                                              epochs=1,
                                              batch_size=self.batch_size,
                                              verbose=self.verbose,
                                              shuffle=self.shuffle)

                self.loss.append(self.History.history['loss'])
                self.val_loss.append(self.History.history['val_loss'])

                self.model.reset_states()

            self.loss = np.array(self.loss).ravel()
            self.val_loss = np.array(self.val_loss).ravel()
        else:
            self.History = self.model.fit(x=x_train, y=y_train,
                                          validation_data=[x_val, y_val],
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          verbose=self.verbose,
                                          shuffle=self.shuffle)

            self.loss = self.History.history['loss']
            self.val_loss = self.History.history['val_loss']

        print('train time: ' +
              str((datetime.datetime.now() - start_time)))

    def reset_weights(self):
        """
        reset the weights to try training new models from scratch

        :return: none, re-initiallizes weights in self.model
        """
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

# test
rnn = RnnNetwork(rnn_units=[1, 1], dense_units=[4, 4], n_classes=y_val.shape[1],
                 lr=0.001, decay=0.0025,
                 l2_rnn_kernel=0.001, l2_rnn_activity=0.001, l2_fc=0.001,
                 clipnorm=1.0, epochs=3, batch_size=2 ** 10,
                 stateful=True, shuffle=True,
                 verbose=1)
rnn.build_model(x_train=x_train_seq)
rnn.fit_model(x_train=x_train_seq, y_train=y_train, x_val=x_val_seq, y_val=y_val)

print(rnn.model.summary())