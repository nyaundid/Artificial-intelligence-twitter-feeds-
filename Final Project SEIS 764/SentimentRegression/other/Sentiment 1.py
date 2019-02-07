
# coding: utf-8

# In[45]:


import tflearn


# In[46]:


from __future__ import division, print_function, absolute_import
import pandas as pd
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


# In[47]:


tweets = pd.read_csv('documents/twcs.csv',encoding='utf-8')
print(tweets.shape)
tweets.head()


# In[48]:


first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]

QnR = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')

# Filter to only outbound replies (from companies)
QnR = QnR[QnR.inbound_y ^ True]
print(f'Data shape: {QnR.shape}')
QnR.head()


# In[49]:


#making sure the dataframe contains only the needed columns
QnR = QnR[["author_id_x","created_at_x","text_x","author_id_y","created_at_y","text_y"]]
QnR.head(5)


# In[50]:


j = QnR.loc[:,["text_x"]] 
j


# In[51]:


QnR


# In[52]:


#Lower case removal

QnR["text_x"] = QnR["text_x"].apply(lambda x: " ".join(x.lower() for x in x.split()))
QnR["text_x"].head()


# In[53]:


QnR["text_x"]


# In[54]:


QnR["text_x"] = [w.lower() for w in QnR["text_x"]]


# In[55]:


QnR["text_x"] 


# In[56]:


#removing puntuation

QnR["text_x"] = QnR["text_x"].str.replace('[^\w\s]','')
QnR["text_x"]


# In[57]:


#removing stopwords

from nltk.corpus import stopwords
stop = stopwords.words('english')
QnR["text_x"] =QnR["text_x"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
QnR["text_x"].head()


# In[58]:


stop


# In[59]:


#removing common words
freq = pd.Series(' '.join(QnR["text_x"]).split()).value_counts()[:10]
freq


# In[60]:


freq = list(freq.index)
QnR["text_x"] = QnR["text_x"].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
QnR["text_x"].head()


# In[61]:


#removing rare words

freq = pd.Series(' '.join(QnR["text_x"]).split()).value_counts()[-10:]
freq


# In[62]:


freq = list(freq.index)
QnR["text_x"] = QnR["text_x"].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
QnR["text_x"].head()


# In[63]:


#TEXTBLOP FOR SENTIMENT SCORE


# In[64]:


#spelling corrections
from textblob import TextBlob
QnR["text_x"][:5].apply(lambda x: str(TextBlob(x).correct()))


# In[65]:


#tokenization
TextBlob(QnR["text_x"][5]).words


# In[66]:


#stemming

from nltk.stem import PorterStemmer
st = PorterStemmer()
QnR["text_x"][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[67]:


QnR["text_x"][:5].apply(lambda x: TextBlob(x).sentiment)


# In[68]:


QnR["setiments"] = QnR["text_x"].apply(lambda x: TextBlob(x).sentiment[0] )
QnR[["text_x","setiments"]].head()


# In[69]:


#def sentiment_type(value):
  #  if value > 0:
   #     return "Positive"
    #elif value < 0:
    #    return "Negitive"
    #else:
     #   return "Neutral"


# In[70]:


#QnR["setiments"] = QnR["setiments"].apply(sentiment_type)


# In[71]:


#QnR["setiments"].value_counts()


# In[72]:


#QnR["setiments"].value_counts().plot(kind = "bar",color = ["blue","red","green"])


# In[73]:


QnR[["text_x","setiments"]]


# In[74]:


QnR


# In[75]:


x = QnR[["text_x","setiments"]]


# In[76]:


x


# In[77]:


import matplotlib.pyplot as plt
x['setiments'].hist()
plt.show()


# In[78]:


x['setiments'].hist()
plt.show()
positive = QnR[["text_x","setiments"]]> 0


# In[79]:


df = QnR[QnR.setiments != 0]


# In[80]:



df


# In[81]:


import matplotlib.pyplot as plt
df['setiments'].hist()
plt.show()


# In[82]:


df


# In[83]:


#def sentiment_type(value):
   # if value > 0:
   #     return "Positive"
   # elif value < 0:
    #    return "Negitive"
   # else:
    #    return "Neutral"


# In[84]:


#QnR["setiments"] = QnR["setiments"].apply(sentiment_type)


# In[85]:


#QnR["setiments"].value_counts()


# In[86]:


#QnR["setiments"].value_counts().plot(kind = "bar",color = ["blue","red","green"])
#plt.title("Sentiment types classification frequency")
#plt.xlabel("Sentiment Types")
#plt.ylabel("Frequency")
#plt.xticks(rotation = "0.5")


# In[87]:


#df["setiments"] = df["setiments"].apply(sentiment_type)


# In[88]:



#df["setiments"].value_counts()


# In[89]:


#df["setiments"].value_counts().plot(kind = "bar",color = ["blue","red","green"])
#plt.title("Sentiment types classification frequency")
#plt.xlabel("Sentiment Types")
#plt.ylabel("Frequency")
#plt.xticks(rotation = "0.5")


# In[90]:


QnR


# In[95]:


n_samples = 300
val_and_test_prop = 0.07
val_and_test_size = int(n_samples * val_and_test_prop)
n_samples += val_and_test_size * 2
twcs = tweets.sample(n=n_samples, random_state=345)


# In[96]:


from keras.models import Sequential
import ast
import datetime
from gensim.models import Word2Vec
import io
from keras import backend as K
from keras import regularizers
from keras.layers import LSTM, GRU, GRUCell, Dense, Flatten, TimeDistributed
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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import string

df = df.sample(n=n_samples, random_state=235)

x_train = df["text_x"].apply(lambda x: x.split(' '))
y_train = df["setiments"]

x_train, x_val, y_train, y_val =     train_test_split(df["text_x"],  df["setiments"],
                     test_size=val_and_test_size,
                     random_state=3135,
                     shuffle=True)

x_train, x_test, y_train, y_test =     train_test_split(x_train, y_train,
                     test_size=val_and_test_size,
                     random_state=3135,
                     shuffle=True)
start = datetime.datetime.now()
x_train.reset_index(inplace=True, drop=True)
x_val.reset_index(inplace=True, drop=True)
x_test.reset_index(inplace=True, drop=True)
print('split time: ' + str(datetime.datetime.now() - start))


# In[97]:


x_val.shape


# In[98]:


x_train.reset_index(inplace=True, drop=True)
x_val.reset_index(inplace=True, drop=True)
x_test.reset_index(inplace=True, drop=True)

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

padding='post'
def to_token(x, vocab_size, max_len, padding):
    # x = x.apply(lambda s: ' '.join(s))
    x = [one_hot(w, vocab_size) for w in x]
    return pad_sequences(x, maxlen=max_len, padding=padding)

start = datetime.datetime.now()
x_train = to_token(x=x_train, vocab_size=vocab_size, max_len=max_sent_len, padding=padding)
x_val = to_token(x=x_val, vocab_size=vocab_size, max_len=max_sent_len, padding=padding)
x_test = to_token(x=x_test, vocab_size=vocab_size, max_len=max_sent_len, padding=padding)
print('to_token time:' + str(datetime.datetime.now() - start))

def plot_training_results(metric, history, nn, x_test, y_test):
    test_loss = nn.evaluate(x=x_test, y=y_test)
    test_result = test_loss if metric == 'loss' else test_acc

    plt.figure(figsize=(8*1.5, 6*1.5))
    plt.plot(history.history[metric], label='train')
    plt.plot(history.history['val_' + metric], label='val')
    plt.legend()
    plt.title(metric.title() + ' by Epoch    |    Test ' + metric.title() + ': ' + str(round(test_result, 3)))
    plt.show()

# y = y_test
# x=x_test
# nn=model


# In[99]:


def plot_confusion_matrix(x, y, nn):
    y = np.argmax(y, axis=1)
    y_hat = nn.predict(x=x)
    y_hat = np.argmax(y_hat, axis=1)

    f1 = np.round(f1_score(y, y_hat, average='micro'), 3)

    conf_matrix = np.log(1 + confusion_matrix(y_true=y, y_pred=y_hat))
    # conf_matrix = confusion_matrix(y_true=y, y_pred=y_hat)

    plt.figure(figsize=(8 * 1.5, 6 * 1.5))
    sns.heatmap(conf_matrix, center=np.median(conf_matrix))
    plt.title('Confusion Matrix (1 + log of count)    |    F1: ' + str(f1))
    plt.show()


# In[100]:


from keras.models import Sequential
# define sequential model
batch_size = 2**7
embed_size = 300
lstm_model = Sequential()
lstm_model.add(Embedding(vocab_size, embed_size, input_length=max_sent_len))
lstm_model.add(LSTM(200, return_sequences=True, stateful=False))
lstm_model.add(LSTM(100, return_sequences=False, stateful=False))
lstm_model.add(Dense(100))
lstm_model.add(Dense(100))
lstm_model.add(Dense(50))
lstm_model.add(Dense(1))
# compile the model
lstm_model.compile(optimizer='adam', loss='mean_absolute_error')
# summarize the model
print(lstm_model.summary())
# fit the model
loss = []
val_loss = []

verbose = 1
train_minutes = 2
start = datetime.datetime.now()
while datetime.datetime.now() - start < datetime.timedelta(minutes=train_minutes):
    history = lstm_model.fit(x=x_train, y=y_train, validation_data=[x_val, y_val],
                             epochs=1, batch_size=batch_size, shuffle=True, verbose=verbose)

    loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
   

history.history['loss'] = np.array(loss).ravel()
history.history['val_loss'] = np.array(val_loss).ravel()


# In[101]:


plot_training_results(metric='loss', history=history, nn=lstm_model, 
                      x_test=x_test, y_test=y_test.values)


# In[102]:


lstm_model.evaluate(x=x_test, y=y_test)


# In[ ]:


#STOOOOOOOOOOOOOOOOPPPPPPP

