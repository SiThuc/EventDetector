import pandas as pd
import numpy as np
import twitter_parser
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import pickle
import glob
from sklearn.utils import shuffle

tokenizer_it = twitter_parser.Tokenizer()

DATASET_PATH = "dataset"
allFiles = glob.glob(DATASET_PATH+"/*.csv")
df = pd.DataFrame()
list_ = []
for file_ in allFiles:
    sub_df = pd.read_csv(file_, index_col=None, header=0)
    list_.append(sub_df)
df = pd.concat(list_)
df = df[['tweet','label']]
print("There are %d items in df"%len(df))
df = df.drop_duplicates(subset=["tweet"]).reset_index()
print("Total unique tweets:%d"%len(df))
df['tokenized'] = df["tweet"].apply(tokenizer_it.tweet_to_tokens)
list_tokenized_tweets = []
for index, row in df.iterrows():
    temp = row['tokenized']
    str = ' '.join(temp)
    list_tokenized_tweets.append(str)

new_column = pd.Series(list_tokenized_tweets)
df['tokenized_text'] = new_column.values
print(df.columns)
print("Statistic to check skew-data:")
print("On-topic Tweets: %d"%len(df[df['label']=='on-topic']))
print("Off-topic Tweets:%d"%len(df[df['label']=='off-topic']))
df['target'] = df.label.map({'off-topic':0., 'on-topic':1.})
df.drop('index', axis=1, inplace=True)
with open('model/T6_Dataset/nrm_df_full.obj', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(df.head())


df.info()

x = df.tokenized_text
y = df.target

from sklearn.cross_validation import train_test_split
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x,y, test_size=0.2, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=0.5, random_state=SEED)
print("Train set has total %d entries"%len(x_train))
print("Train set has total %.2f percent Relevant tweets"%(len(x_train[y==1.])/len(x_train)))
print("Train set has total %.2f percent Not Relevant tweets"%(len(x_train[y==0.])/len(x_train)))
print("--------------------------------------------")
print("Validation set has total %d entries"%len(x_validation))
print("Validation set has total %.2f percent Relevant tweets"%(len(x_validation[y==1.])/len(x_validation)))
print("Validation set has total %.2f percent Not Relevant tweets"%(len(x_validation[y==0.])/len(x_validation)))
print("--------------------------------------------")
print("Test set has total %d entries"%len(x_test))
print("Test set has total %.2f percent Relevant tweets"%(len(x_test[y==1.0])/len(x_test)))
print("Test set has total %.2f percent Not Relevant tweets"%(len(x_test[y==0.])/len(x_test)))
print("--------------------------------------------")

from tqdm import tqdm
tqdm.pandas(desc = "progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

def labelize_tweets_ug(tweets, label):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' %i]))
    return result

all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v = labelize_tweets_ug(all_x, 'all')

cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha

model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])

for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha

model_ug_cbow.save('model/T6_Dataset/w2v_model_ug_cbow.word2vec')
model_ug_sg.save('model/T6_Dataset/w2v_model_ug_sg.word2vec')


from gensim.models import KeyedVectors
model_ug_cbow = KeyedVectors.load('model/T6_Dataset/w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('model/T6_Dataset/w2v_model_ug_sg.word2vec')
print(len(model_ug_cbow.wv.vocab.keys()))

embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
print('Found %s word vectors.' % len(embeddings_index))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

print(len(tokenizer.word_index))

with open('model/T6_Dataset/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


length = []
for x in x_train:
    length.append(len(x.split()))
print(max(length))

x_train_seq = pad_sequences(sequences, maxlen=45)
print('Shape of data tensor:', x_train_seq.shape)

sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=45)

sequences_test = tokenizer.texts_to_sequences(x_test)
x_test_seq = pad_sequences(sequences_test, maxlen=45)
with open('model/T6_Dataset/x_test_seq.obj', 'wb') as handle:
    pickle.dump(x_test_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model/T6_Dataset/y_test.obj', 'wb') as handle:
    pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

num_words = 100000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


seed = 7
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


embed_dim = 200
lstm_out = 196
batch_size = 32
from keras.layers import LSTM
model = Sequential()
model.add(Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True, dropout = 0.2))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_fnr = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(x_val_seq))).round()
        val_targ = y_validation

        cnf_matrix = confusion_matrix(val_targ, val_predict)
        TP = cnf_matrix[1][1]
        FP = cnf_matrix[0][1]
        FN = cnf_matrix[1][0]
        TN = cnf_matrix[0][0]

        # Fall out or false positive rate
        FNR = FN/(FN + TP)

        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_fnr.append(FNR)
        print(" — val_f1: % f — val_precision: % f — val_recall % f -- val_fnr %f" %(_val_f1, _val_precision, _val_recall, FNR))
        return

metric = Metrics()

from keras.callbacks import ModelCheckpoint
filepath="model/T6_Dataset/lstm_T6_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(x_train_seq, y_train, batch_size=64, epochs=5, validation_data=(x_val_seq, y_validation), callbacks = [checkpoint, metric])
