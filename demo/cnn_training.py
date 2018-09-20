import pandas as pd
import twitter_parser
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

tokenizer_it = twitter_parser.Tokenizer()

class CNNTraining(object):


    def __init__(self):
        self._model = None
        self._tokenizer = None

    def training(self, config):
        DATASET_PATH = config['file']['input']['trainingtweets']
        df = pd.read_csv(DATASET_PATH)
        df = df[["choose_one", "text", "choose_one:confidence"]]
        #df = df[df['choose_one:confidence'] == 1.0]
        print("There are %d items in df" % len(df))
        df['target'] = df.choose_one.map({'Relevant': 1, 'Not Relevant': 0})
        df = df[df.target.isnull() == False]
        df = df.drop_duplicates(subset=["text"]).reset_index()
        print("Total unique tweets:%d" % len(df))
        tokenizer_it = twitter_parser.Tokenizer()
        df['tokenized'] = df["text"].apply(tokenizer_it.tweet_to_tokens)
        list_tokenized_tweets = []
        for index, row in df.iterrows():
            temp = row['tokenized']
            str = ' '.join(temp)
            list_tokenized_tweets.append(str)
        new_column = pd.Series(list_tokenized_tweets)
        df['tokenized_text'] = new_column.values
        print(df.columns)
        print("Statistic to check skew-data:")
        print("On-topic Tweets: %d" % len(df[df['target'] == 1]))
        print("Off-topic Tweets:%d" % len(df[df['target'] == 0]))
        df.drop('index', axis=1, inplace=True)
        print(df.head())
        df.info()

        x = df.tokenized_text
        y = df.target
        print(y)

        from sklearn.cross_validation import train_test_split
        SEED = 2000
        x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=0.2,
                                                                                          random_state=SEED)
        x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test,
                                                                      test_size=0.5, random_state=SEED)
        print("Train set has total %d entries" % len(x_train))
        print("Train set has total %.2f percent Relevant tweets" % (len(x_train[y == 1.]) / len(x_train)))
        print("Train set has total %.2f percent Not Relevant tweets" % (len(x_train[y == 0.]) / len(x_train)))
        print("--------------------------------------------")
        print("Validation set has total %d entries" % len(x_validation))
        print(
            "Validation set has total %.2f percent Relevant tweets" % (len(x_validation[y == 1.]) / len(x_validation)))
        print("Validation set has total %.2f percent Not Relevant tweets" % (
                    len(x_validation[y == 0.]) / len(x_validation)))
        print("--------------------------------------------")
        print("Test set has total %d entries" % len(x_test))
        print("Test set has total %.2f percent Relevant tweets" % (len(x_test[y == 1.0]) / len(x_test)))
        print("Test set has total %.2f percent Not Relevant tweets" % (len(x_test[y == 0.]) / len(x_test)))
        print("--------------------------------------------")

        from tqdm import tqdm
        tqdm.pandas(desc="progress-bar")
        import gensim
        from gensim.models.word2vec import Word2Vec
        from gensim.models.doc2vec import TaggedDocument
        import multiprocessing
        from sklearn import utils

        def labelize_tweets_ug(tweets, label):
            result = []
            prefix = label
            for i, t in zip(tweets.index, tweets):
                result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
            return result

        all_x = pd.concat([x_train, x_validation, x_test])
        all_x_w2v = labelize_tweets_ug(all_x, 'all')

        cores = multiprocessing.cpu_count()
        model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065,
                                 min_alpha=0.065)
        model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

        for epoch in range(30):
            model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v),
                                epochs=1)
            model_ug_cbow.alpha -= 0.002
            model_ug_cbow.min_alpha = model_ug_cbow.alpha

        model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065,
                               min_alpha=0.065)
        model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])

        for epoch in range(30):
            model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v),
                              epochs=1)
            model_ug_sg.alpha -= 0.002
            model_ug_sg.min_alpha = model_ug_sg.alpha

        model_ug_cbow.save('../models/CrowFlowd/w2v_model_ug_cbow.word2vec')
        model_ug_sg.save('../models/CrowFlowd/w2v_model_ug_sg.word2vec')

        from gensim.models import KeyedVectors
        model_ug_cbow = KeyedVectors.load('../models/CrowFlowd/w2v_model_ug_cbow.word2vec')
        model_ug_sg = KeyedVectors.load('../models/CrowFlowd/w2v_model_ug_sg.word2vec')
        print(len(model_ug_cbow.wv.vocab.keys()))

        embeddings_index = {}

        import numpy as np
        for w in model_ug_cbow.wv.vocab.keys():
            embeddings_index[w] = np.append(model_ug_cbow.wv[w], model_ug_sg.wv[w])
        print('Found %s word vectors.' % len(embeddings_index))

        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        import pickle

        tokenizer = Tokenizer(num_words=100000)
        tokenizer.fit_on_texts(x_train)
        sequences = tokenizer.texts_to_sequences(x_train)

        print(len(tokenizer.word_index))

        self._tokenizer = tokenizer
        with open('../models/CrowFlowd/tokenizer.pickle', 'wb') as handle:
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
        with open('../models/CrowFlowd/x_test_seq.obj', 'wb') as handle:
            pickle.dump(x_test_seq, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('../models/CrowFlowd/y_test.obj', 'wb') as handle:
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
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall

            def precision(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision

            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

        from keras.layers import Input, Dense, concatenate, Activation, Conv1D, GlobalMaxPooling1D
        from keras.models import Model

        tweet_input = Input(shape=(45,), dtype='int32')
        tweet_encoder = Embedding(100000, 200, weights=[embedding_matrix], input_length=45, trainable=True)(tweet_input)
        bigram_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(tweet_encoder)
        bigram_branch = GlobalMaxPooling1D()(bigram_branch)
        trigram_branch = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(
            tweet_encoder)
        trigram_branch = GlobalMaxPooling1D()(trigram_branch)
        fourgram_branch = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(
            tweet_encoder)
        fourgram_branch = GlobalMaxPooling1D()(fourgram_branch)
        merged = concatenate([bigram_branch, trigram_branch, fourgram_branch], axis=1)

        merged = Dense(256, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(1)(merged)
        output = Activation('sigmoid')(merged)
        model = Model(inputs=[tweet_input], outputs=[output])
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()

        from keras.callbacks import ModelCheckpoint

        import numpy as np
        from keras.callbacks import Callback
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
        class Metrics(Callback):
            def on_train_begin(self, logs={}):
                self.val_f1s = []
                self.val_recalls = []
                self.val_precisions = []

            def on_epoch_end(self, epoch, logs={}):
                val_predict = (np.asarray(self.model.predict(x_val_seq))).round()
                val_targ = y_validation
                _val_f1 = f1_score(val_targ, val_predict)
                _val_recall = recall_score(val_targ, val_predict)
                _val_precision = precision_score(val_targ, val_predict)
                self.val_f1s.append(_val_f1)
                self.val_recalls.append(_val_recall)
                self.val_precisions.append(_val_precision)
                print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
                return

        metric = Metrics()

        filepath = "../models/Crowflowd/CNN_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        model.fit(x_train_seq, y_train, batch_size=64, epochs=5, validation_data=(x_val_seq, y_validation),
                  callbacks=[metric, checkpoint])
        self._model = model
        scores = model.evaluate(x_test_seq, y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))