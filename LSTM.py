import pandas as pd
import numpy as np
import nltk
import gensim.downloader
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

def preprocess_text(text):
    # Tokenize, remove stopwords, and lemmatize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalnum()]
    return ' '.join(filtered_tokens)

def compile_model(embedding_matrix, word_index, mlb, layers=256, dropout=0.2, lr=0.001):# Define the LSTM model
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=300, trainable=False))
    model.add(LSTM(layers, dropout=dropout))
    model.add(Dense(len(mlb.classes_), activation='sigmoid'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    checkpoint_path = "models/LSTM_checkpoint.h5"
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)

    history = model.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(X_val, y_val), callbacks=[model_checkpoint, early_stopping])
    model.load_weights(checkpoint_path)

def hyperparameters(embedding_matrix, word_index, mlb, X, Y, k=3):
    dropouts = [0.3, 0.4]
    hidden = [32, 64, 128, 256, 512]
    learning_rates = [0.00005, 0.0001, 0.0005, 0.001]
    high_score = 0
    best_params = {}

    for count, d in enumerate(dropouts):
        print()
        print('d:', count)
        print('Current best:', high_score)
        print()
        for h in hidden:
            for lr in learning_rates:
                print(d, h, lr)
                best_val = 0
                kf = KFold(n_splits=k, shuffle=True, random_state=42)
                for train_index, test_index in kf.split(X):
                    X_train, X_val = X[train_index], X[test_index]
                    y_train, y_val = Y[train_index], Y[test_index]

                    checkpoint_path = "models/LSTM_checkpoint.h5"
                    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max', verbose=0)
                    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)

                    model = compile_model(embedding_matrix, word_index, mlb, h, d, lr)
                    history = model.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(X_val, y_val), callbacks=[model_checkpoint, early_stopping], verbose=0)
                    best_val += max(history.history['val_accuracy'])
                best_val /= k
                if best_val > high_score:
                    print("High Score of", best_val, "updated with", d, h, lr)
                    high_score = best_val
                    best_params['dropout'] = d
                    best_params['hidden layers'] = h
                    best_params['learning rate'] = lr
    print('Accuracy:', high_score)
    print(best_params)

def main():
    # Load the dataset
    data = pd.read_csv('data/train.csv')

    # Preprocess the text
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    data = data[['movie_name', 'synopsis', 'genre']]
    data['synopsis'] = data['synopsis'].apply(preprocess_text)

    # Preprocess the genres
    data['genre'] = data['genre'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(data['genre'])



    # Load the pre-trained GloVe embeddings
    embed = gensim.downloader.load("glove-wiki-gigaword-100")
    # for line in embed:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs

    # Tokenize and pad the text sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['synopsis'])
    sequences = tokenizer.texts_to_sequences(data['synopsis'])
    word_index = tokenizer.word_index
    padded_sequences = pad_sequences(sequences, maxlen=300)

    # Create the embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        if word in embed:
            embedding_vector = embed[word]
            embedding_matrix[i] = embedding_vector

    X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, genres_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    hyperparameters(embedding_matrix, word_index, mlb, X_train, y_train)

if __name__ == "__main__":
    main()

#contenders: 
# 0.2, 64, 0.0005
# 0.2 256 0.001