from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import concatenate, Concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, GlobalMaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def getTrainTestData():
    # 訓練數據保存在 "data.csv" 文件中
    train_df = pd.read_csv("traindata.csv")
    train_df = train_df.dropna()
    test_df = pd.read_csv('testdata.csv')
    test_df = test_df.dropna()

    # 取出 "sentence" 和 "label" 列
    # train_sentences = train_df["enDescription"].values
    train_labels = train_df["label"].values
    # test_sentences = test_df["enDescription"].values
    test_labels = test_df["label"].values

    # 將字符串類型的標籤轉換為數值類型 (例如將 "true" 轉換為 1，將 "false" 轉換為 0)
    train_labels = np.where(train_labels >= 0.5, 1, 0)
    test_labels = np.where(test_labels >= 0.5, 1, 0)

    # 創建一個詞彙表，並使用詞彙表將句子轉換為數字序列
    vocab_size = 2000  # 詞彙表大小
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(
        " ".join([str(i) for i in train_df["enDescription"]]))
    train_sentences = train_df["enDescription"].transform(
        lambda x: tokenizer.texts_to_sequences([str(x)])[0])

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(
        " ".join([str(i) for i in test_df["enDescription"]]))
    test_sentences = test_df["enDescription"].transform(
        lambda x: tokenizer.texts_to_sequences([str(x)])[0])

    # 將數字序列轉換為等長的張量
    max_length = 100  # 句子的最大長度
    padded_sequences = pad_sequences(train_sentences, maxlen=max_length)
    train_sequences = padded_sequences
    padded_sequences = pad_sequences(test_sentences, maxlen=max_length)
    test_sequences = padded_sequences

    train_sequences, val_sequences, train_labels, val_labelss = train_test_split(
        train_sequences, train_labels, test_size=0.2, random_state=0)

    return train_sequences, train_labels, val_sequences, val_labelss, test_sequences, test_labels


def getModel():
    inputs = keras.Input(shape=(100))
    embed = keras.layers.Embedding(10000, 128)(inputs)
    embed = keras.layers.Reshape((100, 128, 1))(embed)
    t1 = Conv2D(128, 3, activation=tfa.activations.mish)(embed)
    t1 = GlobalMaxPooling2D()(t1)
    t2 = Conv2D(128, 4, activation=tfa.activations.mish)(embed)
    t2 = GlobalMaxPooling2D()(t2)
    t3 = Conv2D(128, 5, activation=tfa.activations.mish)(embed)
    t3 = GlobalMaxPooling2D()(t3)
    combineCnn = Concatenate(axis=-1)([t1, t2, t3])
    flatCnn = keras.layers.Flatten(data_format='channels_last')(combineCnn)
    dense1 = Dense(128, activation=tfa.activations.mish)(flatCnn)
    dropout = keras.layers.Dropout(0.2)(dense1)
    outputs = Dense(2, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(1e-3),
                  metrics=["accuracy"])

    return model


def plot_loss_accuracy(history):
    hs_dic1 = {}
    hs_dic2 = {}
    hs_dic1['loss'] = history.history['loss']
    hs_dic1['accuracy'] = history.history['accuracy']
    hs_dic2['val_loss'] = history.history['val_loss']
    hs_dic2['val_accuracy'] = history.history['val_accuracy']

    historydf1 = pd.DataFrame(hs_dic1, index=history.epoch)
    historydf2 = pd.DataFrame(hs_dic2, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf1.plot(ylim=(0, max(1, historydf1.values.max())))
    loss = hs_dic1['loss'][-1]
    acc = hs_dic1['accuracy'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    plt.savefig('./figure1.jpg')
    plt.show()

    plt.figure(figsize=(8, 6))
    historydf1.plot(ylim=(0, max(1, historydf1.values.max())))
    loss = hs_dic2['val_loss'][-1]
    acc = hs_dic2['val_accuracy'][-1]
    plt.title('Val_Loss: %.3f, Val_Accuracy: %.3f' % (loss, acc))
    plt.savefig('./figure2.jpg')
    plt.show()


if __name__ == '__main__':
    tf.config.list_physical_devices('GPU')
    train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = getTrainTestData()
    model = getModel()

    # filepath = "weights.best.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
    #                              mode='max')
    # callbacks_list = [checkpoint]

    train_history = model.fit(train_sequences, train_labels, batch_size=32,
                              epochs=20, validation_data=(val_sequences, val_labels))
    plot_loss_accuracy(train_history)

    # with open('train_history.pickle', 'wb') as f:
    #     pickle.dump(train_history, f)

    loss, accuracy = model.evaluate(test_sequences, test_labels)
    print("Loss:", loss)
    print("Accuracy:", accuracy)
