import io
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import layers,Sequential
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import numpy as np

def get_data_and_labels(path):
    with io.open(path,"r",encoding="utf-8") as f:
        x_data = []
        y_data = []
        for line in f:
            if(line.__contains__(",,,") and len(line.split(",,,")) == 2):
                line_splitted = line.split(",,,")
                x_data.append(line_splitted[0].strip())
                y_data.append(line_splitted[1].strip())
    return x_data,y_data

#create word to vector mapping from glove vectors
glove_dir = '/home/hunaif/code/hunaif/hands-on-keras/data'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))



max_words = 3000
maxlen = 15
batch_size = 32


print('Loading data...')
(input_train, y_train) = get_data_and_labels("../../data/user_intent_data.txt")
print(len(input_train), 'train sequences')

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(input_train)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

sequences = tokenizer.texts_to_sequences(input_train)

input_train = sequence.pad_sequences(sequences, maxlen=maxlen)

print('input_train shape:', input_train.shape)



#Creating the embedding_matrix of dimension (max_words, embedding_dimension)
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


encoder = LabelEncoder()
encoder.fit(y_train)

encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
one_hot_train_labels = to_categorical(encoded_Y)

model = Sequential()
model.add(layers.Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(layers.LSTM(4))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(5,activation='softmax'))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(input_train,one_hot_train_labels,epochs=50,
                    batch_size=batch_size,
                    validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()