import numpy
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense


#####%& load dataset with only 5000 words
words = 5000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=words)
##len = 500
x_train = sequence.pad_sequences(x_train, maxlen=500)
x_test = sequence.pad_sequences(x_test, maxlen=500)
#%^^$##model
model = Sequential()
model.add(Embedding(words, 32, input_length=500))
model.add(Flatten())

model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
##print(model.summary())
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=2)
# evaluate
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))
