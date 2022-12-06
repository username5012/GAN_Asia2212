import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784, ))
encoded = Dense(32, activation = 'relu')
encoded = encoded(input_img)   # Dense layer를 거쳐서 나온 input_img
decoded = Dense(784, activation = 'sigmoid')   # sigmoid를 쓰는 이유 : 정교화 작업을 통해 0,1 값만 가지기 때문에
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()
# encoder : 데이터 -> 코드
# decoder : 코드 -> 데이터
# encoder와 decoder는 학습된 것들만 처리 가능. => 사용 실효성이 떨어짐

encoder = Model(input_img, encoded)
encoder.summary()

encoder_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

flatted_x_train = x_train.reshape(-1,784)
flatted_x_test = x_test.reshape(-1,784)

fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50,
            batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    # plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()



