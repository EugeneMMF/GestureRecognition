import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(24, 2)),
    tf.keras.layers.SimpleRNN(units=10, return_sequences=True),
    tf.keras.layers.SimpleRNN(units=10, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())