import tensorflow as tf
model = tf.keras.models.load_model('./pspunet_weight.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
