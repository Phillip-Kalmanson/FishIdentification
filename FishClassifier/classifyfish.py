
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model')


test_path = r"D:\CP\Projects\FishApp\FishDatabase\Test Fish\8e57d20a6525398ce5e1fb3f5326fb3d.png" 

img_height = 180
img_width = 180

img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
t = 0

for img in img_array:
    i = 0
    print("Prediction for \"%s\": " % (text))
    for label in labels_index:
        print("\t%s ==> %f" % (label, predictions[t][i]))
        i = i + 1
    t = t + 1