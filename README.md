# Animal-Classification-with-Neural-Networks
The following model works on classifying animals o a given data set for 10 clusters. It works for around 0.6 accuracy. The model is done by using fine tuning, while MobileNetV2 model was taken to be the transformer. The outputs of MobileNetV2 were taken as inputs for few DNN layers with a dropout layer with a rate of 0.4. 
Initially, I tried to train NASNet, but it needed to be lighter and took a lot of time, even on 10 epochs. Thus, I terminated. Besides, I have also tried EfficientNetB0 and EfficientNetB1, but they still needed to be improved from 0.1009 accuracy three epochs on a row. Lastly, I discovered that MobileNet is a comparably suitable and light model for this classification problem; thus, I specifically took MobileNetV2 as a transformer, did fine-tuning on it, and added DNN layers with a dropout rate 0.4 to avoid overfitting. I could manage to run for 10 epochs because of source scarcity. You can see the data importing, training, and testing processes in `MobileNetV2+DNN.ipynb` . The model is saved in `MobileNetV2+DNN.h5` and later can be used by just calling it by the following code part below. 

```
#loading the model
loaded_model = tf.keras.models.load_model("MobileNetV2+DNN.h5")

```
