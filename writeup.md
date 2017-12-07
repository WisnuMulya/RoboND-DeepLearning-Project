# Projet: Follow Me
[//]: # (Image References)

[nn-diagram]: ./docs/misc/nn-diagram.jpg
[training-curve]: ./docs/misc/training-curve.png
[hero]: ./docs/misc/hero.png
[hero-far]: ./docs/misc/hero-far.png
[other-people]: ./docs/misc/other-people.png
[evaluation]: ./docs/misc/evaluation.png

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
## Writeup

### 1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner. The document can be submitted either in either Markdown or a PDF format.

You're reading it1

### 2. The write-up conveys the an understanding of the network architecture.

![Deep Neural Network Diagram][nn-diagram]

The overall neural network that I built consists of four encoding blocks followed by a 1x1 convolution and then followed by four decoding blocks that would output the segmented image. The following is the code to implement the above neural network:

```python
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer

def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([upsampled_layer, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    layer_1 = separable_conv2d_batchnorm(concat_layer, filters)
    layer_2 = separable_conv2d_batchnorm(layer_1, filters)
    
    return layer_2

def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    conv_1 = encoder_block(inputs, 32, 2)
    conv_2 = encoder_block(conv_1, 64, 2)
    conv_3 = encoder_block(conv_2, 128, 2)
    conv_4 = encoder_block(conv_3, 256, 2)

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_5 = conv2d_batchnorm(conv_4, 512, kernel_size=1, strides=1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    upscaled_1 = decoder_block(conv_5, conv_3, 256)
    upscaled_2 = decoder_block(upscaled_1, conv_2, 128)
    upscaled_3 = decoder_block(upscaled_2, conv_1, 64)
    upscaled_4 = decoder_block(upscaled_3, inputs, 32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(upscaled_4)
```

Each of the encoding block consists of one separable convolution and then followed by a batch normalization. The separable convolution in each encoding block uses 3x3 kernel size and strides 2, so that it would output image with half the size of the input. Separable convolution is used preferebly to regular convolution due to its benefits of being more efficient by having less parameters and thus, avoiding overfitting. Further, following the separable convolution is a batch normalization that would normalize the output of separable convolution, so that the next layer would receive a normalized input. This is preferable than to normalize the initial input, since it would provide a faster network train and a bit of regularization.

Next, the output of the encoder part of the network is feeding the 1x1 convolution layer. The output of 1x1 convolution is then feeding the decoder part of the network. Each decoding block consists of one bilinear upsampling layer folowed by layer concatenation and then followed by two layers of separable convolution. The bilinear upsample layer takes the factor of 2 for each row and column, so that it would output an image with double the size of the input. This bilinear upsample layer is not a learnable layer, so that it would speed up the training process and instead, use the two separable convolution layers at the end of the decoding block for the learning part. Following the biliear upsample is a concatenation layer of which it is a method of skip connection with image in the encoder part of the network by the means of concatenation instead of pointwise addition. This concatenation layer would provide higher accuracy in the output segmented image. Then, following the concatenation layer are two separable convolutions that would be the learnable part of the decoding block.

### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.

The parameters I chose for the deep learning are the followings:

```python
learning_rate = 1e-3
batch_size = 21
num_epochs = 20
steps_per_epoch = 200
validation_steps = 50
workers = 2
```

I left the `steps_per_epoch`, `validation_steps`, and `workers` to be at default. The `batch_size` is set to be `21` due to the calculation of `(number of train image: 4132) / (steps_per_epoch: 200)` and this results in a faster training performance than larger `batch_size` like `32` or `64`.

Since I was targeting the loss rate to be less than `0.0200`, the learning rate is suitable to me would be `1e-3` or equal to `0.001` since larger number would have a higher error saturation and smaller number would result in a longer training time. With this learning rate, I was able to train my neural network in a considerably faster time than `1e-4` and at lower epochs, while also have a more accurate model than larger learning rate like `1e-2` with the same training time, since larger learning rate saturates at a higher loss rate.

Here's the training curve of my neural network:
![Neural network training curve][training-curve]

### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

The technique that hasn't been discussed above is regarding the use of 1x1 convolution layer between the encoder part and the decoder part of the neural network. The 1x1 convolution layer is used instead of a fully connected layer to apply a fully convolutional network for the task of segmenting the classes in the image to answer a question like a class lies in the image. 1x1 convolution works, since it preserves the spatial information throughout the network, while a fully connected layer flattens the information, so that the spatial information is lost.

Fully connected layer could be applied in a regular convolutional network and for the task of simply classifying an image.

### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

The image manipulation conducted by the neural network is to produce a segmented image on which there are three types of classifications: hero, other people, and background. The type of neural network for this job is the fully convolutional networks, since it preserves the spatial information throughout the network and also output the image with the same size as the output. On the contrary, regular convolutional network only specify what class is in the image without producing an image showing where are certain classes located in the image.

Fully convolutional network consists of two parts: encoding and decoding. The encoding part works the same like the regular convolutional network, while the decoding part works to upscale the image to return the same size image as the input. Both of the part is connected with a 1x1 convolutional layer, rather than a fully connected layer, since it preserves spatial information.

### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

With the given data, the neural network is only good to follow the hero class and when the hero is in a close proximity. This is justified with the evaluation score on which the IoU of hero in close proximity is 0.9124, while the IoU of hero far away is 0.2255. The IoU of other people is also poor, for which the highest is 0.7503.

For objects other than humans, the current network will not be able to follow them, since it's lack in classifiers of them and the data is not supportive enough to include a better representation of other objects like cars or animals.

## Model

### 1. The model is submitted in the correct format.

Please check out the `data/weights` folder for `.h5` model files.

### 2. The neural network must achieve a minimum level of accuracy for the network implemented.

I have managed to achieve the final score of greater than 40%:

![Evaluation image][evaluation]

## Future Enhancements

For future enhancements, larger epochs would be helpful to produce a more accurate segmentation. Also, a potential layer to be applied would be an inception layer that is discussed in the class, of which it's suggested that it would provide a better result.
