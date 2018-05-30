import math
import time
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics
import pickle

class NeuralNetworkBase:
    def __init__(self,
                 imageSize,
                 imageChannels,
                 classCount,
                 batchSize,
                 convLayers,
                 fcLayers,
                 learningRate,
                 epochs,
                 name,
                 load=False):

        self.image_size = imageSize
        self.image_size_flat = self.image_size * self.image_size * imageChannels
        self.image_shape = (self.image_size, self.image_size)
        self.n_channels = imageChannels
        self.n_classes = classCount
        self.batchSize = batchSize
        self.convLayers = convLayers #|Filer Size | Num Filters | Stride Size
        self.fcLayers = fcLayers
        self.learningRate = learningRate
        self.epochs = epochs
        self.name = name
        self.load = load
        self.buildNetwork()

    def new_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self,length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def flatten_layer(self,layer):
        layer_shape = layer.get_shape()
        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]
        return layer_flat, num_features

    def new_fc_layer(self,
                 input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)
        print('FC Shape:', np.shape(weights))
        layer = tf.matmul(input, weights) + biases
        layer = tf.layers.batch_normalization(layer)
        #if use_relu:
        #    layer = tf.nn.relu(layer)
        layer = tf.layers.dropout(layer, rate=0.2)

        return layer
    def new_conv_layer(self,
                       input,              # The previous layer.
                       num_input_channels, # Num. channels in prev. layer.
                       filter_size,        # Width and height of each filter.
                       num_filters,        # Number of filters.
                       stride_size,        # Stride over x- and y- channel
                       use_pooling=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        print('Conv Shape:%sx%sx%s [x] %s' % (filter_size, filter_size, num_input_channels, num_filters))

        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=num_filters)

        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        strides =[1, stride_size, stride_size, 1]

        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=strides,
                             padding='SAME')

        layer += biases

        layer = tf.layers.batch_normalization(layer)
        layer = tf.nn.relu(layer)
        # Use pooling to down-sample the image resolution?
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')


        return layer, weights

    def buildNetwork(self):
        self.xp = tf.placeholder(tf.float32, shape=[None, self.image_size_flat], name='xp')
        x_image = tf.reshape(self.xp, [-1, self.image_size, self.image_size, self.n_channels])
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y_true')
        y_true_cls = tf.argmax(self.y_true, axis=1)
        tfConvs = []
        tfWeights = []
        for i, convs in enumerate(self.convLayers):
            filterSize = convs[0]
            numFilters = convs[1]
            strides = convs[2]
            inp = x_image
            channels = self.n_channels

            if i > 0:
                inp = tfConvs[i-1]
                channels = self.convLayers[i-1][1]
            use_pooling = True if (i != 0 and i%2 == 1) else False
            c, w = self.new_conv_layer(input=inp,
                                   num_input_channels=channels,
                                   filter_size=filterSize,
                                   num_filters=numFilters,
                                   stride_size=strides,
                                   use_pooling=use_pooling)

            tfConvs.append(c)

            tfWeights.append(w)

        lastConvLayer = tfConvs[-1:][0]
        layer_flat, num_features = self.flatten_layer(lastConvLayer)


        tfFcs = []
        for i, fcNeurons in enumerate(self.fcLayers):
            inp = layer_flat
            inputSize = num_features
            useRelu = True

            if i > 0:
                inp = tfFcs[i-1]
                inputSize = self.fcLayers[i-1]
            if i == len(self.fcLayers):
                useRelu = False

            fc = self.new_fc_layer(input=inp,
                         num_inputs=inputSize,
                         num_outputs=fcNeurons,
                         use_relu=useRelu)

            tfFcs.append(fc)

        lastLayer = tfFcs[-1:][0]

        y_pred = tf.nn.softmax(lastLayer)

        self.y_pred_cls = tf.argmax(y_pred, axis=1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=lastLayer,
                                                        labels=self.y_true)
        cost = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(cost)
        correct_prediction = tf.equal(self.y_pred_cls, y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def optimize(self,X, y, val, val_label):
        saver = tf.train.Saver(max_to_keep=4)
        val_feed = {
            self.xp: val,
            self.y_true: self.oneHot(val_label.astype(np.int8))
        }

        if self.load:
            self.session = tf.Session()
            saver = tf.train.import_meta_graph("models/"+self.name+"/"+"model_cnn")
            saver.restore(self.session, saver)

        for k in range(self.epochs):
            start_time = time.time()
            for i, batch in enumerate(self.getBtches(X,y, self.batchSize)):
                x_batch, y_true_batch = batch
                y_true_batch = self.oneHot(y_true_batch.astype(int))
                # Put the batch into a dict with the proper names
                # for placeholder variables in the TensorFlow graph.
                feed_dict_train = {self.xp: x_batch,
                                   self.y_true: y_true_batch}

                # Run the optimizer using this batch of training data.
                # TensorFlow assigns the variables in feed_dict_train
                # to the placeholder variables and then runs the optimizer.
                extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.session.run([self.optimizer, extra_ops], feed_dict=feed_dict_train)

            # Print status every 100 iterations.
            if k % 1 == 0:
                # Calculate the accuracy on the training-set.
                preds = self.session.run(self.y_pred_cls, feed_dict=val_feed)
                acc = metrics.accuracy_score(val_label, preds)

                # Message for printing.
                msg = "Optimization Iteration: {0:>6}, Validation Accuracy: {1:>6.1%}"
                # Print it.
                print(msg.format(k + 1, acc))
            print('Optimize time taken:%s' % (np.round(time.time() - start_time,2)))

        saver.save(self.session, "models/"+self.name+"/"+"model_cnn")
        #pickle.dump(self, open("models/"+self.name+"self.pkl", 'w+'))

    def predict(self, X, y, **kwargs):
        # Number of images in the test-set.
        num_test = len(X)
        cls_pred = np.zeros(shape=num_test, dtype=np.int)
        i = 0
        while i < num_test:
            j = min(i + self.batchSize, num_test)

            images = X[i:j]
            labels = self.oneHot(y[i:j].astype(int))

            feed_dict = {self.xp: images,
                         self.y_true: labels}

            cls_pred[i:j] = self.session.run(self.y_pred_cls, feed_dict=feed_dict)

            i = j

        return cls_pred

    def oneHot(self, labels):
        vals = np.eye(self.n_classes)[np.array(labels).reshape(-1)]
        return vals

    def reverseOneHot(self, onehotLabels):
        vals = np.argmax(onehotLabels,axis=1)
        return np.atleast_2d(vals).T

    def getBtches(self, X, Y, batchSize):
        m = X.shape[0]
        mini_batches = []

        permutation = list(np.random.permutation(m))

        shuffled_X = X[permutation]
        shuffled_Y = Y[permutation]

        num_complete_minibatches = math.floor(m / batchSize)
        for k in range(0, int(num_complete_minibatches)):
            mini_batch_X = shuffled_X[(k * batchSize): (k + 1) * batchSize]
            mini_batch_Y = shuffled_Y[(k * batchSize): (k + 1) * batchSize]
            mini_batch = (mini_batch_X, mini_batch_Y)
            yield mini_batch

        if m % batchSize != 0:
            mini_batch_X = shuffled_X[-(m % batchSize): m]
            mini_batch_Y = shuffled_Y[-(m % batchSize): m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            yield mini_batch
