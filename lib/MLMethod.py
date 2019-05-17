import tensorflow as tf
from lib.kneedata import *
import time
from datetime import timedelta
import scipy

#tasks
#test without load_prev_model flag
#test tensor objects run session and see variables are set
#problem running optimizations

#changed deviation to qayyum
class ConvolutionMethod:
    def __init__(self,load_prev_model=True):
#        self.knee_data = data_obj
#        self.input_size = input_size
        if load_prev_model:
            self.session = tf.Session()
            saver = tf.train.import_meta_graph("../models/model.meta")
            saver.restore(self.session,tf.train.latest_checkpoint('./'))
            graph = tf.get_default_graph()
            self.x = graph.get_tensor_by_name("x:0")
            self.y_true = graph.get_tensor_by_name("y_true:0")           
            self.layer_fc3 = graph.get_tensor_by_name("layer_fc3:0")
            self.y_pred = graph.get_tensor_by_name("y_pred:0")           
            self.accuracy = graph.get_tensor_by_name("accuracy:0")
            self.optimizer = graph.get_operation_by_name("optimizer")
            self.init_numbers()

        else:
            self.initialize()
            
    def init_numbers(self):
        self.filter_size1 = 11
        self.num_filters1 = 64
        
        self.filter_size2 = 5
        self.num_filters2 = 192
        
        self.filter_size3 = 5
        self.num_filters3 = 384
        
        self.filter_size4 = 3
        self.num_filters4 = 256
        
        self.filter_size5 = 3
        #Change final filter to 128 because fully connected layer otherwise got too many variables
        self.num_filters5 = 128#256       
        
        
        self.fc_size = 4096
        self.final_feature_size = 9
        
        self.stride_length = 1
        
        self.num_channels = 1
        
        self.num_classes = 2
        self.img_size = 200
        
    def initialize(self):
        tf.reset_default_graph()
        self.init_numbers()
        
        self.makeTensor()
        self.initializeSession()
        

        
        
#        self.optimize(int(self.input_size/5))
    
    
    def new_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01))
    
    
    def new_biases(self,length):
        return tf.Variable(tf.constant(0.05, shape=[length]))
    
    def new_conv_layer(self,input,              # The previous layer.
                       num_input_channels, # Num. channels in prev. layer.
                       filter_size,        # Width and height of each filter.
                       num_filters,        # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.
    
        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]
    
        # Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape)
    
        # Create new biases, one for each filter.
        biases = self.new_biases(length=num_filters)
    
        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, self.stride_length, self.stride_length, 1],
                             padding='SAME')
    
        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases
    
        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')
    
        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)
    
        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.
    
        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights
    
    def flatten_layer(self,layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()
    
        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]
    
        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()
        
        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])
    
        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]
    
        # Return both the flattened layer and the number of features.
        return layer_flat, num_features
    
    
    def new_fc_layer(self,input,          # The previous layer.
                     num_inputs,     # Num. inputs from prev. layer.
                     num_outputs,    # Num. outputs.
                     use_relu=True): # Use Rectified Linear Unit (ReLU)?
    
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)
    
        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
    
        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)
    
        return layer
    
    def makeTensor(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size,self.img_size], name='x')
        
        x_image = tf.reshape(self.x, [-1, self.img_size, self.img_size, self.num_channels])
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        y_true_cls = tf.argmax(self.y_true, axis=1)     
        
        layer_conv1, weights_conv1 = \
            self.new_conv_layer(input=x_image,
                          num_input_channels=self.num_channels,
                          filter_size=self.filter_size1,
                          num_filters=self.num_filters1,
                          use_pooling=False)       
        
        layer_conv2, weights_conv2 = \
            self.new_conv_layer(input=layer_conv1,
                   num_input_channels=self.num_filters1,
                   filter_size=self.filter_size2,
                   num_filters=self.num_filters2,
                   use_pooling=True)
        
        layer_conv3, weights_conv3 = \
            self.new_conv_layer(input=layer_conv2,
                   num_input_channels=self.num_filters2,
                   filter_size=self.filter_size3,
                   num_filters=self.num_filters3,
                   use_pooling=True)
        
        layer_conv4, weights_conv4 = \
            self.new_conv_layer(input=layer_conv3,
                   num_input_channels=self.num_filters3,
                   filter_size=self.filter_size4,
                   num_filters=self.num_filters4,
                   use_pooling=False)       
        
        layer_conv5, weights_conv5 = \
            self.new_conv_layer(input=layer_conv4,
                   num_input_channels=self.num_filters4,
                   filter_size=self.filter_size5,
                   num_filters=self.num_filters5,
                   use_pooling=True)       
        
        layer_flat, num_features = self.flatten_layer(layer_conv5)

        layer_fc1 = self.new_fc_layer(input=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=self.fc_size,
                                 use_relu=True)
        
        layer_fc2 = self.new_fc_layer(input=layer_fc1,
                                 num_inputs=self.fc_size,
                                 num_outputs=self.fc_size,
                                 use_relu=True)
        
        #reduced output from 4096 to 9
        self.layer_fc3 = self.new_fc_layer(input=layer_fc2,
                                 num_inputs=self.fc_size,
                                 num_outputs=self.final_feature_size,
                                 use_relu=True)
        
        self.layer_fc3 = tf.identity(self.layer_fc3,name="layer_fc3")
        
        layer_soft_max = tf.nn.log_softmax(self.layer_fc3)
        
        self.y_pred = self.new_fc_layer(input=layer_soft_max,
                                 num_inputs=self.final_feature_size,
                                 num_outputs=self.num_classes,
                                 use_relu=True)
        self.y_pred = tf.identity(self.y_pred,name="y_pred")
        self.y_pred_cls = tf.argmax(self.y_pred, axis=1)       
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred,
                                                        labels=self.y_true)
        
        cost = tf.reduce_mean(cross_entropy)       
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost,name="optimizer")       
#        self.optimizer = tf.identity(self.optimizer,name="optimizer")       
        
        correct_prediction = tf.equal(self.y_pred_cls, y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
#        self.accuracy= tf.identity(self.accuracy,name="accuracy")              
        
    def initializeSession(self):
        self.session = tf.Session()
#        run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
       
        self.session.run(tf.global_variables_initializer())#,options=run_options)
    
    def runSession(self,tf_obj,feed_dict):
        return self.session.run(tf_obj,feed_dict=feed_dict)
        
        
    def optimize(self,data,num_iterations=None):#,data,num_iterations=None):
        # Ensure we update the global variable rather than a local copy.
#        global total_iterations
    
        # Start-time used for printing time-usage below.
        if num_iterations==None:
            num_iterations = int(len(data)/5)
        start_time = time.time()
    
#        for i in range(total_iterations,
#                       total_iterations + num_iterations):
        for i in range(num_iterations):
            print("enter optimization loop")
    
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch = data.getTrainingData(5)           
#            x_batch, y_true_batch = self.knee_data.getTrainingData(5)
    
            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {self.x: x_batch,
                               self.y_true: y_true_batch}
    
            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            self.session.run(self.optimizer, feed_dict=feed_dict_train)
    
            # Print status every 100 iterations.
    #        if i % 100 == 0:
                # Calculate the accuracy on the training-set.
            acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
    
                # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
    
                # Print it.
            print(msg.format(i + 1, acc))
    
        # Update the total number of iterations performed.
    
        # Ending time.
        end_time = time.time()
    
        # Difference between start and end-times.
        time_dif = end_time - start_time
    
        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))   
        saver = tf.train.Saver()
        saver.save(self.session, "../models/model")



    def extractFeatures(self,img):      
        img = np.reshape(img,[-1,self.img_size,self.img_size])       
        feed_dict = {self.x:img}
        return self.session.run(self.layer_fc3,feed_dict=feed_dict)
    
    def distance(self,vec1,vec2):
        return scipy.spatial.distance.euclidean(vec1,vec2)
    
    def testY(self):
        self.makeTensor()
        knee_data = KneeData()
        knee_data.loadData(from_original_files=False)
        knee_data.prepareTrainingData()       
        x_batch, y_true_batch = knee_data.getTrainingData(5)
    
        feed_dict_x= {self.x: x_batch,
                               self.y_true: y_true_batch}
        return self.session.run(self.y_pred,feed_dict=feed_dict_x)
    
    def computeAUC(self,x,y):
#        self.makeTensor()
#        knee_data = KneeData()
#        knee_data.loadData(from_original_files=False)
#        knee_data.prepareTrainingData()
#               
#        x_batch, y_batch = knee_data.getTrainingData(10)
    
        length = len(x)
        index = 0
        auc = None
        
        while index + 10 < length:
            x_batch = x[index:index+10]
            y_batch = tf.convert_to_tensor(y[index:index+10])
            
            feed_dict_x = {self.x: x_batch}
            predictions = self.session.run(self.y_pred,feed_dict=feed_dict_x)
            auc,auc_update = tf.metrics.auc(y_batch,predictions)
            
            self.session.run(tf.local_variables_initializer())
            self.session.run(auc_update)
            index+=10
            
        #get final data
        if index < length-1:
            x_batch = x[index:]
            y_batch = tf.convert_to_tensor(y[index:])
            
            feed_dict_x = {self.x: x_batch}
            predictions = self.session.run(self.y_pred,feed_dict=feed_dict_x)
            auc,auc_update = tf.metrics.auc(y_batch,predictions)
            self.session.run(tf.local_variables_initializer())           
            self.session.run(auc_update) 
                 
        return self.session.run(auc)       
        
    
    
    def extractFromBatch(self,batch,pickled_db_path="MLfeatures.pck"):
        self.matrix = []
        for i in batch:
            self.matrix.append(self.extractFeatures(i))
#            print(i.shape)
        with open(pickled_db_path,'wb') as fp:   
            pickle.dump(self.matrix,fp)      
    
    def loadFeatures(self,pickled_db_path="MLfeatures.pck"):
        with open(pickled_db_path,'rb') as fp:
            self.matrix = pickle.load(fp)

            
    def query(self,query_img,k):
        if self.matrix is []:
            print("empty feature matrix")
            return []
        query_img_features = self.extractFeatures(query_img)
        distances_and_indices = []
        for index,database_img_features in enumerate(self.matrix):
            distance = self.distance(query_img_features,database_img_features)
            distances_and_indices.append((distance,index))
        distances_and_indices.sort(key=lambda pair:pair[0])
        relevant_dist_and_indices = distances_and_indices[:k]
        retval_indices = []
        retval_dist = []
        for i in relevant_dist_and_indices:
            dist,index = i
            retval_indices.append(index)
            retval_dist.append(dist)
        return retval_indices,retval_dist           