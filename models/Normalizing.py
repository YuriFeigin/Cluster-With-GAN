import tensorflow as tf


# batch norm
def batch_norm(input_, name="bn_in", decay=0.1, scale=True, is_training=True, epsilon=1e-4):
    with tf.variable_scope(name):
        shape = input_.get_shape().as_list()
        moving_mean = tf.get_variable('moving_mean',shape[-1], initializer=tf.constant_initializer(0), trainable=False)
        moving_var = tf.get_variable('moving_var',shape[-1], initializer=tf.constant_initializer(1.), trainable=False)
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.))
        if scale:
            gamma = tf.get_variable('gamma', shape[-1], initializer=tf.constant_initializer(1.))
        else:
            gamma = tf.get_variable('gamma', shape[-1], initializer=tf.constant_initializer(1.), trainable=False)

        if is_training:
            out, cur_mean, cur_var = tf.nn.fused_batch_norm(input_, gamma, beta
                                                            , epsilon=epsilon, is_training=is_training)
            #todo check for nan?  tf.check_numerics
            update_mean_op = moving_mean.assign_sub(decay*(moving_mean-cur_mean))
            update_var_op = moving_var.assign_sub(decay*(moving_var-cur_var))
            with tf.control_dependencies([update_mean_op,update_var_op]):
                out = tf.identity(out)
        else:
            out, _, _ = tf.nn.fused_batch_norm(input_, gamma, beta,
                                               moving_mean, moving_var, epsilon=epsilon, is_training=is_training)
    return out

def cond_batchnorm(inputs,name,is_training, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    shape = inputs.get_shape().as_list()
    offset = tf.constant(0,tf.float32,shape=[shape[3]])
    scale = tf.constant(1,tf.float32,shape=[shape[3]])
    result,_,_ = tf.nn.fused_batch_norm(inputs,scale,offset)
    offset_m = tf.get_variable(name+'.offset',[n_labels,shape[3]],initializer=tf.initializers.zeros)
    scale_m = tf.get_variable(name+'.scale',[n_labels,shape[3]],initializer=tf.initializers.ones)
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = result*scale[:,None,None,:] + offset[:,None,None,:]
    return result

def dense_cond_batchnorm(inputs,name,is_training, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    shape = inputs.get_shape().as_list()
    result = tf.layers.batch_normalization(inputs,trainable=False)
    offset_m = tf.get_variable(name+'.offset',[n_labels,shape[1]],initializer=tf.initializers.zeros)
    scale_m = tf.get_variable(name+'.scale',[n_labels,shape[1]],initializer=tf.initializers.ones)
    offset = tf.nn.embedding_lookup(offset_m, labels)
    scale = tf.nn.embedding_lookup(scale_m, labels)
    result = result*scale + offset
    return result