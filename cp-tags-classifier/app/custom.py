import tensorflow as tf

class CustomCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, gamma=0.9, epsilon=1e-7, name="cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        p = y_pred
        q = 1 - y_pred
        
        p = tf.math.maximum(p, self.epsilon)
        q = tf.math.maximum(q, self.epsilon)
        
        pos_loss = -tf.math.log(p ** self.gamma)
        neg_loss = -tf.math.log(q)
        
        y_true = tf.dtypes.cast(y_true, dtype=tf.bool)
        loss = tf.where(y_true, pos_loss, neg_loss)
        
        return tf.math.reduce_mean(loss)
    
    def get_config(self):
        config = {
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        base_config = super().get_config()
        return {**base_config, **config}