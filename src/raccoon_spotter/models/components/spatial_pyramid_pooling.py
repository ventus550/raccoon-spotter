import keras
from keras.layers import *


@keras.saving.register_keras_serializable()
class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum(i * i for i in pool_list)
        self.flatten = Flatten()
        self.concatenate = Concatenate()
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[3]
        super().build(input_shape)

    def call(self, x):
        _, w, h, _ = x.shape

        output = []
        for pool_size in self.pool_list:
            # Calculate stride such that output shape is an integer
            stride = (w // pool_size, h // pool_size)

            # Apply average pooling
            pooled_features = AveragePooling2D(
                pool_size, strides=stride, padding="valid"
            )(x)
            output.append(self.flatten(pooled_features))

        # Concatenate all pooled features
        return self.concatenate(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.channels * self.num_outputs_per_channel)
