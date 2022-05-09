from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Add
from tensorflow_addons.layers import SpectralNormalization
import tensorflow as tf 

from gloro.layers import InvertibleDownsampling
from gloro.layers import MinMax
from gloro.layers import ResnetBlock


def _add_pool(z, pooling_type, activation=None, initialization='orthogonal'):
    if pooling_type == 'avg':
        return AveragePooling2D()(z)

    elif pooling_type == 'conv':
        channels = z.shape[-1]

        z = Conv2D(
            channels, 
            4, 
            strides=2, 
            padding='same', 
            kernel_initializer=initialization)(z)

        return _add_activation(z, activation)

    elif pooling_type == 'invertible':
        return InvertibleDownsampling()(z)

    else:
        raise ValueError(f'unknown pooling type: {pooling_type}')

def _add_activation(z, activation_type='relu'):
    if activation_type == 'relu':
        return Activation('relu')(z)

    elif activation_type == 'elu':
        return Activation('elu')(z)

    elif activation_type == 'softplus':
        return Activation('softplus')(z)

    elif activation_type == 'minmax':
        return MinMax()(z)

    else:
        raise ValueError(f'unknown activation type: {activation_type}')


def cnn_simple(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(256, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    z = Dense(256, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_simple(
    input_shape, 
    num_classes, 
    pooling='invertible', 
    initialization='orthogonal',
    normalize_lc=False,
):
    return cnn_simple(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_2C2F(
    input_shape, 
    num_classes, 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(
        16, 4, strides=2, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)

    z = Conv2D(
        32, 4, strides=2, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    z = Flatten()(z)
    z = Dense(100, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_2C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    initialization='orthogonal',
):
    return cnn_2C2F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_4C3F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_4C3F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    initialization='orthogonal',
):
    return cnn_4C3F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_6C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_6C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    initialization='orthogonal',
):
    return cnn_6C2F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_8C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(256, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_8C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    initialization='orthogonal',
):
    return cnn_8C2F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def alexnet(
    input_shape,
    num_classes,
    pooling='avg',
    activation='relu',
    initialization='orthogonal',
    dropout=False,
):
    x = Input(input_shape)

    z = Conv2D(
        96,
        11,
        padding='same',
        strides=4,
        kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(256, 5, padding='same', kernel_initializer=initialization)(z)
    z = Activation('relu')(z)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(384, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(384, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(384, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    if dropout:
        z = Dropout(0.5)(z)
    z = Dense(4096, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    if dropout:
        z = Dropout(0.5)(z)
    z = Dense(4096, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_alexnet(
    input_shape,
    num_classes,
    pooling='invertible',
    initialization='orthogonal',
    dropout=False,
):
    return alexnet(
        input_shape, num_classes, 
        pooling=pooling,
        activation='minmax',
        initialization=initialization,
        dropout=dropout)


def vgg16(
    input_shape,
    num_classes,
    pooling='avg',
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(4096, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Dense(4096, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_vgg16(
        input_shape,
        num_classes,
        pooling='invertible',
        initialization='orthogonal'):

    return vgg16(
        input_shape, num_classes, 
        pooling=pooling,
        activation='minmax',
        initialization=initialization)


def resnet_tiny(
    input_shape,
    num_classes,
    pooling='avg',
    activation='relu',
    initialization='orthogonal',
    fixup_residual_scaling=False,
    identity_skip=False,
):
    x = Input(input_shape)

    z = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = ResnetBlock(
        filters=(128, 128, 128),
        kernel_sizes=(3, 3, 1),
        stride1=2,
        activation=activation,
        use_invertible_downsample=pooling == 'invertible',
        kernel_initializer=initialization,
        use_fixup_weight_and_bias=fixup_residual_scaling,
        identity_skip=identity_skip)(z)
    z = ResnetBlock(
        filters=(256, 256, 256),
        kernel_sizes=(3, 3, 1),
        stride1=2,
        activation=activation,
        use_invertible_downsample=pooling == 'invertible',
        kernel_initializer=initialization,
        use_fixup_weight_and_bias=fixup_residual_scaling,
        identity_skip=identity_skip)(z)
    z = ResnetBlock(
        filters=(512, 512, 512),
        kernel_sizes=(3, 3, 1),
        stride1=2,
        activation=activation,
        use_invertible_downsample=pooling == 'invertible',
        kernel_initializer=initialization,
        use_fixup_weight_and_bias=fixup_residual_scaling,
        identity_skip=identity_skip)(z)

    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_resnet_tiny(
    input_shape,
    num_classes,
    pooling='invertible',
    initialization='orthogonal',
    fixup_residual_scaling=False,
    identity_skip=False,
):
    return resnet_tiny(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization,
        fixup_residual_scaling=fixup_residual_scaling,
        identity_skip=identity_skip)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = SpectralNormalization(Dense(units, activation=tf.nn.gelu))(x)
        x = Dropout(dropout_rate)(x)
    return x

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = SpectralNormalization(Dense(units=projection_dim))
        self.position_embedding = SpectralNormalization(Embedding(
            input_dim=num_patches, output_dim=projection_dim
        ))

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config

def vit(
    input_shape,
    num_classes,
    patch_size=7,
    projection_dim=128,
    num_patches=16,
    num_heads=8,
    transformer_layers=1, 
    mlp_head_units=[128, 128]
):
    inputs = Input(shape=input_shape)
    transformer_units = [projection_dim * 2, projection_dim,]
    # Augment data.
    # augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = Dense(num_classes)(features)
    # Create the Keras model.
    # model = keras.Model(inputs=inputs, outputs=logits)
    return inputs, logits