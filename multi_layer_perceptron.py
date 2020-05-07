import os
from enum import Enum

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from neuraxle.base import Identity, BaseStep, NonFittableMixin
from neuraxle.hyperparams.distributions import Choice, LogUniform, RandInt, FixedHyperparameter, Uniform
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.metaopt.auto_ml import AutoML, RandomSearchHyperparameterSelectionStrategy, ValidationSplitter, \
    InMemoryHyperparamsRepository
from neuraxle.metaopt.callbacks import ScoringCallback, MetricCallback
from neuraxle.pipeline import Pipeline, MiniBatchSequentialPipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import OneHotEncoder
from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep
from pandas import read_csv
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.losses import sparse_categorical_crossentropy
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.optimizer_v2.adagrad import Adagrad
from tensorflow_core.python.keras.optimizer_v2.adamax import Adamax
from tensorflow_core.python.keras.optimizer_v2.ftrl import Ftrl
from tensorflow_core.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_core.python.keras.optimizer_v2.rmsprop import RMSProp
from tensorflow_core.python.training.adam import AdamOptimizer

from column_transformer_input_output import ColumnTransformerInputOutput
from early_stopping_callback import EarlyStoppingCallback
from metrics import precision_score_weighted, recall_score_weighted, f1_score_weighted, \
    classificaiton_report_imbalanced_metric
from output_transformer_wrapper import OutputTransformerWrapper
import matplotlib.pyplot as plt


def create_model(step: Tensorflow2ModelStep):
    """
    Create a TensorFlow v2 Multi-Layer-Perceptron Model.

    :param step: The base Neuraxle step for TensorFlow v2 (Tensorflow2ModelStep)
    :return: TensorFlow v2 Keras model
    """
    # shape: (batch_size, input_dim)
    inputs = Input(
        shape=(step.hyperparams['input_dim']),
        batch_size=None,
        dtype=tf.dtypes.float32,
        name='inputs',
    )

    dense_layers = [
        Dense(
            units=step.hyperparams['hidden_dim'],
            kernel_initializer=step.hyperparams['kernel_initializer'],
            activation=step.hyperparams['activation'],
            input_shape=(step.hyperparams['input_dim'],)
        )
    ]

    hidden_dim = step.hyperparams['hidden_dim']
    for i in range(step.hyperparams['n_dense_layers'] - 1):
        hidden_dim *= step.hyperparams['hidden_dim_layer_multiplier']
        dense_layers.append(Dense(
            units=int(hidden_dim),
            activation=step.hyperparams['activation'],
            kernel_initializer=step.hyperparams['kernel_initializer']
        ))

    for layer in dense_layers:
        outputs = layer(inputs)

    softmax_layer = Dense(step.hyperparams['n_classes'], activation='softmax')
    outputs = softmax_layer(outputs)

    return Model(inputs=inputs, outputs=outputs)


def create_loss(step: Tensorflow2ModelStep, expected_outputs, predicted_outputs):
    """
    Create a TensorFlow v2 loss

    :param step: The base Neuraxle step for TensorFlow v2 (Tensorflow2ModelStep)
    :return: TensorFlow v2 Keras loss
    """
    return sparse_categorical_crossentropy(
        y_true=expected_outputs,
        y_pred=predicted_outputs,
        from_logits=False,
        axis=-1
    )


class OPTIMIZERS(Enum):
    SGD = 'sgd'
    ADAM = 'adam'
    ADAGRAD = 'adagrad'
    ADAMAX = 'adamax'
    FTRL = 'ftrl'
    NADAM = 'nadam'
    RMSPROP = 'rms_prop'

class ACTIVATIONS(Enum):
    RELU = 'relu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    LEAKY_RELU = 'leaky_relu'
    ELU = 'elu'
    PRELU = 'prelu'

class KERNEL_INITIALIZERS(Enum):
    GLOROT_NORMAL = 'glorot_normal'
    GLOROT_UNIFORM = 'glorot_uniform'
    HE_UNIFORM = 'he_uniform'

def create_optimizer(step: Tensorflow2ModelStep):
    """
    Create a TensorFlow v2 optimizer.

    :param step: The base Neuraxle step for TensorFlow v2 (Tensorflow2ModelStep)
    :return: TensorFlow v2 optimizer
    """
    if step.hyperparams['optimizer'] == 'sgd':
        return SGD(learning_rate=step.hyperparams['learning_rate'])

    if step.hyperparams['optimizer'] == 'adam':
        return AdamOptimizer(learning_rate=step.hyperparams['learning_rate'])

    if step.hyperparams['optimizer'] == 'adagrad':
        return Adagrad(learning_rate=step.hyperparams['learning_rate'])

    if step.hyperparams['optimizer'] == 'adamax':
        return Adamax(learning_rate=step.hyperparams['learning_rate'])

    if step.hyperparams['optimizer'] == 'ftrl':
        return Ftrl(learning_rate=step.hyperparams['learning_rate'])

    if step.hyperparams['optimizer'] == 'nadam':
        return Ftrl(learning_rate=step.hyperparams['learning_rate'])

    if step.hyperparams['optimizer'] == 'rms_prop':
        return RMSProp(learning_rate=step.hyperparams['learning_rate'])

    return AdamOptimizer(learning_rate=step.hyperparams['learning_rate'])


class ToNumpy(NonFittableMixin, BaseStep):
    def __init__(self, dtype):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(self)
        self.dtype = dtype

    def transform(self, data_inputs):
        return data_inputs.astype(self.dtype)

class ExpandDim(NonFittableMixin, BaseStep):
    def transform(self, data_inputs):
        return np.expand_dims(data_inputs, axis=-1)


class PlotDistribution(NonFittableMixin, BaseStep):
    def __init__(self, column):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.column = column

    def transform(self, data_inputs):
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.hist(data_inputs[:, self.column])
        plt.savefig(os.path.join('plots', self.name))
        plt.close()
        return data_inputs

class Resample(NonFittableMixin, BaseStep):
    def __init__(self, column):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)
        self.column = column

    def transform(self, data_inputs):
        NearMiss()
        SMOTE()
        return data_inputs


def main():
    def accuracy(data_inputs, expected_outputs):
        return np.mean(np.argmax(np.array(data_inputs), axis=1) == np.argmax(np.array(expected_outputs), axis=1))

    # load the dataset
    df = read_csv('data/winequality-white.csv', sep=';')
    data_inputs = df.values
    data_inputs[:, -1] = data_inputs[:, -1] - 1
    n_features = data_inputs.shape[1] - 1
    n_classes = 10

    p = Pipeline([
        TrainOnlyWrapper(DataShuffler()),
        ColumnTransformerInputOutput(
            input_columns=[(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ToNumpy(np.float32)
            )],
            output_columns=[(11, Identity())]
        ),
        OutputTransformerWrapper(PlotDistribution(column=-1)),
        MiniBatchSequentialPipeline([
            Tensorflow2ModelStep(
                create_model=create_model,
                create_loss=create_loss,
                create_optimizer=create_optimizer
            ) \
                .set_hyperparams(HyperparameterSamples({
                'n_dense_layers': 2,
                'input_dim': n_features,
                'optimizer': 'adam',
                'activation': 'relu',
                'kernel_initializer': 'he_uniform',
                'learning_rate': 0.01,
                'hidden_dim': 20,
                'n_classes': 3
            })).set_hyperparams_space(HyperparameterSpace({
                'n_dense_layers': RandInt(2, 4),
                'hidden_dim_layer_multiplier': Uniform(0.30, 1),
                'input_dim': FixedHyperparameter(n_features),
                'optimizer': Choice([
                    OPTIMIZERS.ADAM.value,
                    OPTIMIZERS.SGD.value,
                    OPTIMIZERS.ADAGRAD.value
                ]),
                'activation': Choice([
                    ACTIVATIONS.RELU.value,
                    ACTIVATIONS.TANH.value,
                    ACTIVATIONS.SIGMOID.value,
                    ACTIVATIONS.ELU.value,
                ]),
                'kernel_initializer': Choice([
                    KERNEL_INITIALIZERS.GLOROT_NORMAL.value,
                    KERNEL_INITIALIZERS.GLOROT_UNIFORM.value,
                    KERNEL_INITIALIZERS.HE_UNIFORM.value
                ]),
                'learning_rate': LogUniform(0.005, 0.01),
                'hidden_dim': RandInt(3, 80),
                'n_classes': FixedHyperparameter(n_classes)
            }))
        ], batch_size=33),
        OutputTransformerWrapper(Pipeline([
            ExpandDim(),
            OneHotEncoder(nb_columns=n_classes, name='classes')
        ]))
    ])


    auto_ml = AutoML(
        pipeline=p,
        hyperparams_repository=InMemoryHyperparamsRepository(cache_folder='trials'),
        hyperparams_optimizer=RandomSearchHyperparameterSelectionStrategy(),
        validation_splitter=ValidationSplitter(test_size=0.30),
        scoring_callback=ScoringCallback(accuracy, higher_score_is_better=True),
        callbacks=[
            MetricCallback(name='classification_report_imbalanced_metric', metric_function=classificaiton_report_imbalanced_metric, higher_score_is_better=True),
            MetricCallback(name='f1', metric_function=f1_score_weighted, higher_score_is_better=True),
            MetricCallback(name='recall', metric_function=recall_score_weighted, higher_score_is_better=True),
            MetricCallback(name='precision', metric_function=precision_score_weighted, higher_score_is_better=True),
            EarlyStoppingCallback(max_epochs_without_improvement=3)
        ],
        n_trials=200,
        refit_trial=True,
        epochs=75
    )

    auto_ml = auto_ml.fit(data_inputs=data_inputs)


if __name__ == '__main__':
    main()
