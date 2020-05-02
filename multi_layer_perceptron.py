import numpy as np
import tensorflow as tf
from neuraxle.base import Identity, BaseStep, NonFittableMixin
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.metaopt.auto_ml import AutoML, RandomSearchHyperparameterSelectionStrategy, ValidationSplitter, \
    InMemoryHyperparamsRepository
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.pipeline import Pipeline, MiniBatchSequentialPipeline
from neuraxle.steps.data import DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.numpy import OneHotEncoder
from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
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
from output_transformer_wrapper import OutputTransformerWrapper


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
        Dense(units=step.hyperparams['hidden_dim'], activation='relu', kernel_initializer='he_normal',
              input_shape=(step.hyperparams['input_dim'],)),
        Dense(units=8, activation='relu', kernel_initializer='he_normal')
    ]

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


def main():
    def accuracy(data_inputs, expected_outputs):
        return np.mean(np.argmax(np.array(data_inputs), axis=1) == np.argmax(np.array(expected_outputs), axis=1))

    # load the dataset
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    df = read_csv(path, header=None)
    data_inputs = df.values
    data_inputs[:, -1] = LabelEncoder().fit_transform(data_inputs[:, -1])
    n_features = data_inputs.shape[1] - 1
    n_classes = 3

    p = Pipeline([
        TrainOnlyWrapper(DataShuffler()),
        ColumnTransformerInputOutput(
            input_columns=[([0, 1, 2, 3], ToNumpy(np.float32))],
            output_columns=[(4, Identity())]
        ),
        MiniBatchSequentialPipeline([
            Tensorflow2ModelStep(create_model=create_model, create_loss=create_loss, create_optimizer=create_optimizer) \
                .set_hyperparams(HyperparameterSamples({
                'input_dim': n_features,
                'optimizer': 'adam',
                'learning_rate': 0.01,
                'hidden_dim': 20,
                'n_classes': 3
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
        scoring_callback=ScoringCallback(accuracy, higher_score_is_better=False),
        n_trials=1,
        refit_trial=True,
        epochs=150
    )

    auto_ml = auto_ml.fit(data_inputs=data_inputs)


if __name__ == '__main__':
    main()
