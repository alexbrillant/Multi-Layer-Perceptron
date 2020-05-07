from neuraxle.data_container import DataContainer
from neuraxle.metaopt.callbacks import BaseCallback
from neuraxle.metaopt.trial import TrialSplit


class EarlyStoppingCallback(BaseCallback):
    """
    Perform early stopping when there is multiple epochs in a row that didn't improve the performance of the model.

    .. seealso::
        :class:`BaseCallback`,
        :class:`MetaCallback`,
        :class:`IfBestScore`,
        :class:`IfLastStep`,
        :class:`StepSaverCallback`,
        :class:`~neuraxle.metaopt.auto_ml.AutoML`,
        :class:`~neuraxle.metaopt.auto_ml.Trainer`,
        :class:`~neuraxle.metaopt.trial.Trial`,
        :class:`~neuraxle.metaopt.auto_ml.InMemoryHyperparamsRepository`,
        :class:`~neuraxle.metaopt.auto_ml.HyperparamsJSONRepository`,
        :class:`~neuraxle.metaopt.auto_ml.BaseHyperparameterSelectionStrategy`,
        :class:`~neuraxle.metaopt.auto_ml.RandomSearchHyperparameterSelectionStrategy`,
        :class:`~neuraxle.base.HyperparameterSamples`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, max_epochs_without_improvement):
        self.n_epochs_without_improvement = max_epochs_without_improvement
        self.epochs_without_improvement = 0

    def call(
            self,
            trial: TrialSplit,
            epoch_number: int,
            total_epochs: int,
            input_train: DataContainer,
            pred_train: DataContainer,
            input_val: DataContainer,
            pred_val: DataContainer,
            is_finished_and_fitted: bool
    ):
        validation_scores = trial.get_validation_scores()

        if len(validation_scores) > self.n_epochs_without_improvement:
            higher_score_is_better = trial.is_higher_score_better()
            if validation_scores[-1] == 0:
                return False

            if higher_score_is_better:
                if validation_scores[-2] >= validation_scores[-1]:
                    self.epochs_without_improvement += 1
                else:
                    self.epochs_without_improvement = 0

            if not higher_score_is_better:
                if validation_scores[-2] <= validation_scores[-1]:
                    self.epochs_without_improvement += 1
                else:
                    self.epochs_without_improvement = 0

            if self.epochs_without_improvement == self.n_epochs_without_improvement:
                self.epochs_without_improvement = 0
                return True
        return False