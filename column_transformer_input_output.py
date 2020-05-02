from neuraxle.base import TruncableSteps
from neuraxle.steps.column_transformer import ColumnChooserTupleList, ColumnTransformer

from output_transformer_wrapper import OutputTransformerWrapper


class ColumnTransformerInputOutput(TruncableSteps):
    def __init__(
        self,
        input_columns: ColumnChooserTupleList,
        output_columns: ColumnChooserTupleList,
        n_dimension: int = 3
    ):
        TruncableSteps.__init__(self, [
            OutputTransformerWrapper(ColumnTransformer(output_columns, n_dimension), from_data_inputs=True),
            ColumnTransformer(input_columns, n_dimension),
        ])