from neuraxle.pipeline import Pipeline
from neuraxle.steps.column_transformer import ColumnChooserTupleList, ColumnTransformer

from output_transformer_wrapper import OutputTransformerWrapper


class ColumnTransformerInputOutput(Pipeline):
    def __init__(
        self,
        input_columns: ColumnChooserTupleList,
        output_columns: ColumnChooserTupleList,
        n_dimension: int = 3
    ):
        super().__init__([
            OutputTransformerWrapper(ColumnTransformer(output_columns, n_dimension), from_data_inputs=True),
            ColumnTransformer(input_columns, n_dimension),
        ])