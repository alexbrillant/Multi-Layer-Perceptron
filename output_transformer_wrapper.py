from neuraxle.base import ForceHandleOnlyMixin, MetaStepMixin, BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer


class OutputTransformerWrapper(ForceHandleOnlyMixin, MetaStepMixin, BaseStep):
    """
    Transform expected output wrapper step that can sends the expected_outputs to the wrapped step
    so that it can transform the expected outputs.
    """

    def __init__(self, wrapped, from_data_inputs=False, cache_folder_when_no_handle=None):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)
        self.from_data_inputs = from_data_inputs

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle transform by passing expected outputs to the wrapped step transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        new_expected_outputs_data_container = self.wrapped.handle_transform(
            DataContainer(
                current_ids=data_container.current_ids,
                data_inputs=self._get_data_inputs(data_container),
                expected_outputs=None
            ),
            context
        )
        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)

        return data_container

    def _get_data_inputs(self, data_container):
        data_inputs = data_container.expected_outputs
        if self.from_data_inputs:
            data_inputs = data_container.data_inputs
        return data_inputs

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (BaseStep, DataContainer):
        """
        Handle fit by passing expected outputs to the wrapped step fit method.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        :rtype: (BaseStep, DataContainer)
        """
        self.wrapped = self.wrapped.handle_fit(
            DataContainer(
                current_ids=data_container.current_ids,
                data_inputs=self._get_data_inputs(data_container),
                expected_outputs=None
            ),
            context
        )

        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (BaseStep, DataContainer):
        """
        Handle fit transform by passing expected outputs to the wrapped step fit method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :type context: ExecutionContext
        :param data_container: data container to fit on
        :return: self, data container
        :rtype: (BaseStep, DataContainer)
        """
        self.wrapped, new_expected_outputs_data_container = self.wrapped.handle_fit_transform(
            DataContainer(
                current_ids=data_container.current_ids,
                data_inputs=self._get_data_inputs(data_container),
                expected_outputs=None
            ),
            context
        )
        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)

        return self, data_container

    def handle_inverse_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Handle inverse transform by passing expected outputs to the wrapped step inverse transform method.
        Update the expected outputs with the outputs.

        :param context: execution context
        :param data_container:
        :return: data container
        :rtype: DataContainer
        """
        new_expected_outputs_data_container = self.wrapped.handle_inverse_transform(
            DataContainer(
                current_ids=data_container.current_ids,
                data_inputs=self._get_data_inputs(data_container),
                expected_outputs=None
            ),
            context.push(self.wrapped)
        )

        data_container.set_expected_outputs(new_expected_outputs_data_container.data_inputs)

        current_ids = self.hash(data_container)
        data_container.set_current_ids(current_ids)

        return data_container