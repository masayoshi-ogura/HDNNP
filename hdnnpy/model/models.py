# coding: utf-8

"""Neural network potential models."""

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable


class HighDimensionalNNP(chainer.ChainList):
    """High dimensional neural network potential.

    This is one implementation of HDNNP that is proposed by Behler
    *et al* [Ref]_.
    It has a structure in which simple neural networks are arranged in
    parallel.
    Each neural network corresponds to one atom and inputs descriptor
    and outputs property per atom.
    Total value or property is predicted to sum them up.
    """
    def __init__(self, elemental_composition, *args):
        """
        Args:
            elemental_composition (list [str]):
                Create the same number of :class:`SubNNP` instances as
                this. A :class:`SubNNP` with the same element has the
                same parameters synchronized.
            *args: Positional arguments that is passed to `SubNNP`.
        """
        super().__init__(
            *[SubNNP(element, *args) for element in elemental_composition])

    def predict(self, inputs, order):
        """Get prediction from input data in a feed-forward way.

        It accepts 0 or 2 for ``order``.

        Notes:
            0th-order predicted value is not total value, but per-atom
            value.

        Args:
            inputs (list [~numpy.ndarray]):
                Length have to equal to ``order + 1``. Each element is
                correspond to ``0th-order``, ``1st-order``, ...
            order (int):
                Derivative order of prediction by this model.

        Returns:
            list [~chainer.Variable]:
                Predicted values. Each elements is correspond to
                ``0th-order``, ``1st-order``, ...
        """
        assert 0 <= order <= 2
        input_variables = [[Variable(x) for x in data.swapaxes(0, 1)]
                           for data in inputs]
        for nnp in self:
            nnp.results.clear()

        xs = input_variables.pop(0)
        with chainer.force_backprop_mode():
            y_pred = self._predict_y(xs)
        if order == 0:
            return [y_pred]

        dxs = input_variables.pop(0)
        differentiate_more = chainer.config.train or order > 1
        with chainer.force_backprop_mode():
            dy_pred = self._predict_dy(xs, dxs, differentiate_more)
        if order == 1:
            return [y_pred, dy_pred]

        d2xs = input_variables.pop(0)
        differentiate_more = chainer.config.train or order > 2
        with chainer.force_backprop_mode():
            d2y_pred = self._predict_d2y(xs, dxs, d2xs, differentiate_more)
        if order == 2:
            return [y_pred, dy_pred, d2y_pred]

    def get_by_element(self, element):
        """Get all `SubNNP` instances that represent the same element.

        Args:
            element (str): Element symbol that you want to get.

        Returns:
            list [SubNNP]:
                All `SubNNP` instances which represent the same
                ``element`` in this HDNNP instance.
        """
        return [nnp for nnp in self if nnp.element == element]

    def reduce_grad_to(self, master_nnp):
        """Collect calculated gradient of parameters into `MasterNNP`
        for each element.

        Args:
            master_nnp (MasterNNP):
                `MasterNNP` instance where you manage parameters.
        """
        for master in master_nnp.children():
            for nnp in self.get_by_element(master.element):
                master.addgrads(nnp)

    def sync_param_with(self, master_nnp):
        """Synchronize the parameters with `MasterNNP` for each element.

        Args:
            master_nnp (MasterNNP):
                `MasterNNP` instance where you manage parameters.
        """
        for master in master_nnp.children():
            for nnp in self.get_by_element(master.element):
                nnp.copyparams(master)

    def _predict_y(self, xs):
        """Calculate 0th-order prediction for each `SubNNP`.

        Args:
            xs (list [~chainer.Variable]):
                Input data for each `SubNNP` constituting this HDNNP
                instance. The shape of data is
                ``n_atom x (n_sample, n_input)``.

        Returns:
            ~chainer.Variable:
                Output data (per atom) for each `SubNNP` constituting
                this HDNNP instance. The shape of data is
                ``(n_sample, n_output)``.
        """
        for nnp, x in zip(self, xs):
            nnp.feedforward(x)
        return sum([nnp.results['y'] for nnp in self]) / len(self)

    def _predict_dy(self, xs, dxs, differentiate_more):
        """Calculate 1st-order prediction for each `SubNNP`.

        Args:
            xs (list [~chainer.Variable]):
                Input data for each `SubNNP` constituting this HDNNP
                instance. The shape of data is
                ``n_atom x (n_sample, n_input)``.
            dxs (list [~chainer.Variable]):
                Differentiated input data. The shape of data is
                ``n_atom x (n_sample, n_input, n_deriv)``.
            differentiate_more (bool):
                If True, more deep calculation graph will be created for
                back-propagation or higher-order differentiation.

        Returns:
            ~chainer.Variable:
                Differentiated output data. The shape of data is
                ``(n_sample, n_output, n_deriv)``.
        """
        for nnp, x in zip(self, xs):
            nnp.differentiate(x, differentiate_more)
        return sum([F.einsum('soi,six->sox', nnp.results['dy'], dx)
                    for nnp, dx in zip(self, dxs)])

    def _predict_d2y(self, xs, dxs, d2xs, differentiate_more):
        """Calculate 2nd-order prediction for each `SubNNP`.

        Args:
            xs (list [~chainer.Variable]):
                Input data for each `SubNNP` constituting this HDNNP
                instance. The shape of data is
                ``n_atom x (n_sample, n_input)``.
            dxs (list [~chainer.Variable]):
                Differentiated input data. The shape of data is
                ``n_atom x (n_sample, n_input, n_deriv)``.
            d2xs (list [~chainer.Variable]):
                Double differentiated input data. The shape of data is
                ``n_atom x (n_sample, n_input, n_deriv, n_deriv)``.
            differentiate_more (bool):
                If True, more deep calculation graph will be created for
                back-propagation or higher-order differentiation.

        Returns:
            ~chainer.Variable:
                Double differentiated output data. The shape of data is
                ``(n_sample, n_output, n_deriv, n_deriv)``.
        """
        for nnp, x in zip(self, xs):
            nnp.second_differentiate(x, differentiate_more)
        return sum([F.einsum('soij,six,sjy->soxy', nnp.results['d2y'], dx, dx)
                    + F.einsum('soi,sixy->soxy', nnp.results['dy'], d2x)
                    for nnp, dx, d2x in zip(self, dxs, d2xs)])


class MasterNNP(chainer.ChainList):
    """Responsible for managing the parameters of each element."""
    def __init__(self, elements, *args):
        """
        It is implemented as a simple :class:`~chainer.ChainList` of
        `SubNNP`.

        Args:
            elements (list [str]): Element symbols must be unique.
            *args: Positional arguments that is passed to `SubNNP`.
        """
        super().__init__(*[SubNNP(element, *args) for element in elements])

    def dump_params(self):
        """Dump its own parameters as :obj:`str`.

        Returns:
            str: Formed parameters.
        """
        params_str = ''
        for nnp in self:
            element = nnp.element
            depth = len(nnp)
            for i in range(depth):
                weight = getattr(nnp, f'fc_layer{i}').W.data
                bias = getattr(nnp, f'fc_layer{i}').b.data
                activation = getattr(nnp, f'activation_function{i}').__name__
                weight_str = ('\n'+' '*16).join([' '.join(map(str, row))
                                                 for row in weight.T])
                bias_str = ' '.join(map(str, bias))

                params_str += f'''
                {element} {i} {weight.shape[1]} {weight.shape[0]} {activation}
                # weight
                {weight_str}
                # bias
                {bias_str}
                '''

        return params_str


class SubNNP(chainer.Chain):
    """Feed-forward neural network representing one element or atom."""
    def __init__(self, element, n_feature, hidden_layers, n_property):
        """
        | ``element`` is registered as a persistent value.
        | It consists of repetition of fully connected layer and
          activation function.
        | Weight initializer is :obj:`chainer.initializers.HeNormal`.

        Args:
            element (str): Element symbol represented by an instance.
            n_feature (int): Number of nodes of input layer.
            hidden_layers (list [tuple [int, str]]):
                A neural network structure. Last one is output layer,
                and the remains are hidden layers. Each element is a
                tuple ``(# of nodes, activation function)``, for example
                ``(50, 'sigmoid')``. Only activation functions
                implemented in `chainer.functions`_ can be used.
            n_property (int): Number of nodes of output layer.

        .. _`chainer.functions`:
            https://docs.chainer.org/en/stable/reference/functions.html
        """
        super().__init__()
        self.add_persistent('element', element)
        self._n_layer = len(hidden_layers) + 1
        nodes = [n_feature, *[layer[0] for layer in hidden_layers], n_property]
        activations = [*[layer[1] for layer in hidden_layers], 'identity']
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            for i, (in_size, out_size, activation) in enumerate(zip(
                    nodes[:-1], nodes[1:], activations)):
                setattr(self, f'activation_function{i}',
                        eval(f'F.{activation}'))
                setattr(self, f'fc_layer{i}',
                        L.Linear(in_size, out_size, initialW=w))
        self.results = {}

    def __len__(self):
        """Return the number of hidden_layers."""
        return self._n_layer

    def feedforward(self, x):
        """Propagate input data in a feed-forward way.

        Args:
            x (~chainer.Variable):
                Input data which has the shape ``(n_sample, n_input)``.
        """
        h = x
        for i in range(self._n_layer):
            h = eval(f'self.activation_function{i}(self.fc_layer{i}(h))')
        y = h
        self.results['y'] = y

    def differentiate(self, x, enable_double_backprop):
        """Calculate derivative of the output data w.r.t. input data.

        Args:
            x (~chainer.Variable):
                Input data which has the shape ``(n_sample, n_input)``.
            enable_double_backprop (bool):
                Passed to :func:`chainer.grad` to determine whether to
                create more deep calculation graph or not.
        """
        dy = [chainer.grad([output_node], [x],
                           enable_double_backprop=enable_double_backprop)[0]
              for output_node in F.moveaxis(self.results['y'], 0, -1)]
        dy = F.stack(dy, axis=1)
        self.results['dy'] = dy

    def second_differentiate(self, x, enable_double_backprop):
        """Calculate 2nd derivative of the output data w.r.t. input
        data.

        Args:
            x (~chainer.Variable):
                Input data which has the shape ``(n_sample, n_input)``.
            enable_double_backprop (bool):
                Passed to :func:`chainer.grad` to determine whether to
                create more deep calculation graph or not.
        """
        d2y = [[chainer.grad([derivative], [x],
                             enable_double_backprop=enable_double_backprop)[0]
                for derivative in dy_]
               for dy_ in F.moveaxis(self.results['dy'], 0, -1)]
        d2y = F.stack([F.stack(d2y_, axis=1) for d2y_ in d2y], axis=1)
        self.results['d2y'] = d2y
