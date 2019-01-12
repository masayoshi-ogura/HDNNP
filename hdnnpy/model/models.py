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
    def __init__(self, elemental_composition, layers, order):
        """
        It accepts 0 or 2 for ``order``.

        Args:
            elemental_composition (list [str]):
                Create the same number of :class:`SubNNP` instances as
                this. A :class:`SubNNP` with the same element has the
                same parameters synchronized.
            layers (list [tuple [int, str]]):
                A neural network structure. Last one is output layer,
                and the remains are hidden layers. Each element is a
                tuple ``(# of nodes, activation function)``, for example
                ``(50, 'sigmoid')``. Only activation functions
                implemented in `chainer.functions`_ can be used.
            order (int):
                Derivative order of prediction by this model.

        .. _`chainer.functions`:
            https://docs.chainer.org/en/stable/reference/functions.html
        """
        assert 0 <= order <= 2
        super().__init__(
            *[SubNNP(element, layers) for element in elemental_composition])
        self._order = order

    @property
    def order(self):
        """int: Derivative order of prediction by this model."""
        return self._order

    def predict(self, inputs):
        """Get prediction from input data in a feed-forward way.

        Notes:
            0th-order predicted value is not total value, but per-atom
            value.

        Args:
            inputs (list [~numpy.ndarray]):
                Length have to equal to ``order + 1``. Each element is
                correspond to ``0th-order``, ``1st-order``, ...

        Returns:
            list [~chainer.Variable]:
                Predicted values. Each elements is correspond to
                ``0th-order``, ``1st-order``, ...
        """
        input_variables = [[Variable(x) for x in data.swapaxes(0, 1)]
                           for data in inputs]
        ret = []

        if self._order >= 0:
            xs, = input_variables[:1]
            with chainer.force_backprop_mode():
                ys = self._predict_y(xs)
            y_pred = sum(ys) / len(self)
            ret.append(y_pred)

        if self._order >= 1:
            xs, dxs = input_variables[:2]
            with chainer.force_backprop_mode():
                dys = self._predict_dy(xs, dxs, ys)
            dy_pred = sum(dys)
            ret.append(dy_pred)

        if self._order >= 2:
            xs, dxs, d2xs = input_variables[:3]
            with chainer.force_backprop_mode():
                d2ys = self._predict_d2y(xs, dxs, d2xs, dys)
            d2y_pred = sum(d2ys)
            ret.append(d2y_pred)

        return ret

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
            list [~chainer.Variable]:
                Output data for each `SubNNP` constituting this HDNNP
                instance. The shape of data is
                ``n_atom x (n_sample, n_output)``.
        """
        return [nnp.feedforward(x) for nnp, x in zip(self, xs)]

    def _predict_dy(self, xs, dxs, ys):
        """Calculate 1th-order prediction for each `SubNNP`.

        Args:
            xs (list [~chainer.Variable]):
                Input data for each `SubNNP` constituting this HDNNP
                instance. The shape of data is
                ``n_atom x (n_sample, n_input)``.
            dxs (list [~chainer.Variable]):
                Differentiated input data. The shape of data is
                ``n_atom x (n_sample, n_input, n_deriv)``.
            ys (list [~chainer.Variable]):
                Output data for each `SubNNP` constituting this HDNNP
                instance. The shape of data is
                ``n_atom x (n_sample, n_output)``. This can be obtained
                by :meth:`_predict_y`.

        Returns:
            list [~chainer.Variable]:
                Differentiated output data. The shape of data is
                ``n_atom x (n_sample, n_output, n_deriv)``.
        """
        return [F.einsum('soi,six->sox',
                         nnp.first_derivative(x, y), dx)
                for nnp, x, dx, y in zip(self, xs, dxs, ys)]

    def _predict_d2y(self, xs, dxs, d2xs, dys):
        """Calculate 2th-order prediction for each `SubNNP`.

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
            dys (list [~chainer.Variable]):
                Differentiated output data for each `SubNNP`
                constituting this HDNNP instance. The shape of data is
                ``n_atom x (n_sample, n_output, n_input)``.
                This can be obtained by :meth:`_predict_dy`.

        Returns:
            list [~chainer.Variable]:
                Double differentiated output data. The shape of data is
                ``n_atom x (n_sample, n_output, n_deriv, n_deriv)``.
        """
        return [F.einsum('soij,six,sjy->soxy',
                         nnp.second_derivative(x, dy), dx, dx)
                + F.einsum('soi,sixy->soxy', dy, d2x)
                for nnp, x, dx, d2x, dy
                in zip(self, xs, dxs, d2xs, dys)]


class MasterNNP(chainer.ChainList):
    """Responsible for managing the parameters of each element."""
    def __init__(self, elements, layers):
        """
        It is implemented as a simple :class:`~chainer.ChainList` of
        `SubNNP`.

        Args:
            elements (list [str]): Element symbols must be unique.
            layers (list [tuple [int, str]]):
                A neural network structure. Last one is output layer,
                and the remains are hidden layers. Each element is a
                tuple ``(# of nodes, activation function)``, for example
                ``(50, 'sigmoid')``. Only activation functions
                implemented in `chainer.functions`_ can be used.

        .. _`chainer.functions`:
            https://docs.chainer.org/en/stable/reference/functions.html
        """
        super().__init__(*[SubNNP(element, layers) for element in elements])

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
    def __init__(self, element, layers):
        """
        | ``element`` is registered as a persistent value.
        | It consists of repetition of fully connected layer and
          activation function.
        | Weight initializer is :obj:`chainer.initializers.HeNormal`.

        Args:
            element (str): Element symbol represented by an instance.
            layers (list [tuple [int, str]]):
                A neural network structure. Last one is output layer,
                and the remains are hidden layers. Each element is a
                tuple ``(# of nodes, activation function)``, for example
                ``(50, 'sigmoid')``. Only activation functions
                implemented in `chainer.functions`_ can be used.

        .. _`chainer.functions`:
            https://docs.chainer.org/en/stable/reference/functions.html
        """
        super().__init__()
        self.add_persistent('element', element)
        self._n_layer = len(layers)
        nodes = [None] + [layer[0] for layer in layers]
        activations = [layer[1] for layer in layers]
        with self.init_scope():
            w = chainer.initializers.HeNormal()
            for i, (insize, outsize, activation) in enumerate(zip(
                    nodes[:-1], nodes[1:], activations)):
                setattr(self, f'activation_function{i}',
                        eval(f'F.{activation}'))
                setattr(self, f'fc_layer{i}',
                        L.Linear(insize, outsize, initialW=w))

    def __len__(self):
        """Return the number of layers."""
        return self._n_layer

    def feedforward(self, x):
        """Propagate input data in a feed-forward way.

        Args:
            x (~chainer.Variable):
                Input data which has the shape ``(n_sample, n_input)``.

        Returns:
            ~chainer.Variable: Output data.
                Output data which has the shape
                ``(n_sample, n_output)``.
        """
        h = x
        for i in range(self._n_layer):
            h = eval(f'self.activation_function{i}(self.fc_layer{i}(h))')
        y = h
        return y

    @staticmethod
    def first_derivative(x, y):
        """Calculate derivative of the output data w.r.t. input data.

        Args:
            x (~chainer.Variable):
                Input data which has the shape ``(n_sample, n_input)``.
            y (~chainer.Variable):
                Output data which has the shape
                ``(n_sample, n_output)``.

        Returns:
            ~chainer.Variable:
                Derivative of ``y`` w.r.t. ``x`` which has the shape
                ``(n_sample, n_output, n_input)``.
        """
        dy = [chainer.grad([output_node], [x],
                           enable_double_backprop=chainer.config.train
                           )[0]
              for output_node in y.T]
        dy = F.stack(dy, axis=1)
        return dy

    @staticmethod
    def second_derivative(x, dy):
        """Calculate 2nd derivative of the output data w.r.t. input
        data.

        Args:
            x (~chainer.Variable):
                Input data which has the shape ``(n_sample, n_input)``.
            dy (~chainer.Variable):
                1st derivative of output data which has the shape
                ``(n_sample, n_output, n_input)``.

        Returns:
            ~chainer.Variable:
                Derivative of ``dy`` w.r.t. ``x`` which has the shape
                ``(n_sample, n_output, n_input, n_input)``.
        """
        n_sample, n_output, n_input = dy.shape
        d2y = [chainer.grad([derivative], [x])[0]
               for output_node in F.transpose(dy, (1, 2, 0))
               for derivative in output_node]
        d2y = F.stack(d2y, axis=1).reshape(
            (n_sample, n_output, n_input, n_input))
        return d2y
