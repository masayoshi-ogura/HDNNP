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
        It accepts 0 or 1 for ``order``.

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
        assert 0 <= order <= 1
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
        if self._order == 0:
            xs, = inputs
            xs = [Variable(x) for x in xs.transpose(1, 0, 2)]
            ys = self._predict_y(xs)
            y_pred = sum(ys) / len(self)
            return [y_pred]

        elif self._order == 1:
            xs, dxs = inputs
            xs = [Variable(x) for x in xs.transpose(1, 0, 2)]
            dxs = [Variable(dx) for dx in dxs.transpose(1, 0, 2, 3, 4)]
            with chainer.force_backprop_mode():
                ys = self._predict_y(xs)
                dys = self._predict_dy(xs, ys, dxs)
            y_pred = sum(ys) / len(self)
            dy_pred = sum(dys)
            return [y_pred, dy_pred]

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

    def _predict_dy(self, xs, ys, dxs):
        """Calculate 1th-order prediction for each `SubNNP`.

        Args:
            xs (list [~chainer.Variable]):
                Input data for each `SubNNP` constituting this HDNNP
                instance. The shape of data is
                ``n_atom x (n_sample, n_input)``.
            ys (list [~chainer.Variable]):
                Output data for each `SubNNP` constituting this HDNNP
                instance. The shape of data is
                ``n_atom x (n_sample, n_output)``. This can be obtained
                by :meth:`_predict_y`.
            dxs (list [~chainer.Variable]):
                Differentiated input data. The shape of data is
                ``n_atom x (n_sample, n_input, ...)``.

        Returns:
            list [~chainer.Variable]:
                Differentiated output data. The shape of data is
                ``n_atom x (n_sample, n_output, ...)``.
        """
        return [F.einsum('soi,si...->so...',
                         nnp.first_derivative(x, y),
                         dx)
                for nnp, x, y, dx in zip(self, xs, ys, dxs)]


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
