.. module:: hdnnpy.training

Chainer-based training tools
============================

Custom training extensions
--------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~extensions.ScatterPlot
    ~extensions.set_log_scale


Loss functions
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~loss_function.Zeroth
    ~loss_function.First
    ~loss_function.Mix
    ~loss_function.Potential

Loss function base class
------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~loss_function.loss_function_base.LossFunctionBase


Training manager
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~manager.Manager


Updater
-------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    ~updater.Updater
