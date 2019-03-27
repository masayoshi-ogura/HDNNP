How to extend HDNNP
===================

.. contents::
   :local:
   :depth: 2


Dataset
-------------------

HDNNP dataset consists of **Descriptor dataset** and **Property dataset**.




Descriptor dataset
^^^^^^^^^^^^^^^^^^^

| Currently, we have implemented only **symmetry function** dataset.
| If you want to use other descriptor dataset, define a class that inherits
| ``hdnnpy.dataset.descriptor.descriptor_dataset_base.DescriptorDatasetBase``
| It defines several instance variables, properties and instance methods for creating a HDNNP dataset.

In addition, override the following abstract method.

* generate_feature_keys

| It returns a list of unique keys in feature dimension.
| In addition to being able to use it internally,
  it is also used to expand feature dimension and zero-fill in ``hdnnpy.dataset.HDNNPDataset``

* calculate_descriptors

| It is main function for calculating descriptors from a atomic structure, which is a wrapper of ase.Atoms object.





Property dataset
^^^^^^^^^^^^^^^^^^^

| Currently, we have implemented only **interatomic potential** dataset.
| If you want to use other property dataset, define a class that inherits
| ``hdnnpy.dataset.property.property_dataset_base.PropertyDatasetBase``
| It defines several instance variables, properties and instance methods for creating a HDNNP dataset.

In addition, override the following abstract method.

* calculate_properties

| It is main function for getting properties from a atomic structure, which is a wrapper of ase.Atoms object.


Preprocess
-------------------

* PCA
* Scaling
* Standardization


Loss function
-------------------

Currently, we have implemented following loss function for HDNNP training.

* Zeroth
* First

Each loss function uses a 0th/1st order error of property to optimize HDNNP.
``First`` uses both 0th/1st order errors of property weighted by parameter ``mixing_beta`` to optimize HDNNP.

* Potential

It uses 2nd order derivative of descriptor dataset to optimize HDNNP to satisfy following condition:

.. math::

    \rot \bm{F} = 0

Then, there is a scalar potential :math:`\varphi`:

.. math::

    \bm{F} = \mathrm{grad} \varphi

| If you want to use other loss function, define a class that inherits
| ``hdnnpy.training.loss_function.loss_function_base.LossFunctionBase``.
| It defines several instance variables, properties and instance methods.
