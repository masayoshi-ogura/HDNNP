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

* make

| It is main function for calculating descriptors from atomic structures and parameters.
| The calculated descriptors are stored in instance variable ``self._dataset`` as a list of Numpy arrays.





Property dataset
^^^^^^^^^^^^^^^^^^^

| Currently, we have implemented only **interatomic potential** dataset.
| If you want to use other property dataset, define a class that inherits
| ``hdnnpy.dataset.property.property_dataset_base.PropertyDatasetBase``
| It defines several instance variables, properties and instance methods for creating a HDNNP dataset.

In addition, override the following abstract method.

* make

| It is main function for getting properties from atomic structures, which is a wrapper of ase.Atoms object.
| The obtained properties are stored in instance variable ``self._dataset`` as a list of Numpy arrays.


Preprocess
-------------------

* PCA
* Scaling
* Standardization


Loss function
-------------------

Currently, we have implemented following loss function for HDNNP training.

* zeroth_only
* first_only

Each loss function uses a 0th/1st order error of property to optimize HDNNP.

* mix

It uses both 0th/1st order errors of property weighted by parameter ``mixing_beta`` to optimize HDNNP.

If you want to use other loss function, define a function of following form:

.. code-block:: python

   def your_loss_function(model, properties, **kwargs):
       parameterA = kwargs['keyA']
       parameterB = kwargs['keyB']
       observation_keys = ['metricsA', 'metricsB']

       def loss_function(*datasets):
           half = len(datasets) // 2
           inputs, labels = datasets[:half], datasets[half:]
           predictions = model(inputs)
           loss = ...
           observation = {
               observation_keys[0]: ...,
               observation_keys[1]: ...,
               }
           chainer.report(observation, observer=model)
           return loss

       return loss_function, observation_keys


