{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

{% if module.startswith(('hdnnpy.model', 'hdnnpy.training')) %}
.. autoclass:: {{ objname }}
    :no-inherited-members:
{% else %}
.. autoclass:: {{ objname }}
{% endif %}
