{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :inherited-members:

    {% block inheritance %}
    .. rubric:: {{ _('Inheritance diagram') }}
    .. inheritance-diagram:: {{ objname }}
    {% endblock %}

    {% block methods %}
    {% if methods %}
    .. rubric:: {{ _('Members') }}

    .. autosummary::
    {% for item in members %}
    {% if item not in inherited_members and not item.startswith('_') or item == '__init__' %}
        ~{{name}}.{{ item }}
    {% endif %}
    {% endfor %}

    .. rubric:: {{ _('Inherited members') }}

    .. autosummary::
    {% for item in members %}
    {% if item in inherited_members and not item.startswith('_') %}
    {% if item != '__init__' %}
        ~{{ name }}.{{ item }}
    {% endif %}
    {% endif %}
    {% endfor %}

    {% endif %}
    {% endblock %}
