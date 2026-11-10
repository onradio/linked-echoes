# Template Warning: Avoid `{% set %}` for Dynamic Config Values

## Problem

Using `{% set mode = input_block.mode %}` in Jinja2 **creates a static (frozen) copy** of the value at that moment in rendering.  
If the actual value of `input_block.mode` is changed dynamically (e.g. through config overrides), your `{% set %}` variable will **not reflect the updated value**.

This can lead to subtle and confusing bugs — like rendering the wrong prompt block — especially like the example below when toggling between `"single"` and `"bulk"` modes, etc.

---

## Bad Practice

```jinja2
{% set mode = input_block.mode %}
{% if mode == "bulk" %}
  ... render bulk prompt ...
{% endif %}
