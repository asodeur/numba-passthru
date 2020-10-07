Pass-through type for Numba
===========================

Tools to ferry arbitrary Python objects through `nopython` mode. This is a stand-alone version of the Numba internal
implementation [PR 3640](https://github.com/numba/numba/pull/3640). 

This has two typical use-cases:
  1. ferrying data structures not currently supported by Numba into `objmode` blocks via `PassThruContainer`
  2. creating extension types that are simplified representations of the Python class and keep a link to the
     Python object
     
It's not unlikely both can be avoided completely when starting from scratch but potentially require extensive
refactoring when moving Numba into an existing code base.

Ferrying objects into `objmode` using `PassThruContainer`
---------------------------------------------------------
`PassThruContainer` can be used to make Python objects not supported by `nopython` available inside
`objmode` blocks. Outside `objmode` blocks the only supported operation on `PassThruContainer` 
is `==`. Two instances are equal if the wrapped objects are identical, ie `a.obj is b.obj`.

In the following example an object unsupported in `nopython` is ferried into an `objmode` block and
mutated there. 
```python
from __future__ import annotations
from numba import jit, objmode
from numba_passthru import PassThruContainer

class Testee:
    def __init__(self, value: int, invert=False):
        self.value = value
        self.numba_will_not_like_this = compile('pass', 'N/A', 'single')
        
    def __gt__(self, other):    # will be used in next example
        return self.value > other.value

testee = Testee(1)  
container = PassThruContainer(testee)

@jit(nopython=True)
def do_something(container):
    with objmode(value='int64'):
        setattr(container.obj, 'value_2', 2)
        value = container.obj.value
        
    return container, value
    
result, value = do_something(container)

assert container is result
assert value == 1
assert result.obj.value_2 == 2
```

There will be no speed-up for the code inside the `objmode` block and `container` is (un)boxed twice adding further
overhead. Hence, this only makes sense in rarely visited code-path and if refactoring into a more Numba friendly
 form is not an option.
 
Note that the example above already contains the most common pattern that is pretty unpleasant to refactor into a 
Numba friendly form in requiring object identity being preserved through `nopython` (`assert container is result`).
Obviously, this is a highly artificial requirement in this toy example but might get more real if the pass-through
object is part of conditional branches. 

Creating custom passthrough types
---------------------------------
`PassThroughContainer` does not allow attribute access on the wrapped object in `nopython` and there is no way
to dispatch on the type of the wrapped object. To get both you can create a Numba extension type using `pass_thru_type`.
`pass_thru_type` holds a pointer to the `PyObject` and manages references. `pass_thru_type` can be used like any 
mem-managed member on an [extension type](http://numba.pydata.org/numba-doc/latest/extending/index.html). (Some
familiarity with Numba extension types is expected for the following.)  
 
Continuing the example above let's try to get the following code working in `nopython` (another toy
example, no speed-up expected):
```python
def find_max(testees: List[Testee]) -> Testee:
    result = testees[0]  # testees must not be empty
    for testee in testees[1:]:
        if testee > result:
            result = testee

    return result     
``` 
`PassThroughContainer` will not help here as there would be no way to dispatch `>`  to `Testee.__gt__` and there would
be no way to access `.value` from `nopython` inside `Testee.__gt__`. Still, since `Testee.value` is the only attribute 
being accessed from `nopython` there is a realistic chance to get this working. Indeed, assuming we already had the 
(un)boxer this is a straight forward Numba extension type:
```python
from numba import jit, types
from numba.extending import overload
from numba.typing.typeof import typeof_impl
import operator

class TesteeType(PassThruType):
    def __init__(self):
        super(TesteeType, self).__init__(name='Testee')

testee_type = TesteeType()

@typeof_impl.register(Testee)
def type_testee(val, context):
    return testee_type

@overload(operator.gt)
def testee_gt(self, other):
    if self is testee_type and other is testee_type:
        return Testee.__gt__

find_max = jit(nopython=True)(find_max)
```

Trying to implement the (un)boxer to somehow pass the `.numba_will_not_like_this` attribute around `nopython` (sharing
a dict between boxer/unboxer etc) is not straight forward to get working for `find_max` alone and it is impossible
to get the reference counting right in the general case. The clean approach is to have the Numba runtime manage the
references by putting a NRT managed reference to the original Python object onto the extension type's data model.  

`pass_thru_type` helps with the boiler-plate of boxing/unboxing the required `MemInfoPointer`. The `PyObject` 
passed into the unboxer can be unboxed directly into a `pass_thru_type`. On the way out the original `PyObject` is 
recovered  by boxing the `pass_thru_type`. 

```python
from numba import cgutils
from numba.datamodel import models
from numba.extending import make_attribute_wrapper, overload, overload_method, register_model
from numba.pythonapi import NativeValue, unbox, box
from numba.targets.hashing import _Py_hash_t

from numba_passthru import pass_thru_type

@register_model(TesteeType)
class TesteeModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = [
            ('parent', pass_thru_type),
            ('value', types.intp),
        ]
        super(TesteeModel, self).__init__(dmm, fe_typ, members)
        
make_attribute_wrapper(TesteeType, 'value', 'value')

@unbox(TesteeType)
def unbox_testee(typ, obj, context):
    testee = cgutils.create_struct_proxy(typ)(context.context, context.builder)
    
    testee.parent = context.unbox(pass_thru_type, obj).value
    
    value = context.pyapi.object_getattr_string(obj, "value")
    native_value = context.unbox(types.intp, value)
    context.pyapi.decref(value)

    testee.value = native_value.value

    is_error = cgutils.is_not_null(context.builder, context.pyapi.err_occurred())
    return NativeValue(testee._getvalue(), is_error=is_error)
    
@box(TesteeType)
def box_testee(typ, val, context):
    val = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)
    obj = context.box(pass_thru_type, val.parent)

    return obj
```
Given the implementation above `TesteeType` is immutable from `nopython` (`make_attribute_wrapper` creates read-only
attributes). If you made a pass-through type mutable from `nopython` you had to make sure to reflect changes back
to the Python object in the boxer. However, given the [experience with reflected lists and sets](http://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types)
there are good reasons to be careful about this. 

Upward compatibility notice
---------------------------
This is a stand-alone version of Numba [PR 3640](https://github.com/numba/numba/pull/3640). Import of
`PassThruType`, `pass_thru_type`, `PassThruContainer`, `pass_thru_container_type` from `numba` is attempted first 
hence you will get the Numba internal implementations once the PR has landed.

This package contains an overload of `int(Opaque)` (essentially `ptrtoint`) that might break future Numba versions 
if Numba created diverging implementations.

This was considered too unlikely to put a version constraint on the Numba dependency (which would require a new release
of `numba-passthru` every time a new Numba versions is released)
