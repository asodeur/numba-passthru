from contextlib import contextmanager
import gc
from numba import jit, objmode, typed, types, TypingError
from numba.core import cgutils
from numba.core.config import MACHINE_BITS
from numba.core.datamodel import models
from numba.extending import box, NativeValue, register_model, typeof_impl, unbox, make_attribute_wrapper
from numba.core.runtime.nrt import rtsys
from numba_passthru import PassThruContainer, PassThruType, pass_thru_type
import pytest
from sys import getrefcount


@contextmanager
def check_numba_allocations(self, create_tracked_objects=lambda: {}, extra_allocations=0, refcount_changes={}):
    track_refcounts = create_tracked_objects()
    refcounts = {k: getrefcount(v) for k, v in track_refcounts.items()}
    for k, v in refcount_changes.items():
        refcounts[k] += v

    before = rtsys.get_allocation_stats()

    try:
        tracked = tuple(v for k, v in sorted(track_refcounts.items()))
        yield tracked
        del tracked

        refcounts_after = {k: getrefcount(v) for k, v in track_refcounts.items()}

        after = rtsys.get_allocation_stats()
        del track_refcounts

        assert refcounts == refcounts_after
        assert after.alloc - before.alloc - extra_allocations == after.free - before.free
    finally:
        # try to trigger any __del__ that might not been called b/c of exceptions,
        # otherwise these show up randomly later
        gc.collect()


class TestPassThruContainer:
    def test_identity(self):
        obj = dict(a=1)

        @jit(nopython=True)
        def container_identity(c):
            with objmode(a='intp'):
                c.obj['b'] = 2
                a = c.obj['a']

            return c, a

        with check_numba_allocations(self, (lambda: dict(c=PassThruContainer(obj)))) as (c,):
            r, a = container_identity(c)

            assert c is r
            assert a == 1
            assert c.obj['a'] == 1
            assert c.obj['b'] == 2
            del c, r

    def test_forget(self):
        obj = dict(a=1)

        @jit(nopython=True)
        def forget_container(c):
            with objmode(a="intp"):
                a = c.obj['a']
                c.obj['b'] = 2

            return a

        with check_numba_allocations(self, (lambda: dict(c=PassThruContainer(obj)))) as (c,):
            a = forget_container(c)

            assert a == 1
            assert c.obj['a'] == 1
            assert c.obj['b'] == 2
            del c

    def test_eq(self):
        obj1 = dict(a=1)
        obj2 = dict(A=1)

        @jit(nopython=True)
        def container_eq(c1, c2, c3, c4):
            with objmode(a="intp", A="intp"):
                a = c1.obj['a']
                A = c3.obj['A']
                c1.obj['b'] = 2
                c3.obj['b'] = 2

            return (
                a, A, c1 == c1, c1 == c2, c1 == c3, c1 == c4, c2 == c2, c2 == c3, c2 == c4, c3 == c3, c3 == c4, c4 == c4
            )

        with check_numba_allocations(
                self,
                (
                    lambda: dict(
                        c1=PassThruContainer(obj1),
                        c2=PassThruContainer(obj1),
                        c3=PassThruContainer(obj2),
                        c4=PassThruContainer(obj2)
                    )
                )
        ) as (c1, c2, c3, c4):
            a, A, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 = container_eq(c1, c2, c3, c4)

            assert a == 1
            assert A == 1
            assert (
                (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10) ==
                (c1 == c1, c1 == c2, c1 == c3, c1 == c4, c2 == c2, c2 == c3, c2 == c4, c3 == c3, c3 == c4, c4 == c4)
            )
            assert c1.obj['a'] == 1
            assert c1.obj['b'] == 2
            assert c3.obj['A'] == 1
            assert c3.obj['b'] == 2
            del c1, c2, c3, c4

    def test_eq_any(self):
        obj1 = dict(a=1)
        c1=PassThruContainer(obj1)

        @jit(nopython=True)
        def container_eq_any(x, y):
            return x == y

        with pytest.raises(NotImplementedError):
            c1 == 1

        with pytest.raises(TypingError) as context:
            r = container_eq_any(c1, 1)

        msg = str(context.value)
        assert (
            'Failed in nopython mode pipeline (step: nopython frontend)' in msg
            and 'No implementation of function Function(<built-in function eq>) found for signature:' in msg
            and 'eq(PassThruContainerType, int64)' in msg
        )

        with pytest.raises(NotImplementedError):
            1 == c1

        with pytest.raises(TypingError) as context:
            r = container_eq_any(1, c1)

        msg = str(context.value)
        assert (
            'Failed in nopython mode pipeline (step: nopython frontend)' in msg
            and 'No implementation of function Function(<built-in function eq>) found for signature:' in msg
            and 'eq(int64, PassThruContainerType)' in msg
        )

    def test_hash(self):
        obj = dict(a=1)

        @jit(nopython=True)
        def container_hash(c):
            return hash(c)

        with check_numba_allocations(self, (lambda: dict(c=PassThruContainer(obj)))) as (c,):
            res = container_hash(c)

            assert hash(c) == res
            del c


################################ MyPassThru ################################
# simple pass through extension type, no attributes accessible from nopython
# use case for this might be to dispatch on this type
############################################################################
class MyPassThru(object):
    def __init__(self):
        self.something_not_boxable = object()


my_pass_thru_type = PassThruType('MyPassThruType')


@typeof_impl.register(MyPassThru)
def type_my_passthru(val, context):
    return my_pass_thru_type


class TestMyPassThru:
    def test_forget(self):
        with check_numba_allocations(self, (lambda: dict(o=MyPassThru()))) as (o,):
            _id = forget(o)
            assert _id == id(o.something_not_boxable) & (2**MACHINE_BITS - 1)
            del o

    def test_identity(self):
        with check_numba_allocations(self, (lambda: dict(o=MyPassThru()))) as (o,):
            o2 = identity(o)
            assert o is o2
            del o, o2

    def test_doubling(self):
        with check_numba_allocations(self, (lambda: dict(o=MyPassThru()))) as (o,):
            o2, o3 = double(o)
            assert o is o2
            assert o is o3
            del o, o2, o3

    def test_list_forget(self):
        @jit(nopython=True)
        def forget_list(x):
            return 1

        def create_tracked():
            o = MyPassThru()
            l = typed.List([o, o, o])

            return dict(l=l, o=o)

        with check_numba_allocations(self, create_tracked) as (l, o):
            one = forget_list(l)
            assert one is 1
            del o, l

    def test_list_identity(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            l = typed.List([x, y, z])
            l2 = identity(l)
            for ii in range(3):
                assert l[ii] is l2[ii]
            del x, y, z, l, l2

    def test_list_create(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru()))) as (x, y):
            l = create_passthru_list(x, y)

            assert l[0] is x
            assert l[1] is y

            del l, x, y

    def test_list_copy(self):
        @jit(nopython=True)
        def copy_list(x):
            the_copy = typed.List()
            for ii in range(len(x)):
                v = x[ii]
                the_copy.append(v)

            return the_copy  # y

        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru()))) as (x, y):
            l1 = typed.List([x, y])

            l2 = copy_list(l1)

            assert l2[0] is x
            assert l2[1] is y

            del l1, l2, x, y

    def test_list_doubling(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            l = typed.List([x, y, z])

            l2, l3 = double(l)
            for ii in range(3):
                assert l2[ii] is l[ii]
                assert l3[ii] is l[ii]

            del x, y, z, l, l2, l3


############################# PassThruComplex #############################
# pass through extension type with several attrs accessible from nopython,
# including another pass through type and a list, changes to any attribute
# from the nopython side will NOT be reflected back to Python. The direct
# members are read-only in nopython but there is currently no way to stop
# mutating, for example, the list attribute in nopython.
###########################################################################

class PassThruComplex(object):
    def __init__(self, int_attr, passthru_attr, list_attr):
        self.int_attr = int_attr
        self.passthru_attr = passthru_attr
        assert isinstance(list_attr, typed.List)
        self.list_attr = list_attr

        self.something_not_boxable = object()


class PassThruComplexType(PassThruType):
    def __init__(self):
        super(PassThruComplexType, self).__init__()


@register_model(PassThruComplexType)
class PassThruComplexModel(models.StructModel):
    def __init__(self, dmm, fe_typ):
        members = [
            ('parent', pass_thru_type),
            ('int_attr', types.intp),
            ('passthru_attr', pass_thru_type),
            ('list_attr', types.ListType(pass_thru_type))
        ]
        super(PassThruComplexModel, self).__init__(dmm, fe_typ, members)


make_attribute_wrapper(PassThruComplexType, 'int_attr', 'int_attr')
make_attribute_wrapper(PassThruComplexType, 'passthru_attr', 'passthru_attr')
make_attribute_wrapper(PassThruComplexType, 'list_attr', 'list_attr')


@typeof_impl.register(PassThruComplex)
def type_my_pass_thru_complex(val, context):
    return PassThruComplexType()


@unbox(PassThruComplexType)
def unbox_passthru_complex(typ, obj, context):
    pass_thru = cgutils.create_struct_proxy(typ)(context.context, context.builder)

    pass_thru.parent = context.unbox(pass_thru_type, obj).value

    int_attr = context.pyapi.object_getattr_string(obj, 'int_attr')
    pass_thru.int_attr = context.unbox(types.intp, int_attr).value
    context.pyapi.decref(int_attr)

    passthru_attr = context.pyapi.object_getattr_string(obj, 'passthru_attr')
    pass_thru.passthru_attr = context.unbox(pass_thru_type, passthru_attr).value
    context.pyapi.decref(passthru_attr)

    list_attr = context.pyapi.object_getattr_string(obj, 'list_attr')
    pass_thru.list_attr = context.unbox(types.ListType(pass_thru_type), list_attr).value
    context.pyapi.decref(list_attr)

    # should have done this more often, errors most likely erased by subsequent pyapi call anyway ...
    is_error = cgutils.is_not_null(context.builder, context.pyapi.err_occurred())

    return NativeValue(pass_thru._getvalue(), is_error=is_error)


@box(PassThruComplexType)
def box_passthru_complex(typ, val, context):
    pass_thru = cgutils.create_struct_proxy(typ)(context.context, context.builder, value=val)

    # just unwrap the parent Python object, no attempt at reflecting attribute mutations back
    # incref pass_thru.parent, then box pass_thru.parent (which will decref again),
    # finally decref val
    context.context.nrt.incref(context.builder, pass_thru_type, pass_thru.parent)
    obj = context.box(pass_thru_type, pass_thru.parent)

    context.context.nrt.decref(context.builder, typ, val)

    return obj


class TestPassThruComplex:
    def test_forget(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, typed.List([y, z]))

            _id = forget(o)
            assert _id == id(o.something_not_boxable) & (2**MACHINE_BITS - 1)
            del o, x, y, z

    def test_identity(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, typed.List([y, z]))

            o2 = identity(o)
            assert o is o2

            del o, o2, x, y, z

    def test_doubling(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, typed.List([y, z]))

            o2, o3 = double(o)
            assert o is o2
            assert o is o3

            del o, o2, o3, x, y, z

    def test_attr_access(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, typed.List([y, z]))

            value, x2, (y2, z2) = attr_access(o, 'int_attr', 'passthru_attr', 'list_attr')

            assert value == 42
            assert x2 is x
            assert y2 is y
            assert z2 is z

            del o, x, x2, y, y2, z, z2

    def test_list_create(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o1 = PassThruComplex(42, x, typed.List([y, z]))
            o2 = PassThruComplex(43, x, typed.List([y, z]))

            l = create_passthru_list(o1, o2)

            assert o1.int_attr == 42
            assert o2.int_attr == 43
            assert o1.passthru_attr is x
            assert o1.list_attr[0] is y
            assert o1.list_attr[1] is z

            del l, o1, o2, x, y, z

    def test_list_copy(self):
        @jit(nopython=True)
        def copy_list(x):
            the_copy = []
            for ii in range(len(x.list_attr)):
                v = x.list_attr[ii]
                the_copy.append(v)

            return the_copy  # y

        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, typed.List([y, z]))

            l2 = copy_list(o)

            assert l2[0] is y
            assert l2[1] is z

            del o, l2, x, y, z

    def test_list_identity(self):
        with check_numba_allocations(self, (lambda: dict(x=MyPassThru(), y=MyPassThru(), z=MyPassThru()))) as (x, y, z):
            o = PassThruComplex(42, x, typed.List([y, z]))

            l = typed.List([o, o, o])
            l2 = identity(l)
            for ii in range(3):
                assert l[ii] is l2[ii]

            del o, x, y, z, l, l2

    def test_list_forget(self):
        @jit(nopython=True)
        def forget_list(x):
            return 1

        def create_tracked():
            x = MyPassThru()
            y = MyPassThru()
            z = MyPassThru()
            o = PassThruComplex(42, x, typed.List([y, z]))
            l = typed.List([o, o, o])

            return dict(l=l, o=o, x=x, y=y, z=z)

        with check_numba_allocations(self, create_tracked) as (l, o, x, y, z):
            one = forget_list(l)
            one = forget_list(l)
            # one = forget_list(l)
            assert one is 1
            del x, y, z, l, o

    def test_pass_thru_type_eq(self):
        def create_tracked():
            x = MyPassThru()
            y = MyPassThru()
            z = MyPassThru()
            a = PassThruComplex(42, x, typed.List([y, z]))
            b = PassThruComplex(42, y, typed.List([z, y]))

            return dict(a=a, b=b)

        @jit(nopython=True)
        def pass_thru_type_eq(a, b):
            return a.passthru_attr == b.passthru_attr, a.passthru_attr != b.passthru_attr

        with check_numba_allocations(self, create_tracked) as (x, y):
            for a in (x, y):
                for b in (x, y):
                    assert pass_thru_type_eq(a, b) == pass_thru_type_eq.py_func(a, b)

            del a, b, x, y


############################ test functions ############################
# test function for basic allocation tests
########################################################################
@jit(nopython=True)
def forget(o):
    with objmode(x='uintp'):
        # TODO: cannot get 32bit to work w/o truncation
        x = id(o.something_not_boxable) & (2**MACHINE_BITS - 1)

    return x


@jit(nopython=True)
def identity(x):
    return x


@jit(nopython=True)
def double(x):
    return x, x


def attr_access(o, *names):
    body = ", ".join("o.{}".format(n) for n in names)
    f = jit(nopython=True)(eval("lambda o: ({})".format(body)))

    return f(o)


@jit(nopython=True)
def create_passthru_list(list_attr1, list_attr2):
    return [list_attr1, list_attr2]
