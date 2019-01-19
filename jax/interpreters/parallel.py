# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as onp
import six

from .. import core
from .. import linear_util as lu
from ..core import Trace, Tracer, Primitive, new_master
from ..abstract_arrays import ShapedArray, ConcreteArray, make_shaped_array
from ..util import safe_zip, unzip2, unzip3, partialmethod, prod
from ..lib import xla_bridge as xb
from . import partial_eval as pe
from . import xla
from . import batching

zip = safe_zip

def identity(x): return x


### pmap


def pmap(fun, name, in_vals, in_axes, out_axis_target):
  sizes = reduce(set.union, map(batching.dimsize, in_axes, in_vals))
  if not sizes:
    return fun.call_wrapped(*in_vals)
  elif len(sizes) == 1:
    size = sizes.pop()
    if size % xb.get_replica_count():
      msg = ("pmap requires mapped axis to be divisible by num_replicas, "
             "got axis size {} for {} replicas.")
      raise TypeError(msg.format(size, xb.get_replica_count()))

    out_val, out_axis = pmap_transform(fun).call_wrapped(name, in_vals, in_axes)
    return batching.moveaxis(size, out_axis_target, out_axis, out_val)
  else:
    raise TypeError("got inconsistent map dimension sizes: {}".format(sizes))

@lu.transformation
def pmap_transform(name, vals, axes):
  with new_master(PmapTrace) as master:
    trace = PmapTrace(master, core.cur_sublevel())
    in_tracers = map(partial(PmapTracer, trace, name), vals, axes)
    ans = yield in_tracers
    out_tracer = trace.full_raise(ans)
    out_val, out_axis = out_tracer.val, out_tracer.axis
    del master
  yield out_val, out_axis

@lu.transformation_with_aux
def pmap_subtrace(master, name, axes, *vals):
  trace = PmapTrace(master, core.cur_sublevel())
  ans = yield map(partial(PmapTracer, trace, name), vals, axes)
  out_tracer = trace.full_raise(ans)
  out_val, out_axis = out_tracer.val, out_tracer.axis
  yield out_val, out_axis

class PmapTracer(Tracer):
  def __init__(self, trace, name, val, axis):
    self.trace = trace
    self.name = name
    self.val = val
    self.axis = axis

  @property
  def aval(self):
    batched_aval = batching.get_aval(self.val)
    return batching.remove_batch_dim_from_aval(self.axis, batched_aval)

  def unpack(self):
    t = type(self.axis)
    if t is tuple:
      axes = self.axis
    elif t is int:
      axes = [self.axis] * len(self.val)
    elif t is type(None):
      return tuple(self.val)
    else:
      raise TypeError(t)
    return map(partial(PmapTracer, self.trace, self.name), self.val, axes)

  def full_lower(self):
    if self.axis is None:
      return core.full_lower(self.val)
    else:
      return self

class PmapTrace(Trace):
  def pure(self, val):
    return PmapTracer(self, None, val, None)

  def lift(self, val):
    return PmapTracer(self, None, val, None)

  def sublift(self, val):
    return PmapTracer(self, val.name, val.val, val.axis)

  def process_primitive(self, primitive, tracers, params):
    names_in, vals_in, axes_in = unzip3((t.name, t.val, t.axis) for t in tracers)
    if all(axis is None for axis in axes_in):
      return primitive.bind(*vals_in, **params)
    else:
      # act just like vmap except when we hit a collective reduction
      if primitive in pmap_primitive_rules:
        tracer, = tracers
        val, = vals_in
        axis, = axes_in
        name = params['axis_name']

        assert axis is not None
        if tracer.name == name:
          return pmap_primitive_rules[primitive](val, axis)
        else:
          return primitive.bind(val, axis_name=name)
      else:
        rule = batching.get_primitive_batcher(primitive)
        val_out, axis_out = rule(vals_in, axes_in, **params)
        name = next(name for name in names_in if name is not None)
        return PmapTracer(self, name, val_out, axis_out)

  def process_call(self, call_primitive, f, tracers, params):
    # TODO do something special for xla_call if it's abstracted over 'name'
    names, vals, axes = unzip3((t.name, t.val, t.axis) for t in tracers)
    if all(axis is None for axis in axes):
      return call_primitive.bind(f, *vals, **params)
    else:
      name = next(name for name in names if name is not None)
      f, axis_out = pmap_subtrace(f, self.master, name, axes)
      val_out = call_primitive.bind(f, *vals, **params)
      return PmapTracer(self, self.name, val_out, axis_out())

  def post_process_call(self, _, out_tracer):
    raise NotImplementedError  # TODO(mattjj,dougalm)

  def pack(self, tracers):
    vals = pack([t.val for t in tracers])
    axis = tuple(t.axis for t in tracers)
    return PmapTracer(self, self.name, vals, axis)


def unbound_name_error(x, axis_name):
  raise NameError("axis name '{}' is unbound".format(axis_name))

def PmapPrimitive(name):
  prim = Primitive(name)
  prim.def_impl(unbound_name_error)
  return prim


pmap_primitive_rules = {}


def psum(x, axis_name):
  return psum_p.bind(x, axis_name=axis_name)

psum_p = PmapPrimitive('psum')
pmap_primitive_rules[psum_p] = lambda val, axis: val.sum(axis)


### papply

newvar = pe.gensym('_axis')

def papply(fun, in_vals, in_axes):
  sizes = reduce(set.union, map(batching.dimsize, in_axes, in_vals))
  if not sizes:
    return fun.call_wrapped(*in_vals)
  elif len(sizes) == 1:
    name = newvar()
    size = sizes.pop()
    if size % xb.get_replica_count():
      msg = ("papply requires mapped axis to be divisible by num_replicas, "
             "got axis size {} for {} replicas.")
      raise TypeError(msg.format(size, xb.get_replica_count()))

    out_val = papply_transform(fun).call_wrapped(name, in_vals, in_axes)
    return out_val, name
  else:
    raise TypeError("got inconsistent map dimension sizes: {}".format(sizes))

@lu.transformation
def papply_transform(name, args, axes):
  with new_master(PapplyTrace) as master:
    trace = PapplyTrace(master, core.cur_sublevel())
    in_tracers = map(partial(PapplyTracer, trace, name), args, axes)
    out_tracer = yield in_tracers
    out_tracer = trace.full_raise(out_tracer)
    out_val = out_tracer.val
    del master
  yield out_val

class PapplyTracer(Tracer):
  def __init__(self, trace, name, val, axis):
    self.trace = trace
    self.name = name
    self.val = val
    self.axis = axis

  @property
  def aval(self):
    return batching.get_aval(self.val)

  def unpack(self):
    raise NotImplementedError  # TODO

  def full_lower(self):
    if self.axis is None:
      return core.full_lower(self.val)
    else:
      return self

class PapplyTrace(Trace):
  def pure(self, val):
    return PapplyTrace(self, None, val, None)

  def lift(self, val):
    return PapplyTracer(self, None, val, None)

  def sublift(self, val):
    return PapplyTracer(self, val.name, val.val, val.axis)

  def process_primitive(self, primitive, tracers, params):
    names, vals, axes = unzip3((t.name, t.val, t.axis) for t in tracers)
    if all(axis is None for axis in axes):
      return primitive.bind(*vals, **params)
    else:
      rule = papply_primitive_rules[primitive]
      name, val, axis = rule(names, vals, axes)
      return PapplyTracer(self, name, val, axis)

  def process_call(self, call_primitive, f, tracers, params):
    raise NotImplementedError  # TODO(mattjj)

  def post_process_call(self, _, out_tracer):
    raise NotImplementedError  # TODO(mattjj)

  def pack(self, tracers):
    vals = core.pack([t.val for t in tracers])
    axis = tuple(t.axis for t in tracers)
    name = tuple(t.name for t in tracers)
    return PapplyTracer(self, name, vals, axis)


papply_primitive_rules = {}


def defvectorized(prim):
  papply_primitive_rules[prim] = partial(vectorized_papply, prim)

def vectorized_papply(prim, vals, dims, **params):
  assert all(dims[0] == d for d in dims[1:])
  return prim.bind(*vals, **params), dims[0]



# def parallel_xla_call_impl(fun, *args, **kwargs):
#   in_axes = kwargs.pop('in_axes')
#   assert not kwargs
#   compiled_fun = replicated_callable(fun, in_axes, *map(xla.abstractify, args))
#   return compiled_fun(*args)

# parallel_xla_call_p = Primitive('parallel_xla_call')
# parallel_xla_call = partial(core.call_bind, parallel_xla_call_p)
# parallel_xla_call_p.def_custom_bind(parallel_xla_call)
# parallel_xla_call_p.def_impl(parallel_xla_call_impl)

# xla.translations[parallel_xla_call_p] = xla.xla_call_translation_rule


# @lu.memoize
# def replicated_callable(fun, in_axes, *abstract_args):
#   fun, out_axis = papply_transform(fun, in_axes)
#   pvals = [pe.PartialVal((aval, core.unit)) for aval in abstract_args]
#   with core.new_master(pe.JaxprTrace, True) as master:
#     jaxpr, (pval, consts, env) = pe.trace_to_subjaxpr(fun, master).call_wrapped(pvals)
#     assert not env  # no subtraces here (though cond might eventually need them)
#     sharded_avals = map(shard_aval, in_axes, abstract_args)
#     compiled, sharded_result_shape = xla.compile_jaxpr(jaxpr, consts, *sharded_avals)
#     del master, pvals, consts, jaxpr, env
#   handle_result = result_handler(out_axis(), sharded_result_shape)
#   return partial(execute_replicated, compiled, in_axes, pval, handle_result)


# ### sharded device values


# def device_put(shards):
#   if type(shards) is ShardedDeviceArray:
#     return shards.device_buffers
#   elif type(shards) is list and type(shards[0]) is onp.ndarray:
#     num_replicas = xb.get_replica_count()
#     return list(map(xb.device_put, shards, range(num_replicas)))
#   else:
#     raise NotImplementedError  # TODO

# class ShardedDeviceValue(object):
#   __slots__ = ["device_buffers"]
#   def __init__(self, axis, device_buffers):
#     self.axis = axis
#     self.device_buffers = device_buffers

# class ShardedDeviceArray(ShardedDeviceValue):
#   __slots__ = ["shape", "dtype", "ndim", "size", "_npy_value"]
#   __array_priority__ = 100.

#   def __init__(self, axis, device_buffers, shape, dtype, ndim, size):
#     self.axis = axis
#     self.device_buffers = device_buffers

#     self.shape = shape
#     self.dtype = dtype
#     self.ndim = ndim
#     self.size = size

#     self._npy_value = None

#   @property
#   def _value(self):
#     if self._npy_value is None:
#       npy_shards = [buf.to_py() for buf in self.device_buffers]
#       self._npy_value = stack_pyval_shards(self.axis, npy_shards)
#     return self._npy_value

#   def copy(self):
#     """Returns an ndarray (backed by host memory, not device memory)."""
#     return onp.asarray(self)

#   def __len__(self):
#     try:
#       return self.shape[0]
#     except IndexError:
#       raise TypeError("len() of unsized object")  # same as numpy error

#   def __format__(self, format_spec):
#     # Simulates behavior of https://github.com/numpy/numpy/pull/9883
#     if self.ndim == 0:
#       return format(self._value[()], format_spec)
#     else:
#       return format(self._value, format_spec)

#   __array__ = partialmethod(xla.forward_to_value, onp.asarray)
#   __str__ = partialmethod(xla.forward_to_value, str)
#   __repr__ = partialmethod(xla.forward_to_value, repr)
#   __bool__ = __nonzero__ = partialmethod(xla.forward_to_value, bool)
#   __float__ = partialmethod(xla.forward_to_value, float)
#   __int__ = partialmethod(xla.forward_to_value, int)
#   if six.PY2:
#     __long__ = partialmethod(xla.forward_to_value, long)  # noqa: F821
#   __complex__ = partialmethod(xla.forward_to_value, complex)
#   __hex__ = partialmethod(xla.forward_to_value, hex)
#   __oct__ = partialmethod(xla.forward_to_value, oct)

#   def __hash__(self):
#     # TODO(mattjj): this is not semantically correct because it is possible
#     # __eq__ is true for values with unequal __hash__ values. However, the
#     # main use case at the moment is memoization for which false negatives are
#     # fine.
#     return id(self)

# def stack_pyval_shards(axis, shards):
#   assert shards
#   t = type(shards[0])
#   if t is onp.ndarray:
#     return onp.concatenate(shards, axis)
#   elif t is tuple:
#     return tuple(map(partial(stack_pyval_shards, axis), shards))
#   else:
#     raise TypeError(t)

# core.pytype_aval_mappings[ShardedDeviceArray] = ConcreteArray  # TODO
# xla.pytype_aval_mappings[ShardedDeviceArray] = make_shaped_array
# xla.canonicalize_dtype_handlers[ShardedDeviceArray] = identity




# # Replicated execution


# def execute_replicated(compiled, in_axes, pval, handle_result, *args):
#   sharded_args = zip(*map(shard_arg, in_axes, args))
#   input_bufs = [device_put(sharded_arg) for sharded_arg in sharded_args]
#   out_bufs = compiled.ExecutePerReplica(input_bufs)
#   return pe.merge_pvals(handle_result(out_bufs), pval)

# def result_handler(axis, sharded_result_shape):
#   result_shape = unshard_result_shape(axis, sharded_result_shape)
#   if type(result_shape) is xla.ResultArray:
#     def handle_result(out_bufs):
#       return ShardedDeviceArray(axis, out_bufs, *result_shape)
#   else:
#     raise NotImplementedError  # TODO
#   return handle_result

# def shard_aval(axis, aval):
#   if isinstance(aval, ShapedArray):
#     assert aval.shape[axis] % xb.get_replica_count() == 0
#     shard_shape = tuple(s // xb.get_replica_count() if i == axis else s
#                         for i, s in enumerate(aval.shape))
#     return ShapedArray(shard_shape, aval.dtype)
#   else:
#     raise NotImplementedError  # TODO

# def unshard_result_shape(axis, result_shape):
#   if type(result_shape) is xla.ResultArray:
#     shape, dtype, ndim, size = result_shape
#     shape = tuple(s * xb.get_replica_count() if i == axis else s
#                   for i, s in enumerate(shape))
#     return xla.ResultArray((shape, dtype, ndim, size))
#   else:
#     raise NotImplementedError  # TODO

# def shard_arg(axis, x):
#   t = type(x)
#   if t is ShardedDeviceArray:
#     return x
#   elif t is onp.ndarray:
#     num_replicas = xb.get_replica_count()
#     return onp.split(x, num_replicas, axis)
#   elif t in (xla.DeviceArray, xla.DeviceConstant):
#     # TODO(mattjj): can probably improve these implementations
#     return shard_arg(axis, onp.asarray(x))
#   elif t is core.JaxTuple:
#     return list(map(partial(shard_arg, axis), t))
#   else:
#     raise TypeError(t)
