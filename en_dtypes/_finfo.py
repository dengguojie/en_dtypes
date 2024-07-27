# Copyright 2024 The en_dtypes Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Overload of numpy.finfo to handle dtypes defined in en_dtypes."""

from typing import Dict

from en_dtypes._en_dtypes_ext import float4_e2m1
from en_dtypes._en_dtypes_ext import float4_e1m2
from en_dtypes._en_dtypes_ext import float6_e2m3
from en_dtypes._en_dtypes_ext import float6_e3m2
from en_dtypes._en_dtypes_ext import float8_e8m0
import numpy as np

_float4_e2m1_dtype = np.dtype(float4_e2m1)
_float4_e1m2_dtype = np.dtype(float4_e1m2)

_float6_e2m3_dtype = np.dtype(float6_e2m3)
_float6_e3m2_dtype = np.dtype(float6_e3m2)

_float8_e8m0_dtype = np.dtype(float8_e8m0)


class _Float4E2m1MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p0")  # 1.0
    self.smallest_normal = float4_e2m1(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-1")  # 0.5
    self.smallest_subnormal = float4_e2m1(smallest_subnormal)


class _Float4E1m2MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p0")  # 1.0
    self.smallest_normal = float4_e1m2(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-2")  # 0.25
    self.smallest_subnormal = float4_e1m2(smallest_subnormal)


class _Float6E2m3MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p0")  # 1.0
    self.smallest_normal = float6_e2m3(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-3")  # 0.125
    self.smallest_subnormal = float6_e2m3(smallest_subnormal)


class _Float6E3m2MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-2")  # 0.25
    self.smallest_normal = float6_e3m2(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-4")  # 0.0625
    self.smallest_subnormal = float6_e3m2(smallest_subnormal)


class _Float8E8m0MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-127")  # 2^-127
    self.smallest_normal = float8_e8m0(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-127")  # 2^-127, actually no subnormal
    self.smallest_subnormal = float8_e8m0(smallest_subnormal)


class finfo(np.finfo):  # pylint: disable=invalid-name,missing-class-docstring
  __doc__ = np.finfo.__doc__
  _finfo_cache: Dict[np.dtype, np.finfo] = {}

  @staticmethod
  def _float4_e2m1_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p0")  # smallest_normal = 1
    resolution = 1  # 10**(-precision)
    eps = float.fromhex("0x1p-1")  # 0.5: difference between 1.0 and the next representable value.
    epsneg = float.fromhex("0x1p-1")  # 0.5: difference between 1.0 and the previous representable value.
    max_ = float.fromhex("0x1.8p2")  # 6 = (1+8/16) * 2^2, it must be 0x1.*pN

    obj = object.__new__(np.finfo)
    obj.dtype = _float4_e2m1_dtype
    obj.bits = 4
    obj.eps = float4_e2m1(eps)
    obj.epsneg = float4_e2m1(epsneg)
    obj.machep = -1  # np.log10(eps) / np.log10(2)
    obj.negep = -1  # np.log10(epsneg) / np.log10(2)
    obj.max = float4_e2m1(max_)
    obj.min = float4_e2m1(-max_)
    obj.nexp = 2
    obj.nmant = 1
    obj.iexp = obj.nexp
    obj.maxexp = 3  # N+1 in max_, where max_=float.fromhex("0x1.*pN")
    obj.minexp = 0  # N in tiny, where tiny=float.fromhex("0x1.*pN")
    obj.precision = 0  # int(-log10(eps))
    obj.resolution = float4_e2m1(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float4E2m1MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float4_e2m1(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float4_e1m2_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p0")  # smallest_normal = 1
    resolution = 1  # 10**(-precision)
    eps = float.fromhex("0x1p-2")  # 0.25
    epsneg = float.fromhex("0x1p-2")  # 0.25
    max_ = float.fromhex("0x1.Cp0")  # 1.75 = (1+12/16) * 2^0

    obj = object.__new__(np.finfo)
    obj.dtype = _float4_e1m2_dtype
    obj.bits = 4
    obj.eps = float4_e1m2(eps)
    obj.epsneg = float4_e1m2(epsneg)
    obj.machep = -2
    obj.negep = -2
    obj.max = float4_e1m2(max_)
    obj.min = float4_e1m2(-max_)
    obj.nexp = 1
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 1
    obj.minexp = 0
    obj.precision = 0  # int(-log10(eps))
    obj.resolution = float4_e1m2(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float4E1m2MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float4_e1m2(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float6_e2m3_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p0")  # smallest_normal = 1
    resolution = 1  # 10**(-precision)
    eps = float.fromhex("0x1p-3")  # 0.125
    epsneg = float.fromhex("0x1p-3")  # 0.125
    max_ = float.fromhex("0x1.Ep2")  # 7.5 = (1+14/16) * 2^2

    obj = object.__new__(np.finfo)
    obj.dtype = _float6_e2m3_dtype
    obj.bits = 6
    obj.eps = float6_e2m3(eps)
    obj.epsneg = float6_e2m3(epsneg)
    obj.machep = -3
    obj.negep = -3
    obj.max = float6_e2m3(max_)
    obj.min = float6_e2m3(-max_)
    obj.nexp = 2
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 3
    obj.minexp = 0
    obj.precision = 0  # int(-log10(eps))
    obj.resolution = float6_e2m3(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float6E2m3MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float6_e2m3(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float6_e3m2_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-2")  # smallest_normal = 0.25
    resolution = 1  # 10**(-precision)
    eps = float.fromhex("0x1p-2")  # 0.25
    epsneg = float.fromhex("0x1p-3")  # 0.125
    max_ = float.fromhex("0x1.Cp4")  # 28 = (1+12/16) * 2^4

    obj = object.__new__(np.finfo)
    obj.dtype = _float6_e3m2_dtype
    obj.bits = 6
    obj.eps = float6_e3m2(eps)
    obj.epsneg = float6_e3m2(epsneg)
    obj.machep = -2
    obj.negep = -3
    obj.max = float6_e3m2(max_)
    obj.min = float6_e3m2(-max_)
    obj.nexp = 3
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 5
    obj.minexp = -2
    obj.precision = 0  # int(-log10(eps))
    obj.resolution = float6_e3m2(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float6E3m2MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float6_e3m2(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e8m0_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-127")  # smallest_normal = 2^-127
    resolution = 1  # 10**(-precision) TODO
    eps = float.fromhex("0x1p0")  # 1.0
    epsneg = float.fromhex("0x1p-1")  # 0.5
    max_ = float.fromhex("0x1.p127")  # 2 ^ 127

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e8m0_dtype
    obj.bits = 8
    obj.eps = float8_e8m0(eps)
    obj.epsneg = float8_e8m0(epsneg)
    obj.machep = 0  # np.log10(eps) / np.log10(2)
    obj.negep = -1  # np.log10(epsneg) / np.log10(2)
    obj.max = float8_e8m0(max_)
    obj.min = float8_e8m0(tiny)
    obj.nexp = 8
    obj.nmant = 0
    obj.iexp = obj.nexp
    obj.maxexp = 128
    obj.minexp = -127
    obj.precision = 0  # int(-log10(eps))
    obj.resolution = float8_e8m0(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E8m0MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e8m0(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  def __new__(cls, dtype):
    if (isinstance(dtype, str) and dtype == "float4_e2m1" or dtype == _float4_e2m1_dtype):
      if _float4_e2m1_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float4_e2m1_dtype] = (cls._float4_e2m1_finfo())
      return cls._finfo_cache[_float4_e2m1_dtype]
    if (isinstance(dtype, str) and dtype == "float4_e1m2" or dtype == _float4_e1m2_dtype):
      if _float4_e1m2_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float4_e1m2_dtype] = cls._float4_e1m2_finfo()
      return cls._finfo_cache[_float4_e1m2_dtype]
    if (isinstance(dtype, str) and dtype == "float6_e2m3" or dtype == _float6_e2m3_dtype):
      if _float6_e2m3_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float6_e2m3_dtype] = (cls._float6_e2m3_finfo())
      return cls._finfo_cache[_float6_e2m3_dtype]
    if (isinstance(dtype, str) and dtype == "float6_e3m2" or dtype == _float6_e3m2_dtype):
      if _float6_e3m2_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float6_e3m2_dtype] = cls._float6_e3m2_finfo()
      return cls._finfo_cache[_float6_e3m2_dtype]
    if (isinstance(dtype, str) and dtype == "float8_e8m0" or dtype == _float8_e8m0_dtype):
      if _float8_e8m0_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float8_e8m0_dtype] = cls._float8_e8m0_finfo()
      return cls._finfo_cache[_float8_e8m0_dtype]
    return super().__new__(cls, dtype)
