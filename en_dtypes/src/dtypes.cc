/* Copyright 2024 The en_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


// Enable cmath defines on Windows
#define _USE_MATH_DEFINES

// Must be included first
// clang-format off
#include "src/numpy.h" //NOLINT
// clang-format on

#include <array>    // NOLINT
#include <cmath>    // NOLINT
#include <cstdint>  // NOLINT
#include <limits>   // NOLINT
#include <locale>   // NOLINT

// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "Eigen/Core"
#include "src/en_float.h"
#include "include/floatn.h"
#include "include/hifloat.h"

namespace en_dtypes {

template <>
struct TypeDescriptor<float4_e2m1>
    : EnFloatType<float4_e2m1> {
  typedef float4_e2m1 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float4_e2m1";
  static constexpr const char* kQualifiedTypeName = "en_dtypes.float4_e2m1";
  static constexpr const char* kTpDoc = "float4_e2m1 floating-point values";
  // We must register float4_e2m1 with a kind other than "f", because
  // numpy considers two types with the same kind and size to be equal, and we
  // expect multiple 1 byte floating point types.
  // The downside of this is that NumPy scalar promotion does not work with
  // float4 values.
  static constexpr char kNpyDescrKind = 'V';
  // there doesn't seem to be a way of guaranteeing a type character is unique.
  // numpy used type is define in numpy/_core/include/numpy/ndarraytypes.h
  static constexpr char kNpyDescrType = 'r';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float4_e1m2>
    : EnFloatType<float4_e1m2> {
  typedef float4_e1m2 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float4_e1m2";
  static constexpr const char* kQualifiedTypeName = "en_dtypes.float4_e1m2";
  static constexpr const char* kTpDoc = "float4_e1m2 floating-point values";
  // We must register float4_e1m2 with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float4 values.
  static constexpr char kNpyDescrKind = 'V';
  // there doesn't seem to be a way of guaranteeing a type character is unique.
  static constexpr char kNpyDescrType = 'w';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float6_e2m3>
    : EnFloatType<float6_e2m3> {
  typedef float6_e2m3 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float6_e2m3";
  static constexpr const char* kQualifiedTypeName = "en_dtypes.float6_e2m3";
  static constexpr const char* kTpDoc = "float6_e2m3 floating-point values";
  // We must register float6_e2m3 with a kind other than "f", because
  // numpy considers two types with the same kind and size to be equal, and we
  // expect multiple 1 byte floating point types.
  // The downside of this is that NumPy scalar promotion does not work with
  // float6 values.
  static constexpr char kNpyDescrKind = 'V';
  // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
  // character is unique.
  static constexpr char kNpyDescrType = 'x';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float6_e3m2>
    : EnFloatType<float6_e3m2> {
  typedef float6_e3m2 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float6_e3m2";
  static constexpr const char* kQualifiedTypeName = "en_dtypes.float6_e3m2";
  static constexpr const char* kTpDoc = "float6_e3m2 floating-point values";
  // We must register float6_e3m2 with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float6 values.
  static constexpr char kNpyDescrKind = 'V';
  // there doesn't seem to be a way of guaranteeing a type character is unique.
  static constexpr char kNpyDescrType = 'y';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<float8_e8m0>
    : EnFloatType<float8_e8m0> {
  typedef float8_e8m0 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "float8_e8m0";
  static constexpr const char* kQualifiedTypeName = "en_dtypes.float8_e8m0";
  static constexpr const char* kTpDoc = "float8_e8m0 floating-point values";
  // We must register float8_e8m0 with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8 values.
  static constexpr char kNpyDescrKind = 'V';
  // there doesn't seem to be a way of guaranteeing a type character is unique.
  static constexpr char kNpyDescrType = 'Z';
  static constexpr char kNpyDescrByteorder = '=';
};

template <>
struct TypeDescriptor<hifloat8>
    : EnFloatType<hifloat8> {
  typedef hifloat8 T;
  static constexpr bool is_floating = true;
  static constexpr bool is_integral = false;
  static constexpr const char* kTypeName = "hifloat8";
  static constexpr const char* kQualifiedTypeName = "en_dtypes.hifloat8";
  static constexpr const char* kTpDoc = "hifloat8 floating-point values";
  // We must register hifloat8 with a unique kind, because numpy
  // considers two types with the same kind and size to be equal.
  // The downside of this is that NumPy scalar promotion does not work with
  // float8 values.
  static constexpr char kNpyDescrKind = 'V';
  // there doesn't seem to be a way of guaranteeing a type character is unique.
  static constexpr char kNpyDescrType = '7';
  static constexpr char kNpyDescrByteorder = '=';
};

namespace {

// Performs a NumPy array cast from type 'From' to 'To' via `Via`.
template <typename From, typename To, typename Via>
void PyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
            void* toarr) {
  const auto* from = static_cast<From*>(from_void);
  auto* to = static_cast<To*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] = static_cast<To>(static_cast<Via>(from[i]));
  }
}

template <typename Type1, typename Type2, typename Via>
bool RegisterTwoWayCustomCast() {
  int nptype1 = TypeDescriptor<Type1>::npy_type;
  int nptype2 = TypeDescriptor<Type2>::npy_type;
  PyArray_Descr* descr1 = PyArray_DescrFromType(nptype1);
  if (PyArray_RegisterCastFunc(descr1, nptype2, PyCast<Type1, Type2, Via>) <
      0) {
    return false;
  }
  PyArray_Descr* descr2 = PyArray_DescrFromType(nptype2);
  if (PyArray_RegisterCastFunc(descr2, nptype1, PyCast<Type2, Type1, Via>) <
      0) {
    return false;
  }
  return true;
}

template <typename Type1, typename Type2, typename Via>
bool RegisterOneWayCustomCast() {
  int nptype1 = TypeDescriptor<Type1>::npy_type;
  int nptype2 = TypeDescriptor<Type2>::npy_type;
  PyArray_Descr* descr1 = PyArray_DescrFromType(nptype1);
  if (PyArray_RegisterCastFunc(descr1, nptype2, PyCast<Type1, Type2, Via>) <
      0) {
    return false;
  }
  return true;
}

}  // namespace

// Initializes the module.
bool Initialize() {
  en_dtypes::ImportNumpy();
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  if (!RegisterFloatDtype<float4_e2m1>(numpy.get())) {
    return false;
  }
  if (!RegisterFloatDtype<float4_e1m2>(numpy.get())) {
    return false;
  }

  if (!RegisterFloatDtype<float6_e2m3>(numpy.get())) {
    return false;
  }
  if (!RegisterFloatDtype<float6_e3m2>(numpy.get())) {
    return false;
  }

  if (!RegisterFloatDtype<float8_e8m0>(numpy.get())) {
    return false;
  }

  if (!RegisterFloatDtype<hifloat8>(numpy.get())) {
    return false;
  }

  // Register casts between pairs of custom float dtypes.
  bool success = true;
  success &=
      RegisterTwoWayCustomCast<float4_e2m1, float4_e1m2, float>();
  success &=
      RegisterTwoWayCustomCast<float6_e2m3, float6_e3m2, float>();
  return success;
}

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_en_dtypes_ext",
};

// TODO(phawkins): PyMODINIT_FUNC handles visibility correctly in Python 3.9+.
// Just use PyMODINIT_FUNC after dropping Python 3.8 support.
#if defined(WIN32) || defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

extern "C" EXPORT_SYMBOL PyObject* PyInit__en_dtypes_ext() {
  Safe_PyObjectPtr m = make_safe(PyModule_Create(&module_def));
  if (!m) {
    return nullptr;
  }
  if (!Initialize()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load _en_dtypes_ext module.");
    }
    return nullptr;
  }

  if (PyObject_SetAttrString(m.get(), "float4_e2m1",
                            reinterpret_cast<PyObject*>(
                                TypeDescriptor<float4_e2m1>::type_ptr)) < 0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(m.get(), "float4_e1m2",
                             reinterpret_cast<PyObject*>(
                                 TypeDescriptor<float4_e1m2>::type_ptr)) < 0) {
    return nullptr;
  }

  if (PyObject_SetAttrString(m.get(), "float6_e2m3",
                            reinterpret_cast<PyObject*>(
                                TypeDescriptor<float6_e2m3>::type_ptr)) < 0) {
    return nullptr;
  }
  if (PyObject_SetAttrString(m.get(), "float6_e3m2",
                             reinterpret_cast<PyObject*>(
                                 TypeDescriptor<float6_e3m2>::type_ptr)) < 0) {
    return nullptr;
  }

  if (PyObject_SetAttrString(m.get(), "float8_e8m0",
                            reinterpret_cast<PyObject*>(
                                TypeDescriptor<float8_e8m0>::type_ptr)) < 0) {
    return nullptr;
  }

  if (PyObject_SetAttrString(m.get(), "hifloat8",
                            reinterpret_cast<PyObject*>(
                                TypeDescriptor<hifloat8>::type_ptr)) < 0) {
    return nullptr;
  }

  return m.release();
}
}  // namespace en_dtypes


