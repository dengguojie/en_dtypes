# en_dtypes

`en_dtypes` is a stand-alone implementation of several Experimental NumPy dtype Extensions used in machine learning libraries, including:

- [Microscaling (MX)-compliant data formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):
  * `float4_e2m1`
  * `float4_e1m2`
  * `float6_e2m3`
  * `float6_e3m2`
  * `float8_e8m0`
- [Ascend HiFloat8](https://arxiv.org/pdf/2409.16626):
  * `hifloat8`

See below for specifications of these number formats.

## Installation

The `en_dtypes` package is tested with Python versions 3.7-3.12, and can be installed with the following command:
```
pip3 install en-dtypes
```
To build from source, clone the repository and run:
```
git submodule init
git submodule update
pip3 install .
```

## Example Usage

```python
>>> from en_dtypes import float4_e2m1, hifloat8
>>> import numpy as np
>>> np.zeros(4, dtype=float4_e2m1)
array([0, 0, 0, 0], dtype=float4_e2m1)
>>> np.zeros(4, dtype=hifloat8)
array([0, 0, 0, 0], dtype=hifloat8)
```

## Microscaling (MX)-compliant data formats

### `float4_e2m1`

Exponent: 2, Mantissa: 1, bias: 1.

Extended range: no inf, no NaN.

Microscaling format, 4 bits (encoding: `0bSEEM`) using byte storage (higher 4 bits are unused). NaN representation is undefined.

Possible absolute values: [`0`, `0.5`, `1`, `1.5`, `2`, `3`, `4`, `6`]

### `float4_e1m2`

Exponent: 1, Mantissa: 2, bias: 1.

Extended range: no inf, no NaN.

Microscaling format, 4 bits (encoding: `0bSEMM`) using byte storage (higher 4 bits are unused). NaN representation is undefined.

Possible absolute values: [`0`, `0.25`, `0.5`, `0.75`, `1`, `1.25`, `1.5`, `1.75`]

### `float6_e2m3`

Exponent: 2, Mantissa: 3, bias: 1.

Extended range: no inf, no NaN.

Microscaling format, 6 bits (encoding: `0bSEEMMM`) using byte storage (higher 2 bits are unused). NaN representation is undefined.

Possible values range: [`-7.5`; `7.5`]

### `float6_e3m2`

Exponent: 3, Mantissa: 2, bias: 3.

Extended range: no inf, no NaN.

Microscaling format, 4 bits (encoding: `0bSEEEMM`) using byte storage (higher 2 bits are unused). NaN representation is undefined.

Possible values range: [`-28`; `28`]

### `float8_e8m0`

scale format E8M0, which has the following properties:
  * Unsigned format
  * 8 exponent bits
  * Exponent range from -127 to 127
  * No zero and infinity
  * Single NaN value (0xFF).

## Ascend data formats

### `hifloat8`

HiFloat8 is a 8-bit floating point format used in Ascend devices. It has the following properties:
  * Sign field with 1 bit
  * Dot field with 2 to 4 bits
  * Exponent field with 0 to 4 bits implicated by dot field
  * Mantissa field with 1 to 3 bits implicated by dot field
  * Exponent bias only exist in sub-normal format as 23
  * Zero is `0b0'0000'000` without negative zero
  * Infinity is `0bS'11'0111'1`
  * NaN is `0b1'0000'000`

## License

The `en_dtypes` source code is licensed under the Apache 2.0 license (see [LICENSE](LICENSE)). Pre-compiled wheels are built with the [EIGEN](https://eigen.tuxfamily.org/) project, which is released under the MPL 2.0 license (see [LICENSE.eigen](LICENSE.eigen)). Implemention is drawn on the [ml_dtypes](https://github.com/jax-ml/ml_dtypes) project, which is released under the Apache 2.0 license.

