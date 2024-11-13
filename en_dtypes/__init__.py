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

__version__ = "0.0.3"  # Keep in sync with pyproject.toml:version
__all__ = [
    "__version__",
    "finfo",
    "float4_e2m1",
    "float4_e1m2",
    "float6_e2m3",
    "float6_e3m2",
    "float8_e8m0",
    "hifloat8",
]

from typing import Type

from en_dtypes._finfo import finfo
from en_dtypes._en_dtypes_ext import float4_e2m1
from en_dtypes._en_dtypes_ext import float4_e1m2
from en_dtypes._en_dtypes_ext import float6_e2m3
from en_dtypes._en_dtypes_ext import float6_e3m2
from en_dtypes._en_dtypes_ext import float8_e8m0
from en_dtypes._en_dtypes_ext import hifloat8
import numpy as np

float4_e2m1: Type[np.generic]
float4_e1m2: Type[np.generic]

float6_e2m3: Type[np.generic]
float6_e3m2: Type[np.generic]

float8_e8m0: Type[np.generic]

hifloat8: Type[np.generic]


del np, Type

