# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import ray
import cudf
import cupy
import numpy as np
import cupy as cp
from modin.engines.base.frame.partition import BaseFramePartition
from pandas.core.dtypes.common import is_list_like


class cuDFOnRayFramePartition(BaseFramePartition):

    _length_cache = None
    _width_cache = None

    @property
    def __constructor__(self):
        return type(self)

    def __init__(self, gpu_manager, key, length=None, width=None):
        self.gpu_manager = gpu_manager
        self.key = key
        self._length_cache = length
        self._width_cache = width

    def apply(self, func, **kwargs):
        return self.gpu_manager.apply.remote(self.get_key(), None, func, **kwargs)

    def apply_result_not_dataframe(self, func, **kwargs):
        return self.gpu_manager.apply_result_not_dataframe.remote(
            self.get_key(), func, **kwargs
        )

    @classmethod
    def preprocess_func(cls, func):
        return ray.put(func)

    def length(self):
        if self._length_cache:
            return self._length_cache
        return self.gpu_manager.length.remote(self.get_key())

    def width(self):
        if self._width_cache:
            return self._width_cache
        return self.gpu_manager.width.remote(self.get_key())

    def reduce(self, others, func, axis=0):
        return self.gpu_manager.reduce.remote(self.get_key(), others, func, axis=1)

    def mask(self, row_indices, col_indices):
        def func(df, row_indices, col_indices):
            # CuDF currently does not support indexing multiindices with arrays,
            # so we have to create a boolean array where the desire indices are true
            if isinstance(df.index, cudf.core.multiindex.MultiIndex) and is_list_like(
                row_indices
            ):
                new_row_indices = cp.full(
                    (1, df.index.size), False, dtype=bool
                ).squeeze()
                new_row_indices[row_indices] = True
                row_indices = new_row_indices
            return df.iloc[row_indices, col_indices]

        func = ray.put(func)
        key_future = self.gpu_manager.apply.remote(
            self.get_key(),
            func,
            col_indices=col_indices,
            row_indices=row_indices,
        )
        return key_future

    def get_gpu_manager(self):
        return self.gpu_manager

    def get_key(self):
        return ray.get(self.key) if isinstance(self.key, ray.ObjectRef) else self.key

    @classmethod
    def put(cls, gpu_manager, pandas_dataframe):
        key_future = gpu_manager.put.remote(pandas_dataframe)
        return key_future

    def get_object_id(self):
        return self.gpu_manager.get_object_id.remote(self.get_key())

    def get(self):
        return self.gpu_manager.get.remote(self.get_key())

    def to_pandas(self):
        return self.gpu_manager.apply_non_persistent.remote(
            self.get_key(), None, cudf.DataFrame.to_pandas
        )

    def to_numpy(self):
        def convert(df):
            if len(df.columns == 1):
                df = df.iloc[:, 0]
            if isinstance(df, cudf.Series):  # convert to column vector
                return cupy.asnumpy(df.to_array())[:, np.newaxis]
            elif isinstance(
                df, cudf.DataFrame
            ):  # dataframes do not support df.values with strings
                return cupy.asnumpy(df.values)

        return self.gpu_manager.apply_result_not_dataframe.remote(
            self.get_key(),
            convert,
        )

    def free(self):
        self.gpu_manager.free.remote(self.get_key())

    def copy(self):
        new_key = self.gpu_manager.apply.remote(
            self.get_key(),
            lambda x: x,
        )
        new_key = ray.get(new_key)
        return self.__constructor__(self.gpu_manager, new_key)

    # TODO(kvu35): buggy garbage collector reference issue #43
    # def __del__(self):
    #     self.free()
