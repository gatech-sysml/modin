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

import pandas
import ray
import cudf
import cupy
import numpy as np
import cupy as cp
from pandas.core.dtypes.common import is_list_like

class cuDFOnRayFramePartition(object):

    _length_cache = None
    _width_cache = None

    @property
    def __constructor__(self):
        return type(self)

    def __init__(self, gpu_manager, key):
        self.gpu_manager = gpu_manager
        self.key = key

    def apply_result_not_dataframe(self, func, **kwargs):
        return self.gpu_manager.apply_result_not_dataframe.remote(self.key, func, **kwargs)

    @classmethod
    def preprocess_func(cls, func):
        return ray.put(func)

    def length(self):
        return self.gpu_manager.length.remote(self.key)

    def width(self):
        return self.gpu_manager.width.remote(self.key)

    def mask(self, row_indices, col_indices):
        def func(df, row_indices, col_indices):
            # CuDF currently does not support indexing multiindices with arrays,
            # so we have to create a boolean array where the desire indices are true
            if (
                isinstance(df.index, cudf.core.multiindex.MultiIndex)
                and is_list_like(row_indices)
            ):
                new_row_indices = cp.full((1, df.index.size), False, dtype=bool).squeeze()
                new_row_indices[row_indices] = True
                row_indices = new_row_indices
            return df.iloc[row_indices, col_indices]
        func = ray.put(func)
        key_future = self.gpu_manager.apply.remote(
            self.key,
            func,
            col_indices=col_indices,
            row_indices=row_indices,
        )
        return key_future

    def get_gpu_manager(self):
        return self.gpu_manager

    def get_key(self):
        return self.key

    @classmethod
    def put(cls, gpu_manager, pandas_dataframe):
        key_future = gpu_manager.put.remote(pandas_dataframe)
        return key_future

    def get_object_id(self):
        return self.gpu_manager.get_object_id.remote(self.key)

    def get(self):
        return self.gpu_manager.get.remote(self.key)

    def to_pandas(self):
        return self.gpu_manager.to_pandas.remote(self.key)

    def to_numpy(self):
        def convert(df):
            if len(df.columns == 1):
                df = df.iloc[:,0]
            if isinstance(df, cudf.Series): # convert to column vector
                return cupy.asnumpy(df.to_array())[:,np.newaxis]
            elif isinstance(df, cudf.DataFrame): # dataframes do not support df.values with strings
                return cupy.asnumpy(df.values)
        return self.gpu_manager.apply_result_not_dataframe.remote(
            self.key,
            convert,
        )

    @classmethod
    def length_extraction_fn(cls):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def width_extraction_fn(cls):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def empty(cls):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def apply(self, func, **kwargs):
        return self.gpu_manager.apply.remote(self.key, func, **kwargs)

    def free(self):
        self.gpu_manager.free.remote(self.key)

    def copy(self):
        new_key = self.gpu_manager.apply.remote(
            self.key,
            lambda x : x,
        )
        new_key = ray.get(new_key)
        return self.__constructor__(self.gpu_manager, new_key)

    def __del__(self):
        self.free()
