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

import numpy as np
import pandas

from modin.error_message import ErrorMessage

from pandas.api.types import union_categoricals

from .axis_partition import (
    cuDFOnRayFrameColumnPartition,
    cuDFOnRayFrameRowPartition,
)
from .partition import cuDFOnRayFramePartition

from modin.config import NPartitions
from modin.data_management.utils import split_result_of_axis_func_pandas
from modin.config import GpuCount

from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
import ray


@ray.remote(num_cpus=1, num_gpus=0.5)
def func(df, other, apply_func):
    return apply_func(ray.get(df.get.remote()), ray.get(other.get.remote()))


class cuDFOnRayFrameManager(RayFrameManager):

    _partition_class = cuDFOnRayFramePartition
    _column_partitions_class = cuDFOnRayFrameColumnPartition
    _row_partition_class = cuDFOnRayFrameRowPartition

    @classmethod
    def _get_gpu_managers(cls):
        from modin.pandas import GPU_MANAGERS

        return GPU_MANAGERS

    @classmethod
    def from_pandas(cls, df, return_dims=False):
        num_splits = GpuCount.get()
        put_func = cls._partition_class.put
        # For now, we default to row partitioning
        pandas_dfs = split_result_of_axis_func_pandas(0, num_splits, df)
        keys = [
            put_func(cls._get_gpu_managers()[i], pandas_dfs[i])
            for i in range(num_splits)
        ]
        keys = ray.get(keys)
        parts = cls._create_partitions(keys, cls._get_gpu_managers()).reshape(
            (num_splits, 1)
        )
        if not return_dims:
            return parts
        else:
            row_lengths = [len(df.index) for df in pandas_dfs]
            col_widths = [
                len(df.columns)
            ]  # single value since we only have row partitions
            return parts, row_lengths, col_widths

    @classmethod
    def _create_partitions(cls, keys, gpu_managers):
        return np.array(
            [
                cls._partition_class(gpu_managers[i], keys[i])
                for i in range(len(gpu_managers))
            ]
        )
