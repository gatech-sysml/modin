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

from modin.engines.base.frame.partition import BaseFramePartition
from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.ray.utils import handle_ray_task_error

import ray
from ray.services import get_node_ip_address
from ray.worker import RayTaskError


class PandasOnRayFramePartition(BaseFramePartition):
    def __init__(self, object_id, length=None, width=None, ip=None, call_queue=None):
        assert type(object_id) is ray.ObjectRef

        self.oid = object_id
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width
        self.ip = ip

    def get(self):
        """Gets the object out of the plasma store.

        Returns
        -------
            The object from the plasma store.
        """
        if len(self.call_queue):
            self.drain_call_queue()
        try:
            return ray.get(self.oid)
        except RayTaskError as e:
            handle_ray_task_error(e)

    def apply(self, func, **kwargs):
        """Apply a function to the object stored in this partition.

        Note: It does not matter if func is callable or an ObjectRef. Ray will
            handle it correctly either way. The keyword arguments are sent as a
            dictionary.

        Parameters
        ----------
            func: The function to apply.

        Returns
        -------
            A RayRemotePartition object.
        """
        oid = self.oid
        call_queue = self.call_queue + [(func, kwargs)]
        result, length, width, ip = deploy_ray_func.remote(call_queue, oid)
        return PandasOnRayFramePartition(result, length, width, ip)

    def add_to_apply_calls(self, func, **kwargs):
        """Add the function to the apply function call stack.

        Note: This function will be executed when apply is called. It will be executed
        in the order inserted; apply's func operates the last and return

        Parameters
        ----------
        func : callable
            The function to apply.

        Returns
        -------
            A new `PandasOnRayFramePartition` with the function added to the call queue.
        """
        return PandasOnRayFramePartition(
            self.oid, call_queue=self.call_queue + [(func, kwargs)]
        )

    def drain_call_queue(self):
        """Execute all functionality stored in the call queue."""
        if len(self.call_queue) == 0:
            return
        oid = self.oid
        call_queue = self.call_queue
        (
            self.oid,
            self._length_cache,
            self._width_cache,
            self.ip,
        ) = deploy_ray_func.remote(call_queue, oid)
        self.call_queue = []

    def __copy__(self):
        """
        Create a copy of the data frame partition.
        """
        return PandasOnRayFramePartition(
            self.oid, self._length_cache, self._width_cache, call_queue=self.call_queue
        )

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns
        -------
            A Pandas DataFrame.
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series
        return dataframe

    def to_numpy(self, **kwargs):
        """
        Convert the object stored in this partition to a NumPy array.

        Returns
        -------
            A NumPy array.
        """
        return self.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).get()

    def mask(self, row_indices, col_indices):
        """Lazily create a mask that extracts the indices provided.

        Parameters
        ----------
            row_indices: The indices for the rows to extract.
            col_indices: The indices for the columns to extract.

        Returns
        -------
            A `BaseFramePartition` object.
        """
        if (
            (isinstance(row_indices, slice) and row_indices == slice(None))
            or (
                not isinstance(row_indices, slice)
                and self._length_cache is not None
                and len(row_indices) == self._length_cache
            )
        ) and (
            (isinstance(col_indices, slice) and col_indices == slice(None))
            or (
                not isinstance(col_indices, slice)
                and self._width_cache is not None
                and len(col_indices) == self._width_cache
            )
        ):
            return self.__copy__()

        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        if not isinstance(row_indices, slice):
            new_obj._length_cache = len(row_indices)
        if not isinstance(col_indices, slice):
            new_obj._width_cache = len(col_indices)
        return new_obj

    @classmethod
    def put(cls, obj):
        """Put an object in the Plasma store and wrap it in this object.

        Parameters
        ----------
            obj: The object to be put.

        Returns
        -------
            A `RayRemotePartition` object.
        """
        return PandasOnRayFramePartition(ray.put(obj), len(obj.index), len(obj.columns))

    @classmethod
    def preprocess_func(cls, func):
        """Put a callable function into the plasma store for use in `apply`.

        Parameters
        ----------
            func: The function to preprocess.

        Returns
        -------
            A ray.ObjectRef.
        """
        return ray.put(func)

    def length(self):
        """Return the length of partition."""
        if self._length_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = get_index_and_columns.remote(
                    self.oid
                )
        if isinstance(self._length_cache, ray.ObjectRef):
            try:
                self._length_cache = ray.get(self._length_cache)
            except RayTaskError as e:
                handle_ray_task_error(e)
        return self._length_cache

    def width(self):
        """Return the width of partition."""
        if self._width_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = get_index_and_columns.remote(
                    self.oid
                )
        if isinstance(self._width_cache, ray.ObjectRef):
            try:
                self._width_cache = ray.get(self._width_cache)
            except RayTaskError as e:
                handle_ray_task_error(e)
        return self._width_cache

    @classmethod
    def length_extraction_fn(cls):
        """Compute the length of the object in this partition.

        Returns
        -------
            A callable function.
        """
        return length_fn_pandas

    @classmethod
    def width_extraction_fn(cls):
        """Compute the width of the object in this partition.

        Returns
        -------
            A callable function.
        """
        return width_fn_pandas

    @classmethod
    def empty(cls):
        """Create an empty partition.

        Returns
        -------
            An empty partition
        """
        return cls.put(pandas.DataFrame())


@ray.remote(num_returns=2)
def get_index_and_columns(df):
    """Return the index and columns of partition."""
    return len(df.index), len(df.columns)


@ray.remote(num_returns=4)
def deploy_ray_func(call_queue, partition):  # pragma: no cover
    """
    Remotely run functions in the call queue on the given partition with Ray.

    Parameters
    ----------
        call_queue
            The call queue contains a list of functions to be run.
        partition
            The partition on which functions are applied/called.

    Returns
    -------
        Results of the functions called.
    """
    def deserialize(obj):
        if isinstance(obj, ray.ObjectRef):
            return ray.get(obj)
        return obj

    if len(call_queue) > 1:
        for func, kwargs in call_queue[:-1]:
            func = deserialize(func)
            kwargs = deserialize(kwargs)
            try:
                partition = func(partition, **kwargs)
            except ValueError:
                partition = func(partition.copy(), **kwargs)
    func, kwargs = call_queue[-1]
    func = deserialize(func)
    kwargs = deserialize(kwargs)
    try:
        result = func(partition, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
    # don't want the error to propagate to the user, and we want to avoid copying unless
    # we absolutely have to.
    except ValueError:
        result = func(partition.copy(), **kwargs)
    return (
        result,
        len(result) if hasattr(result, "__len__") else 0,
        len(result.columns) if hasattr(result, "columns") else 0,
        get_node_ip_address(),
    )
