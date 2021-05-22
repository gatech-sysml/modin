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
import pyarrow

import ray


class OmnisciOnRayFramePartition(BaseFramePartition):
    def __init__(
        self, object_id=None, frame_id=None, arrow_table=None, length=None, width=None
    ):
        """Initialize the OmnisciOnRayFramePartition object."""
        assert type(object_id) is ray.ObjectRef

        self.oid = object_id
        self.frame_id = frame_id
        self.arrow_table = arrow_table
        self._length_cache = length
        self._width_cache = width

    def to_pandas(self):
        """
        Convert the object stored in this partition to a Pandas DataFrame.

        Note: If the underlying object is a Pandas DataFrame, this will likely
            only need to call `get`

        Returns
        -------
            A Pandas DataFrame.
        """
        obj = self.get()
        if isinstance(obj, (pandas.DataFrame, pandas.Series)):
            return obj
        assert isinstance(obj, pyarrow.Table)
        return obj.to_pandas()

    def get(self):
        """
        Return the object wrapped by this one to the original format.

        Note: This is the opposite of the classmethod `put`.
            E.g. if you assign `x = cuDFOnRayFramePartition.put(1)`, `x.get()` should
            always return 1.

        Returns
        -------
            The object that was `put`.
        """
        if self.arrow_table is not None:
            return self.arrow_table
        return ray.get(self.oid)

    def wait(self):
        ray.wait([self.oid])

    @classmethod
    def put(cls, obj):
        """
        Format a given object.
        Store the given object obj in Ray, and put its details into a new frame
        partition object, which is then returned.

        Parameters
        ----------
            obj
                The object to be stored in Ray.

        Returns
        -------
            An OmnisciOnRayFramePartition object with details on the object ID in Ray,
            and number of rows and columns.
        """
        return OmnisciOnRayFramePartition(
            object_id=ray.put(obj), length=len(obj.index), width=len(obj.columns)
        )

    @classmethod
    def put_arrow(cls, obj):
        """
        Format a given object.
        Store the given pyarrow object obj in Ray, and put its details into a new frame
        partition object, which is then returned.

        Parameters
        ----------
            obj
                The pyarrow object to be stored in Ray.

        Returns
        -------
            An OmnisciOnRayFramePartition object with details on the object ID in Ray,
            arrow table, and number of rows and columns.
        """
        return OmnisciOnRayFramePartition(
            object_id=ray.put(None),
            arrow_table=obj,
            length=len(obj),
            width=len(obj.columns),
        )
