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

from .partition import cuDFOnRayFramePartition
from .partition_manager import cuDFOnRayFrameManager

from modin.engines.ray.pandas_on_ray.frame.data import PandasOnRayFrame
from modin.error_message import ErrorMessage


class cuDFOnRayFrame(PandasOnRayFrame):

    _frame_mgr_cls = cuDFOnRayFrameManager

    def _apply_index_objs(self, axis=None):
        """Eagerly applies the index object (Index or Columns) to the partitions.

        Args:
            axis: The axis to apply to, None applies to both axes.

        Returns
        -------
            A new 2D array of partitions that have the index assignment added to the
            call queue.
        """
        ErrorMessage.catch_bugs_and_request_email(
            axis is not None and axis not in [0, 1]
        )

        cum_row_lengths = np.cumsum([0] + self._row_lengths)
        cum_col_widths = np.cumsum([0] + self._column_widths)

        def apply_idx_objs(df, idx, cols, axis):
            # cudf does not support set_axis. It only supports rename with 1-to-1 mapping.
            # Therefore, we need to create the dictionary that have the relationship between
            # current index and new ones.
            idx = {df.index[i]: idx[i] for i in range(len(idx))}
            cols = {df.index[i]: cols[i] for i in range(len(cols))}

            if axis == 0:
                return df.rename(index=idx)
            elif axis == 1:
                return df.rename(columns=cols)
            else:
                return df.rename(index=idx, columns=cols)

        keys = np.array(
            [
                [
                    self._partitions[i][j].apply(
                        apply_idx_objs,
                        idx=self.index[
                            slice(cum_row_lengths[i], cum_row_lengths[i + 1])
                        ],
                        cols=self.columns[
                            slice(cum_col_widths[j], cum_col_widths[j + 1])
                        ],
                        axis=axis,
                    )
                    for j in range(len(self._partitions[i]))
                ]
                for i in range(len(self._partitions))
            ]
        )

        self._partitions = np.array(
            [
                [
                    cuDFOnRayFramePartition(
                        self._partitions[i][j].get_gpu_manager(),
                        keys[i][j],
                        self._partitions[i][j]._length_cache,
                        self._partitions[i][j]._width_cache,
                    )
                    for j in range(len(keys[i]))
                ]
                for i in range(len(keys))
            ]
        )
    
    def broadcast_apply(
        self, axis, func, other, join_type="left", preserve_labels=True, dtypes=None
    ):
        """
        Broadcast partitions of other dataframe partitions and apply a function.

        Parameters
        ----------
            axis: int,
                The axis to broadcast over.
            func: callable,
                The function to apply.
            other: BasePandasFrame
                The Modin DataFrame to broadcast.
            join_type: str (optional)
                The type of join to apply.
            preserve_labels: boolean (optional)
                Whether or not to keep labels from this Modin DataFrame.
            dtypes: "copy" or None (optional)
                 Whether to keep old dtypes or infer new dtypes from data.

        Returns
        -------
            BasePandasFrame
        """

        self_index = self.axes[axis]
        others_index = other.axes[axis]
        joined_index, _ = self._join_index_objects(
            axis, [self_index, others_index], how=join_type, sort=False
        )

        new_frame = self._frame_mgr_cls.broadcast_apply(
            axis, func, self._partitions, other._partitions
        )

        if dtypes == "copy":
            dtypes = self._dtypes
        new_index = self.index
        new_columns = self.columns
        if not preserve_labels:
            if axis == 1:
                new_columns = joined_index
            else:
                new_index = joined_index
        return self.__constructor__(
            new_frame, new_index, new_columns, None, None, dtypes=dtypes
        )

    
    def _compute_axis_labels(self, axis: int, partitions=None):
        """
        Compute the labels for specific `axis`.

        Parameters
        ----------
        axis: int
            Axis to compute labels along
        partitions: numpy 2D array (optional)
            Partitions from which labels will be grabbed,
            if no specified, partitions will be considered as `self._partitions`

        Returns
        -------
        Pandas.Index
            Labels for the specified `axis`
        """
        if partitions is None:
            partitions = self._partitions
        
        def get_index_labels(df):
            if axis == 0:
                return df.index
            elif axis == 1:
                return df.columns
            else:
                raise ValueError("Only 0, 1")
        return self._frame_mgr_cls.get_indices(
            axis, partitions, get_index_labels
        )
