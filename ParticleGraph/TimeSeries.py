import os
from collections.abc import Sequence
from typing import List, Tuple, Self

import torch
from torch_geometric.data import Data


class TimeSeries(Sequence):
    """
    Class to represent a time series of :py:class:`torch_geometric.data.Data` objects.
    The time series can be indexed like a list to access individual time steps or slices of time steps. It also holds
    an array of time points at which the data was recorded in the attribute `time`. The fields of the data objects are
    described by a dictionary of :py:class:`FieldDescriptor` objects in the attribute `fields`.
    """
    def __init__(
            self,
            time: torch.Tensor,
            data: Sequence[Data],
    ):
        self.time = time
        self._data = data

    def __len__(self) -> int:
        """
        Get the number of time steps in the time series.
        :return: The number of time steps.
        """
        return len(self.time)

    def __getitem__(self, idx: int | slice) -> Data | Self:
        """
        Get a single time step or a slice of time steps from the time series.
        :param idx: A single index or a slice of the desired time steps.
        :return: A :py:class:`torch_geometric.data.Data` object if a single index is given, or a new
            :py:class:`TimeSeries` object if a slice is given.
        """
        if isinstance(idx, slice):
            # torch does not support slicing of tensors with negative step size, so make indices explicit
            torch_idx = torch.arange(*idx.indices(len(self)))
            return TimeSeries(self.time[torch_idx], self._data[idx])
        elif isinstance(idx, int):
            return self._data[idx]
        else:
            raise TypeError("Index must be an integer or a slice.")

    @staticmethod
    def load(path: str) -> 'TimeSeries':
        """
        Load a time series from a directory. The time series is expected to be stored in the format written by
        :py:meth:`TimeSeries.save`.
        :param path: The directory to load the time series from.
        :return: A :py:class:`TimeSeries` object containing the loaded data.
        :raises ValueError: If the data could not be loaded from the given path.
        """
        try:
            torch.load(os.path.join(path, 'fields.pt'))
            time = torch.load(os.path.join(path, 'time.pt'))

            n_time_steps = len(time)
            n_digits = len(str(n_time_steps - 1))

            data = []
            for i in range(n_time_steps):
                data.append(torch.load(os.path.join(path, f'data_{str(i).zfill(n_digits)}.pt')))
        except Exception as e:
            raise ValueError(f"Could not load data from {path}.") from e

        return TimeSeries(time, data)

    @staticmethod
    def save(time_series: 'TimeSeries', path: str):
        """
        Save a time series to a directory. The time series is stored as 'fields.pt', 'time.pt', and 'data_*.pt' files,
        where * is a zero-padded index of the time step.
        :param time_series: The time series to save.
        :param path: The directory to save the time series to.
        :raises ValueError: If the data could not be saved to the given path.
        """
        try:
            os.makedirs(path, exist_ok=False)
            torch.save(time_series.time, os.path.join(path, 'time.pt'))

            n_time_steps = len(time_series)
            n_digits = len(str(n_time_steps - 1))

            for i, d in enumerate(time_series):
                torch.save(d, os.path.join(path, f'data_{str(i).zfill(n_digits)}.pt'))
        except Exception as e:
            raise ValueError(f"Could not save data to {path}.") from e

    def compute_derivative(
            self,
            field_name: str,
            *,
            id_name: str = None,
    ) -> List[torch.Tensor] | Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute the backward difference quotient of a field in a time series.
        :param field_name: The field for which to compute the difference quotient.
        :param id_name: If given, this field is used to match data points between time steps. Ids are assumed to be
            unique.
        :return: A list of tensors containing the difference quotient at each time step. Where the difference quotient
            could not be computed, the corresponding entry is Nan. If id_name is given, a list of masks is also
            returned, indicating which entries could not be computed.
        """
        difference_quotients = [torch.full_like(getattr(self[0], field_name), torch.nan)]
        for i in range(1, len(self)):
            x_current = getattr(self[i], field_name)
            x_previous = getattr(self[i - 1], field_name)
            delta_t = self.time[i] - self.time[i - 1]

            if id_name is None:
                difference_quotients.append((x_current - x_previous) / delta_t)
            else:
                id_current = getattr(self[i], id_name)
                id_previous = getattr(self[i - 1], id_name)

                # Compute a set of global unique ids
                all_ids = torch.cat((id_current, id_previous))
                _, indices, counts = torch.unique(all_ids, return_inverse=True, return_counts=True)

                # Compute the difference quotient in the global id space
                indices_current = indices[:len(id_current)]
                indices_previous = indices[len(id_current):]
                all_differences = torch.zeros((len(counts), x_current.shape[1]), device=x_current.device)
                for j in range(x_current.shape[1]):
                    all_differences[:, j] = (torch.bincount(indices_current, x_current[:, j], minlength=len(counts))
                                             - torch.bincount(indices_previous, x_previous[:, j], minlength=len(counts)))

                # Only consider ids that are present in both time steps and map to current ids
                all_differences[counts.ne(2)] = torch.nan
                difference_quotient = all_differences[indices_current] / delta_t
                difference_quotients.append(difference_quotient)

        if id_name is None:
            return difference_quotients
        else:
            # Compute a mask which entries could not be computed
            mask = [torch.logical_not(torch.any(torch.isnan(dq), dim=1)) for dq in difference_quotients]
            return difference_quotients, mask
