# pylint: disable=eq-without-hash,invalid-name
import abc
from typing import Union, List

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

Number = Union[int, float]


class TimeseriesEstimator:
    """
    Abstract class for time-series anomaly detection.

    Every subclass will provide, `learn`, `assess` and `assess_learn` methods.
    None of these functions will receive time as one of its parameters, as we
    will assume that the timeseries consists of equally distanced points.

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
            self, points_per_day: Number, history_days: int, model=None,
            history: List[Number] = None, residuals: List[Number] = None, scores: List[Number] = None,
            collection_type=list
    ):
        """

        Parameters
        ----------
        points_per_day
            We assume equidistant data points. Some applications will want to know the distance between the points.
            This argument provides this information. For example, `points_per_day=1` means that the data is collected
            every day, `points_per_day=24` means every hour, and `points_per_day=1/7.0` means that the data is
            collected every week
        history_days
            How many days of history should we consider?
        model
            If you already have a fitted model, provide it using this argument.
        history
            If you already have historical data, provide it in this list
        residuals
            If you already have data on differences between a fitted model and the observations, provide it
        scores
            If you already have historical data on previous scores, provide it here
        collection_type
            Container type. If you want to keep your lists in something other than python `list`

        """

        self._collection_type = collection_type
        self.points_per_day = float(points_per_day)
        self.history_days = int(history_days)
        self.history = self._collection_type() if history is None else self._collection_type(history)
        self.residuals = self._collection_type() if residuals is None else self._collection_type(residuals)
        self.scores = self._collection_type() if scores is None else self._collection_type(scores)
        self.model = model
        self.model_type = 'UNDEFINED'

        optional_arguments = ('history', 'residuals', 'scores')
        optional_defined = [getattr(self, a) is not None for a in optional_arguments]
        if np.any(optional_defined):
            assert np.all(optional_defined)
            optional_arrays = ('history', 'residuals', 'scores')
            length_values = set()
            for a in optional_arrays:
                arr = getattr(self, a)
                length_values.add(len(arr))
            assert len(length_values) == 1

    @abc.abstractmethod
    def learn(self, X: np.array):
        """
        Learn the data. Return `self`

        Parameters
        ----------
        X
            Should have at least two columns. The first column is the time,
            the subsequent columns represent the actual data. The time column is required even if
            it is not used by the particular implementation.

        Returns
        -------
        self
        """
        return self

    @staticmethod
    def validate_x(X):
        X = np.asanyarray(X)
        assert X.ndim > 1
        for i in range(1, X.shape[1]):
            X[:, i] = X[:, i].astype(float)
        assert hasattr(X, 'shape') and (len(X.shape) > 1) and (X.shape[1] > 1)
        return X

    @abc.abstractmethod
    def assess(self, X: np.array,
               return_expectation: bool = False):  # pylint: disable=redundant-returns-doc,missing-raises-doc
        """
        Assess the anomaly score

        Parameters
        ----------
        X
            `X` should have exactly two columns. The first column is the time,
            the second column represents the actual data. The time column is required even though
            it is not used by this algorithm.
        return_expectation
            Switch determining nature of return value. When it is False (the
            default) just the anomaly scores are returned, when True values expected by the underlying
            model are returned, along with the scores.

        Returns
        -------
        ndarray
            shape (M,) or a tuple of two such arrays, where M is the length of X, depending on the
            value of `return_expectation`. Anomaly scores, with or without the expected values
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def assess_learn(self, X, return_expectation=False, reset_first=False):
        pass

    @abc.abstractmethod
    def predict_timesteps(self, n_steps, **kwargs):
        """ Predict the next `n_steps` time steps of the time series"""

    @abc.abstractmethod
    def simulate_history(self):
        """ Use the current model state to simulate the data for the available history """

    def simulate_and_predict(self, n_steps):
        """ Use the current model state to simulate the data for the available history, and to predict the future """
        reconstructed = self.simulate_history()
        forecast = self.predict_timesteps(n_steps)
        ret = np.concatenate((reconstructed, forecast))
        return ret

    @classmethod
    @abc.abstractmethod
    def load(cls, data):
        pass

    @abc.abstractmethod
    def serialize(self):
        pass

    def save(self, output, writemode='w'):
        """
        Save the object using

        Parameters
        ----------

        output
            either an open filehandle or a file name
        writemode
            what mode should be used to open the file?
        """
        need_to_close = False
        if not hasattr(output, 'write'):
            output = open(output, mode=writemode)
            need_to_close = True
        output.write(self.serialize())
        if need_to_close:
            # to prevent warnings
            output.close()

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """ Reset the internal state of the model """

    @staticmethod
    def create_x_matrix(values, ts=None):
        if ts is None:
            x = np.asanyarray(values)
            if x.ndim == 2:
                pass
            elif x.ndim == 1:
                x = x.reshape(-1, 1)
            else:
                raise RuntimeError()
            if x.shape[1] == 2:
                pass
            elif x.shape[1] == 1:
                x = np.hstack((np.arange(len(x)).reshape(-1, 1), x))
            elif x.shape[0] == 2:
                x = x.T
            else:
                raise RuntimeError()
            return x
        else:
            return np.vstack((ts, values)).T

    @staticmethod
    def align_arrays(x_array, y_array, at_the_end=True):
        if len(x_array) == len(y_array):
            # nothing to do for us
            pass
        elif len(x_array) < len(y_array) and at_the_end:
            # align both arrays at the end
            y_array = y_array[-len(x_array):]
        elif len(x_array) < len(y_array) and not at_the_end:
            # align both arrays at the beginning
            y_array = y_array[:len(x_array):]
        elif len(x_array) > len(y_array) and at_the_end:
            # align both arrays at the end
            x_array = x_array[-len(y_array):]
        else:
            # align both arrays at the beginning
            x_array = x_array[:len(y_array):]
        return (x_array, y_array)

    def analyze_correlation(self, x_array, y_array, correlation_type='any'):
        assert correlation_type in ['any', 'linear']
        (x_array, y_array) = self.align_arrays(x_array, y_array)
        if correlation_type == 'linear':
            return pearsonr(x_array, y_array)
        return spearmanr(x_array, y_array)
