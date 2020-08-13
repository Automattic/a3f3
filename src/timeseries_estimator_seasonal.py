import json
import os
from typing import Union, Iterable, Dict

import numpy as np
import seasonal
from scipy.stats import norm


from src.timeseries_estimator import TimeseriesEstimator


Number = Union[int, float]

class SeasonalModel:
    def __init__(self, seasons=None, trend=None):
        self.seasons = [float(s) for s in seasons] if seasons is not None else None
        self.trend = [float(t) for t in trend] if trend is not None else None

    def is_empty(self):
        return (not bool(self.seasons)) and (not bool(self.trend))

    def to_dict(self):
        if self.seasons is None:
            seasons = None
        else:
            seasons = [s for s in self.seasons]
        if self.trend is None:
            trend = None
        else:
            trend = [t for t in self.trend]
        return {'seasons': seasons, 'trend': trend}

    def copy(self):
        seasons = self.seasons[:] if self.seasons is not None else None
        trend = self.trend[:] if self.trend is not None else None
        return SeasonalModel(seasons=seasons, trend=trend)


class TimeseriesEstimator_seasonal(TimeseriesEstimator):  # pylint: disable=invalid-name
    """
    Use the `seasonal`[1] module to model the data. Assess anomaly using the residuals

    [1]: https://github.com/welch/seasonal

    """

    def __init__(self, points_per_day, period_days, history_days, trend_forecast_method='average', model=None,
                 history=None, residuals=None, scores=None):

        super().__init__(
            points_per_day=points_per_day, history_days=history_days,
            model=model, history=history, residuals=residuals, scores=scores
        )

        if period_days is not None:
            period_days = float(period_days)
            if period_days * self.points_per_day < 2:
                msg = 'Period of %f days, given %d points per day makes no sense' % (period_days, points_per_day)
                raise ValueError(msg)
        self.period_days = period_days
        self.model = SeasonalModel() if model is None else model
        supported_trend_forecast_methods = ('last', 'average')  # pylint: disable=invalid-name
        if trend_forecast_method not in supported_trend_forecast_methods:
            msg = 'Trend forecast method "%s" is not supported. Use one of %s' % (
                trend_forecast_method, ', '.join(supported_trend_forecast_methods)
            )
            raise RuntimeError(msg)
        self.trend_forecast_method = trend_forecast_method
        self.model_type = 'seasonal'

        # call to `super` above asserts that proper values of `history`, `residuals`,
        # and `scores`. In this case, we need to add `model` to the similar constraint.
        # Unlike the previous case, here we will test the arguments to the
        # constructor, and not the actual object memebrs.
        optional_defined = [
            history is not None,
            residuals is not None,
            model is not None,
            scores is not None
        ]
        if np.any(optional_defined):
            assert np.all(optional_defined)

    @staticmethod
    def _get_model(x, period_days, points_per_day):
        """
        Perform the actual learning
        """
        if len(x) > 5:
            # When performing the fitting, `fit_seasons` uses at least two `period`
            # data points for trend and season estimation. The function isn't robust
            # enough to handle the situations where there are too few data points,
            # and `period is not None`. By setting `period = None` we let  `fit_seasons`
            # estimate the period by itself
            if period_days is not None:
                period = int(period_days * points_per_day)
                if len(x) < 2 * period:
                    period = None
            else:
                period = None

            # When performing the fitting, `fit_seasons` calls `fit_trend` that uses
            # median filtering to clean up the data. When the fitted data is too short
            # (less than 3 data points), the default filtering mechanism fails due to
            # illegal list indices that stem from integer division errors. By specifying
            # `trend=None`, instead of the default `trend='spine'`, we skip this process
            trend = 'spline'
            if len(x) < 3:
                trend = None
            seasons, trend = seasonal.fit_seasons(
                x,
                trend=trend,
                period=period
            )
        else:
            # There is no enough data, nothing to learn
            trend = x
            seasons = np.zeros(len(trend))
        model = SeasonalModel(seasons, trend)
        return model

    def simulate_history(self)-> Iterable[Number]:
        """
        Apply the learned model to object's history and return the predicted values

        Returns
        -------
        An iterable of modeled numbers
        """
        super(TimeseriesEstimator_seasonal, self).simulate_history()
        n_history = len(self.history)
        if n_history:
            if self.model.seasons is not None:
                residuals = np.ceil(n_history / float(len(self.model.seasons)))
                seasonal_component = np.tile(self.model.seasons, int(residuals))[0:n_history]
            else:
                seasonal_component = 0
            reconstructed = np.asanyarray(self.model.trend) + seasonal_component
        else:
            reconstructed = np.array([])
        return reconstructed

    def predict_timesteps(self, n_steps:int, model=None)->Iterable[Number]:  # pylint: disable=arguments-differ
        """
        Apply this model to predict the specified number of time steps.

        Parameters
        ---------
        n_steps
            Number of time steps to predict

        model
            If not None, use a different model for prediction

        Returns
        ------
            An iterable of length `timesteps_forward` with the forecast
        """
        super(TimeseriesEstimator_seasonal, self).predict_timesteps(n_steps)

        if model is None:
            model = self.model
        if (model is None) or (model.trend is None):
            n_history = 0
        else:
            n_history = len(model.trend)
        if not n_history:
            forecast = np.zeros(n_steps)
        elif n_history == 1:
            forecast = [model.trend[0]] * n_steps
        else:
            # We have no other choice but to assume constant trend slope.
            # There may be several possibilities assess this slope:
            #  * the slope between the last two known points
            #  * average slope over the entire history
            if self.trend_forecast_method == 'last':
                trend_delta_y = model.trend[-1] - model.trend[-2]
            elif self.trend_forecast_method == 'average':
                trend_delta_y = np.polyfit(np.arange(len(model.trend)), model.trend, 1)[0]
            if model.seasons is not None:
                residuals = np.ceil(float(n_history + n_steps) / len(model.seasons) + 1).astype(int)
                seasons = np.tile(
                    model.seasons,
                    residuals
                )[(n_history + 1):]
            else:
                seasons = np.zeros(n_steps)
            last_y_trend_value = model.trend[-1]
            forecast = np.empty(n_steps)
            for i in range(n_steps):
                delta_season = seasons[i]
                new_y = last_y_trend_value + trend_delta_y * (i + 1) + delta_season

                forecast[i] = new_y
        return forecast

    @staticmethod
    def score_from_residual(current_residual, residuals):
        if len(residuals) < 3:
            return 0.0
        loc = 0.0  # in the ideal world, the average residual should be 0
        try:
            scale = np.std(residuals, ddof=2)
        except BaseException:
            scale = 0
        scale += 1e-32
        score = norm.cdf(current_residual, loc=loc, scale=scale)
        if score > 0.5:
            score = 1 - score
        score = 1.0 - 2 * score
        score *= np.sign(current_residual)
        return score

    def _observation_handler(self, X:np.ndarray, return_expectation:bool, update_state:bool, ongoing_learning:bool):
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

        update_state
            Whether the state should be updated after the call
        ongoing_learning
            If True, learn while estimating

        Returns
        -------
        p : ndarray, shape (M,) or a tuple of two such arrays, where M is the length of X,
        depending on the value of `return_expectation`. Anomaly scores, with or without the expected values
        """
        X = self.validate_x(X)
        observed = X[:, 1]
        copy_residuals = self.residuals[:]
        copy_history = self.history[:]
        copy_scores = self.scores[:]
        current_model = self.model.copy()
        ret_scores = []
        ret_expectations = []
        n_keep = int(self.history_days * self.points_per_day)
        for obs_value in observed:
            pred_value = self.predict_timesteps(1, model=current_model)[0]
            residual = obs_value - pred_value
            current_score = self.score_from_residual(current_residual=residual, residuals=copy_residuals)
            ret_scores.append(current_score)
            ret_expectations.append(pred_value)
            copy_residuals.append(residual)
            copy_history.append(obs_value)
            copy_scores.append(current_score)
            copy_residuals = copy_residuals[-n_keep:]
            copy_history = copy_history[-n_keep:]
            copy_scores = copy_scores[-n_keep:]
            if ongoing_learning:
                current_model = self._get_model(x=copy_history, period_days=self.period_days,
                                                points_per_day=self.points_per_day)
        ret_scores = np.array(ret_scores)
        ret_expectations = np.array(ret_expectations)
        if update_state:
            self.history = copy_history
            self.residuals = copy_residuals
            self.scores = copy_scores
            self.model = current_model
        if return_expectation:
            return ret_scores, ret_expectations
        return ret_scores

    def learn(self, X):
        update_state = True
        ongoing_learning = True
        _ = self._observation_handler(
            X=X, return_expectation=False, update_state=update_state, ongoing_learning=ongoing_learning
        )
        return self

    def assess(self, X, return_expectation=False):
        update_state = False
        ongoing_learning = False
        ret = self._observation_handler(
            X=X, return_expectation=return_expectation, update_state=update_state, ongoing_learning=ongoing_learning
        )
        return ret

    def assess_learn(self, X, return_expectation=False, reset_first=False):
        if reset_first:
            self.reset()
        update_state = True
        ongoing_learning = True
        ret = self._observation_handler(
            X=X, return_expectation=return_expectation, update_state=update_state, ongoing_learning=ongoing_learning
        )
        return ret

    def reset(self):
        self.model = SeasonalModel()
        self.history = []
        self.residuals = []

    def __eq__(self, other):
        if self is other:
            return True
        attributes = ['points_per_day', 'period_days', 'history_days', 'model', 'history', 'residuals']
        for attr in attributes:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def serialize(self):
        return self.to_json()

    def to_json(self, indent=None):
        """ Return JSON representation of the object """
        return json.dumps(self.to_dict(), indent=indent)

    def to_dict(self):
        ret = dict(
            history_days=self.history_days,
            period_days=self.period_days,
            points_per_day=self.points_per_day,
            history=list(self.history),
            scores=list(self.scores),
            model=self.model.to_dict() if self.model is not None else None,
            residuals=list(self.residuals)
        )
        return ret

    @classmethod
    def load(cls, data):
        if hasattr(data, 'items'):
            dct = data
        else:
            dct = cls._dict_from_json(data)
        if ('model' in dct.keys()) and hasattr(dct['model'], 'keys'):
            dct['model'] = SeasonalModel(**dct['model'])
        return cls(**dct)

    @staticmethod
    def _dict_from_json(data)->Dict:
        """
        Create an instance from JSON data

        Parameters
        ----------
        data
            either a filename, an open filehandle or a JSON string

        Returns
        -------
        the created object
        """
        if hasattr(data, 'items'):
            # it's already a dict
            return data
        else:
            if hasattr(data, 'read'):
                data = data.read()
            elif os.path.exists(data):
                with open(data) as inp:
                    data = inp.read()
            return json.loads(data)

    def __repr__(self):
        ret = dict(
            model_type=self.model_type,
            points_per_day=self.points_per_day,
            period_days=self.period_days,
            history_days=self.history_days
        )
        return '%s: %s' % (self.__class__.__name__, json.dumps(ret))
