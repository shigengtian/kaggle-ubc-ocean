import numpy as np
import pandas as pd
import pandas.api.types

# import kaggle_metric_utilities

import sklearn.metrics

from typing import Sequence, Union, Optional


class ParticipantVisibleError(Exception):
    pass

import numpy as np
import pandas as pd
import pandas.api.types

from typing import Union


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def treat_as_participant_error(error_message: str, solution: Union[pd.DataFrame, np.ndarray]) -> bool:
    '''
    Many metrics can raise more errors than can be handled manually. This function attempts
    to identify errors that can be treated as ParticipantVisibleError without leaking any competition data.

    If the solution is purely numeric, and there are no numbers in the error message,
    then the error message is sufficiently unlikely to leak usable data and can be shown to participants.

    We expect this filter to reject many safe messages. It's intended only to reduce the number of errors we need to manage manually.
    '''
    # This check treats bools as numeric
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all([pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
        solution_has_bools = any([pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values])
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if 'true' in error_message.lower() or 'false' in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    '''
    Call score. If that raises an error and that already been specifically handled, just raise it.
    Otherwise make a conservative attempt to identify potential participant visible errors.
    '''
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == 'ParticipantVisibleError':
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == 'HostVisibleError':
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err
    return score_result

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, weights_column_name: Optional[str]=None, adjusted: bool=False) -> float:
    '''
    Wrapper for https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.

    The best value is 1 and the worst value is 0 when ``adjusted=False``.

    Parameters
    ----------
    solution : 1d DataFrame
    Ground truth (correct) target values.

    submission : 1d DataFrame
    Estimated targets as returned by a classifier.

    weights_column_name: optional str, the name of the sample weights column in the solution file.

    adjusted : bool, default=False
    When true, the result is adjusted for chance, so that random
    performance would score 0, while keeping perfect performance at a score
    of 1.

    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
    The balanced accuracy and its posterior distribution.
    Proceedings of the 20th International Conference on Pattern
    Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
    `Fundamentals of Machine Learning for Predictive Data Analytics:
    Algorithms, Worked Examples, and Case Studies
    <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.

    Examples
    --------

    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.625
    '''
    # Skip sorting and equality checks for the row_id_column since that should already be handled
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    sample_weight = None
    if weights_column_name:
        if weights_column_name not in solution.columns:
            raise ValueError(f'The solution weights column {weights_column_name} is not found')
        sample_weight = solution.pop(weights_column_name).values
        if not pandas.api.types.is_numeric_dtype(sample_weight):
            raise ParticipantVisibleError('The solution weights are not numeric')

    if len(submission.columns) > 1:
        raise ParticipantVisibleError(f'The submission can only include one column of predictions. Found {len(submission.columns)}')

    solution = solution.values
    submission = submission.values

    score_result = safe_call_score(sklearn.metrics.balanced_accuracy_score, solution, submission, sample_weight=sample_weight, adjusted=adjusted)

    return score_result