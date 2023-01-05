import numpy as np
from typing import Optional, Union, List

def get_labels_np_array(preds: np.ndarray) -> np.ndarray:
    """
    Returns the label of the most probable class given a array of class confidences.

    :param preds: Array of class confidences, nb of instances as first dimension.
    :return: Labels.
    """
    if len(preds.shape) >= 2:
        preds_max = np.amax(preds, axis=1, keepdims=True)
    else:
        preds_max = np.round(preds)
    y = preds == preds_max
    y = y.astype(np.uint8)
    return y

def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical

def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int], return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes. If None the number of classes is determined automatically.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    labels_return = labels

    if len(labels.shape) == 2 and labels.shape[1] > 1:  # multi-class, one-hot encoded
        if not return_one_hot:
            labels_return = np.argmax(labels, axis=1)
            labels_return = np.expand_dims(labels_return, axis=1)
    elif len(labels.shape) == 2 and labels.shape[1] == 1:
        if nb_classes is None:
            nb_classes = np.max(labels) + 1
        if nb_classes > 2:  # multi-class, index labels
            if return_one_hot:
                labels_return = to_categorical(labels, nb_classes)
            else:
                labels_return = np.expand_dims(labels, axis=1)
        elif nb_classes == 2:  # binary, index labels
            if return_one_hot:
                labels_return = to_categorical(labels, nb_classes)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, "
                "nb_classes)"
            )
    elif len(labels.shape) == 1:  # index labels
        if return_one_hot:
            labels_return = to_categorical(labels, nb_classes)
        else:
            labels_return = np.expand_dims(labels, axis=1)
    else:
        raise ValueError(
            "Shape of labels not recognised."
            "Please provide labels in shape (nb_samples,) or (nb_samples, "
            "nb_classes)"
        )

    return labels_return

def projection_l1_1(values: np.ndarray, eps: Union[int, float, np.ndarray]) -> np.ndarray:
    """
    This function computes the orthogonal projections of a batch of points on L1-balls of given radii
    The batch size is  m = values.shape[0].  The points are flattened to dimension
    n = np.prod(value.shape[1:]).  This is required to facilitate sorting.

    If a[0] <= ... <= a[n-1], then the projection can be characterized using the largest  j  such that
    a[j+1] +...+ a[n-1] - a[j]*(n-j-1) >= eps. The  ith  coordinate of projection is equal to  0
    if i=0,...,j.

    :param values:  A batch of  m  points, each an ndarray
    :param eps:  The radii of the respective L1-balls
    :return: projections
    """
    # pylint: disable=C0103

    shp = values.shape
    a = values.copy()
    n = np.prod(a.shape[1:])
    m = a.shape[0]
    a = a.reshape((m, n))
    sgns = np.sign(a)
    a = np.abs(a)

    a_argsort = a.argsort(axis=1)
    a_sorted = np.zeros((m, n))
    for i in range(m):
        a_sorted[i, :] = a[i, a_argsort[i, :]]
    a_argsort_inv = a.argsort(axis=1).argsort(axis=1)
    mat = np.zeros((m, 2))

    #   if  a_sorted[i, n-1]  >= a_sorted[i, n-2] + eps,  then the projection is  [0,...,0,eps]
    done = early_done = False
    active = np.array([1] * m)
    after_vec = np.zeros((m, n))
    proj = a_sorted.copy()
    j = n - 2
    while j >= 0:
        mat[:, 0] += a_sorted[:, j + 1]  # =  sum(a_sorted[: i] :  i = j + 1,...,n-1
        mat[:, 1] = a_sorted[:, j] * (n - j - 1) + eps
        #  Find the max in each problem  max{ sum{a_sorted[:, i] : i=j+1,..,n-1} , a_sorted[:, j] * (n-j-1) + eps }
        row_maxes = np.max(mat, axis=1)
        #  Set to  1  if  max >  a_sorted[:, j] * (n-j-1) + eps  >  sum ;  otherwise, set to  0
        ind_set = np.sign(np.sign(row_maxes - mat[:, 0]))
        #  ind_set = ind_set.reshape((m, 1))
        #   Multiplier for activation
        act_multiplier = (1 - ind_set) * active
        act_multiplier = np.transpose([np.transpose(act_multiplier)] * n)
        #  if done, the projection is supported by the current indices  j+1,..,n-1   and the amount by which each
        #  has to be reduced is  delta
        delta = (mat[:, 0] - eps) / (n - j - 1)
        #    The vector of reductions
        delta_vec = np.transpose(np.array([delta] * (n - j - 1)))
        #   The sub-vectors:  a_sorted[:, (j+1):]
        a_sub = a_sorted[:, (j + 1) :]
        #   After reduction by delta_vec
        a_after = a_sub - delta_vec
        after_vec[:, (j + 1) :] = a_after
        proj += act_multiplier * (after_vec - proj)
        active = active * ind_set
        if sum(active) == 0:
            done = early_done = True
            break
        j -= 1
    if not early_done:
        delta = (mat[:, 0] + a_sorted[:, 0] - eps) / n
        ind_set = np.sign(np.maximum(delta, 0))
        act_multiplier = ind_set * active
        act_multiplier = np.transpose([np.transpose(act_multiplier)] * n)
        delta_vec = np.transpose(np.array([delta] * n))
        a_after = a_sorted - delta_vec
        proj += act_multiplier * (a_after - proj)
        done = True
    if not done:
        proj = active * (a_sorted - proj)

    for i in range(m):
        proj[i, :] = proj[i, a_argsort_inv[i, :]]

    proj = sgns * proj
    proj = proj.reshape(shp)

    return proj


def projection_l1_2(values: np.ndarray, eps: Union[int, float, np.ndarray]) -> np.ndarray:
    """
    This function computes the orthogonal projections of a batch of points on L1-balls of given radii
    The batch size is  m = values.shape[0].  The points are flattened to dimension
    n = np.prod(value.shape[1:]).  This is required to facilitate sorting.

    Starting from a vector  a = (a1,...,an)  such that  a1 >= ... >= an >= 0,  a1 + ... + an > 1,
    we first move to  a' = a - (t,...,t)  such that either a1 + ... + an >= 1 ,  an >= 0,
    and  min( a1 + ... + an  - nt - 1, an -t ) = 0.  This means  t = min( (a1 + ... + an - 1)/n, an).
    If  t = (a1 + ... + an - 1)/n , then  a' is the desired projection.  Otherwise, the problem is reduced to
    finding the projection of  (a1 - t, ... , a{n-1} - t ).

    :param values:  A batch of  m  points, each an ndarray
    :param eps:  The radii of the respective L1-balls
    :return: projections
    """
    # pylint: disable=C0103
    shp = values.shape
    a = values.copy()
    n = np.prod(a.shape[1:])
    m = a.shape[0]
    a = a.reshape((m, n))
    sgns = np.sign(a)
    a = np.abs(a)
    a_argsort = a.argsort(axis=1)
    a_sorted = np.zeros((m, n))
    for i in range(m):
        a_sorted[i, :] = a[i, a_argsort[i, :]]

    a_argsort_inv = a.argsort(axis=1).argsort(axis=1)
    row_sums = np.sum(a, axis=1)
    mat = np.zeros((m, 2))
    mat0 = np.zeros((m, 2))
    a_var = a_sorted.copy()
    for j in range(n):
        mat[:, 0] = (row_sums - eps) / (n - j)
        mat[:, 1] = a_var[:, j]
        mat0[:, 1] = np.min(mat, axis=1)
        min_t = np.max(mat0, axis=1)
        if np.max(min_t) < 1e-8:
            continue
        row_sums = row_sums - a_var[:, j] * (n - j)
        a_var[:, (j + 1) :] = a_var[:, (j + 1) :] - np.matmul(min_t.reshape((m, 1)), np.ones((1, n - j - 1)))
        a_var[:, j] = a_var[:, j] - min_t
    proj = np.zeros((m, n))
    for i in range(m):
        proj[i, :] = a_var[i, a_argsort_inv[i, :]]

    proj = sgns * proj
    proj = proj.reshape(shp)
    return proj

def projection(values: np.ndarray, eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]) -> np.ndarray:
    """
    Project `values` on the L_p norm ball of size `eps`.

    :param values: Array of perturbations to clip.
    :param eps: Maximum norm allowed.
    :param norm_p: L_p norm to use for clipping.
            Only 1, 2 , `np.Inf` 1.1 and 1.2 supported for now.
            1.1 and 1.2 compute orthogonal projections on l1-ball, using two different algorithms
    :return: Values of `values` after projection.
    """
    # Pick a small scalar to avoid division by 0
    tol = 10e-8
    values_tmp = values.reshape((values.shape[0], -1))

    if norm_p == 2:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError("The parameter `eps` of type `np.ndarray` is not supported to use with norm 2.")

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1
        )

    elif norm_p == 1:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError("The parameter `eps` of type `np.ndarray` is not supported to use with norm 1.")

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)),
            axis=1,
        )
    elif norm_p == 1.1:
        values_tmp = projection_l1_1(values_tmp, eps)
    elif norm_p == 1.2:
        values_tmp = projection_l1_2(values_tmp, eps)

    elif norm_p in [np.inf, "inf"]:
        if isinstance(eps, np.ndarray):
            eps = eps * np.ones_like(values)
            eps = eps.reshape([eps.shape[0], -1])  # type: ignore

        values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)

    else:
        raise NotImplementedError(
            'Values of `norm_p` different from 1, 2, `np.inf` and "inf" are currently not ' "supported."
        )

    values = values_tmp.reshape(values.shape)

    return values