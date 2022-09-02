"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """
    Floats multiplication.

    Args:
        x: a float to be multiplied with another float
        y: another float to be multiplied with the first float argument

    Returns:
        Resulting float after the multiplication
    """
    return x * y


def id(x: float) -> float:
    """
    Float indentification

    Args:
        x: a float to be identified

    Returns:
        The identified float. Identical to the input float.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Floats addition.

    Args:
        x: a float to be added with another float
        y: another float to be added with the first float argument

    Returns:
        Resulting float after the addition
    """
    return x + y


def neg(x: float) -> float:
    """
    Float negation.

    Args:
        x: a float to be inversed in sign

    Returns:
        -x, the same float value with opposite sign as originally inputted
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    Floats comparison.

    Args:
        x: a float to be compared in size with the second float
        y: the second float to be compared in size with the first float

    Returns:
        Float, 1.0 if x is greater than y, otherwise 0.0
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    Floats equality comparison.

    Args:
        x: a float to be compared with the second float
        y: the second float to be compared with the first float

    Returns:
        float: 1.0 if x is identical to y, otherwise 0.0
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    Maximum float identifier.

    Args:
        x: a float to be compared with the second float in size
        y: the second float to be compared with the first float in size

    Returns:
        float: x or y, whichever is greater in size
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    Floats proximity comparison.

    Args:
        x: a float to be compared with the second float
        y: the second float to be compared with the first float

    Returns:
        float: 1.0 if x is close to y, otherwise 0.0. The closeness is determined by whether their difference is less than 1e-2.
    """
    return abs(x - y) < 10 ** (-2)


def sigmoid(x: float) -> float:
    """
    Sigmoid computation.

    Args:
        x: a float to be calculated the sigmoid value of

    Returns:
        float: sigmoid value of the input value, calculated using two different equations depending on the sign of the input float.
    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """
    Rectifier computation.

    Args:
        x: a float to be rectified

    Returns:
        float: returns the larger value between 0 and x
    """
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    """
    Log computation.

    Args:
        x: a float whose log value to compute

    Returns:
        float: returns the log value of the input float
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """
    Exponent computation.

    Args:
        x: a float whose exponential value to compute

    Returns:
        float: returns the exponential value of the input float
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    Derivative of a log computation.

    Args:
        x: a float whose derivative to compute
        d: a float to be multiplied to the derivative of the first argument

    Returns:
        float: returns the product of the derivative of the first input argument with the second input argument
    """
    return d * 1 / (x * math.log(10))


def inv(x: float) -> float:
    """
    Inverse computation.

    Args:
        x: a float whose inverse to compute

    Returns:
        float: returns the inverse of the inputted float
    """
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """
    Inverse value derivation computation.

    Args:
        x: a float whose derivation to compute
        d: a float to be multiplied with the derivative value of the first input

    Returns:
        float: returns the derivation of the inputted float, multiplied by the second input value.
    """
    return d * -1 / (x**2)


def relu_back(x: float, d: float) -> float:
    """
    Rectifier derivative computation.

    Args:
        x: a float whose derivation to compute
        d: a float to be multiplied with the derivative value of the first input

    Returns:
        float: returns d if x is greater than 0. Returns 0 otherwise.
    """
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order mapping Function

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    A function to negate each element in a list.

    Args:
        ls: a list whose elements to be negated

    Returns:
        Another list with each element negated
    """
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Zipping

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.
    """

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Adding elements of two lists using 'zipWith' and 'add'

    Args:
        ls1: first list to be added with the second list
        ls2: second list to be added with the first list

    Returns:
        Another list with elements from the two input lists
    """
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def apply(ls: Iterable[float]) -> float:
        ans = start
        for x in ls:
            ans = fn(ans, x)
        return ans

    return apply


def sum(ls: Iterable[float]) -> float:
    """
    Summing function for elemnts in a list using `reduce` and `add`

    Args:
        ls: a list whose elements to be summed

    Returns:
        Float, which is the result of summing the input list
    """
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    """
    Finding product of a list using 'reduce' and 'mul'

    Args:
        ls: an iterable list whose elements to be multiplied with each other

    Returns:
        Float, which is the product of elements in the input list
    """
    return reduce(mul, 1)(ls)
