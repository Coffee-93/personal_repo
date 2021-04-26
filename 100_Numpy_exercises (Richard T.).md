# 100 numpy exercises - Richard T.

### Instructions:

This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow
and in the numpy documentation. The goal of this collection is to offer a quick reference for both old
and new users but also to provide a set of exercises for those who teach.


If you find an error or think you've a better way to solve some of them, feel
free to open an issue at <https://github.com/rougier/numpy-100>.

File automatically generated. See the documentation to update questions/answers/hints programmatically.

Run the `initialize.py` module, then for each question you can query the
answer or an hint with `hint(n)` or `answer(n)` for `n` question number.


```python
%run initialise.py
```

#### 1. Import the numpy package under the name `np` (★☆☆)


```python
import numpy as np
```

#### 2. Print the numpy version and the configuration (★☆☆)


```python
np.__version__
```




    '1.19.1'




```python
np.show_config()
```

    blas_mkl_info:
      NOT AVAILABLE
    blis_info:
      NOT AVAILABLE
    openblas_info:
        library_dirs = ['D:\\a\\1\\s\\numpy\\build\\openblas_info']
        libraries = ['openblas_info']
        language = f77
        define_macros = [('HAVE_CBLAS', None)]
    blas_opt_info:
        library_dirs = ['D:\\a\\1\\s\\numpy\\build\\openblas_info']
        libraries = ['openblas_info']
        language = f77
        define_macros = [('HAVE_CBLAS', None)]
    lapack_mkl_info:
      NOT AVAILABLE
    openblas_lapack_info:
        library_dirs = ['D:\\a\\1\\s\\numpy\\build\\openblas_lapack_info']
        libraries = ['openblas_lapack_info']
        language = f77
        define_macros = [('HAVE_CBLAS', None)]
    lapack_opt_info:
        library_dirs = ['D:\\a\\1\\s\\numpy\\build\\openblas_lapack_info']
        libraries = ['openblas_lapack_info']
        language = f77
        define_macros = [('HAVE_CBLAS', None)]
    

#### 3. Create a null vector of size 10 (★☆☆)


```python
np.zeros((10,))
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
np.zeros((10,)).shape
```




    (10,)




```python
x = np.zeros((10,))
```

#### 4. How to find the memory size of any array (★☆☆)


```python
print("Size of array: ", x.size)
```

    Size of array:  10
    


```python
print("Memory size of array element (in bytes): ", x.itemsize)
```

    Memory size of array element (in bytes):  8
    


```python
print("Memory size of array(in bytes): ", x.size * x.itemsize)
```

    Memory size of array(in bytes):  80
    

#### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)


```python
np.info(np.add)
```

    add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
    Add arguments element-wise.
    
    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.
    
    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.
    
    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])
    

#### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)


```python
z = np.zeros(10)
```


```python
z[4] = 1
```


```python
z
```




    array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])



#### 7. Create a vector with values ranging from 10 to 49 (★☆☆)


```python
np.arange(10,50)
```




    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
           27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
           44, 45, 46, 47, 48, 49])




```python
y = np.arange(10,50)
```

#### 8. Reverse a vector (first element becomes last) (★☆☆)


```python
y[::-1]
```




    array([49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
           32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
           15, 14, 13, 12, 11, 10])



#### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)


```python
a = np.arange(0,9)
```


```python
a.reshape(3,3)
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])



#### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)


```python
b = np.array([1,2,0,0,4,0])
```


```python
np.nonzero(b)
```




    (array([0, 1, 4], dtype=int64),)



#### 11. Create a 3x3 identity matrix (★☆☆)


```python
np.eye(3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])



#### 12. Create a 3x3x3 array with random values (★☆☆)


```python
np.random.rand(3,3,3)
```




    array([[[0.20195606, 0.72574115, 0.72971648],
            [0.2764139 , 0.67931363, 0.93459592],
            [0.414943  , 0.78578787, 0.18040016]],
    
           [[0.34886065, 0.08449621, 0.75937294],
            [0.22700628, 0.65537544, 0.36643547],
            [0.61042388, 0.01326999, 0.04482956]],
    
           [[0.98795561, 0.58060967, 0.44310073],
            [0.81097125, 0.51335856, 0.61644834],
            [0.90428715, 0.82527   , 0.78434922]]])



#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)


```python
c = np.random.rand(10,10)
```


```python
np.amin(c), np.amax(c)
```




    (0.0006571333908146348, 0.9929577720482958)




```python
c
```




    array([[8.47056841e-01, 1.78927769e-01, 4.59054488e-01, 1.97543162e-01,
            4.09551491e-01, 4.58185329e-02, 8.54040114e-01, 9.78645033e-01,
            4.33391272e-01, 2.66194053e-01],
           [4.02327820e-01, 1.99097961e-01, 6.23507714e-01, 2.22119050e-01,
            2.42998400e-01, 3.99129964e-01, 5.72443442e-01, 6.93101769e-01,
            5.96550156e-01, 1.69604316e-01],
           [1.02725821e-01, 4.12415423e-01, 9.48870770e-02, 2.63616961e-01,
            5.58570817e-01, 2.31872215e-01, 7.27561555e-01, 9.10912026e-01,
            2.84234937e-01, 9.92957772e-01],
           [7.50976042e-01, 1.52732590e-01, 1.85216422e-01, 5.78405089e-01,
            5.28387094e-01, 7.61653089e-01, 3.23369761e-01, 5.55825782e-01,
            5.83194075e-03, 1.55713735e-01],
           [9.91893648e-01, 7.65634097e-02, 8.21508126e-01, 4.29967163e-01,
            4.13692556e-01, 8.10162996e-02, 8.16785310e-01, 3.42367764e-01,
            1.01116565e-01, 6.88017583e-01],
           [7.93596920e-01, 4.00502948e-01, 2.16452980e-01, 3.77254628e-02,
            9.64150410e-01, 6.03321525e-01, 1.68668366e-01, 8.11848340e-01,
            4.66183805e-01, 4.79786759e-01],
           [8.38951435e-01, 4.61894586e-01, 8.96832328e-01, 6.96822839e-01,
            7.03937468e-02, 1.05895034e-01, 7.56165396e-01, 4.26205724e-01,
            2.08226639e-01, 6.90522167e-01],
           [9.33279092e-01, 9.62188574e-02, 6.65289400e-01, 2.47776826e-01,
            3.79199810e-01, 5.79313253e-01, 2.89296531e-01, 6.70972716e-01,
            3.06226616e-01, 6.39294948e-01],
           [9.31703300e-01, 7.97355258e-01, 8.13424881e-01, 8.56428855e-01,
            2.57938162e-01, 1.47623833e-02, 3.97186155e-01, 1.93885233e-01,
            6.57133391e-04, 2.31455577e-01],
           [3.42993359e-01, 4.91821205e-01, 7.43802654e-02, 5.06477669e-01,
            3.05646730e-02, 2.32922522e-01, 1.76459868e-01, 7.85525163e-01,
            1.10821248e-01, 7.64352981e-02]])



#### 14. Create a random vector of size 30 and find the mean value (★☆☆)


```python
d = np.random.rand(30,)
```


```python
np.mean(d)
```




    0.514632531847768




```python
d
```




    array([0.48050592, 0.68634216, 0.69097435, 0.80981739, 0.93968173,
           0.1734264 , 0.63606344, 0.21969857, 0.72719349, 0.63578504,
           0.8557641 , 0.12952217, 0.64983819, 0.73016002, 0.79617604,
           0.06170542, 0.90235332, 0.06880447, 0.42598221, 0.39346752,
           0.04014788, 0.68430442, 0.65362925, 0.09782423, 0.58392169,
           0.23827668, 0.51854002, 0.78552182, 0.03541402, 0.78813401])



#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)


```python
e = np.ones((5,5))
```


```python
e
```




    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])




```python
e[1:-1, 1:-1] = 0
```


```python
e
```




    array([[1., 1., 1., 1., 1.],
           [1., 0., 0., 0., 1.],
           [1., 0., 0., 0., 1.],
           [1., 0., 0., 0., 1.],
           [1., 1., 1., 1., 1.]])



#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)


```python
f = np.ones((3,3))
```


```python
f
```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])




```python
np.pad(f, [(1,1), (1,1)], mode='constant')
```




    array([[0., 0., 0., 0., 0.],
           [0., 1., 1., 1., 0.],
           [0., 1., 1., 1., 0.],
           [0., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0.]])



#### 17. What is the result of the following expression? (★☆☆)
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```


```python
0 * np.nan
```




    nan




```python
np.nan == np.nan
```




    False




```python
np.inf > np.nan
```




    False




```python
np.nan - np.nan
```




    nan




```python
np.nan in set([np.nan])
```




    True




```python
0.3 == 3 * 0.1
```




    False



#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)


```python
g = np.array([1,2,3,4])
```


```python
np.diag(g)
```




    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])




```python
np.diag(g, k=-1)
```




    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 2, 0, 0, 0],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 4, 0]])



#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)


```python
h = np.zeros((8,8),dtype=int)
```


```python
h
```




    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])




```python
h[1::2, ::2] = 1
```


```python
h
```




    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 1, 0, 1, 0]])




```python
h[0::2, 1::2] = 1
```


```python
h
```




    array([[0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0]])



#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?


```python
np.unravel_index(100, (6,7,8))
```




    (1, 5, 4)



#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)


```python
a = np.array([0, 1])
```


```python
a
```




    array([0, 1])




```python
np.tile(a,(2,2))
```




    array([[0, 1, 0, 1],
           [0, 1, 0, 1]])




```python
np.tile(a,(4,4))
```




    array([[0, 1, 0, 1, 0, 1, 0, 1],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [0, 1, 0, 1, 0, 1, 0, 1]])



#### 22. Normalize a 5x5 random matrix (★☆☆)


```python
i = np.random.rand(5,5)
```


```python
i
```




    array([[0.41480072, 0.32624731, 0.75023633, 0.81364664, 0.73676915],
           [0.40826884, 0.96178642, 0.84681194, 0.85097785, 0.69000424],
           [0.22151212, 0.05936116, 0.30879648, 0.55324725, 0.39389252],
           [0.04010275, 0.4324319 , 0.84971871, 0.5425782 , 0.81986193],
           [0.5818911 , 0.89608551, 0.42154648, 0.39072563, 0.38848092]])




```python
i = (i - np.mean(i))
```


```python
hint(22)
```

    hint: (x -mean)/std
    

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)


```python
#red, green, blue, alpha values

color = np.dtype([("r", np.ubyte),
                  ("g", np.ubyte),
                  ("b", np.ubyte),
                  ("a", np.ubyte)
                 ])
```


```python
color
```




    dtype([('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])



#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)


```python
a = np.ones((5,3))
```


```python
b = np.ones((3,2))
```


```python
a, b
```




    (array([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]),
     array([[1., 1.],
            [1., 1.],
            [1., 1.]]))




```python
np.dot(a, b)
```




    array([[3., 3.],
           [3., 3.],
           [3., 3.],
           [3., 3.],
           [3., 3.]])



#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)


```python
j = np.arange(0,10)
```


```python
j
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
j[(3 < j) & (j < 8)] *= -1
```


```python
j
```




    array([ 0,  1,  2,  3, -4, -5, -6, -7,  8,  9])



#### 26. What is the output of the following script? (★☆☆)
```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```


```python
print(sum(range(5),-1))
```

    9
    


```python
from numpy import *
```


```python
print(sum(range(5),-1))
```

    10
    

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```


```python
Z = np.array([1,2,3])
```


```python
Z
```




    array([1, 2, 3])




```python
# legal
Z**Z
```




    array([ 1,  4, 27], dtype=int32)




```python
# legal
2 << Z >> 2
```




    array([1, 2, 4], dtype=int32)




```python
# legal
Z <- Z
```




    array([False, False, False])




```python
# legal
1j*Z
```




    array([0.+1.j, 0.+2.j, 0.+3.j])




```python
# legal
Z/1/1
```




    array([1., 2., 3.])




```python
# illegal
Z<Z>Z
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-44-6d2bd9eb1fd1> in <module>
    ----> 1 Z<Z>Z
    

    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()


#### 28. What are the result of the following expressions?
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```


```python
np.array(0) / np.array(0)
```

    <ipython-input-46-3585dcb7ab9b>:1: RuntimeWarning: invalid value encountered in true_divide
      np.array(0) / np.array(0)
    




    nan




```python
np.array(0) // np.array(0)
```

    <ipython-input-47-4764261090d0>:1: RuntimeWarning: divide by zero encountered in floor_divide
      np.array(0) // np.array(0)
    




    0




```python
np.array([np.nan]).astype(int).astype(float)
```




    array([-2.14748365e+09])



#### 29. How to round away from zero a float array ? (★☆☆)


```python
z = np.random.uniform(-10,+10,10)
```


```python
z
```




    array([-0.22449264, -0.72062735, -3.95650839, -4.48340875, -7.21484279,
            6.37547818,  8.95210565,  7.97824584,  6.98578044, -0.45132205])




```python
np.where(z > 0, np.ceil(z), np.floor(z))
```




    array([-1., -1., -4., -5., -8.,  7.,  9.,  8.,  7., -1.])



#### 30. How to find common values between two arrays? (★☆☆)


```python
z1 = np.random.randint(0,10,10)
```


```python
z2 = np.random.randint(0,10,10)
```


```python
z1, z2
```




    (array([5, 5, 2, 8, 0, 0, 9, 4, 5, 2]), array([5, 5, 5, 1, 2, 4, 8, 9, 4, 0]))




```python
np.intersect1d(z1,z2)
```




    array([0, 2, 4, 5, 8, 9])



#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)


```python
# ignore all numpy warnings
defaults = np.seterr(all="ignore")
```


```python
z = np.ones(1) / 0
```


```python
# go back to default
_ = np.seterr(**defaults)
```


```python
# confirm error message present
z = np.ones(1) / 0
```

    <ipython-input-9-dedc404ac13b>:2: RuntimeWarning: divide by zero encountered in true_divide
      z = np.ones(1) / 0
    

#### 32. Is the following expressions true? (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

    <ipython-input-12-16339fbc685e>:1: RuntimeWarning: invalid value encountered in sqrt
      np.sqrt(-1) == np.emath.sqrt(-1)
    




    False



#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)


```python
# today's date
today = np.datetime64('today')
print(today)
```

    2021-04-24
    


```python
# yesterday's date
yesterday = np.datetime64('today') - np.timedelta64(1)
print(yesterday)
```

    2021-04-23
    


```python
# tomorrow's date
tomorrow = np.datetime64('today') + np.timedelta64(1)
print(tomorrow)
```

    2021-04-25
    

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)


```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
```


```python
print(Z)
```

    ['2016-07-01' '2016-07-02' '2016-07-03' '2016-07-04' '2016-07-05'
     '2016-07-06' '2016-07-07' '2016-07-08' '2016-07-09' '2016-07-10'
     '2016-07-11' '2016-07-12' '2016-07-13' '2016-07-14' '2016-07-15'
     '2016-07-16' '2016-07-17' '2016-07-18' '2016-07-19' '2016-07-20'
     '2016-07-21' '2016-07-22' '2016-07-23' '2016-07-24' '2016-07-25'
     '2016-07-26' '2016-07-27' '2016-07-28' '2016-07-29' '2016-07-30'
     '2016-07-31']
    

#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)


```python
A = np.ones(3)*1
```


```python
A
```




    array([1., 1., 1.])




```python
B = np.ones(3)*2
```


```python
B
```




    array([2., 2., 2.])




```python
# computations in place
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
```




    array([-1.5, -1.5, -1.5])



#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)


```python
Z = np.random.uniform(0,10,10)
```


```python
print(Z - Z%1)
```

    [2. 7. 0. 7. 8. 6. 3. 6. 7. 7.]
    


```python
print(Z // 1)
```

    [2. 7. 0. 7. 8. 6. 3. 6. 7. 7.]
    


```python
print(np.floor(Z))
```

    [2. 7. 0. 7. 8. 6. 3. 6. 7. 7.]
    


```python
print(Z.astype(int))
```

    [2 7 0 7 8 6 3 6 7 7]
    

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)


```python
k = np.zeros((5,5))
```


```python
k
```




    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])




```python
k += np.arange(5)
```


```python
k
```




    array([[0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.]])



#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)


```python
def generate():
    for x in range(10):
        yield x
```


```python
l = np.fromiter(generate(),dtype=float,count=-1)
```


```python
l
```




    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])



#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)


```python
m = np.linspace(0, 1, 11, endpoint=False)[1:]
```


```python
m
```




    array([0.09090909, 0.18181818, 0.27272727, 0.36363636, 0.45454545,
           0.54545455, 0.63636364, 0.72727273, 0.81818182, 0.90909091])



#### 40. Create a random vector of size 10 and sort it (★★☆)


```python
n = np.random.random(10)
```


```python
n.sort()
```


```python
print(n)
```

    [0.09344588 0.17312059 0.17559889 0.27828809 0.30398433 0.43367415
     0.44670456 0.51326004 0.81625951 0.82083992]
    

#### 41. How to sum a small array faster than np.sum? (★★☆)


```python
o = np.arange(10)
```


```python
o
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# reduce function
np.add.reduce(o)
```




    45



#### 42. Consider two random array A and B, check if they are equal (★★☆)


```python
A = np.random.randint(0,2,5)
```


```python
A
```




    array([0, 1, 1, 0, 1])




```python
B = np.random.randint(0,2,5)
```


```python
B
```




    array([1, 1, 0, 0, 1])




```python
# allclose function - identical shape and tolerance for comparison of values
equal = np.allclose(A, B)
print(equal)
```

    False
    


```python
# array_equal function - both shape and element values, no tolerance
equal = np.array_equal(A, B)
print(equal)
```

    False
    

#### 43. Make an array immutable (read-only) (★★☆)


```python
p = np.zeros(10)
```


```python
p
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
# make immutable
p.flags.writeable = False
```


```python
# read-only now
p[0] = 1
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-73-5e3dc289d6e4> in <module>
          1 # read-only now
    ----> 2 p[0] = 1
    

    ValueError: assignment destination is read-only


#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)


```python
q = np.random.random((10,2))
q
```




    array([[0.59609329, 0.02881368],
           [0.34655247, 0.64805595],
           [0.10370091, 0.30831773],
           [0.05913228, 0.88364477],
           [0.46246505, 0.74795948],
           [0.47220916, 0.57308306],
           [0.99370782, 0.77893249],
           [0.16251291, 0.25564089],
           [0.22404398, 0.07912515],
           [0.98108828, 0.04023436]])




```python
X,Y = q[:,0], q[:,1]
```


```python
# pythagoras theorem to find hypotenuse
R = np.sqrt(X**2 + Y**2)
```


```python
# tangent function for angle
T = np.arctan2(Y,X)
```


```python
print(R)
print(T)
```

    [0.59678927 0.73489804 0.32529018 0.88562109 0.87938461 0.74256696
     1.26261279 0.30292361 0.23760575 0.98191294]
    [0.04829995 1.07973125 1.24633835 1.50397734 1.01702786 0.88160331
     0.66482462 1.00453405 0.33949426 0.04098696]
    

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)


```python
r = np.random.random(10)
```


```python
r
```




    array([0.45288372, 0.29930519, 0.47564946, 0.5477067 , 0.1978866 ,
           0.0955493 , 0.58892752, 0.74020481, 0.12103364, 0.79591178])




```python
# argmax function - returns index of max value
r[r.argmax()] = 0
```


```python
r
```




    array([0.45288372, 0.29930519, 0.47564946, 0.5477067 , 0.1978866 ,
           0.0955493 , 0.58892752, 0.74020481, 0.12103364, 0.        ])



#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)


```python
s = np.zeros((5,5), [('x',float),('y',float)])
```


```python
s
```




    array([[(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
           [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
           [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
           [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
           [(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)]],
          dtype=[('x', '<f8'), ('y', '<f8')])




```python
s['x'], s['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
```


```python
print(s)
```

    [[(0.  , 0.  ) (0.25, 0.  ) (0.5 , 0.  ) (0.75, 0.  ) (1.  , 0.  )]
     [(0.  , 0.25) (0.25, 0.25) (0.5 , 0.25) (0.75, 0.25) (1.  , 0.25)]
     [(0.  , 0.5 ) (0.25, 0.5 ) (0.5 , 0.5 ) (0.75, 0.5 ) (1.  , 0.5 )]
     [(0.  , 0.75) (0.25, 0.75) (0.5 , 0.75) (0.75, 0.75) (1.  , 0.75)]
     [(0.  , 1.  ) (0.25, 1.  ) (0.5 , 1.  ) (0.75, 1.  ) (1.  , 1.  )]]
    

#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))


```python
X = np.arange(8)
```


```python
X
```




    array([0, 1, 2, 3, 4, 5, 6, 7])




```python
Y = X + 0.5
```


```python
Y
```




    array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])




```python
C = 1.0 / np.subtract.outer(X, Y)
```


```python
C
```




    array([[-2.        , -0.66666667, -0.4       , -0.28571429, -0.22222222,
            -0.18181818, -0.15384615, -0.13333333],
           [ 2.        , -2.        , -0.66666667, -0.4       , -0.28571429,
            -0.22222222, -0.18181818, -0.15384615],
           [ 0.66666667,  2.        , -2.        , -0.66666667, -0.4       ,
            -0.28571429, -0.22222222, -0.18181818],
           [ 0.4       ,  0.66666667,  2.        , -2.        , -0.66666667,
            -0.4       , -0.28571429, -0.22222222],
           [ 0.28571429,  0.4       ,  0.66666667,  2.        , -2.        ,
            -0.66666667, -0.4       , -0.28571429],
           [ 0.22222222,  0.28571429,  0.4       ,  0.66666667,  2.        ,
            -2.        , -0.66666667, -0.4       ],
           [ 0.18181818,  0.22222222,  0.28571429,  0.4       ,  0.66666667,
             2.        , -2.        , -0.66666667],
           [ 0.15384615,  0.18181818,  0.22222222,  0.28571429,  0.4       ,
             0.66666667,  2.        , -2.        ]])




```python
print(np.linalg.det(C))
```

    3638.163637117973
    

#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)


```python
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
```

    -128
    127
    -2147483648
    2147483647
    -9223372036854775808
    9223372036854775807
    


```python
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)
```

    -3.4028235e+38
    3.4028235e+38
    1.1920929e-07
    -1.7976931348623157e+308
    1.7976931348623157e+308
    2.220446049250313e-16
    

#### 49. How to print all the values of an array? (★★☆)


```python
# set_printoptions function
np.set_printoptions(threshold=float("inf"))
```


```python
t = np.zeros((40,40))
t
```




    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0.]])



#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)


```python
u = np.arange(100)
```


```python
u
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
           51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
           68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
           85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])




```python
v = np.random.uniform(0,100)
```


```python
v
```




    61.714647806186576




```python
index = (np.abs(u-v)).argmin()
```


```python
print(u[index])
```

    62
    

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)


```python
w = np.zeros(10, [('position', [('x', float, 1),
                                ('y', float, 1)]),
                  ('color',    [('r', float, 1),
                                ('g', float, 1),
                                ('b', float, 1)])
                 ])
```

    <ipython-input-6-5f430dda51c3>:1: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      w = np.zeros(10, [('position', [('x', float, 1),
    


```python
w
```




    array([((0., 0.), (0., 0., 0.)), ((0., 0.), (0., 0., 0.)),
           ((0., 0.), (0., 0., 0.)), ((0., 0.), (0., 0., 0.)),
           ((0., 0.), (0., 0., 0.)), ((0., 0.), (0., 0., 0.)),
           ((0., 0.), (0., 0., 0.)), ((0., 0.), (0., 0., 0.)),
           ((0., 0.), (0., 0., 0.)), ((0., 0.), (0., 0., 0.))],
          dtype=[('position', [('x', '<f8'), ('y', '<f8')]), ('color', [('r', '<f8'), ('g', '<f8'), ('b', '<f8')])])



#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)


```python
x = np.random.random((10,2))
```


```python
X,Y = np.atleast_2d(x[:,0], x[:,1])
```


```python
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
```


```python
D
```




    array([[0.        , 0.64837941, 0.34493727, 0.65828458, 0.4438798 ,
            0.24802834, 0.70262019, 0.9136641 , 0.56045362, 0.2981511 ],
           [0.64837941, 0.        , 0.57155892, 0.43953532, 0.52923108,
            0.49144523, 0.42427967, 0.29350213, 0.57760699, 0.39555395],
           [0.34493727, 0.57155892, 0.        , 0.7989238 , 0.11135082,
            0.14030178, 0.82435325, 0.86497992, 0.78446877, 0.43887558],
           [0.65828458, 0.43953532, 0.7989238 , 0.        , 0.81565215,
            0.66589352, 0.05669611, 0.47519293, 0.2065163 , 0.37864048],
           [0.4438798 , 0.52923108, 0.11135082, 0.81565215, 0.        ,
            0.20873068, 0.83401688, 0.81831538, 0.82859728, 0.48172933],
           [0.24802834, 0.49144523, 0.14030178, 0.66589352, 0.20873068,
            0.        , 0.69460911, 0.78167424, 0.64426127, 0.29952089],
           [0.70262019, 0.42427967, 0.82435325, 0.05669611, 0.83401688,
            0.69460911, 0.        , 0.42913368, 0.26200035, 0.41598157],
           [0.9136641 , 0.29350213, 0.86497992, 0.47519293, 0.81831538,
            0.78167424, 0.42913368, 0.        , 0.67276553, 0.62998479],
           [0.56045362, 0.57760699, 0.78446877, 0.2065163 , 0.82859728,
            0.64426127, 0.26200035, 0.67276553, 0.        , 0.3475469 ],
           [0.2981511 , 0.39555395, 0.43887558, 0.37864048, 0.48172933,
            0.29952089, 0.41598157, 0.62998479, 0.3475469 , 0.        ]])




```python
# scipi method
import scipy.spatial
```


```python
x = np.random.random((10,2))
```


```python
D = scipy.spatial.distance.cdist(x,x)
```


```python
D
```




    array([[0.        , 0.48193004, 0.74885187, 0.46398736, 0.54886244,
            0.98642744, 0.15253965, 0.20984829, 0.54667491, 0.52260442],
           [0.48193004, 0.        , 0.65647877, 0.82193334, 0.60686191,
            0.95112546, 0.6339972 , 0.6361851 , 0.08658154, 0.8786251 ],
           [0.74885187, 0.65647877, 0.        , 0.66080834, 0.2401842 ,
            0.29495592, 0.8248367 , 0.70348665, 0.61015955, 0.68353609],
           [0.46398736, 0.82193334, 0.66080834, 0.        , 0.42593018,
            0.76738572, 0.39457112, 0.2585732 , 0.8499335 , 0.0596731 ],
           [0.54886244, 0.60686191, 0.2401842 , 0.42593018, 0.        ,
            0.43757546, 0.60187511, 0.47108497, 0.59170197, 0.45476708],
           [0.98642744, 0.95112546, 0.29495592, 0.76738572, 0.43757546,
            0.        , 1.03214858, 0.89263633, 0.90475142, 0.76735962],
           [0.15253965, 0.6339972 , 0.8248367 , 0.39457112, 0.60187511,
            1.03214858, 0.        , 0.14951993, 0.69677639, 0.44736907],
           [0.20984829, 0.6361851 , 0.70348665, 0.2585732 , 0.47108497,
            0.89263633, 0.14951993, 0.        , 0.68399621, 0.3157126 ],
           [0.54667491, 0.08658154, 0.61015955, 0.8499335 , 0.59170197,
            0.90475142, 0.69677639, 0.68399621, 0.        , 0.90459469],
           [0.52260442, 0.8786251 , 0.68353609, 0.0596731 , 0.45476708,
            0.76735962, 0.44736907, 0.3157126 , 0.90459469, 0.        ]])



#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?


```python
Z = (np.random.rand(10)*100).astype(np.float32)
```


```python
Y = Z.view(np.int32)
```


```python
Y[:] = Z
```


```python
Y
```




    array([11, 46,  8, 85, 42, 87, 33, 63, 92, 52])



#### 54. How to read the following file? (★★☆)
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```


```python
# use StringIO library
from io import StringIO
```


```python
# fake file
s = StringIO('''1, 2, 3, 4, 5
                6,  ,  , 7, 8
                 ,  , 9, 10,11

''')
```


```python
Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
print(Z)
```

    [[ 1  2  3  4  5]
     [ 6 -1 -1  7  8]
     [-1 -1  9 10 11]]
    

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)


```python
A = np.arange(9).reshape(3,3)
```


```python
A
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
for index, value in np.ndenumerate(A):
    print(index, value)
```

    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) 3
    (1, 1) 4
    (1, 2) 5
    (2, 0) 6
    (2, 1) 7
    (2, 2) 8
    


```python
for index in np.ndindex(A.shape):
    print(index, A[index])
```

    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) 3
    (1, 1) 4
    (1, 2) 5
    (2, 0) 6
    (2, 1) 7
    (2, 2) 8
    

#### 56. Generate a generic 2D Gaussian-like array (★★☆)


```python
X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
```


```python
D = np.sqrt(X*X+Y*Y)
```


```python
sigma, mu = 1.0, 0.0
```


```python
G = np.exp(-( (D-mu)**2 / (2.0 * sigma**2) ) )
```


```python
G
```




    array([[0.36787944, 0.44822088, 0.51979489, 0.57375342, 0.60279818,
            0.60279818, 0.57375342, 0.51979489, 0.44822088, 0.36787944],
           [0.44822088, 0.54610814, 0.63331324, 0.69905581, 0.73444367,
            0.73444367, 0.69905581, 0.63331324, 0.54610814, 0.44822088],
           [0.51979489, 0.63331324, 0.73444367, 0.81068432, 0.85172308,
            0.85172308, 0.81068432, 0.73444367, 0.63331324, 0.51979489],
           [0.57375342, 0.69905581, 0.81068432, 0.89483932, 0.9401382 ,
            0.9401382 , 0.89483932, 0.81068432, 0.69905581, 0.57375342],
           [0.60279818, 0.73444367, 0.85172308, 0.9401382 , 0.98773022,
            0.98773022, 0.9401382 , 0.85172308, 0.73444367, 0.60279818],
           [0.60279818, 0.73444367, 0.85172308, 0.9401382 , 0.98773022,
            0.98773022, 0.9401382 , 0.85172308, 0.73444367, 0.60279818],
           [0.57375342, 0.69905581, 0.81068432, 0.89483932, 0.9401382 ,
            0.9401382 , 0.89483932, 0.81068432, 0.69905581, 0.57375342],
           [0.51979489, 0.63331324, 0.73444367, 0.81068432, 0.85172308,
            0.85172308, 0.81068432, 0.73444367, 0.63331324, 0.51979489],
           [0.44822088, 0.54610814, 0.63331324, 0.69905581, 0.73444367,
            0.73444367, 0.69905581, 0.63331324, 0.54610814, 0.44822088],
           [0.36787944, 0.44822088, 0.51979489, 0.57375342, 0.60279818,
            0.60279818, 0.57375342, 0.51979489, 0.44822088, 0.36787944]])



#### 57. How to randomly place p elements in a 2D array? (★★☆)


```python
n = 10
p = 3
Z = np.zeros((n,n))
```


```python
np.put(Z, np.random.choice(range(n*n), p, replace = False), 1)
```


```python
Z
```




    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])



#### 58. Subtract the mean of each row of a matrix (★★☆)


```python
X = np.random.rand(5, 10)
```


```python
X
```




    array([[0.60625586, 0.24352194, 0.20003086, 0.93227335, 0.82599863,
            0.34743783, 0.73837222, 0.66763635, 0.86390564, 0.03123113],
           [0.53212493, 0.2446421 , 0.45633678, 0.76941971, 0.92969427,
            0.92172859, 0.58781254, 0.38125938, 0.38786159, 0.7678965 ],
           [0.29786993, 0.4448536 , 0.96678221, 0.87916596, 0.20453326,
            0.65393774, 0.97522861, 0.72006299, 0.60423742, 0.54900022],
           [0.17770586, 0.9093526 , 0.41722348, 0.13813328, 0.42561158,
            0.74951525, 0.05482845, 0.18472084, 0.78517784, 0.13563615],
           [0.78469146, 0.59582493, 0.45603746, 0.0799815 , 0.83416176,
            0.49758756, 0.16124058, 0.33755538, 0.52215527, 0.05423376]])




```python
Y = X - X.mean(axis=1, keepdims=True)
```


```python
Y
```




    array([[ 0.06058948, -0.30214444, -0.34563552,  0.38660696,  0.28033225,
            -0.19822855,  0.19270584,  0.12196997,  0.31823926, -0.51443525],
           [-0.06575271, -0.35323554, -0.14154086,  0.17154207,  0.33181663,
             0.32385095, -0.01006509, -0.21661826, -0.21001605,  0.17001886],
           [-0.33169726, -0.18471359,  0.33721501,  0.24959877, -0.42503394,
             0.02437055,  0.34566142,  0.0904958 , -0.02532978, -0.08056697],
           [-0.22008467,  0.51156207,  0.01943295, -0.25965725,  0.02782105,
             0.35172471, -0.34296209, -0.21306969,  0.3873873 , -0.26215438],
           [ 0.3523445 ,  0.16347796,  0.0236905 , -0.35236547,  0.4018148 ,
             0.06524059, -0.27110639, -0.09479158,  0.08980831, -0.37811321]])



#### 59. How to sort an array by the nth column? (★★☆)


```python
Z = np.random.randint(0, 10, (5,5))
```


```python
Z
```




    array([[8, 1, 4, 3, 5],
           [7, 9, 7, 8, 1],
           [8, 4, 8, 1, 7],
           [6, 4, 2, 5, 9],
           [7, 2, 4, 9, 6]])




```python
# sorting by last column
Z[Z[:, 4].argsort()]
```




    array([[7, 9, 7, 8, 1],
           [8, 1, 4, 3, 5],
           [7, 2, 4, 9, 6],
           [8, 4, 8, 1, 7],
           [6, 4, 2, 5, 9]])



#### 60. How to tell if a given 2D array has null columns? (★★☆)


```python
Z = np.random.randint(0, 3, (3, 10))
```


```python
Z
```




    array([[1, 0, 2, 2, 2, 1, 2, 0, 2, 2],
           [1, 0, 0, 1, 2, 1, 1, 0, 0, 0],
           [0, 0, 1, 0, 2, 2, 1, 1, 2, 2]])




```python
(~Z.any(axis=0)).any()
```




    True



#### 61. Find the nearest value from a given value in an array (★★☆)


```python
Z = np.random.uniform(0,1,10)
```


```python
Z
```




    array([0.73645307, 0.15969755, 0.15100995, 0.22561198, 0.96431667,
           0.74831511, 0.29940875, 0.14886429, 0.8899544 , 0.47256232])




```python
z = 0.5
```


```python
m = Z.flat[np.abs(Z - z).argmin()]
```


```python
m
```




    0.4725623223242146



#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)


```python
A = np.arange(3).reshape(3,1)
A
```




    array([[0],
           [1],
           [2]])




```python
B = np.arange(3).reshape(1,3)
B
```




    array([[0, 1, 2]])




```python
#iterator
it = np.nditer([A, B, None])
for x,y,z in it: z[...] = x + y
```


```python
print(it.operands[2])
```

    [[0 1 2]
     [1 2 3]
     [2 3 4]]
    

#### 63. Create an array class that has a name attribute (★★☆)


```python
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")
```


```python
Z = NamedArray(np.arange(10), "range_10")
```


```python
Z
```




    NamedArray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
Z.name
```




    'range_10'



#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)


```python
Z = np.ones(10)
Z
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
I = np.random.randint(0, len(Z), 20)
I
```




    array([8, 5, 9, 9, 9, 6, 5, 1, 1, 3, 7, 7, 5, 9, 6, 2, 3, 9, 9, 9])




```python
np.add.at(Z, I, 1)
Z
```




    array([1., 3., 2., 3., 1., 4., 3., 3., 2., 8.])



#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)


```python
answer(65)
```

    # Author: Alan G Isaac
    
    X = [1,2,3,4,5,6]
    I = [1,3,9,3,4,1]
    F = np.bincount(I,X)
    print(F)
    


```python
X = [1, 2, 3 , 4, 5, 6]
I = [1, 3, 9, 3, 4, 1]
```


```python
F = np.bincount(I, X)
```


```python
F
```




    array([0., 7., 0., 6., 5., 0., 0., 0., 0., 3.])



#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)


```python
w, h = 256, 256
```


```python
I = np.random.randint(0, 4, (w, h, 3)).astype(np.ubyte)
```


```python
I
```




    array([[[3, 2, 1],
            [0, 0, 3],
            [0, 1, 0],
            ...,
            [1, 3, 3],
            [2, 3, 0],
            [3, 1, 0]],
    
           [[1, 2, 2],
            [2, 1, 1],
            [3, 3, 2],
            ...,
            [0, 0, 1],
            [0, 2, 2],
            [3, 1, 0]],
    
           [[1, 1, 1],
            [1, 0, 2],
            [3, 1, 0],
            ...,
            [0, 3, 3],
            [2, 2, 0],
            [3, 3, 3]],
    
           ...,
    
           [[0, 2, 1],
            [3, 0, 0],
            [3, 0, 0],
            ...,
            [2, 3, 1],
            [0, 2, 1],
            [0, 2, 2]],
    
           [[2, 1, 3],
            [1, 3, 2],
            [2, 3, 3],
            ...,
            [3, 3, 2],
            [3, 1, 3],
            [2, 2, 2]],
    
           [[0, 3, 3],
            [0, 0, 1],
            [1, 3, 2],
            ...,
            [3, 1, 1],
            [2, 1, 1],
            [3, 1, 2]]], dtype=uint8)




```python
colors = np.unique(I.reshape(-1, 3), axis=0)
colors
```




    array([[0, 0, 0],
           [0, 0, 1],
           [0, 0, 2],
           [0, 0, 3],
           [0, 1, 0],
           [0, 1, 1],
           [0, 1, 2],
           [0, 1, 3],
           [0, 2, 0],
           [0, 2, 1],
           [0, 2, 2],
           [0, 2, 3],
           [0, 3, 0],
           [0, 3, 1],
           [0, 3, 2],
           [0, 3, 3],
           [1, 0, 0],
           [1, 0, 1],
           [1, 0, 2],
           [1, 0, 3],
           [1, 1, 0],
           [1, 1, 1],
           [1, 1, 2],
           [1, 1, 3],
           [1, 2, 0],
           [1, 2, 1],
           [1, 2, 2],
           [1, 2, 3],
           [1, 3, 0],
           [1, 3, 1],
           [1, 3, 2],
           [1, 3, 3],
           [2, 0, 0],
           [2, 0, 1],
           [2, 0, 2],
           [2, 0, 3],
           [2, 1, 0],
           [2, 1, 1],
           [2, 1, 2],
           [2, 1, 3],
           [2, 2, 0],
           [2, 2, 1],
           [2, 2, 2],
           [2, 2, 3],
           [2, 3, 0],
           [2, 3, 1],
           [2, 3, 2],
           [2, 3, 3],
           [3, 0, 0],
           [3, 0, 1],
           [3, 0, 2],
           [3, 0, 3],
           [3, 1, 0],
           [3, 1, 1],
           [3, 1, 2],
           [3, 1, 3],
           [3, 2, 0],
           [3, 2, 1],
           [3, 2, 2],
           [3, 2, 3],
           [3, 3, 0],
           [3, 3, 1],
           [3, 3, 2],
           [3, 3, 3]], dtype=uint8)




```python
n = len(colors)
```


```python
n
```




    64



#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)


```python
Z = np.random.randint(0, 10, (3, 4, 3, 4))
Z
```




    array([[[[6, 9, 0, 9],
             [4, 8, 3, 0],
             [0, 8, 2, 2]],
    
            [[4, 7, 9, 8],
             [6, 5, 6, 4],
             [3, 3, 0, 0]],
    
            [[8, 0, 1, 5],
             [7, 2, 5, 7],
             [6, 7, 1, 8]],
    
            [[6, 2, 4, 8],
             [7, 2, 9, 3],
             [4, 4, 8, 8]]],
    
    
           [[[5, 0, 7, 6],
             [3, 1, 6, 0],
             [1, 1, 0, 3]],
    
            [[0, 7, 9, 2],
             [4, 8, 1, 4],
             [7, 0, 2, 1]],
    
            [[1, 3, 7, 8],
             [4, 0, 8, 1],
             [6, 9, 0, 0]],
    
            [[6, 3, 8, 5],
             [9, 9, 8, 5],
             [7, 4, 8, 9]]],
    
    
           [[[1, 3, 3, 0],
             [7, 9, 7, 5],
             [0, 6, 9, 4]],
    
            [[2, 3, 6, 0],
             [0, 6, 0, 8],
             [2, 4, 9, 1]],
    
            [[1, 5, 7, 5],
             [8, 9, 8, 2],
             [8, 3, 3, 8]],
    
            [[4, 1, 4, 5],
             [8, 3, 7, 3],
             [4, 0, 5, 4]]]])




```python
# solution by passing a tuple of axes (introduced in numpy 1.7.0)
sum = Z.sum(axis=(-2,-1))
print(sum)
```

    [[51 55 57 65]
     [33 45 47 81]
     [54 41 67 48]]
    

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)


```python
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
```


```python
D
```




    array([0.27394315, 0.3486959 , 0.52883025, 0.13973558, 0.93943906,
           0.11653039, 0.839077  , 0.58945026, 0.12843178, 0.94986587,
           0.75048923, 0.51280231, 0.81171304, 0.84203423, 0.79580562,
           0.79686438, 0.66317505, 0.57408294, 0.57901675, 0.6675771 ,
           0.81423271, 0.56299419, 0.33252542, 0.37249117, 0.97753431,
           0.27404393, 0.14885222, 0.14510231, 0.34786307, 0.09947907,
           0.47947526, 0.23384606, 0.53607798, 0.36614381, 0.01715548,
           0.05655896, 0.11141143, 0.13444047, 0.27406891, 0.20814668,
           0.22343957, 0.33017279, 0.54434893, 0.52861736, 0.85138979,
           0.29047871, 0.20142409, 0.59104457, 0.34093488, 0.87517047,
           0.93762921, 0.66643949, 0.53053657, 0.88472425, 0.12128706,
           0.64733496, 0.53427024, 0.04321629, 0.32090843, 0.72916846,
           0.63410158, 0.2184891 , 0.33456943, 0.67710159, 0.51475913,
           0.23408947, 0.5520451 , 0.32892111, 0.59077084, 0.73173187,
           0.4449122 , 0.73721605, 0.04297324, 0.54887854, 0.2998898 ,
           0.95095099, 0.88433346, 0.20797315, 0.24691517, 0.86759234,
           0.18279069, 0.62631   , 0.62913905, 0.49716727, 0.70073223,
           0.22665706, 0.70111962, 0.73807589, 0.76036652, 0.43021061,
           0.29675466, 0.21787817, 0.94895763, 0.12296417, 0.47372866,
           0.75309868, 0.06505724, 0.97297956, 0.29265948, 0.27572825])




```python
S
```




    array([3, 5, 9, 6, 4, 5, 6, 6, 9, 7, 4, 7, 0, 2, 2, 6, 3, 9, 2, 2, 8, 7,
           7, 3, 3, 7, 9, 8, 6, 5, 9, 3, 9, 6, 2, 9, 5, 3, 2, 3, 4, 6, 5, 3,
           3, 6, 1, 4, 5, 7, 5, 8, 8, 6, 1, 8, 0, 8, 3, 4, 7, 1, 6, 5, 0, 9,
           3, 6, 9, 5, 9, 2, 3, 3, 2, 5, 4, 2, 8, 4, 5, 5, 0, 6, 1, 4, 5, 3,
           6, 3, 2, 6, 0, 4, 7, 5, 6, 3, 6, 0])




```python
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)
```

    [0.61909456 0.31048312 0.47174917 0.49060346 0.59279199 0.5087238
     0.44257057 0.57690405 0.44196821 0.37220819]
    

#### 69. How to get the diagonal of a dot product? (★★★)


```python
A = np.random.uniform(0, 1, (5,5))
A
```




    array([[0.56824833, 0.82252255, 0.48605141, 0.86963482, 0.89495829],
           [0.86384724, 0.58404561, 0.0129606 , 0.84906904, 0.80974793],
           [0.23961901, 0.78736007, 0.04451528, 0.83566219, 0.79468801],
           [0.21233741, 0.98830363, 0.74899557, 0.74891564, 0.16509075],
           [0.3445228 , 0.46713394, 0.2502314 , 0.73941473, 0.32696304]])




```python
B = np.random.uniform(0, 1, (5,5))
B
```




    array([[0.85724146, 0.28563067, 0.18653118, 0.40444455, 0.5142549 ],
           [0.32035693, 0.52724248, 0.70450174, 0.44951541, 0.26108481],
           [0.33521859, 0.06625462, 0.11367659, 0.68646576, 0.79941748],
           [0.35670433, 0.74043248, 0.27817287, 0.07951202, 0.58419087],
           [0.57057542, 0.39722468, 0.96769399, 0.08680222, 0.37845915]])




```python
np.diag(np.dot(A, B))
```




    array([1.73440401, 1.50586378, 1.60592665, 1.11817428, 1.05487496])



#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)


```python
Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
```

    [1. 0. 0. 0. 2. 0. 0. 0. 3. 0. 0. 0. 4. 0. 0. 0. 5.]
    

#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)


```python
A = np.ones((5,5,3))
A
```




    array([[[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]])




```python
B = 2*np.ones((5,5))
B
```




    array([[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]])




```python
A * B[:,:,None]
```




    array([[[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]],
    
           [[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]],
    
           [[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]],
    
           [[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]],
    
           [[2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.]]])



#### 72. How to swap two rows of an array? (★★★)


```python
Z = np.arange(25).reshape(5,5)
Z
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])




```python
Z[[0,1]] = Z[[1,0]]
Z
```




    array([[ 5,  6,  7,  8,  9],
           [ 0,  1,  2,  3,  4],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])



#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)


```python
faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
```

    [( 1, 12) ( 1, 58) ( 3, 71) ( 3, 81) (12, 58) (14, 34) (14, 36) (14, 87)
     (14, 89) (18, 55) (18, 57) (21, 53) (21, 97) (27, 68) (27, 85) (34, 87)
     (35, 38) (35, 61) (35, 65) (35, 95) (36, 89) (38, 95) (53, 97) (55, 57)
     (61, 65) (63, 71) (63, 73) (68, 85) (71, 73) (71, 81)]
    

#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)


```python
C = np.bincount([1,1,2,3,4,4,6])
```


```python
A = np.repeat(np.arange(len(C)), C)
```


```python
A
```




    array([1, 1, 2, 3, 4, 4, 6])



#### 75. How to compute averages using a sliding window over an array? (★★★)


```python
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
```


```python
Z = np.arange(20)
Z
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19])




```python
moving_average(Z, n=3)
```




    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
           14., 15., 16., 17., 18.])



#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)


```python
from numpy.lib import stride_tricks
```


```python
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
```


```python
Z = rolling(np.arange(10), 3)
```


```python
Z
```




    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])



#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)


```python
Z = np.random.uniform(-1.0, 1.0, 100)
```


```python
Z
```




    array([-0.34611318, -0.71719537,  0.64678347,  0.72135426, -0.73316015,
           -0.97301848,  0.53429307,  0.37227378,  0.39523452,  0.18731851,
            0.27723302,  0.02567807, -0.47555274, -0.27226658, -0.44200397,
           -0.83635284,  0.95823258,  0.68446333, -0.41952863, -0.09411462,
            0.27883162,  0.53446742, -0.63649996,  0.32533875, -0.18105624,
            0.69720115, -0.79992016,  0.11597821, -0.85542173, -0.78372551,
           -0.83956249, -0.75048994,  0.21063211,  0.30942751,  0.74473739,
            0.71497038,  0.42404883,  0.52716638,  0.63590643, -0.53899071,
            0.04123708, -0.80678767, -0.82690185,  0.29716293,  0.62505578,
           -0.26257781, -0.58775934, -0.47358865, -0.45276948, -0.28673119,
            0.73845231, -0.38575665, -0.91107123,  0.62881096,  0.15082664,
           -0.34121672, -0.33867871, -0.2150675 , -0.15451252, -0.44636934,
            0.00501078, -0.86298582,  0.82771523,  0.51444203, -0.35119647,
            0.35087327, -0.6048804 , -0.88213953,  0.29213304, -0.04493586,
           -0.52424089, -0.72346504, -0.80251526,  0.80287418,  0.42070933,
           -0.95118419, -0.68624416,  0.19590782, -0.55708524,  0.62288427,
            0.61985391,  0.25297482,  0.16918647,  0.58131366, -0.1874613 ,
           -0.28521121,  0.78145181, -0.11089944,  0.70994965, -0.54914984,
            0.29288732,  0.4559433 , -0.70348531, -0.60546294,  0.75151746,
            0.89688061,  0.61130762,  0.27884765,  0.9222451 , -0.85945903])




```python
np.negative(Z, out=Z)
```




    array([ 0.34611318,  0.71719537, -0.64678347, -0.72135426,  0.73316015,
            0.97301848, -0.53429307, -0.37227378, -0.39523452, -0.18731851,
           -0.27723302, -0.02567807,  0.47555274,  0.27226658,  0.44200397,
            0.83635284, -0.95823258, -0.68446333,  0.41952863,  0.09411462,
           -0.27883162, -0.53446742,  0.63649996, -0.32533875,  0.18105624,
           -0.69720115,  0.79992016, -0.11597821,  0.85542173,  0.78372551,
            0.83956249,  0.75048994, -0.21063211, -0.30942751, -0.74473739,
           -0.71497038, -0.42404883, -0.52716638, -0.63590643,  0.53899071,
           -0.04123708,  0.80678767,  0.82690185, -0.29716293, -0.62505578,
            0.26257781,  0.58775934,  0.47358865,  0.45276948,  0.28673119,
           -0.73845231,  0.38575665,  0.91107123, -0.62881096, -0.15082664,
            0.34121672,  0.33867871,  0.2150675 ,  0.15451252,  0.44636934,
           -0.00501078,  0.86298582, -0.82771523, -0.51444203,  0.35119647,
           -0.35087327,  0.6048804 ,  0.88213953, -0.29213304,  0.04493586,
            0.52424089,  0.72346504,  0.80251526, -0.80287418, -0.42070933,
            0.95118419,  0.68624416, -0.19590782,  0.55708524, -0.62288427,
           -0.61985391, -0.25297482, -0.16918647, -0.58131366,  0.1874613 ,
            0.28521121, -0.78145181,  0.11089944, -0.70994965,  0.54914984,
           -0.29288732, -0.4559433 ,  0.70348531,  0.60546294, -0.75151746,
           -0.89688061, -0.61130762, -0.27884765, -0.9222451 ,  0.85945903])



#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)


```python
P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
```


```python
P0
```




    array([[ 0.01207417, -8.54799724],
           [-3.5200497 ,  4.78039776],
           [-3.8392601 ,  6.37562563],
           [-3.6231077 ,  5.67347386],
           [ 6.2768762 , -6.47429219],
           [-9.17253152, -5.82746844],
           [-1.13384473, -5.67640085],
           [-6.66882769, -6.80118858],
           [ 8.02642692, -4.96488265],
           [-3.2353472 , -0.6331608 ]])




```python
P1
```




    array([[ 0.14018735,  6.63915397],
           [-4.87045288,  4.42776118],
           [ 3.98596206, -9.26866407],
           [-6.49102311, -8.67837376],
           [ 2.69179003, -9.74004937],
           [-8.21857527,  1.73151647],
           [-7.01257699, -9.15105525],
           [ 5.28141026,  4.10933178],
           [ 9.59485091, -9.87135989],
           [ 8.04887386, -8.53049055]])




```python
p
```




    array([[-1.3475746 ,  7.44757618]])




```python
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))
```


```python
distance(P0, P1, p)
```




    array([ 1.49452832,  2.0317397 ,  2.70799761,  1.8837731 , 15.42635859,
            6.10122983, 11.40680821,  6.9348981 ,  5.1494838 ,  7.70286256])



#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)


```python
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
```


```python
# based on distance function from previous question
np.array([distance(P0,P1,p_i) for p_i in p])
```




    array([[ 8.59849534,  0.30227508,  0.83463273,  6.77978322,  3.39928686,
             7.70561732,  4.05782776,  0.84602647,  2.72286964,  0.40441707],
           [ 5.58417029, 11.25039834,  9.85427305,  4.11201624,  8.90905312,
             2.89212024, 13.13273194,  7.58279957,  7.02453004, 11.09334345],
           [ 0.31129217,  4.46588408,  6.13569772,  2.19834339,  5.99895028,
            13.10671825,  2.68055562,  8.01576642,  6.11792543,  4.65367696],
           [ 5.69637771,  3.04139976,  3.89314955,  3.81596208,  5.65317273,
            10.79060169,  0.65147103,  4.33952142,  0.93488625,  3.16867287],
           [ 6.21481214,  2.58408566,  3.37759663,  4.34450644,  5.28310113,
            10.27017907,  1.23194662,  3.74224453,  0.29947113,  2.70680531],
           [ 8.00039403,  6.78416796,  3.58424598,  9.5230672 ,  6.860233  ,
             3.53875282,  4.50751281,  1.77271892,  5.3959753 ,  6.49275719],
           [ 2.51442556,  5.04755269,  7.01517621,  4.40560099,  6.18607163,
            14.01291356,  3.95440982,  9.41356489,  8.05200479,  5.25742075],
           [ 3.94708206,  2.78402788,  3.89881646,  2.08390515,  5.06995727,
            10.82066277,  0.27009075,  4.84706671,  2.06588501,  2.93026794],
           [ 6.456126  ,  1.63758587,  0.72304895,  4.70486422,  1.0319656 ,
             6.18499672,  5.01401396,  0.10704643,  2.25184079,  1.508964  ],
           [ 8.26922809,  7.64244749, 10.34823716, 10.20185202,  7.76916962,
            17.41100475,  8.22404063, 13.96544887, 13.70915015,  7.90782601]])



#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)


```python
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)
Z
```




    array([[9, 4, 6, 6, 8, 2, 3, 3, 5, 3],
           [7, 1, 0, 8, 2, 2, 1, 0, 7, 9],
           [5, 4, 3, 1, 3, 2, 5, 9, 6, 7],
           [4, 6, 6, 5, 3, 2, 7, 7, 2, 1],
           [8, 7, 9, 6, 0, 7, 5, 0, 3, 4],
           [1, 3, 5, 9, 1, 9, 9, 3, 4, 1],
           [0, 0, 6, 3, 7, 9, 3, 8, 9, 7],
           [2, 2, 5, 2, 2, 0, 4, 3, 6, 2],
           [4, 8, 9, 2, 1, 7, 7, 5, 4, 4],
           [8, 3, 1, 0, 9, 6, 4, 1, 0, 3]])




```python
R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)
R
```




    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])




```python
R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2
```


```python
R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

```


```python
r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
```

    <ipython-input-32-8a0bbea14d07>:3: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      R[r] = Z[z]
    


```python
Z
```




    array([[9, 4, 6, 6, 8, 2, 3, 3, 5, 3],
           [7, 1, 0, 8, 2, 2, 1, 0, 7, 9],
           [5, 4, 3, 1, 3, 2, 5, 9, 6, 7],
           [4, 6, 6, 5, 3, 2, 7, 7, 2, 1],
           [8, 7, 9, 6, 0, 7, 5, 0, 3, 4],
           [1, 3, 5, 9, 1, 9, 9, 3, 4, 1],
           [0, 0, 6, 3, 7, 9, 3, 8, 9, 7],
           [2, 2, 5, 2, 2, 0, 4, 3, 6, 2],
           [4, 8, 9, 2, 1, 7, 7, 5, 4, 4],
           [8, 3, 1, 0, 9, 6, 4, 1, 0, 3]])




```python
R
```




    array([[0, 0, 0, 0, 0],
           [0, 9, 4, 6, 6],
           [0, 7, 1, 0, 8],
           [0, 5, 4, 3, 1],
           [0, 4, 6, 6, 5]])



#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)


```python
Z = np.arange(1,15,dtype=np.uint32)
Z
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
          dtype=uint32)




```python
R = stride_tricks.as_strided(Z, (11,4), (4,4))
```


```python
R
```




    array([[ 1,  2,  3,  4],
           [ 2,  3,  4,  5],
           [ 3,  4,  5,  6],
           [ 4,  5,  6,  7],
           [ 5,  6,  7,  8],
           [ 6,  7,  8,  9],
           [ 7,  8,  9, 10],
           [ 8,  9, 10, 11],
           [ 9, 10, 11, 12],
           [10, 11, 12, 13],
           [11, 12, 13, 14]], dtype=uint32)



#### 82. Compute a matrix rank (★★★)


```python
Z = np.random.uniform(0, 1, (10,10))
Z
```




    array([[0.1232169 , 0.9760086 , 0.96859617, 0.49424797, 0.99347647,
            0.06328458, 0.40341133, 0.4818836 , 0.3683665 , 0.79385758],
           [0.33311252, 0.50895318, 0.94124538, 0.16518617, 0.66071865,
            0.04032933, 0.79996411, 0.96467682, 0.86988914, 0.85223907],
           [0.41526189, 0.80085612, 0.87579205, 0.16275738, 0.89923306,
            0.58738437, 0.38016826, 0.4395107 , 0.42194846, 0.57700535],
           [0.07526055, 0.6213567 , 0.19946256, 0.9029077 , 0.74111276,
            0.48314876, 0.5684592 , 0.82064205, 0.88224062, 0.7546564 ],
           [0.52952458, 0.540589  , 0.12983148, 0.55084929, 0.56092643,
            0.39652212, 0.68838262, 0.38671469, 0.56364358, 0.36953082],
           [0.30748391, 0.88151202, 0.62705113, 0.08157428, 0.36870331,
            0.5122246 , 0.73396942, 0.25803297, 0.69503513, 0.80627679],
           [0.44702553, 0.8500453 , 0.60359613, 0.72389502, 0.03332621,
            0.10554721, 0.91396044, 0.80132533, 0.2238569 , 0.55316985],
           [0.21725211, 0.87824935, 0.56062164, 0.0295559 , 0.18943425,
            0.33550158, 0.75916366, 0.73154605, 0.31034145, 0.54133   ],
           [0.14962789, 0.79475107, 0.16893675, 0.6976553 , 0.54604567,
            0.69231195, 0.43908179, 0.29956995, 0.21750865, 0.15411748],
           [0.01165984, 0.1470515 , 0.26495535, 0.34002974, 0.32619088,
            0.23325613, 0.01197093, 0.16944055, 0.01088037, 0.24747941]])




```python
U, S, V = np.linalg.svd(Z) #singular value decomposition
```


```python
rank = np.sum(S > 1e-10)
```


```python
rank
```




    10



#### 83. How to find the most frequent value in an array?


```python
Z = np.random.randint(0, 10, 50)
```


```python
Z
```




    array([6, 2, 3, 7, 3, 5, 3, 7, 1, 0, 6, 6, 4, 2, 6, 8, 7, 8, 7, 0, 5, 8,
           4, 5, 7, 4, 7, 6, 9, 2, 2, 4, 0, 5, 8, 5, 9, 3, 9, 9, 4, 7, 8, 8,
           4, 9, 4, 9, 9, 7])




```python
np.bincount(Z).argmax()
```




    7



#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)


```python
Z = np.random.randint(0, 5, (10,10))
Z
```




    array([[3, 4, 0, 1, 2, 4, 0, 3, 1, 1],
           [3, 3, 3, 0, 0, 1, 4, 4, 1, 0],
           [1, 3, 0, 0, 0, 0, 0, 0, 3, 0],
           [0, 2, 3, 1, 3, 2, 4, 2, 4, 0],
           [1, 4, 0, 2, 4, 4, 3, 4, 4, 1],
           [0, 0, 0, 3, 1, 3, 4, 2, 0, 3],
           [2, 0, 4, 1, 2, 1, 1, 1, 0, 3],
           [1, 4, 3, 0, 0, 2, 4, 4, 0, 3],
           [0, 0, 1, 1, 4, 2, 3, 0, 3, 0],
           [2, 0, 3, 4, 4, 0, 0, 3, 0, 3]])




```python
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
```


```python
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
```


```python
C
```




    array([[[[3, 4, 0],
             [3, 3, 3],
             [1, 3, 0]],
    
            [[4, 0, 1],
             [3, 3, 0],
             [3, 0, 0]],
    
            [[0, 1, 2],
             [3, 0, 0],
             [0, 0, 0]],
    
            [[1, 2, 4],
             [0, 0, 1],
             [0, 0, 0]],
    
            [[2, 4, 0],
             [0, 1, 4],
             [0, 0, 0]],
    
            [[4, 0, 3],
             [1, 4, 4],
             [0, 0, 0]],
    
            [[0, 3, 1],
             [4, 4, 1],
             [0, 0, 3]],
    
            [[3, 1, 1],
             [4, 1, 0],
             [0, 3, 0]]],
    
    
           [[[3, 3, 3],
             [1, 3, 0],
             [0, 2, 3]],
    
            [[3, 3, 0],
             [3, 0, 0],
             [2, 3, 1]],
    
            [[3, 0, 0],
             [0, 0, 0],
             [3, 1, 3]],
    
            [[0, 0, 1],
             [0, 0, 0],
             [1, 3, 2]],
    
            [[0, 1, 4],
             [0, 0, 0],
             [3, 2, 4]],
    
            [[1, 4, 4],
             [0, 0, 0],
             [2, 4, 2]],
    
            [[4, 4, 1],
             [0, 0, 3],
             [4, 2, 4]],
    
            [[4, 1, 0],
             [0, 3, 0],
             [2, 4, 0]]],
    
    
           [[[1, 3, 0],
             [0, 2, 3],
             [1, 4, 0]],
    
            [[3, 0, 0],
             [2, 3, 1],
             [4, 0, 2]],
    
            [[0, 0, 0],
             [3, 1, 3],
             [0, 2, 4]],
    
            [[0, 0, 0],
             [1, 3, 2],
             [2, 4, 4]],
    
            [[0, 0, 0],
             [3, 2, 4],
             [4, 4, 3]],
    
            [[0, 0, 0],
             [2, 4, 2],
             [4, 3, 4]],
    
            [[0, 0, 3],
             [4, 2, 4],
             [3, 4, 4]],
    
            [[0, 3, 0],
             [2, 4, 0],
             [4, 4, 1]]],
    
    
           [[[0, 2, 3],
             [1, 4, 0],
             [0, 0, 0]],
    
            [[2, 3, 1],
             [4, 0, 2],
             [0, 0, 3]],
    
            [[3, 1, 3],
             [0, 2, 4],
             [0, 3, 1]],
    
            [[1, 3, 2],
             [2, 4, 4],
             [3, 1, 3]],
    
            [[3, 2, 4],
             [4, 4, 3],
             [1, 3, 4]],
    
            [[2, 4, 2],
             [4, 3, 4],
             [3, 4, 2]],
    
            [[4, 2, 4],
             [3, 4, 4],
             [4, 2, 0]],
    
            [[2, 4, 0],
             [4, 4, 1],
             [2, 0, 3]]],
    
    
           [[[1, 4, 0],
             [0, 0, 0],
             [2, 0, 4]],
    
            [[4, 0, 2],
             [0, 0, 3],
             [0, 4, 1]],
    
            [[0, 2, 4],
             [0, 3, 1],
             [4, 1, 2]],
    
            [[2, 4, 4],
             [3, 1, 3],
             [1, 2, 1]],
    
            [[4, 4, 3],
             [1, 3, 4],
             [2, 1, 1]],
    
            [[4, 3, 4],
             [3, 4, 2],
             [1, 1, 1]],
    
            [[3, 4, 4],
             [4, 2, 0],
             [1, 1, 0]],
    
            [[4, 4, 1],
             [2, 0, 3],
             [1, 0, 3]]],
    
    
           [[[0, 0, 0],
             [2, 0, 4],
             [1, 4, 3]],
    
            [[0, 0, 3],
             [0, 4, 1],
             [4, 3, 0]],
    
            [[0, 3, 1],
             [4, 1, 2],
             [3, 0, 0]],
    
            [[3, 1, 3],
             [1, 2, 1],
             [0, 0, 2]],
    
            [[1, 3, 4],
             [2, 1, 1],
             [0, 2, 4]],
    
            [[3, 4, 2],
             [1, 1, 1],
             [2, 4, 4]],
    
            [[4, 2, 0],
             [1, 1, 0],
             [4, 4, 0]],
    
            [[2, 0, 3],
             [1, 0, 3],
             [4, 0, 3]]],
    
    
           [[[2, 0, 4],
             [1, 4, 3],
             [0, 0, 1]],
    
            [[0, 4, 1],
             [4, 3, 0],
             [0, 1, 1]],
    
            [[4, 1, 2],
             [3, 0, 0],
             [1, 1, 4]],
    
            [[1, 2, 1],
             [0, 0, 2],
             [1, 4, 2]],
    
            [[2, 1, 1],
             [0, 2, 4],
             [4, 2, 3]],
    
            [[1, 1, 1],
             [2, 4, 4],
             [2, 3, 0]],
    
            [[1, 1, 0],
             [4, 4, 0],
             [3, 0, 3]],
    
            [[1, 0, 3],
             [4, 0, 3],
             [0, 3, 0]]],
    
    
           [[[1, 4, 3],
             [0, 0, 1],
             [2, 0, 3]],
    
            [[4, 3, 0],
             [0, 1, 1],
             [0, 3, 4]],
    
            [[3, 0, 0],
             [1, 1, 4],
             [3, 4, 4]],
    
            [[0, 0, 2],
             [1, 4, 2],
             [4, 4, 0]],
    
            [[0, 2, 4],
             [4, 2, 3],
             [4, 0, 0]],
    
            [[2, 4, 4],
             [2, 3, 0],
             [0, 0, 3]],
    
            [[4, 4, 0],
             [3, 0, 3],
             [0, 3, 0]],
    
            [[4, 0, 3],
             [0, 3, 0],
             [3, 0, 3]]]])



#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)


```python
class Symmetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symmetric, self).__setitem__((i,j), value)
        super(Symmetric, self).__setitem__((j,i), value)
```


```python
def symmetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symmetric)
```


```python
S = symmetric(np.random.randint(0,10,(5,5)))
S
```




    Symmetric([[ 3, 13, 11, 14, 13],
               [13,  0,  8, 11,  9],
               [11,  8,  7,  0,  3],
               [14, 11,  0,  4, 10],
               [13,  9,  3, 10,  8]])




```python
S[2,3] = 42
```


```python
S
```




    Symmetric([[ 3, 13, 11, 14, 13],
               [13,  0,  8, 11,  9],
               [11,  8,  7, 42,  3],
               [14, 11, 42,  4, 10],
               [13,  9,  3, 10,  8]])



#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)


```python
p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
```


```python
S
```




    array([[200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.],
           [200.]])



#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)


```python
Z = np.ones((16,16))
Z
```




    array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])




```python
k = 4
```


```python
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)
```


```python
S
```




    array([[16., 16., 16., 16.],
           [16., 16., 16., 16.],
           [16., 16., 16., 16.],
           [16., 16., 16., 16.]])



#### 88. How to implement the Game of Life using numpy arrays? (★★★)


```python
def iterate(Z):
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z
```


```python
Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
```


```python
Z
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 1, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]])



#### 89. How to get the n largest values of an array (★★★)


```python
Z = np.arange(1000)
np.random.shuffle(Z)
n = 5
```


```python
Z[np.argsort(Z)[-n:]]
```




    array([995, 996, 997, 998, 999])



#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)


```python
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix
```


```python
cartesian(([1, 2, 3], [4, 5], [6, 7]))
```




    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])



#### 91. How to create a record array from a regular array? (★★★)


```python
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
Z
```




    array([['Hello', '2.5', '3'],
           ['World', '3.6', '2']], dtype='<U5')




```python
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
```


```python
R
```




    rec.array([(b'Hello', 2.5, 3), (b'World', 3.6, 2)],
              dtype=[('col1', 'S8'), ('col2', '<f8'), ('col3', '<i8')])



#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)


```python
x = np.random.rand(int(5e7))
x
```




    array([0.33427621, 0.20730357, 0.71252141, ..., 0.88998156, 0.98027757,
           0.61647723])




```python
%timeit np.power(x,3)
```

    1.72 s ± 12.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    


```python
%timeit x*x*x
```

    298 ms ± 6.16 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    


```python
%timeit np.einsum('i,i,i->i',x,x,x)
```

    159 ms ± 3.22 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)


```python
A = np.random.randint(0,5,(8,3))
A
```




    array([[0, 3, 0],
           [2, 0, 2],
           [4, 4, 1],
           [4, 4, 4],
           [3, 2, 4],
           [0, 1, 1],
           [1, 3, 4],
           [2, 1, 3]])




```python
B = np.random.randint(0,5,(2,2))
B
```




    array([[2, 3],
           [1, 1]])




```python
C = (A[..., np.newaxis, np.newaxis] == B)
C
```




    array([[[[False, False],
             [False, False]],
    
            [[False,  True],
             [False, False]],
    
            [[False, False],
             [False, False]]],
    
    
           [[[ True, False],
             [False, False]],
    
            [[False, False],
             [False, False]],
    
            [[ True, False],
             [False, False]]],
    
    
           [[[False, False],
             [False, False]],
    
            [[False, False],
             [False, False]],
    
            [[False, False],
             [ True,  True]]],
    
    
           [[[False, False],
             [False, False]],
    
            [[False, False],
             [False, False]],
    
            [[False, False],
             [False, False]]],
    
    
           [[[False,  True],
             [False, False]],
    
            [[ True, False],
             [False, False]],
    
            [[False, False],
             [False, False]]],
    
    
           [[[False, False],
             [False, False]],
    
            [[False, False],
             [ True,  True]],
    
            [[False, False],
             [ True,  True]]],
    
    
           [[[False, False],
             [ True,  True]],
    
            [[False,  True],
             [False, False]],
    
            [[False, False],
             [False, False]]],
    
    
           [[[ True, False],
             [False, False]],
    
            [[False, False],
             [ True,  True]],
    
            [[False,  True],
             [False, False]]]])




```python
rows = np.where(C.any((3,1)).all(1))[0]
```


```python
rows
```




    array([6, 7], dtype=int64)



#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)


```python
Z = np.random.randint(0,5,(10,3))
print(Z)
```

    [[4 3 0]
     [4 0 1]
     [4 4 0]
     [1 2 2]
     [0 2 4]
     [4 2 3]
     [1 4 1]
     [4 2 0]
     [0 0 4]
     [1 1 1]]
    


```python
E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)
```

    [[4 3 0]
     [4 0 1]
     [4 4 0]
     [1 2 2]
     [0 2 4]
     [4 2 3]
     [1 4 1]
     [4 2 0]
     [0 0 4]]
    

#### 95. Convert a vector of ints into a matrix binary representation (★★★)


```python
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))
```

    [[0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 1 0]
     [0 0 0 0 0 0 1 1]
     [0 0 0 0 1 1 1 1]
     [0 0 0 1 0 0 0 0]
     [0 0 1 0 0 0 0 0]
     [0 1 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0]]
    

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)


```python
Z = np.random.randint(0,2,(6,3))
Z
```




    array([[0, 0, 0],
           [0, 1, 1],
           [1, 0, 0],
           [0, 0, 1],
           [0, 0, 0],
           [0, 1, 0]])




```python
uZ = np.unique(Z, axis=0)
print(uZ)
```

    [[0 0 0]
     [0 0 1]
     [0 1 0]
     [0 1 1]
     [1 0 0]]
    

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)


```python
A = np.random.uniform(0,1,10)
A
```




    array([0.76394913, 0.83031562, 0.45191313, 0.30545302, 0.54974677,
           0.10941989, 0.84681732, 0.6967188 , 0.85989466, 0.49422301])




```python
B = np.random.uniform(0,1,10)
B
```




    array([0.94739856, 0.82898565, 0.49914158, 0.04325412, 0.21457481,
           0.96745154, 0.96514776, 0.46931813, 0.71167131, 0.93539178])




```python
np.einsum('i->', A)       # np.sum(A)
```




    5.90845134211319




```python
np.einsum('i,i->i', A, B) # A * B
```




    array([0.7237643 , 0.68831974, 0.22556863, 0.0132121 , 0.11796181,
           0.10585844, 0.81730383, 0.32698277, 0.61196236, 0.46229214])




```python
np.einsum('i,i', A, B)    # np.inner(A, B)
```




    4.093226116871653




```python
np.einsum('i,j->ij', A, B)    # np.outer(A, B)
```




    array([[0.7237643 , 0.63330286, 0.38131877, 0.03304394, 0.16392424,
            0.73908376, 0.73732378, 0.35853518, 0.54368068, 0.71459173],
           [0.78663982, 0.68831974, 0.41444505, 0.03591457, 0.17816482,
            0.80329013, 0.80137726, 0.38968218, 0.59091181, 0.77667041],
           [0.42814185, 0.3746295 , 0.22556863, 0.0195471 , 0.09696918,
            0.43720405, 0.43616294, 0.21209103, 0.32161361, 0.42271583],
           [0.28938575, 0.25321617, 0.1524643 , 0.0132121 , 0.06554252,
            0.29551099, 0.29480729, 0.14335464, 0.21738215, 0.28571824],
           [0.5208293 , 0.45573218, 0.27440147, 0.02377881, 0.11796181,
            0.53185336, 0.53058686, 0.25800613, 0.391239  , 0.51422861],
           [0.10366425, 0.09070752, 0.05461602, 0.00473286, 0.02347875,
            0.10585844, 0.10560637, 0.05135274, 0.077871  , 0.10235047],
           [0.8022735 , 0.7019994 , 0.42268173, 0.03662833, 0.18170567,
            0.81925472, 0.81730383, 0.39742672, 0.60265559, 0.79210596],
           [0.66007039, 0.57756989, 0.34776132, 0.03013596, 0.14949831,
            0.67404168, 0.67243659, 0.32698277, 0.49583478, 0.65170504],
           [0.81466296, 0.71284033, 0.42920917, 0.03719398, 0.18451173,
            0.83190641, 0.8299254 , 0.40356415, 0.61196236, 0.80433839],
           [0.46822616, 0.40970378, 0.24668725, 0.02137718, 0.10604781,
            0.47813681, 0.47699822, 0.23194782, 0.35172433, 0.46229214]])



#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?


```python
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)
```


```python
dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
```


```python
r[1:] = np.cumsum(dr)                # integrate path
```


```python
r_int = np.linspace(0, r.max(), 200) # regular spaced path
```


```python
x_int = np.interp(r_int, r, x)       # integrate path
```


```python
y_int = np.interp(r_int, r, y)
```


```python
x_int
```




    array([ 0.00000000e+00, -3.73131229e-01, -2.59817608e+00, -3.26212050e+00,
           -2.18442687e+00, -2.98929946e-02,  2.42923642e+00,  4.54913599e+00,
            5.92318348e+00,  6.35117933e+00,  5.82369277e+00,  4.46259540e+00,
            2.47320794e+00,  1.09577220e-01, -2.36575300e+00, -4.71261671e+00,
           -6.72701769e+00, -8.25541575e+00, -9.18486120e+00, -9.46381505e+00,
           -9.11085788e+00, -8.12875279e+00, -6.63306046e+00, -4.69271059e+00,
           -2.44736165e+00, -2.05444585e-02,  2.46101146e+00,  4.86841760e+00,
            7.08937968e+00,  9.02539126e+00,  1.05948609e+01,  1.17357250e+01,
            1.24068974e+01,  1.25885805e+01,  1.22815267e+01,  1.15053927e+01,
            1.02963689e+01,  8.70429550e+00,  6.78948686e+00,  4.61716636e+00,
            2.25853448e+00, -1.98731680e-01, -2.68040566e+00, -5.11543300e+00,
           -7.41973991e+00, -9.53891040e+00, -1.14237629e+01, -1.29919305e+01,
           -1.42355069e+01, -1.51243232e+01, -1.56061571e+01, -1.57219415e+01,
           -1.54217066e+01, -1.47579136e+01, -1.37255236e+01, -1.23634834e+01,
           -1.07024632e+01, -8.78327367e+00, -6.65029558e+00, -4.35514246e+00,
           -1.94290275e+00,  5.28038914e-01,  3.00985904e+00,  5.45450275e+00,
            7.80093594e+00,  1.00179639e+01,  1.20595602e+01,  1.38732623e+01,
            1.54564938e+01,  1.67497894e+01,  1.77435802e+01,  1.84439878e+01,
            1.87990820e+01,  1.88271003e+01,  1.85472755e+01,  1.79378850e+01,
            1.70105456e+01,  1.58085777e+01,  1.43501808e+01,  1.26222364e+01,
            1.06905445e+01,  8.58498641e+00,  6.33562529e+00,  3.96179025e+00,
            1.51802643e+00, -9.61699044e-01, -3.44349301e+00, -5.88951759e+00,
           -8.26006582e+00, -1.05270865e+01, -1.26610269e+01, -1.46343020e+01,
           -1.64206033e+01, -1.79708499e+01, -1.92894561e+01, -2.03605967e+01,
           -2.11716587e+01, -2.17133504e+01, -2.19797677e+01, -2.19626531e+01,
           -2.16476737e+01, -2.10622561e+01, -2.02145809e+01, -1.91158467e+01,
           -1.77800930e+01, -1.62239930e+01, -1.44666197e+01, -1.25291882e+01,
           -1.04347777e+01, -8.20773276e+00, -5.86978851e+00, -3.45563299e+00,
           -9.92556869e-01,  1.49191203e+00,  3.97031733e+00,  6.41557503e+00,
            8.80126359e+00,  1.11019001e+01,  1.32931995e+01,  1.53523150e+01,
            1.72580567e+01,  1.89910866e+01,  2.05340895e+01,  2.18719167e+01,
            2.29917039e+01,  2.38829603e+01,  2.45376309e+01,  2.49501314e+01,
            2.51173551e+01,  2.50386548e+01,  2.47157984e+01,  2.41529019e+01,
            2.33563391e+01,  2.23346314e+01,  2.10983193e+01,  1.96598169e+01,
            1.80332532e+01,  1.62343003e+01,  1.42799934e+01,  1.21885425e+01,
            9.97914013e+00,  7.67176574e+00,  5.28699023e+00,  2.84441658e+00,
            3.68190102e-01, -2.11846865e+00, -4.59426061e+00, -7.03829202e+00,
           -9.43024209e+00, -1.17505186e+01, -1.39803999e+01, -1.61021634e+01,
           -1.80991985e+01, -1.99561058e+01, -2.16587804e+01, -2.31850397e+01,
           -2.45166960e+01, -2.56568097e+01, -2.65976904e+01, -2.73332424e+01,
           -2.78589555e+01, -2.81718833e+01, -2.82706110e+01, -2.81552119e+01,
           -2.78066635e+01, -2.72341677e+01, -2.64578863e+01, -2.54844118e+01,
           -2.43214986e+01, -2.29779690e+01, -2.14636172e+01, -1.97837046e+01,
           -1.79341987e+01, -1.59545785e+01, -1.38585628e+01, -1.16602783e+01,
           -9.37416511e+00, -7.01488958e+00, -4.59184632e+00, -2.12995331e+00,
            3.50691071e-01,  2.83498918e+00,  5.30813450e+00,  7.75565814e+00,
            1.01522531e+01,  1.24813227e+01,  1.47337756e+01,  1.68973642e+01,
            1.89604991e+01,  2.09071945e+01,  2.26943693e+01,  2.43420179e+01,
            2.58422998e+01,  2.71881423e+01,  2.83731992e+01,  2.93376609e+01,
            3.01222558e+01,  3.07280750e+01,  3.11531219e+01,  3.13960177e+01])




```python
y_int
```




    array([ 0.00000000e+00,  1.74026724e+00,  9.81816584e-01, -1.34251287e+00,
           -3.53191891e+00, -4.70449474e+00, -4.59573427e+00, -3.33831870e+00,
           -1.28956083e+00,  1.14234685e+00,  3.55645601e+00,  5.62188218e+00,
            7.09389488e+00,  7.82951803e+00,  7.78700678e+00,  6.99685666e+00,
            5.55535240e+00,  3.60417006e+00,  1.30506373e+00, -1.15819722e+00,
           -3.61328132e+00, -5.89130465e+00, -7.87065053e+00, -9.41689057e+00,
           -1.04714836e+01, -1.09903787e+01, -1.09354306e+01, -1.03330912e+01,
           -9.22690166e+00, -7.67489830e+00, -5.75257980e+00, -3.54832201e+00,
           -1.15854355e+00,  1.31709956e+00,  3.78023622e+00,  6.13780326e+00,
            8.30540149e+00,  1.02098724e+01,  1.17910419e+01,  1.29972950e+01,
            1.37774249e+01,  1.41319792e+01,  1.40585758e+01,  1.35636816e+01,
            1.26345244e+01,  1.13409083e+01,  9.72216360e+00,  7.79464305e+00,
            5.64544223e+00,  3.32503222e+00,  8.88107041e-01, -1.59329248e+00,
           -4.05992507e+00, -6.45313746e+00, -8.71327144e+00, -1.07900854e+01,
           -1.26380653e+01, -1.42147006e+01, -1.54892299e+01, -1.64390858e+01,
           -1.70351433e+01, -1.72938948e+01, -1.71685631e+01, -1.67219979e+01,
           -1.59044166e+01, -1.47824153e+01, -1.33665280e+01, -1.16678134e+01,
           -9.75246391e+00, -7.63091465e+00, -5.35345831e+00, -2.96905840e+00,
           -5.09631905e-01,  1.97507060e+00,  4.44430515e+00,  6.85344809e+00,
            9.15903461e+00,  1.13337844e+01,  1.33467125e+01,  1.51336120e+01,
            1.66966787e+01,  1.80162922e+01,  1.90746708e+01,  1.98127969e+01,
            2.02634499e+01,  2.04221761e+01,  2.02877631e+01,  1.98449702e+01,
            1.90974465e+01,  1.80799531e+01,  1.68069397e+01,  1.52959771e+01,
            1.35665801e+01,  1.16227312e+01,  9.51601869e+00,  7.27388893e+00,
            4.92519240e+00,  2.49981337e+00,  2.82932820e-02, -2.45817963e+00,
           -4.92460367e+00, -7.34030894e+00, -9.67635928e+00, -1.19050601e+01,
           -1.40002645e+01, -1.59376556e+01, -1.76950019e+01, -1.92523833e+01,
           -2.05923861e+01, -2.16995543e+01, -2.25478860e+01, -2.31432552e+01,
           -2.34800725e+01, -2.35556687e+01, -2.33702741e+01, -2.29269662e+01,
           -2.22315865e+01, -2.12926288e+01, -2.01211000e+01, -1.87303570e+01,
           -1.71359212e+01, -1.53552740e+01, -1.34076356e+01, -1.13137308e+01,
           -9.09554383e+00, -6.77606668e+00, -4.37904252e+00, -1.92870828e+00,
            5.50461201e-01,  3.03400451e+00,  5.49771780e+00,  7.91788934e+00,
            1.02715222e+01,  1.25365431e+01,  1.46919953e+01,  1.67182148e+01,
            1.85969872e+01,  2.03116866e+01,  2.18473923e+01,  2.31909872e+01,
            2.43312337e+01,  2.52588296e+01,  2.59664444e+01,  2.64391214e+01,
            2.66748505e+01,  2.66796333e+01,  2.64545680e+01,  2.60026765e+01,
            2.53288265e+01,  2.44396400e+01,  2.33433894e+01,  2.20498837e+01,
            2.05703459e+01,  1.89172843e+01,  1.71043584e+01,  1.51396391e+01,
            1.30387952e+01,  1.08294252e+01,  8.52903617e+00,  6.15551380e+00,
            3.72697869e+00,  1.26164946e+00, -1.22228693e+00, -3.70679651e+00,
           -6.17014862e+00, -8.59028675e+00, -1.09516293e+01, -1.32379536e+01,
           -1.54338179e+01, -1.75246263e+01, -1.94966815e+01, -2.13317519e+01,
           -2.29952001e+01, -2.44987532e+01, -2.58335652e+01, -2.69919481e+01,
           -2.79673583e+01, -2.87543764e+01, -2.93186910e+01, -2.96679958e+01,
           -2.98181354e+01, -2.97693731e+01, -2.95229769e+01, -2.90811595e+01,
           -2.84133854e+01, -2.75430665e+01, -2.64934034e+01, -2.52713717e+01,
           -2.38845608e+01, -2.23353782e+01, -2.06051319e+01, -1.87442619e+01,
           -1.67635423e+01, -1.46739620e+01, -1.24866789e+01, -1.01933401e+01,
           -7.83474328e+00, -5.42495146e+00, -2.97607515e+00, -5.00072086e-01])



#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)


```python
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
```


```python
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
```


```python
M &= (X.sum(axis=-1) == n)
```


```python
print(X[M])
```

    [[2. 0. 1. 1.]]
    

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)


```python
answer(100)
```

    # Author: Jessica B. Hamrick
    
    X = np.random.randn(100) # random 1D array
    N = 1000 # number of bootstrap samples
    idx = np.random.randint(0, X.size, (N, X.size))
    means = X[idx].mean(axis=1)
    confint = np.percentile(means, [2.5, 97.5])
    print(confint)
    


```python
X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
```


```python
idx = np.random.randint(0, X.size, (N, X.size))
```


```python
means = X[idx].mean(axis=1)
```


```python
confint = np.percentile(means, [2.5, 97.5])
```


```python
confint
```




    array([-0.2394537 ,  0.15132318])




```python

```
