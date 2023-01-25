from polarityjam.polarityjam_logging import get_logger

PERMUTATIONS = 999
import numpy as np
import scipy.stats as stats

from polarityjam.model.weights import W


# ### copied from pysal package esda moran see https://pysal.org/esda/generated/esda.Moran.html
# or github https://github.com/pysal/esda/blob/c6a78c3fd4583ef7097970ff04a1d56b6dfbc7f7/esda/moran.py
# pointing to a commit, not to main for reproducibility
class Moran(object):
    """Moran's I Global Autocorrelation Statistic
    Parameters
    ----------
    y               : array
                      variable measured across n spatial units
    w               : W
                      spatial weights instance
    transformation  : string
                      weights transformation,  default is row-standardized "r".
                      Other options include "B": binary,  "D":
                      doubly-standardized,  "U": untransformed
                      (general weights), "V": variance-stabilizing.
    permutations    : int
                      number of random permutations for calculation of
                      pseudo-p_values
    two_tailed      : boolean
                      If True (default) analytical p-values for Moran are two
                      tailed, otherwise if False, they are one-tailed.
    Attributes
    ----------
    y            : array
                   original variable
    w            : W
                   original w object
    permutations : int
                   number of permutations
    I            : float
                   value of Moran's I
    EI           : float
                   expected value under normality assumption
    VI_norm      : float
                   variance of I under normality assumption
    seI_norm     : float
                   standard deviation of I under normality assumption
    z_norm       : float
                   z-value of I under normality assumption
    p_norm       : float
                   p-value of I under normality assumption
    VI_rand      : float
                   variance of I under randomization assumption
    seI_rand     : float
                   standard deviation of I under randomization assumption
    z_rand       : float
                   z-value of I under randomization assumption
    p_rand       : float
                   p-value of I under randomization assumption
    two_tailed   : boolean
                   If True p_norm and p_rand are two-tailed, otherwise they
                   are one-tailed.
    sim          : array
                   (if permutations>0)
                   vector of I values for permuted samples
    p_sim        : array
                   (if permutations>0)
                   p-value based on permutations (one-tailed)
                   null: spatial randomness
                   alternative: the observed I is extreme if
                   it is either extremely greater or extremely lower
                   than the values obtained based on permutations
    EI_sim       : float
                   (if permutations>0)
                   average value of I from permutations
    VI_sim       : float
                   (if permutations>0)
                   variance of I from permutations
    seI_sim      : float
                   (if permutations>0)
                   standard deviation of I under permutations.
    z_sim        : float
                   (if permutations>0)
                   standardized I based on permutations
    p_z_sim      : float
                   (if permutations>0)
                   p-value based on standard normal approximation from
                   permutations
    Notes
    -----
    Technical details and derivations can be found in :cite:`cliff81`.
    Examples
    --------
    >>> import libpysal
    >>> w = libpysal.io.open(libpysal.examples.get_path("stl.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("stl_hom.txt"))
    >>> y = np.array(f.by_col['HR8893'])
    >>> from esda.moran import Moran
    >>> mi = Moran(y,  w)
    >>> round(mi.I, 3)
    0.244
    >>> mi.EI
    -0.012987012987012988
    >>> mi.p_norm
    0.00027147862770937614
    SIDS example replicating OpenGeoda
    >>> w = libpysal.io.open(libpysal.examples.get_path("sids2.gal")).read()
    >>> f = libpysal.io.open(libpysal.examples.get_path("sids2.dbf"))
    >>> SIDR = np.array(f.by_col("SIDR74"))
    >>> mi = Moran(SIDR,  w)
    >>> round(mi.I, 3)
    0.248
    >>> mi.p_norm
    0.0001158330781489969
    One-tailed
    >>> mi_1 = Moran(SIDR,  w, two_tailed=False)
    >>> round(mi_1.I, 3)
    0.248
    >>> round(mi_1.p_norm, 4)
    0.0001
    """

    def __init__(
            self, y, w, transformation="r", permutations=PERMUTATIONS, two_tailed=True
    ):
        y = np.asarray(y).flatten()
        self.y = y
        w.transform = transformation
        self.w = w
        self.permutations = permutations
        self.__moments()
        self.I = self.__calc(self.z)
        self.z_norm = (self.I - self.EI) / self.seI_norm
        self.z_rand = (self.I - self.EI) / self.seI_rand

        if self.z_norm > 0:
            self.p_norm = 1 - stats.norm.cdf(self.z_norm)
            self.p_rand = 1 - stats.norm.cdf(self.z_rand)
        else:
            self.p_norm = stats.norm.cdf(self.z_norm)
            self.p_rand = stats.norm.cdf(self.z_rand)

        if two_tailed:
            self.p_norm *= 2.0
            self.p_rand *= 2.0

        if permutations:
            sim = [
                self.__calc(np.random.permutation(self.z)) for i in range(permutations)
            ]
            self.sim = sim = np.array(sim)
            above = sim >= self.I
            larger = above.sum()
            if (self.permutations - larger) < larger:
                larger = self.permutations - larger
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.EI_sim = sim.sum() / permutations
            self.seI_sim = np.array(sim).std()
            self.VI_sim = self.seI_sim ** 2
            if self.seI_sim != 0:
                self.z_sim = (self.I - self.EI_sim) / self.seI_sim
            else:
                self.z_sim = (self.I - self.EI_sim) / np.finfo(float).eps
            if self.z_sim > 0:
                self.p_z_sim = 1 - stats.norm.cdf(self.z_sim)
            else:
                self.p_z_sim = stats.norm.cdf(self.z_sim)

        # provide .z attribute that is znormalized
        sy = y.std()
        self.z /= sy

    def __moments(self):
        self.n = len(self.y)
        y = self.y
        z = y - y.mean()
        self.z = z
        self.z2ss = (z * z).sum()
        self.EI = -1.0 / (self.n - 1)
        n = self.n
        n2 = n * n
        s1 = self.w.s1
        s0 = self.w.s0
        s2 = self.w.s2
        s02 = s0 * s0
        v_num = n2 * s1 - n * s2 + 3 * s02
        v_den = (n - 1) * (n + 1) * s02
        self.VI_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
        self.seI_norm = self.VI_norm ** (1 / 2.0)

        # variance under randomization
        xd4 = z ** 4
        xd2 = z ** 2
        k_num = xd4.sum() / n
        k_den = (xd2.sum() / n) ** 2
        k = k_num / k_den
        EI = self.EI
        A = n * ((n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02)
        B = k * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02)
        VIR = (A - B) / ((n - 1) * (n - 2) * (n - 3) * s02) - EI * EI
        self.VI_rand = VIR
        self.seI_rand = VIR ** (1 / 2.0)

    def __calc(self, z):
        zl = slag(self.w, z)
        inum = (z * zl).sum()
        return self.n / self.w.s0 * inum / self.z2ss

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics"""
        return self.I

    @classmethod
    def by_col(
            cls, df, cols, w=None, inplace=False, pvalue="sim", outvals=None, **stat_kws
    ):
        """
        Function to compute a Moran statistic on a dataframe

        Arguments
        ---------
        df          :   pandas.DataFrame
                        a pandas dataframe with a geometry column
        cols        :   string or list of string
                        name or list of names of columns to use to compute the statistic
        w           :   pysal weights object
                        a weights object aligned with the dataframe. If not provided, this
                        is searched for in the dataframe's metadata
        inplace     :   bool
                        a boolean denoting whether to operate on the dataframe inplace or to
                        return a series contaning the results of the computation. If
                        operating inplace, the derived columns will be named
                        'column_moran'
        pvalue      :   string
                        a string denoting which pvalue should be returned. Refer to the
                        the Moran statistic's documentation for available p-values
        outvals     :   list of strings
                        list of arbitrary attributes to return as columns from the
                        Moran statistic
        **stat_kws  :   keyword arguments
                        options to pass to the underlying statistic. For this, see the
                        documentation for the Moran statistic.

        Returns
        --------
        If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
        returns a copy of the dataframe with the relevant columns attached.

        """
        return _univariate_handler(
            df,
            cols,
            w=w,
            inplace=inplace,
            pvalue=pvalue,
            outvals=outvals,
            stat=cls,
            swapname=cls.__name__.lower(),
            **stat_kws
        )

# copied from pysal package
# see https://github.com/pysal/libpysal/blob/1d6326129904776b0f67597603f3d0af05a9da4c/libpysal/weights/spatial_lag.py
# pointing to a commit, not to main for reproducibility
# renamed from lag_spatial to slag
def slag(w, y):
    """
    Spatial lag operator.

    If w is row standardized, returns the average of each observation's neighbors;
    if not, returns the weighted sum of each observation's neighbors.

    Parameters
    ----------

    w                   : W
                          libpysal spatial weightsobject
    y                   : array
                          numpy array with dimensionality conforming to w (see examples)

    Returns
    -------

    wy                  : array
                          array of numeric values for the spatial lag

    Examples
    --------

    Setup a 9x9 binary spatial weights matrix and vector of data; compute the
    spatial lag of the vector.

    >>> import libpysal
    >>> import numpy as np
    >>> w = libpysal.weights.lat2W(3, 3)
    >>> y = np.arange(9)
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([ 4.,  6.,  6., 10., 16., 14., 10., 18., 12.])

    Row standardize the weights matrix and recompute the spatial lag

    >>> w.transform = 'r'
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([2.        , 2.        , 3.        , 3.33333333, 4.        ,
           4.66666667, 5.        , 6.        , 6.        ])


    Explicitly define data vector as 9x1 and recompute the spatial lag

    >>> y.shape = (9, 1)
    >>> yl = libpysal.weights.lag_spatial(w, y)
    >>> yl
    array([[2.        ],
           [2.        ],
           [3.        ],
           [3.33333333],
           [4.        ],
           [4.66666667],
           [5.        ],
           [6.        ],
           [6.        ]])


    Take the spatial lag of a 9x2 data matrix

    >>> yr = np.arange(8, -1, -1)
    >>> yr.shape = (9, 1)
    >>> x = np.hstack((y, yr))
    >>> yl = libpysal.weights.lag_spatial(w, x)
    >>> yl
    array([[2.        , 6.        ],
           [2.        , 6.        ],
           [3.        , 5.        ],
           [3.33333333, 4.66666667],
           [4.        , 4.        ],
           [4.66666667, 3.33333333],
           [5.        , 3.        ],
           [6.        , 2.        ],
           [6.        , 2.        ]])

    """
    return w.sparse * y


# copied from pysal package
# see https://github.com/pysal/esda/blob/c6a78c3fd4583ef7097970ff04a1d56b6dfbc7f7/esda/tabular.py
# pointing to a commit, not to main for reproducibility
def _univariate_handler(df, cols, stat=None, w=None, inplace=True,
                        pvalue='sim', outvals=None, swapname='', **kwargs):
    """
    Compute a univariate descriptive statistic `stat` over columns `cols` in
    `df`.

    Parameters
    ----------
    df          : pandas.DataFrame
                  the dataframe containing columns to compute the descriptive
                  statistics
    cols        : string or list of strings
                  one or more names of columns in `df` to use to compute
                  exploratory descriptive statistics.
    stat        : callable
                  a function that takes data as a first argument and any number
                  of configuration keyword arguments and returns an object
                  encapsulating the exploratory statistic results
    w           : pysal.weights.W
                  the spatial weights object corresponding to the dataframe
    inplace     : bool
                  a flag denoting whether to add the statistic to the dataframe
                  in memory, or to construct a copy of the dataframe and append
                  the results to the copy
    pvalue      : string
                  the name of the pvalue on the results object wanted
    outvals     : list of strings
                  names of attributes of the dataframe to attempt to flatten
                  into a column
    swapname    : string
                  suffix to replace generic identifier with. Each caller of this
                  function should set this to a unique column suffix
    **kwargs    : optional keyword arguments
                  options that are passed directly to the statistic
    """
    ### Preprocess
    if not inplace:
        new_df = df.copy()
        _univariate_handler(new_df, cols, stat=stat, w=w, pvalue=pvalue,
                            inplace=True, outvals=outvals,
                            swapname=swapname, **kwargs)
        return new_df
    if w is None:
        for name in df._metadata:
            this_obj = df.__dict__.get(name)
            if isinstance(this_obj, W):
                w = this_obj
    if w is None:
        raise Exception('Weights not provided and no weights attached to frame!'
                        ' Please provide a weight or attach a weight to the'
                        ' dataframe')
    ### Prep indexes
    if outvals is None:
        outvals = []
    outvals.insert(0, '_statistic')
    if pvalue.lower() in ['all', 'both', '*']:
        raise NotImplementedError("If you want more than one type of PValue,add"
                                  " the targeted pvalue type to outvals. For example:"
                                  " Geary(df, cols=['HOVAL'], w=w, outvals=['p_z_sim', "
                                  "'p_rand']")
    # this is nontrivial, since we
    # can't know which p_value types are on the object without computing it.
    # This is because we don't flag them with @properties, so they're just
    # arbitrarily assigned post-facto. One solution might be to post-process the
    # objects, determine which pvalue types are available, and then grab them
    # all if needed.

    if pvalue != '':
        outvals.append('p_' + pvalue.lower())
    if isinstance(cols, str):
        cols = [cols]

    ### Make closure around weights & apply columnwise
    def column_stat(column):
        return stat(column.values, w=w, **kwargs)

    stat_objs = df[cols].apply(column_stat)

    ### Assign into dataframe
    for col in cols:
        stat_obj = stat_objs[col]
        y = kwargs.get('y')
        if y is not None:
            col += '-' + y.name
        outcols = ['_'.join((col, val)) for val in outvals]
        for colname, attname in zip(outcols, outvals):
            df[colname] = stat_obj.__getattribute__(attname)
    if swapname != '':
        df.columns = [_swap_ending(col, swapname) if col.endswith('_statistic') else col
                      for col in df.columns]


# copied from pysal package
# see https://github.com/pysal/esda/blob/c6a78c3fd4583ef7097970ff04a1d56b6dfbc7f7/esda/tabular.py
# pointing to a commit, not to main for reproducibility
def _swap_ending(s, ending, delim='_'):
    """
    Replace the ending of a string, delimited into an arbitrary
    number of chunks by `delim`, with the ending provided

    Parameters
    ----------
    s       :   string
                string to replace endings
    ending  :   string
                string used to replace ending of `s`
    delim   :   string
                string that splits s into one or more parts

    Returns
    -------
    new string where the final chunk of `s`, delimited by `delim`, is replaced
    with `ending`.
    """
    parts = [x for x in s.split(delim)[:-1] if x != '']
    parts.append(ending)
    return delim.join(parts)


def run_morans(rag, foi):
    """Run morans I, measure of spatial correlation and significance.

    Parameters
    ----------
    rag     :   RAG
                Region adjacency graph. (https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag.html)
    foi     :   string
                feature of interest

    Returns
    -------
    Morans I class object.

    """
    get_logger().info("Calculating morans I group statistic for FOI \"%s\"..." % foi)

    # extract weights
    weights = W.from_networkx(rag)

    # extract the feature of interest from the rag
    morans_features = [rag.nodes[nodes_idx][foi] for nodes_idx in list(rag.nodes)]

    morans_i = Moran(morans_features, weights, two_tailed=False)

    get_logger().info("Morans I value: %s " % morans_i.I)
    get_logger().info("Morans I p norm: %s " % morans_i.p_norm)

    return morans_i
