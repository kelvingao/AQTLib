"""Several objects defintions."""

import os
import pandas as pd
import logging

from stat import S_IWRITE


__all__ = ('Object DataStore').split()


class Object:
    """
    Base object, with:
    * __slots__ to avoid typos;
    * A general constructor;
    * A general string representation;
    """
    __slots__ = ('__weakref__',)
    defaults: dict = {}

    def __init__(self, *args, **kwargs):
        """
        Attribute values can be given positionally or as keyword.
        If an attribute is not given it will take its value from the
        'defaults' class member. If an attribute is given both positionally
        and as keyword, the keyword wins.
        """
        defaults = self.__class__.defaults
        d = {**defaults, **dict(zip(defaults, args)), **kwargs}
        for k, v in d.items():
            setattr(self, k, v)

    def __repr__(self):
        clsName = self.__class__.__qualname__
        kwargs = ', '.join(f'{k}={v!r}' for k, v in self.dict().items())
        return f'{clsName}({kwargs})'

    __str__ = __repr__

    def dict(self):
        """
        Return key-value pairs as a dictionary.
        """
        return {k: getattr(self, k) for k in self.__class__.defaults}

    def update(self, **kwargs):
        """
        Update key values.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def nonDefaults(self):
        """
        Get a dictionary of all attributes that differ from the default.
        """
        nonDefaults = {}
        for k, d in self.__class__.defaults.items():
            v = getattr(self, k)
            if v != d and (v == v or d == d):  # tests for NaN too
                nonDefaults[k] = v
        return nonDefaults

# ---------------------------------------------

def chmod(f):
    """ change mod to writeable """
    try:
        os.chmod(f, S_IWRITE)  # windows (cover all)
    except Exception as e:
        pass
    try:
        os.chmod(f, 0o777)  # *nix
    except Exception as e:
        pass


class DataStore:
    def __init__(self, output_file=None):
        self.output_file = output_file
        self.recorded = pd.DataFrame()
        self.rows = []
        self._logger = logging.getLogger(self.__class__.__name__)

        self.callbacks = []

    def register_callback(self, callback):
        self.callbacks.append(callback)
        return callback

    def notify(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)

    def record(self, timestamp, *args, **kwargs):
        """ add custom data to data store """
        if self.output_file is None:
            return

        data = self._new_data(timestamp, *args, **kwargs)

        # append to rows
        self.rows.append(pd.DataFrame(data=data, index=[timestamp]))

        # wait bar data
        if "symbol" not in self.rows[-1].columns:
            return

        # create a record
        record = self._create_record()

        # make this public
        self.recorded = self.recorded.append(record, sort=True)
        self.notify(self.recorded)

        self.rows.clear()

        # save
        if ".csv" in self.output_file:
            self.recorded.to_csv(self.output_file)
        elif ".h5" in self.output_file:
            self.recorded.to_hdf(self.output_file, 0)
        elif (".pickle" in self.output_file) | (".pkl" in self.output_file):
            self.recorded.to_pickle(self.output_file)

        chmod(self.output_file)

    def _new_data(self, timestamp, *args, **kwargs):

        data = {}

        # append all data
        if len(args) == 1:
            if isinstance(args[0], dict):
                data.update(dict(args[0]))
            elif isinstance(args[0], pd.DataFrame):
                data.update(args[0][-1:].to_dict(orient='records')[0])

        # add kwargs
        if kwargs:
            data.update(dict(kwargs))

        new_data = {}

        if "symbol" not in data.keys():
            new_data = dict(data)

        # BAR...
        else:
            sym = data["symbol"]
            new_data["symbol"] = sym
            for key in data.keys():
                if key not in ['datetime', 'symbol_group', 'asset_class']:
                    new_data[sym + '_' + str(key).upper()] = data[key]

        new_data['datetime'] = timestamp

        return new_data

    def _create_record(self):

        recorded = pd.concat(self.rows, sort=True)

        # group by symbol
        recorded['datetime'] = recorded.index

        data = recorded.groupby(['symbol', 'datetime'], as_index=False).sum()
        data.set_index('datetime', inplace=True)

        symbols = data['symbol'].unique().tolist()
        data.drop(columns=['symbol'], inplace=True)

        # cleanup:
        # remove symbols
        recorded.drop(['symbol'] + [sym + '_SYMBOL' for sym in symbols],
            axis=1, inplace=True)

        recorded = recorded.groupby(recorded['datetime']).first()

        for sym in symbols:
            if sym + '_POSITION' not in recorded.columns:
                recorded[sym + '_POSITION'] = self.recorded[sym + '_POSITION'][-1]

        self._logger.debug('CREATE RECORD\n{}'.format(recorded))

        return recorded
