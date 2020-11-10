"""Algo class definition."""

import os
import sys
import logging
import argparse
import pandas as pd

from datetime import datetime
from aqtlib import utils, Broker, Porter
from aqtlib.objects import DataStore
from abc import abstractmethod
from .instrument import Instrument


__all__ = ['Algo']


class Algo(Broker):
    """Algo class initilizer.

    Args:
        instruments : list
            List of IB contract tuples.
        resolution : str
            Desired bar resolution (using pandas resolution: 1T, 1H, etc).
            Default is 1T (1min)
        bars_window : int
            Length of bars lookback window to keep. Defaults to 120
        timezone : str
            Convert IB timestamps to this timezone (eg. US/Central).
            Defaults to UTC
        backtest: bool
            Whether to operate in Backtest mode (default: False)
        start: str
            Backtest start date (YYYY-MM-DD [HH:MM:SS[.MS]). Default is None
        end: str
            Backtest end date (YYYY-MM-DD [HH:MM:SS[.MS]). Default is None
        data : str
            Path to the directory with AQTLib-compatible CSV files (Backtest)
        output: str
            Path to save the recorded data (default: None)
    """

    defaults = dict(
        instruments=[],
        resolution="1D",
        bars_window=120,
        timezone='UTC',
        backtest=False,
        start=None,
        end=None,
        data=None,
        output=None
    )

    def __init__(self, instruments, *args, **kwargs):
        super(Algo, self).__init__(instruments, *args, **kwargs)

        # strategy name
        self.name = self.__class__.__name__

        # initilize strategy logger
        self._logger = logging.getLogger(self.name)

        # override args with (non-default) command-line args
        self.update(**self.load_cli_args())

        self.backtest_csv = self.data

        # sanity checks for backtesting mode
        if self.backtest:
            self._check_backtest_args()

        # initilize output file
        self.record_ts = None
        if self.output:
            self.datastore = DataStore(self.output)

        self.bars = pd.DataFrame()
        self.bar_hashes = {}

        # -----------------------------------
        # signal collector
        self.signals = {}
        for sym in self.symbols:
            self.signals[sym] = pd.DataFrame()

        self.initialize()

    # ---------------------------------------
    def _check_backtest_args(self):
        if self.output is None:
            self._logger.error(
                "Must provide an output file for Backtest mode")
            sys.exit(0)

        if self.start is None:
            self._logger.error(
                "Must provide start date for Backtest mode")
            sys.exit(0)

        if self.end is None:
            self.end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.backtest_csv is not None:
            self.backtest_csv = os.path.expanduser(self.backtest_csv)
            if not os.path.exists(self.backtest_csv):
                self._logger.error(
                    "CSV directory cannot be found ({dir})".format(dir=self.backtest_csv))
                sys.exit(0)
            elif self.backtest_csv.endswith("/"):
                self.backtest_csv = self.backtest_csv[:-1]

    # ---------------------------------------
    def load_cli_args(self):
        """
        Parse command line arguments and return only the non-default ones

        :Retruns: dict
            a dict of any non-default args passed on the command-line.
        """
        parser = argparse.ArgumentParser(
            description='AQTLib Algorithm',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--backtest', default=self.defaults['backtest'],
                            help='Work in Backtest mode (flag)',
                            action='store_true')
        parser.add_argument('--start', default=self.defaults['start'],
                            help='Backtest start date')
        parser.add_argument('--end', default=self.defaults['end'],
                            help='Backtest end date')
        parser.add_argument('--data', default=self.defaults['data'],
                            help='Path to backtester CSV files')
        parser.add_argument('--output', default=self.defaults['output'],
                            help='Path to save the recorded data')

        # only return non-default cmd line args
        # (meaning only those actually given)
        cmd_args, _ = parser.parse_known_args()
        args = {arg: val for arg, val in vars(
            cmd_args).items() if val != parser.get_default(arg)}
        return args

    # ---------------------------------------
    def run(self):
        """Starts the algo

        Connects to the Porter, processes data and passes
        bar data to the ``on_bar`` function.

        """
        history = pd.DataFrame()

        if self.backtest:
            self._logger.info('Algo start backtesting...')
            # history from csv dir
            if self.backtest_csv:
                dfs = self._fetch_csv()

                # prepare history data
                history = Porter.prepare_bars_history(
                    data=pd.concat(dfs, sort=True),
                    resolution=self.resolution,
                    tz=self.timezone
                )

                history = history[(history.index >= self.start) & (history.index <= self.end)]

            else:
                # history from porter
                import nest_asyncio
                nest_asyncio.apply()

                # connect to database
                self.porter.connect_sql()
                history = self.porter.get_history(
                    symbols=self.symbols,
                    start=self.start,
                    end=self.end if self.end else datetime.now(),
                    resolution=self.resolution,
                    tz=self.timezone
                )

                history = utils.prepare_data(('AAPL', 'STK'), history, index=history.datetime)

            # optimize pandas
            if not history.empty:
                history['symbol'] = history['symbol'].astype('category')
                history['symbol_group'] = history['symbol_group'].astype('category')
                history['asset_class'] = history['asset_class'].astype('category')

            # drip history
            Porter.drip(history, self._bar_handler)

    # ---------------------------------------
    def _fetch_csv(self):
        """
        Get bars history from AQTLib-compatible csv file.

        """
        dfs = []
        for symbol in self.symbols:
            file = "{data}/{symbol}.{kind}.csv".format(data=self.backtest_csv, symbol=symbol, kind="BAR")
            if not os.path.exists(file):
                self._logger.error(
                    "Can't load data for {symbol} ({file} doesn't exist)".format(
                        symbol=symbol, file=file))
                sys.exit(0)
            try:
                df = pd.read_csv(file)
                if not Porter.validate_csv(df, "BAR"):
                    self._logger.error("{file} isn't a AQTLib-compatible format".format(file=file))
                    sys.exit(0)

                if df['symbol'].values[-1] != symbol:
                    self._logger.error(
                        "{file} doesn't content data for {symbol}".format(file=file, symbol=symbol))
                    sys.exit(0)

                dfs.append(df)

            except Exception as e:
                self._logger.error(
                    "Error reading data for {symbol} ({errmsg})", symbol=symbol, errmsg=e)
                sys.exit(0)
        return dfs

    # ---------------------------------------
    def _bar_handler(self, bar):
        """
        Invoked on every bar captured for the selected instrument.

        """
        symbol = bar['symbol'].values
        if len(symbol) == 0:
            return
        symbol = symbol[0]

        # self_bars = self.bars.copy()  # work on copy

        self.bars = self._update_window(self.bars, bar,
                                        window=self.bars_window)

        # optimize pandas
        if len(self.bars) == 1:
            self.bars['symbol'] = self.bars['symbol'].astype('category')
            self.bars['symbol_group'] = self.bars['symbol_group'].astype('category')
            self.bars['asset_class'] = self.bars['asset_class'].astype('category')

        # new bar?
        hash_string = bar[:1]['symbol'].to_string().translate(
            str.maketrans({key: None for key in "\n -:+"}))
        this_bar_hash = abs(hash(hash_string)) % (10 ** 8)

        newbar = True
        if symbol in self.bar_hashes.keys():
            newbar = self.bar_hashes[symbol] != this_bar_hash
        self.bar_hashes[symbol] = this_bar_hash

        if newbar:
            if self.bars[(self.bars['symbol'] == symbol) | (
                    self.bars['symbol_group'] == symbol)].empty:
                return

            instrument = self.get_instrument(symbol)
            if instrument:
                self.record_ts = bar.index[0]
                self._logger.debug('BAR TIME: {}'.format(self.record_ts))
                self.on_bar(instrument)
                self.record(bar)

    # ---------------------------------------
    def _update_window(self, df, data, window=None, resolution=None):
        """
        No. of bars to keep.

        """
        df = df.append(data, sort=True) if df is not None else data

        # return
        if window is None:
            return df

        return self._get_window_per_symbol(df, window)

    # ---------------------------------------
    @staticmethod
    def _get_window_per_symbol(df, window):
        """
        Truncate bars window per symbol.

        """
        dfs = []
        for symbol in list(df["symbol"].unique()):
            dfs.append(df[df['symbol'] == symbol][-window:])

        return pd.concat(dfs, sort=True).sort_index()

    # ---------------------------------------
    def get_instrument(self, symbol):
        """
        A string subclass that provides easy access to misc
        symbol-related methods and information using shorthand.

        Call from within your strategy:
        ``instrument = self.get_instrument("SYMBOL")``

        """
        instrument = Instrument(symbol)
        instrument.attach_strategy(self)

        return instrument

    # ---------------------------------------
    @abstractmethod
    def on_bar(self, instrument):
        """
        Invoked on every bar captured for the selected instrument.
        This is where you'll write your strategy logic for bar events.

        """
        # raise NotImplementedError("Should implement on_bar()")
        pass

    # ---------------------------------------
    @abstractmethod
    def initialize(self):
        """
        Invoked once when algo starts. Used for when the strategy
        needs to initialize parameters upon starting.

        """
        # raise NotImplementedError("Should implement initialize()")
        pass

    def order(self, signal, symbol, quantity=0, **kwargs):
        """ Send an order for the selected instrument

        :Parameters:

            direction : string
                Order Type (BUY/SELL, EXIT/FLATTEN)
            symbol : string
                instrument symbol
            quantity : int
                Order quantiry

        :Optional:

            limit_price : float
                In case of a LIMIT order, this is the LIMIT PRICE
            expiry : int
                Cancel this order if not filled after *n* seconds
                (default 60 seconds)
            order_type : string
                Type of order: Market (default),
                LIMIT (default when limit_price is passed),
                MODIFY (required passing or orderId)
            orderId : int
                If modifying an order, the order id of the modified order
            target : float
                Target (exit) price
            initial_stop : float
                Price to set hard stop
            stop_limit: bool
                Flag to indicate if the stop should be STOP or STOP LIMIT.
                Default is ``False`` (STOP)
            trail_stop_at : float
                Price at which to start trailing the stop
            trail_stop_type : string
                Type of traiing stop offset (amount, percent).
                Default is ``percent``
            trail_stop_by : float
                Offset of trailing stop distance from current price
            fillorkill: bool
                Fill entire quantiry or none at all
            iceberg: bool
                Is this an iceberg (hidden) order
            tif: str
                Time in force (DAY, GTC, IOC, GTD). default is ``DAY``
        """
        self._logger.debug('ORDER: %s %4d %s %s', signal,
            quantity, symbol, kwargs)

        position = self.get_positions(symbol)
        if signal.upper() == "EXIT" or signal.upper() == "FLATTEN":
            if position['position'] == 0:
                return

            kwargs['symbol'] = symbol
            kwargs['quantity'] = abs(position['position'])
            kwargs['direction'] = "BUY" if position['position'] < 0 else "SELL"

            # print("EXIT", kwargs)

            try:
                self.record({symbol + '_POSITION': 0})
            except Exception as e:
                pass

        else:
            if quantity == 0:
                return

            kwargs['symbol'] = symbol
            kwargs['quantity'] = abs(quantity)
            kwargs['direction'] = signal.upper()

            # print(signal.upper(), kwargs)

            # record
            try:
                quantity = abs(quantity)
                if kwargs['direction'] != "BUY":
                    quantity = -quantity
                self.record({symbol + '_POSITION': quantity + position['position']})
            except Exception as e:
                pass

    # ---------------------------------------
    def record(self, *args, **kwargs):
        """Records data for later analysis.
        Values will be logged to the file specified via
        ``--output [file]`` (along with bar data) as
        csv/pickle/h5 file.

        Call from within your strategy:
        ``self.record(key=value)``

        :Parameters:
            ** kwargs : mixed
                The names and values to record

        """
        if self.output:
            try:
                self.datastore.record(self.record_ts, *args, **kwargs)
            except Exception as e:
                pass
