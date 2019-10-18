#!/usr/bin/env python3
#
# MIT License
#
# Copyright (c) 2019 Kelvin Gao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import sys
import logging
import argparse
import pandas as pd

from datetime import datetime
from aqtlib import util, Object, Garner
from abc import abstractmethod
from .instrument import Instrument


util.createLogger(__name__, level=logging.INFO)

__all__ = ['Algo']


class Algo(Object):
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
        resolution="1T",
        bars_window=120,
        timezone='UTC',
        backtest=False,
        start=None,
        end=None,
        data=None,
        output=None
    )

    def __init__(self, instruments, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)

        # strategy name
        self.name = self.__class__.__name__

        # initilize strategy logger
        self._logger = logging.getLogger(__name__)

        # override args with (non-default) command-line args
        self.update(**self.load_cli_args())

        # sanity checks for backtesting mode
        if self.backtest:
            self._check_backtest_args()

        self.instruments = instruments
        self.symbols = list([x[0] for x in self.instruments])

        self.bars = pd.DataFrame()

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
            self.end = datetime.now()
        if self.data is not None:
            if not os.path.exists(self.data):
                self._logger.error(
                    "CSV directory cannot be found ({dir})".format(dir=self.data))
                sys.exit(0)
            elif self.data.endswith("/"):
                self.data = self.data[:-1]

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

        parser.add_argument('--backtest', default=self.backtest,
                            help='Work in Backtest mode (flag)',
                            action='store_true')
        parser.add_argument('--start', default=self.start,
                            help='Backtest start date')
        parser.add_argument('--end', default=self.end,
                            help='Backtest end date')
        parser.add_argument('--data', default=self.data,
                            help='Path to backtester CSV files')
        parser.add_argument('--output', default=self.output,
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

        Connects to the Garner, processes data and passes
        bar data to the ``on_bar`` function.

        """
        history = pd.DataFrame()

        if self.backtest:
            # get history from csv dir
            if self.data:
                dfs = self._get_history()

                # prepare history data
                params = (
                    pd.concat(dfs, sort=True), self.resolution, self.start, self.end, self.timezone)
                history = Garner.prepare_bars_history(*params)

            # initiate strategy
            self.on_start()

            # drip history
            Garner.drip(history, self._bar_handler)

    # ---------------------------------------
    def _get_history(self):
        """
        Get bars history from AQTLib-compatible csv file.

        """
        dfs = []
        for symbol in self.symbols:
            file = "{data}/{symbol}.{kind}.csv".format(data=self.data, symbol=symbol, kind="BAR")
            if not os.path.exists(file):
                self._logger.error(
                    "Can't load data for {symbol} ({file} doesn't exist)".format(
                        symbol=symbol, file=file))
                sys.exit(0)
            try:
                df = pd.read_csv(file)
                if not Garner.validate_csv(df, "BAR"):
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

        self.bars = self._update_window(self.bars, bar,
                                        window=self.bars_window)

        instrument = self.get_instrument(symbol)
        self.on_bar(instrument)

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
        instrument._bind_strategy(self)

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
    def on_start(self):
        """
        Invoked once when algo starts. Used for when the strategy
        needs to initialize parameters upon starting.

        """
        # raise NotImplementedError("Should implement on_start()")
        pass
