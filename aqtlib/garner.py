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

import sys
import asyncio
import argparse
import logging

import pandas as pd
import numpy as np

from sqlalchemy.engine.url import URL

from typing import List, Awaitable
from aqtlib import Object, PG, util
from aqtlib.schema import metadata, ticks

from ib_insync import IB, Forex

# configure logging
util.createLogger(__name__, logging.DEBUG)

__all__ = ['Garner']


class Garner(Object):
    """Garner class initilizer

    Args:
        symbols : str
            IB contracts CSV database (default: ./symbols.csv)
        ib_port : int
            TWS/GW Port to use (default: 4002)
        ib_client : int
            TWS/GW Client ID (default: 100)
        ib_server : str
            IB TWS/GW Server hostname (default: localhost)
        db_host : str
            PostgreSQL server hostname (default: localhost)
        db_port : str
            PostgreSQL server port (default: 3306)
        db_name : str
            PostgreSQL server database (default: aqtlib_db)
        db_user : str
            PostgreSQL server username (default: aqtlib_user)
        db_pass : str
            PostgreSQL server password (default: aqtlib_pass)
        db_skip : str
            Skip PostgreSQL logging (default: False)
    """

    RequestTimeout = 0

    defaults = dict(
        symbols='sybmols.csv',
        ib_port=4002,  # 7496/7497 = TWS, 4001/4002 = IBGateway
        ib_client=100,
        ib_server='localhost',
        db_host='localhost',
        db_port=5432,
        db_name='aqtlib_db',
        db_user='aqtlib_user',
        db_pass='aqtlib_pass',
        db_skip=False
    )
    __slots__ = defaults.keys()

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)

        # initilize class logger
        self._logger = logging.getLogger(__name__)

        # override with (non-default) command-line args
        self.update(**self.load_cli_args())

        # PostgreSQL manager
        settings = dict(
            drivername='postgres',
            database=self.db_name,
            username=self.db_user,
            password=self.db_pass,
            host=self.db_host
        )
        self.pg = PG(str(URL(**settings)), metadata)

        # sync/async framework for Interactive Brokers
        self.ib = IB()
        self.ib.pendingTickersEvent += self.onPendingTickers

        # do not act on first tick (incorrect)
        self.first_tick = True

        self._loop = asyncio.get_event_loop()

    def onPendingTickers(self, tickers):
        """
        Handling and recording tickers form Interactive Brokers.

        """
        # do not act on first incorrect tick
        if self.first_tick:
            self.first_tick = False
            return

        fields = ['bid', 'bidSize', 'ask', 'askSize', 'time']

        clip_tickers_attrs_generator = (
            # retrive sub attributes from the sequence of Ticker objects.
            # a list of fields is given and only retain those fields.
            {k: v for k, v in ticker.dict().items() if k in fields} for ticker in tickers)

        data = list(clip_tickers_attrs_generator)
        asyncio.ensure_future(self.pg.execute(ticks.insert().values(data)))

    # -------------------------------------------
    def load_cli_args(self):
        parser = argparse.ArgumentParser(
            description='AQTLib Garner',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--ib_port', default=self.ib_port,
                            help='TWS/GW Port to use', required=False)
        parser.add_argument('--ib_client', default=self.ib_client,
                            help='TWS/GW Client ID', required=False)
        parser.add_argument('--ib_server', default=self.ib_server,
                            help='IB TWS/GW Server hostname', required=False)
        parser.add_argument('--orderbook', action='store_true',
                            help='Get Order Book (Market Depth) data',
                            required=False)
        parser.add_argument('--db_host', default=self.db_host,
                            help='PostgreSQL server hostname', required=False)
        parser.add_argument('--db_port', default=self.db_port,
                            help='PostgreSQL server port', required=False)
        parser.add_argument('--db_name', default=self.db_name,
                            help='PostgreSQL server database', required=False)
        parser.add_argument('--db_user', default=self.db_user,
                            help='PostgreSQL server username', required=False)
        parser.add_argument('--db_pass', default=self.db_pass,
                            help='PostgreSQL server password', required=False)
        parser.add_argument('--db_skip', default=self.db_skip,
                            required=False, help='Skip PostgreSQL logging (flag)',
                            action='store_true')

        # only return non-default cmd line args
        # (meaning only those actually given)
        cmd_args, _ = parser.parse_known_args()
        args = {arg: val for arg, val in vars(
            cmd_args).items() if val != parser.get_default(arg)}
        return args

    def _run(self, *awaitables: List[Awaitable]):
        return util.run(*awaitables, timeout=self.RequestTimeout)

    def run(self):
        """Starts the garner

        Connects to the TWS/GW, processes and logs market data.

        """
        # initilize and create PostgreSQL schema tables
        self._logger.info("Initialize PostgreSQL...")
        self._run(self.pg.init_pool())

        # connect to Interactive Brokers
        # FIXME: if disconnect, try reconnecting
        self._logger.info("Connecting to Interactive Brokers...")
        self.ib.connect(
            self.ib_server, self.ib_port, self.ib_client)
        self._logger.info("Connection established.")

        try:
            contract = Forex('EURUSD')
            if contract and self.ib.qualifyContracts(contract):
                tickerInfo = self.ib.reqMktData(contract, '', False, False, None)
                self._logger.info('{} requested...'.format(tickerInfo))

        except (KeyboardInterrupt, SystemExit):
            self.quitting = True  # don't display connection errors on ctrl+c
            print(
                "\n\n>>> Interrupted with Ctrl-c...\n\n")
            # FIXME: quit gracefully
            sys.exit(1)

        self._loop.run_forever()

    # ---------------------------------------------
    @staticmethod
    def validate_csv(df: pd.DataFrame, kind: str = "BAR") -> bool:
        """
        Check if a AQTLib-compatible CSV file.

        """
        _BARS_COLS = ('asset_class', 'open', 'high', 'low', 'close', 'volume')

        for el in _BARS_COLS:
            if el not in df.columns:
                raise ValueError('Column {el} not found'.format(el=el))
                return False

        return True

    # -------------------------------------------
    @staticmethod
    def prepare_bars_history(df, resolution="1T", start=None, end=None, tz="UTC"):

        # setup dataframe
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)

        # meta data
        meta_data = df.groupby(["symbol"])[
            ['symbol', 'symbol_group', 'asset_class']].last()

        combined = []

        bars_ohlc_dict = {
            'open':           'first',
            'high':           'max',
            'low':            'min',
            'close':          'last',
            'volume':         'sum'
        }

        for symbol in meta_data.index.values:
            bar_dict = {}

            for col in df[df['symbol'] == symbol].columns:
                if col in bars_ohlc_dict.keys():
                    bar_dict[col] = bars_ohlc_dict[col]

            resampled = df[df['symbol'] == symbol].resample(
                        resolution).apply(bar_dict).fillna(value=np.nan)

            resampled['symbol'] = symbol
            resampled['symbol_group'] = meta_data[meta_data.index == symbol]['symbol_group'].values[0]
            resampled['asset_class'] = meta_data[meta_data.index == symbol]['asset_class'].values[0]

            combined.append(resampled)

        data = pd.concat(combined, sort=True)
        data['volume'] = data['volume'].astype(int)

        # convert timezone
        if tz:
            data.index = data.index.tz_convert(tz)

        return data

    # -------------------------------------------
    @staticmethod
    def drip(history, handler):
        """
        Replaying history data, and handling each record.

        """
        try:
            for i in range(len(history)):
                handler(history.iloc[i:i + 1])

            print("\n\n>>> Backtesting Completed.")

        except (KeyboardInterrupt, SystemExit):
            print(
                "\n\n>>> Interrupted with Ctrl-c...\n\n")
            print(".\n.\n.\n")
            sys.exit(1)
