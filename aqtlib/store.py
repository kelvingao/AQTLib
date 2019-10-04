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

from sqlalchemy.engine.url import URL

from typing import List, Awaitable
from aqtlib import Object, PG, util
from aqtlib.schema import metadata, ticks

from ib_insync import IB, Forex

# configure logging
util.createLogger(__name__, logging.DEBUG)

__all__ = ['Store']


class Store(Object):
    """Store class initilizer

    """

    RequestTimeout = 0

    defaults = dict(
        symbols='sybmols.csv',
        ib_port=4001,  # 7496/7497 = TWS, 4001/4002 = IBGateway
        ib_client=308,
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
        self.pg = PG()

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
            description='AQTLib Store',
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

        # initilize and create PostgreSQL schema tables
        self._logger.info("Initialize PostgreSQL...")
        self.pg.init(str(URL(
                    drivername='postgres',
                    database=self.db_name,
                    username=self.db_user,
                    password=self.db_pass,
                    host=self.db_host)
                ), metadata)

        # connect to Interactive Brokers
        # FIXME: if there is no ibgateway, try reconnecting
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
                "\n\n>>> Interrupted with Ctrl-c...\n(waiting for running tasks to be completed)\n")
            # FIXME: quit gracefully
            sys.exit(1)

        self._loop.run_forever()
