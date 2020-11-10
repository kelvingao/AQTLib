"""Porter class definition."""

import os
import sys
import asyncio
import argparse
import logging

import pandas as pd
import numpy as np

from sqlalchemy.engine.url import URL
from sqlalchemy import select, and_
from datetime import datetime
from typing import List, Awaitable
from aqtlib import Object, utils
from apgsa import PG
from ib_insync import IB, Forex

from .schema import metadata, symbols, bars, ticks


__all__ = ['Porter']


class Porter(Object):
    """Porter class initilizer

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
    # __slots__ = defaults.keys()

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)

        # initilize class logger
        self._logger = logging.getLogger(__name__)

        # override with (non-default) command-line args
        self.update(**self.load_cli_args())

        # database manager
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
            description='AQTLib Porter',
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

    def _run(self, *awaitables: Awaitable):
        return utils.run(*awaitables, timeout=self.RequestTimeout)

    def run(self):
        """Starts the Porter

        Connects to the TWS/GW, processes and logs market data.

        """

        self._loop.run_forever()

    def connect_sql(self):
        # connect to PostgreSQL
        self.pg.connect(
            self.db_host, self.db_name, self.db_user, self.db_pass)
        self._logger.info("PostgreSQL {}:{} Connected.".format(self.db_host, self.db_port))

    async def get_symbol_id_async(self, symbol):
        # start
        asset_class = utils.gen_asset_class(symbol)
        symbol_group = utils.gen_symbol_group(symbol)
        clean_symbol = symbol.replace("_" + asset_class, "")
        expiry = None

        async def querySymbolIdAsync(asset_class, symbol_group, clean_symbol):
            sql = select([symbols]).where(and_(
                symbols.c.symbol == clean_symbol,
                symbols.c.symbol_group == symbol_group,
                symbols.c.asset_class == asset_class)
            )

            return await self.pg.fetchrow(sql)

        # symbol already in db
        row = await querySymbolIdAsync(asset_class, symbol_group, clean_symbol)
        if row is not None:
            return row[0]

        # symbol/expiry not in db... insert new/update expiry
        else:
            # need to update the expiry?
            # TODO: add expiry

            # insert new symbol
            data = {
                'symbol': clean_symbol,
                'symbol_group': symbol_group,
                'asset_class': asset_class,
                'expiry': expiry
            }
            sql = symbols.insert([data])
            await self.pg.execute(sql)

            row = await querySymbolIdAsync(asset_class, symbol_group, clean_symbol)
            return row[0]

    async def store_data_async(self, df, kind="BAR"):
        # validate columns
        valid_cols = utils.validate_columns(df, kind)
        if not valid_cols:
            raise ValueError('Invalid Column list')

        # loop through symbols and save in db
        for symbol in list(df['symbol'].unique()):
            data = df[df['symbol'] == symbol]
            symbol_id = await self.get_symbol_id_async(symbol)

            # prepare columns for insert
            data.loc[:, 'datetime'] = data.index
            data.loc[:, 'symbol_id'] = symbol_id
            data = data.drop(['symbol', 'symbol_group', 'asset_class', 'expiry'], axis=1)

            # insert row by row to handle greeks
            data = data.to_dict(orient="records")

            if kind == "BAR":
                for _, row in enumerate(data):
                    sql = bars.insert().values([row])
                    await self.pg.execute(sql)
            else:
                pass

        return True

    # -------------------------------------------
    async def get_data_async(self, sql) -> pd.DataFrame:
        # async with self.pg.pool.acquire() as conn:
        # stmt = await conn.prepare(sql)
        data = await self.pg.fetch(sql)
        if not data:
            return pd.DataFrame()

        columns = [k for k in data[0].keys()]
        return pd.DataFrame(data, columns=columns)

    def get_history(self, symbols, start, end=None, resolution="1T", tz="UTC", continuous=True):
        if end is None:
            end = datetime.now()

        sql_query = select([
            bars.c.datetime, bars.c.open, bars.c.high, bars.c.low, bars.c.close, bars.c.volume]).where(
                and_(bars.c.datetime >= start, bars.c.datetime <= end))

        return utils.run(self.get_data_async(sql_query))

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
    def prepare_data(instrument, data, output_path=None,
                 index=None, colsmap=None, kind="BAR", resample="1T") -> pd.DataFrame:
        """
        Converts given DataFrame to a AQTLib-compatible format csv file.

        :Parameters:
            instrument : mixed
                IB contract tuple / string (same as that given to strategy)
            data : pd.DataFrame
                Pandas DataDrame with that instrument's market data
            output_path : str
                Path to where the resulting CSV should be saved (optional)
            index : pd.Series
                Pandas Series that will be used for df's index (optioanl)
            colsmap : dict
                Dict for mapping df's columns to those used by AQTLib
                (default assumes same naming convention as AQTLib's)
            kind : str
                Is this ``BAR`` or ``TICK`` data
            resample : str
                Pandas resolution (defaults to 1min/1T)
        :Returns:
            data : pd.DataFrame
                Pandas DataFrame in a AQTLib-compatible format and timezone

        """

        df = data.copy()

        # jquant's csv?
        if set(df.columns) == set(['close', 'open', 'high', 'low', 'volume', 'money']):
            df.index = df.index.tz_localize(utils.get_timezone()).tz_convert('UTC')

        # FIXME: generate a valid ib tuple
        symbol = instrument[0] + '_' + instrument[1]
        symbol_group = instrument[0]
        asset_class = instrument[1]

        df.loc[:, 'symbol'] = symbol
        df.loc[:, 'symbol_group'] = symbol_group
        df.loc[:, 'asset_class'] = asset_class

        # TODO: validate, remove and map columns

        df.index.rename("datetime", inplace=True)

        # save csv
        if output_path is not None:
            output_path = os.path.expanduser(output_path)
            output_path = output_path[:-1] if output_path.endswith('/') else output_path
            df.to_csv("{path}/{symbol}.{kind}.csv".format(
                path=output_path, symbol=symbol, kind=kind))

        return df

    # -------------------------------------------
    @staticmethod
    def prepare_bars_history(data, resolution="1T", tz=None):

        # setup dataframe
        data.set_index('datetime', inplace=True)
        data.index = pd.to_datetime(data.index, utc=True)

        # meta data
        meta_data = data.groupby(["symbol"])[
            ['symbol', 'symbol_group', 'asset_class']].last()

        combined = []

        bars_ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        for symbol in meta_data.index.values:
            bar_dict = {}

            for col in data[data['symbol'] == symbol].columns:
                if col in bars_ohlc_dict.keys():
                    bar_dict[col] = bars_ohlc_dict[col]

            # convert timezone
            if tz:
                data.index = data.index.tz_convert(tz)

            resampled = data[data['symbol'] == symbol].resample(resolution).apply(bar_dict)

            # drop NANs
            resampled.dropna(inplace=True)

            resampled['symbol'] = symbol
            resampled['symbol_group'] = meta_data[meta_data.index == symbol]['symbol_group'].values[0]
            resampled['asset_class'] = meta_data[meta_data.index == symbol]['asset_class'].values[0]

            combined.append(resampled)

        data = pd.concat(combined, sort=True)
        data['volume'] = data['volume'].astype(int)

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
