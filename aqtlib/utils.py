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
import datetime
import time
import asyncio
import colorlog
import logging
import pandas as pd

from dateutil.parser import parse as parse_date
from dateutil import relativedelta
from pytz import timezone
from apgsa import PG

import eventkit as ev
import numpy as np

globalErrorEvent = ev.Event()
"""
Event to emit global exceptions.
"""


def get_timezone(as_timedelta=False):
    """ utility to get the machine's timezone """
    try:
        offset_hour = -(time.altzone if time.daylight else time.timezone)
    except Exception as e:
        offset_hour = -(datetime.datetime.now() -
                        datetime.datetime.utcnow()).seconds

    offset_hour = offset_hour // 3600
    offset_hour = offset_hour if offset_hour < 10 else offset_hour // 10

    if as_timedelta:
        return datetime.timedelta(hours=offset_hour)

    return 'Etc/GMT%+d' % -offset_hour


def set_timezone(data, tz=None, from_local=False):
    """ change the timeozone to specified one without converting """
    # pandas object?
    if isinstance(data, pd.DataFrame) | isinstance(data, pd.Series):
        try:
            try:
                data.index = data.index.tz_convert(tz)
            except Exception as e:
                if from_local:
                    data.index = data.index.tz_localize(
                        get_timezone()).tz_convert(tz)
                else:
                    data.index = data.index.tz_localize('UTC').tz_convert(tz)
        except Exception as e:
            pass

    # not pandas...
    else:
        if isinstance(data, str):
            data = parse_date(data)
        try:
            try:
                data = data.astimezone(tz)
            except Exception as e:
                data = timezone('UTC').localize(data).astimezone(timezone(tz))
        except Exception as e:
            pass

    return data


def run(*awaitables, timeout: float = None):
    """
    By default run the event loop forever.

    When awaitables (like Tasks, Futures or coroutines) are given then
    run the event loop until each has completed and return their results.

    An optional timeout (in seconds) can be given that will raise
    asyncio.TimeoutError if the awaitables are not ready within the
    timeout period.
    """
    loop = asyncio.get_event_loop()
    if not awaitables:
        if loop.is_running():
            return
        loop.run_forever()
        f = asyncio.gather(*asyncio.Task.all_tasks())
        f.cancel()
        result = None
        try:
            loop.run_until_complete(f)
        except asyncio.CancelledError:
            pass
    else:
        if len(awaitables) == 1:
            future = awaitables[0]
        else:
            future = asyncio.gather(*awaitables)
        if timeout:
            future = asyncio.wait_for(future, timeout)
        task = asyncio.ensure_future(future)

        def onError(_):
            task.cancel()

        globalErrorEvent.connect(onError)
        try:
            result = loop.run_until_complete(task)
        except asyncio.CancelledError as e:
            raise globalErrorEvent.value() or e
        finally:
            globalErrorEvent.disconnect(onError)

    return result


def patchAsyncio():
    """Patch asyncio to allow nested event loops."""
    import nest_asyncio
    nest_asyncio.apply()


def startLoop():
    """
    Use nested asyncio event loop for Jupyter notebooks.

    This is not needed anymore in Jupyter versions 5 or higher.
    """
    def _ipython_loop_asyncio(kernel):
        """Use asyncio event loop for the given IPython kernel."""
        loop = asyncio.get_event_loop()

        def kernel_handler():
            kernel.do_one_iteration()
            loop.call_later(kernel._poll_interval, kernel_handler)

        loop.call_soon(kernel_handler)
        try:
            if not loop.is_running():
                loop.run_forever()
        finally:
            if not loop.is_running():
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

    patchAsyncio()
    loop = asyncio.get_event_loop()
    if not loop.is_running():
        from ipykernel.eventloops import register_integration, enable_gui
        register_integration('asyncio')(_ipython_loop_asyncio)
        enable_gui('asyncio')


def logToConsole(level=logging.INFO, color=True):
    """Create a colorlog handler that logs to the console."""
    logger = colorlog.getLogger()
    logger.setLevel(level)

    if color:
        formatter = colorlog.ColoredFormatter(
            '%(bold)s%(log_color)s%(asctime)s [%(levelname)s] %(name)s %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },)
        handler = colorlog.StreamHandler()
    else:
        formatter = logging.Formatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s')
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)

    logger.handlers = [
        h for h in logger.handlers
        if type(h) is not (colorlog.StreamHandler if color else logging.StreamHandler)]
    logger.addHandler(handler)

# ---------------------------------------------
# data preparation methods
# ---------------------------------------------
_BARS_COLSMAP = {
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
    'opt_price': 'opt_price',
    'opt_underlying': 'opt_underlying',
    'opt_dividend': 'opt_dividend',
    'opt_volume': 'opt_volume',
    'opt_iv': 'opt_iv',
    'opt_oi': 'opt_oi',
    'opt_delta': 'opt_delta',
    'opt_gamma': 'opt_gamma',
    'opt_vega': 'opt_vega',
    'opt_theta': 'opt_theta'
}

def prepare_data(instrument, data, output_path=None,
                 index=None, colsmap=None, kind="BAR", resample=None) -> pd.DataFrame:
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

    global _BARS_COLSMAP

    # work on copy
    df = data.copy()

    # lower case columns
    df.columns = map(str.lower, df.columns)

    # yahoo's csv?
    if 'adj close' in set(df.columns):
        ratio = df["close"] / df["adj close"]
        df["close"] = df["adj close"]
        df["open"] = df["open"] / ratio
        df["high"] = df["high"] / ratio
        df["low"] = df["low"] / ratio

    # set index
    if index is None:
        index = df.index

    # set defaults columns
    if not isinstance(colsmap, dict):
        colsmap = {}

    _colsmap = _BARS_COLSMAP
    for el in _colsmap:
        if el not in colsmap:
            colsmap[el] = _colsmap[el]

    # generate a valid ib tuple
    instrument = create_ib_tuple(instrument)

    # # create contract string (no need for connection)
    contract_string = contractString(instrument)
    asset_class = gen_asset_class(contract_string)
    symbol_group = gen_symbol_group(contract_string)

    # add symbol data
    df.loc[:, 'symbol'] = contract_string
    df.loc[:, 'symbol_group'] = symbol_group
    df.loc[:, 'asset_class'] = asset_class

    # validate columns
    valid_cols = validate_columns(df, kind)
    if not valid_cols:
        raise ValueError('Invalid Column list')

    # rename columns to map
    df.rename(columns=colsmap, inplace=True)

    # remove all other columns
    known_cols = list(colsmap.values()) + \
        ['symbol', 'symbol_group', 'asset_class', 'expiry']
    for col in df.columns:
        if col not in known_cols:
            df.drop(col, axis=1, inplace=True)

    # set UTC index
    df.index = pd.to_datetime(index)
    df = set_timezone(df, "UTC")
    df.index.rename("datetime", inplace=True)

    # resample
    if resample and kind == "BAR":
        df = resample_data(df, resolution=resample, tz="UTC")

    # add expiry
    df.loc[:, 'expiry'] = np.nan
    if asset_class in ("FUT", "OPT", "FOP"):
        df.loc[:, 'expiry'] = contract_expiry_from_symbol(contract_string)

    # save csv
    if output_path is not None:
        output_path = output_path[
            :-1] if output_path.endswith('/') else output_path
        df.to_csv("%s/%s.%s.csv" % (output_path, contract_string, kind))

    return df


dataTypes = {
    "MONTH_CODES": ['', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'],
}

def contract_expiry_from_symbol(symbol):
    expiry = None
    symbol, asset_class = symbol.split("_")

    if asset_class == "FUT":
        expiry = str(symbol)[-5:]
        y = int(expiry[-4:])
        m = dataTypes["MONTH_CODES"].index(expiry[:1])
        day = datetime(y, m, 1)
        expiry = day + relativedelta.relativedelta(weeks=2, weekday=relativedelta.FR)
        expiry = expiry.strftime("%Y-%m-%d")

    elif asset_class in ("OPT", "FOP"):
        expiry = str(symbol)[-17:-9]
        expiry = expiry[:4] + "-" + expiry[4:6] + "-" + expiry[6:]

    return expiry


def create_ib_tuple(instrument):
    """ create ib contract tuple """

    # tuples without strike/right
    if len(instrument) <= 7:
        instrument_list = list(instrument)
        if len(instrument_list) < 3:
            instrument_list.append("SMART")
        if len(instrument_list) < 4:
            instrument_list.append("USD")
        if len(instrument_list) < 5:
            instrument_list.append("")
        if len(instrument_list) < 6:
            instrument_list.append(0.0)
        if len(instrument_list) < 7:
            instrument_list.append("")

        try:
            instrument_list[4] = int(instrument_list[4])
        except Exception as e:
            pass

        instrument_list[5] = 0. if isinstance(instrument_list[5], str) \
            else float(instrument_list[5])

        instrument = tuple(instrument_list)

    return instrument

# -----------------------------------------
def contractString(contract, seperator="_"):
    """ returns string from contract tuple """

    localSymbol = ""
    contractTuple = contract

    # build identifier
    try:
        if contractTuple[1] in ("OPT", "FOP"):
            strike = '{:0>5d}'.format(int(contractTuple[5])) + \
                format(contractTuple[5], '.3f').split('.')[1]

            contractString = (contractTuple[0] + str(contractTuple[4]) +
                contractTuple[6][0] + strike, contractTuple[1])

        elif contractTuple[1] == "CASH":
            contractString = (contractTuple[0] + contractTuple[3], contractTuple[1])

        else:  # STK
            contractString = (contractTuple[0], contractTuple[1])

        # construct string
        contractString = seperator.join(
            str(v) for v in contractString).replace(seperator + "STK", "")

    except Exception:
        contractString = contractTuple[0]

    return contractString.replace(" ", "_").upper()


def gen_asset_class(sym):
    sym_class = str(sym).split("_")
    if len(sym_class) > 1:
        return sym_class[-1].replace("CASH", "CSH")
    return "STK"


def gen_symbol_group(sym):
    sym = sym.strip()

    if "_FUT" in sym:
        sym = sym.split("_FUT")
        return sym[0][:-5] + "_F"

    elif "_CASH" in sym:
        return "CASH"

    if "_FOP" in sym or "_OPT" in sym:
        return sym[:-12]

    return sym


def validate_columns(df, kind="BAR", raise_errors=True):
    global _BARS_COLSMAP
    # validate columns
    if "asset_class" not in df.columns:
        if raise_errors:
            raise ValueError('Column asset_class not found')
        return False

    is_option = "OPT" in list(df['asset_class'].unique())

    colsmap = _BARS_COLSMAP

    for el in colsmap:
        col = colsmap[el]
        if col not in df.columns:
            if "opt_" in col and is_option:
                if raise_errors:
                    raise ValueError('Column %s not found' % el)
                return False
            elif "opt_" not in col and not is_option:
                if raise_errors:
                    raise ValueError('Column %s not found' % el)
                return False
    return True

def resample_data(data, resolution="1T", tz=None, ffill=True, dropna=False,
             sync_last_timestamp=True):

    def __finalize(data, tz=None):
        # figure out timezone
        try:
            tz = data.index.tz if tz is None else tz
        except Exception as e:
            pass

        if str(tz) != 'None':
            try:
                data.index = data.index.tz_convert(tz)
            except Exception as e:
                data.index = data.index.tz_localize('UTC').tz_convert(tz)

        # sort by index (datetime)
        data.sort_index(inplace=True)

        # drop duplicate rows per instrument
        data.loc[:, '_idx_'] = data.index
        data.drop_duplicates(
            subset=['_idx_', 'symbol', 'symbol_group', 'asset_class'],
            keep='last', inplace=True)
        data.drop('_idx_', axis=1, inplace=True)

        return data
        # return data[~data.index.duplicated(keep='last')]

    def __resample_ticks(data, freq=1000, by='last'):
        """
        function that re-samples tick data into an N-tick or N-volume OHLC format
        df = pandas pd.dataframe of raw tick data
        freq = resoltuin grouping
        by = the column name to resample by
        """

        data.fillna(value=np.nan, inplace=True)

        # get only ticks and fill missing data
        try:
            df = data[['last', 'lastsize', 'opt_underlying', 'opt_price',
                       'opt_dividend', 'opt_volume', 'opt_iv', 'opt_oi',
                       'opt_delta', 'opt_gamma', 'opt_theta', 'opt_vega']].copy()
            price_col = 'last'
            size_col = 'lastsize'
        except Exception as e:
            df = data[['close', 'volume', 'opt_underlying', 'opt_price',
                       'opt_dividend', 'opt_volume', 'opt_iv', 'opt_oi',
                       'opt_delta', 'opt_gamma', 'opt_theta', 'opt_vega']].copy()
            price_col = 'close'
            size_col = 'volume'

        # add group indicator evey N df
        if by == 'size' or by == 'lastsize' or by == 'volume':
            df['cumvol'] = df[size_col].cumsum()
            df['mark'] = round(
                round(round(df['cumvol'] / .1) * .1, 2) / freq) * freq
            df['diff'] = df['mark'].diff().fillna(0).astype(int)
            df['grp'] = np.where(df['diff'] >= freq - 1,
                                 (df['mark'] / freq), np.nan)
        else:
            df['grp'] = [np.nan if i %
                         freq else i for i in range(len(df[price_col]))]

        df.loc[:1, 'grp'] = 0

        df.fillna(method='ffill', inplace=True)

        # print(df[['lastsize', 'cumvol', 'mark', 'diff', 'grp']].tail(1))

        # place timestamp index in T colums
        # (to be used as future df index)
        df['T'] = df.index

        # make group the index
        df = df.set_index('grp')

        # grop df
        groupped = df.groupby(df.index, sort=False)

        # build ohlc(v) pd.dataframe from new grp column
        newdf = pd.DataFrame({
            'open': groupped[price_col].first(),
            'high': groupped[price_col].max(),
            'low': groupped[price_col].min(),
            'close': groupped[price_col].last(),
            'volume': groupped[size_col].sum(),

            'opt_price': groupped['opt_price'].last(),
            'opt_underlying': groupped['opt_underlying'].last(),
            'opt_dividend': groupped['opt_dividend'].last(),
            'opt_volume': groupped['opt_volume'].last(),
            'opt_iv': groupped['opt_iv'].last(),
            'opt_oi': groupped['opt_oi'].last(),
            'opt_delta': groupped['opt_delta'].last(),
            'opt_gamma': groupped['opt_gamma'].last(),
            'opt_theta': groupped['opt_theta'].last(),
            'opt_vega': groupped['opt_vega'].last()
        })

        # set index to timestamp
        newdf['datetime'] = groupped.T.head(1)
        newdf.set_index(['datetime'], inplace=True)

        return newdf

    if data.empty:
        return __finalize(data, tz)

    # ---------------------------------------------
    # force same last timestamp to all symbols before resampling
    if sync_last_timestamp:
        data.loc[:, '_idx_'] = data.index
        start_date = str(data.groupby(["symbol"])[
                         ['_idx_']].min().max().values[-1]).replace('T', ' ')
        end_date = str(data.groupby(["symbol"])[
                       ['_idx_']].max().min().values[-1]).replace('T', ' ')
        data = data[(data.index >= start_date) & (data.index <= end_date)
                    ].drop_duplicates(subset=['_idx_', 'symbol',
                                              'symbol_group', 'asset_class'],
                                      keep='first')

    # ---------------------------------------------
    # resample
    periods = int("".join([s for s in resolution if s.isdigit()]))
    meta_data = data.groupby(["symbol"])[
        ['symbol', 'symbol_group', 'asset_class']].last()
    combined = []

    if "K" in resolution:
        if periods > 1:
            for sym in meta_data.index.values:
                symdata = __resample_ticks(data[data['symbol'] == sym].copy(),
                                           freq=periods, by='last')
                symdata['symbol'] = sym
                symdata['symbol_group'] = meta_data[
                    meta_data.index == sym]['symbol_group'].values[0]
                symdata['asset_class'] = meta_data[
                    meta_data.index == sym]['asset_class'].values[0]

                # cleanup
                symdata.dropna(inplace=True, subset=[
                    'open', 'high', 'low', 'close', 'volume'])
                if sym[-3:] in ("OPT", "FOP"):
                    symdata.dropna(inplace=True)

                combined.append(symdata)

            data = pd.concat(combined, sort=True)

    elif "V" in resolution:
        if periods > 1:
            for sym in meta_data.index.values:
                symdata = __resample_ticks(data[data['symbol'] == sym].copy(),
                                           freq=periods, by='lastsize')
                symdata['symbol'] = sym
                symdata['symbol_group'] = meta_data[
                    meta_data.index == sym]['symbol_group'].values[0]
                symdata['asset_class'] = meta_data[
                    meta_data.index == sym]['asset_class'].values[0]

                # cleanup
                symdata.dropna(inplace=True, subset=[
                    'open', 'high', 'low', 'close', 'volume'])
                if sym[-3:] in ("OPT", "FOP"):
                    symdata.dropna(inplace=True)

                combined.append(symdata)

            data = pd.concat(combined, sort=True)

    # continue...
    else:
        ticks_ohlc_dict = {
            'lastsize': 'sum',
            'opt_price': 'last',
            'opt_underlying': 'last',
            'opt_dividend': 'last',
            'opt_volume': 'last',
            'opt_iv': 'last',
            'opt_oi': 'last',
            'opt_delta': 'last',
            'opt_gamma': 'last',
            'opt_theta': 'last',
            'opt_vega': 'last'
        }
        bars_ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'opt_price': 'last',
            'opt_underlying': 'last',
            'opt_dividend': 'last',
            'opt_volume': 'last',
            'opt_iv': 'last',
            'opt_oi': 'last',
            'opt_delta': 'last',
            'opt_gamma': 'last',
            'opt_theta': 'last',
            'opt_vega': 'last'
        }

        for sym in meta_data.index.values:

            if "last" in data.columns:
                tick_dict = {}
                for col in data[data['symbol'] == sym].columns:
                    if col in ticks_ohlc_dict.keys():
                        tick_dict[col] = ticks_ohlc_dict[col]

                ohlc = data[data['symbol'] == sym]['last'].resample(
                    resolution).ohlc()
                symdata = data[data['symbol'] == sym].resample(
                    resolution).apply(tick_dict).fillna(value=np.nan)
                symdata.rename(
                    columns={'lastsize': 'volume'}, inplace=True)

                symdata['open'] = ohlc['open']
                symdata['high'] = ohlc['high']
                symdata['low'] = ohlc['low']
                symdata['close'] = ohlc['close']

            else:
                bar_dict = {}
                for col in data[data['symbol'] == sym].columns:
                    if col in bars_ohlc_dict.keys():
                        bar_dict[col] = bars_ohlc_dict[col]

                original_length = len(data[data['symbol'] == sym])
                symdata = data[data['symbol'] == sym].resample(
                    resolution).apply(bar_dict).fillna(value=np.nan)

                # deal with new rows caused by resample
                if len(symdata) > original_length:
                    # volume is 0 on rows created using resample
                    symdata['volume'].fillna(0, inplace=True)
                    symdata.ffill(inplace=True)

                    # no fill / return original index
                    if ffill:
                        symdata['open'] = np.where(symdata['volume'] <= 0,
                            symdata['close'], symdata['open'])
                        symdata['high'] = np.where(symdata['volume'] <= 0,
                            symdata['close'], symdata['high'])
                        symdata['low'] = np.where(symdata['volume'] <= 0,
                            symdata['close'], symdata['low'])
                    else:
                        symdata['open'] = np.where(symdata['volume'] <= 0,
                            np.nan, symdata['open'])
                        symdata['high'] = np.where(symdata['volume'] <= 0,
                            np.nan, symdata['high'])
                        symdata['low'] = np.where(symdata['volume'] <= 0,
                            np.nan, symdata['low'])
                        symdata['close'] = np.where(symdata['volume'] <= 0,
                            np.nan, symdata['close'])

                # drop NANs
                if dropna:
                    symdata.dropna(inplace=True)

            symdata['symbol'] = sym
            symdata['symbol_group'] = meta_data[meta_data.index == sym]['symbol_group'].values[0]
            symdata['asset_class'] = meta_data[meta_data.index == sym]['asset_class'].values[0]

            # cleanup
            symdata.dropna(inplace=True, subset=[
                'open', 'high', 'low', 'close', 'volume'])
            if sym[-3:] in ("OPT", "FOP"):
                symdata.dropna(inplace=True)

            combined.append(symdata)

        data = pd.concat(combined, sort=True)
        data['volume'] = data['volume'].astype(int)

    return __finalize(data, tz)
