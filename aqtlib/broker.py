"""Broker class definition."""

import pandas as pd
import logging

from aqtlib.objects import Object
from aqtlib import Porter, utils

__all__ = ['Broker']


class Broker(Object):
    defaults = dict(
        instruments=None,
        ibclient=998,
        ibport=4002,
        ibserver='localhost'
    )

    def __init__(self, instruments, *args, **kwargs):
        super(Broker, self).__init__(instruments, *args, **kwargs)
        self._logger = logging.getLogger(__name__)

        self.porter = Porter()

    @property
    def instruments(self):
        # print("Getting instruments...")
        return self._instruments

    @instruments.setter
    def instruments(self, instruments):
        # print("Setting instruments: {}".format(instruments))
        instrument_tuples_dict = {}
        for instrument in instruments:
            try:
                instrument = utils.create_ib_tuple(instrument)
                contractString = utils.contractString(instrument)
                instrument_tuples_dict[contractString] = instrument
                utils.createContract(instrument)
            except Exception as e:
                pass

        self._instruments = instrument_tuples_dict
        self.symbols = list(self.instruments.keys())

    # ---------------------------------------
    @staticmethod
    def get_symbol(symbol):
        if not isinstance(symbol, str):
            if isinstance(symbol, dict):
                symbol = symbol['symbol']
            elif isinstance(symbol, pd.DataFrame):
                symbol = symbol[:1]['symbol'].values[0]

        return symbol

    # ---------------------------------------
    def get_positions(self, symbol):
        symbol = self.get_symbol(symbol)

        if self.backtest:
            position = 0
            avgCost = 0.0

            if not self.datastore.recorded.empty:
                data = self.datastore.recorded
                col = symbol.upper() + '_POSITION'
                position = data[col].values[-1]
                if position != 0:
                    pos = data[col].diff()
                    avgCost = data[data.index.isin(pos[pos != 0][-1:].index)
                                   ][symbol.upper() + '_OPEN'].values[-1]

            self._logger.debug('GET {} CURRENT POSITION: {}'.format(symbol, position))

            return {
                "symbol": symbol,
                "position": position,
                "avgCost": avgCost,
                "account": "Backtest"
            }

        return {
            "symbol": symbol,
            "position": 0,
            "avgCost": 0.0,
            "account": None
        }
