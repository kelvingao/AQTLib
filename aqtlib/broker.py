
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

import pandas as pd
import logging

from aqtlib.objects import Object
from aqtlib import util

__all__ = ['Broker']


class Broker(Object):
    def __init__(self, instruments, *args, **kwargs):
        super(Broker, self).__init__(instruments, *args, **kwargs)

        self._logger = logging.getLogger(__name__)
        # -----------------------------------
        # create contracts
        instrument_tuples_dict = {}
        for instrument in instruments:
            try:
                instrument = util.create_ib_tuple(instrument)
                contractString = util.contractString(instrument)
                instrument_tuples_dict[contractString] = instrument
                util.createContract(instrument)
            except Exception as e:
                pass

        self.instruments = instrument_tuples_dict
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
