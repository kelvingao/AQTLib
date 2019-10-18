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

__all__ = ['Instrument']


class Instrument(str):
    """
    A string subclass that provides easy access to misc
    symbol-related methods and information.

    """
    strategy = None

    # ---------------------------------------
    def _bind_strategy(self, strategy):
        """
        Sets the strategy object to communicate with.

        """
        self.strategy = strategy

    # ---------------------------------------
    def get_bars(self, lookback=None, as_dict=False):
        """
        Get bars for this instrument and return as a dataframe or dict.

        Args:
            lookback : int
                Max number of bars to get (None = all available bars)
            as_dict : bool
                Return a dict or a pd.DataFrame object

        """
        bars = self.strategy.bars

        lookback = self.strategy.bars_window if lookback is None else lookback
        bars = bars[-lookback:]

        if as_dict:
            bars.reset_index(inplace=True)
            bars = bars.to_dict(orient='records')
            if lookback == 1:
                bars = None if not bars else bars[0]

        return bars
