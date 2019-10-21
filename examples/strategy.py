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

from aqtlib.algo import Algo


class TestStrategy(Algo):
    """
    Example: This Strategy buys/sells single contract of the
    SPDR S&P 500 ETF TRUST when moving average price cross above/below close price.
    """

    positions = 0

    # ---------------------------------------
    def on_start(self):
        """ initilize positions """
        self.positions = 0

    # ---------------------------------------
    def on_bar(self, instrument):
        # get bars history
        bars = instrument.get_bars(lookback=20)

        # make sure we have at least 20 bars to work with
        if len(bars) < 20:
            return

        # compute averages using internal rolling_mean
        bars['long_ma'] = bars['close'].rolling(window=20).mean()

        # trading logic - entry signal
        if bars['close'][-1] > bars['long_ma'][-1]:
            if self.positions == 0:
                # send a buy signal
                self.logger.info("Entry signal captured")
                self.positions = 1

        # trading logic - exit signal
        elif bars['close'][-1] < bars['long_ma'][-1]:
            if self.positions != 0.0:
                self.logger.info("Exit signal captured")
                self.positions = 0


# ===========================================
if __name__ == "__main__":

    strategy = TestStrategy(
        instruments=[("SPY", "STK", "ARCA", "USD")],
        resolution="1D",
        bars_window=120
    )
    strategy.run()
