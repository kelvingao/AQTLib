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

import logging

__all__ = ['Instrument']


class Instrument(str):
    """
    A string subclass that provides easy access to misc
    symbol-related methods and information.

    """
    strategy = None
    _logger = logging.getLogger(__name__)

    # ---------------------------------------
    def attach_strategy(self, strategy):
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

        return bars.copy()

    # ---------------------------------------
    def get_positions(self, attr=None):
        """Get the positions data for the instrument
        :Optional:
            attr : string
                Position attribute to get
                (optional attributes: symbol, position, avgCost, account)
        :Retruns:
            positions : dict (positions) / float/str (attribute)
                positions data for the instrument
        """
        pos = self.strategy.get_positions(self)

        try:
            if attr is not None:
                attr = attr.replace("quantity", "position")
            return pos[attr]
        except Exception as e:
            return pos

    # ---------------------------------------
    def buy(self, quantity, **kwargs):
        """ Shortcut for ``instrument.order("BUY", ...)`` and accepts all of its
        `optional parameters <#qtpylib.instrument.Instrument.order>`_
        :Parameters:
            quantity : int
                Order quantity
        """
        self.strategy.order("BUY", self, quantity=quantity, **kwargs)

    # ---------------------------------------
    def exit(self):
        """ Shortcut for ``instrument.order("EXIT", ...)``
        (accepts no parameters)"""
        self.strategy.order("EXIT", self)

    # ---------------------------------------
    def order(self, direction, quantity, **kwargs):
        """ Send an order for this instrument
        :Parameters:
            direction : string
                Order Type (BUY/SELL, EXIT/FLATTEN)
            quantity : int
                Order quantity
        :Optional:
            limit_price : float
                In case of a LIMIT order, this is the LIMIT PRICE
            expiry : int
                Cancel this order if not filled after *n* seconds (default 60 seconds)
            order_type : string
                Type of order: Market (default), LIMIT (default when limit_price is passed),
                MODIFY (required passing or orderId)
            orderId : int
                If modifying an order, the order id of the modified order
            target : float
                target (exit) price
            initial_stop : float
                price to set hard stop
            stop_limit: bool
                Flag to indicate if the stop should be STOP or STOP LIMIT (default False=STOP)
            trail_stop_at : float
                price at which to start trailing the stop
            trail_stop_by : float
                % of trailing stop distance from current price
            fillorkill: bool
                fill entire quantiry or none at all
            iceberg: bool
                is this an iceberg (hidden) order
            tif: str
                time in force (DAY, GTC, IOC, GTD). default is ``DAY``
        """
        self.strategy.order(direction.upper(), self, quantity, **kwargs)
