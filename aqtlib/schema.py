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

from sqlalchemy import MetaData, Table, Column, DateTime, DECIMAL, ForeignKey
from sqlalchemy.dialects.postgresql import SMALLINT, VARCHAR, FLOAT, INTEGER, DATE

metadata = MetaData()

symbols = Table(
    'symbols', metadata,
    Column('id', SMALLINT, primary_key=True),
    Column('symbol', VARCHAR(24)),
    Column('symbol_group', VARCHAR(18), index=True),
    Column('asset_class', VARCHAR(3), index=True),
    Column('expiry', DATE, index=True)
)

ticks = Table(
    'ticks', metadata,
    Column('id', INTEGER, primary_key=True),
    Column('symbol_id', ForeignKey('symbols.id'), nullable=True, index=True),
    Column('bid', FLOAT(asdecimal=True)),
    Column('bidSize', INTEGER),
    Column('ask', FLOAT(asdecimal=True)),
    Column('askSize', INTEGER),
    Column('time', DateTime(timezone=True), nullable=True, index=True, unique=True)
)

trades = Table(
    'trades', metadata,
    Column('id', INTEGER, primary_key=True),
    Column('algo', VARCHAR(32), index=True),
    Column('symbol', VARCHAR(12), index=True),
    Column('direction', VARCHAR(5)),
    Column('quantity', INTEGER),
    Column('entry_time', DateTime(timezone=True), index=True, unique=True),
    Column('exit_time', DateTime(timezone=True), index=True, unique=True),
    Column('exit_reason', VARCHAR(8), index=True),
    Column('order_type', VARCHAR(6), index=True),
    Column('market_price', FLOAT(asdecimal=True), index=True),
    Column('target', FLOAT(asdecimal=True)),
    Column('stop', FLOAT(asdecimal=True)),
    Column('entry_price', FLOAT(asdecimal=True), index=True),
    Column('exit_price', FLOAT(asdecimal=True), index=True),
    Column('realized_pnl', FLOAT(asdecimal=True))
)

bars = Table(
    'bars', metadata,
    Column('id', INTEGER, primary_key=True),
    Column('datetime', DateTime(timezone=True), nullable=False, index=True, unique=True),
    Column('symbol_id', ForeignKey('symbols.id'), nullable=False, index=True),
    Column('open', FLOAT(asdecimal=True)),
    Column('high', FLOAT(asdecimal=True)),
    Column('low', FLOAT(asdecimal=True)),
    Column('close', FLOAT(asdecimal=True)),
    Column('volume', INTEGER)
)


greeks = Table(
    'greeks', metadata,
    Column('id', INTEGER, primary_key=True),
    Column('tick_id', ForeignKey('ticks.id'), index=True),
    Column('bar_id', ForeignKey('bars.id'), index=True),
    Column('price', FLOAT(asdecimal=True)),
    Column('underlying', FLOAT(asdecimal=True)),
    Column('dividend', FLOAT(asdecimal=True)),
    Column('volume', INTEGER),
    Column('iv', FLOAT(asdecimal=True)),
    Column('oi', FLOAT(asdecimal=True)),
    Column('delta', DECIMAL(3, 2)),
    Column('gamma', DECIMAL(3, 2)),
    Column('theta', DECIMAL(3, 2)),
    Column('vega', DECIMAL(3, 2))
)
