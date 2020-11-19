"""Sqlalchemy modals definition."""

from sqlalchemy import MetaData, Table, Column, DateTime, DECIMAL, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import SMALLINT, VARCHAR, FLOAT, INTEGER, DATE

metadata = MetaData()

symbols = Table(
    'symbols', metadata,
    Column('id', SMALLINT, primary_key=True),
    Column('symbol', VARCHAR(24), index=True),
    Column('symbol_group', VARCHAR(18), index=True),
    Column('asset_class', VARCHAR(3), index=True),
    Column('expiry', DATE, index=True)
)

ticks = Table(
    'ticks', metadata,
    Column('id', INTEGER, primary_key=True),
    Column('symbol_id', ForeignKey('symbols.id'), nullable=False, index=True),
    Column('bid', FLOAT(asdecimal=True)),
    Column('bidsize', INTEGER),
    Column('ask', FLOAT(asdecimal=True)),
    Column('asksize', INTEGER),
    Column('last', FLOAT(asdecimal=True)),
    Column('lastsize', INTEGER),
    Column('datetime', DateTime(timezone=True), nullable=False, index=True),
    UniqueConstraint('datetime', 'symbol_id')
)

trades = Table(
    'trades', metadata,
    Column('id', INTEGER, primary_key=True),
    Column('algo', VARCHAR(32), index=True),
    Column('symbol', VARCHAR(12), index=True),
    Column('direction', VARCHAR(5)),
    Column('quantity', INTEGER),
    Column('entry_time', DateTime(timezone=True), index=True),
    Column('exit_time', DateTime(timezone=True), index=True),
    Column('exit_reason', VARCHAR(8), index=True),
    Column('order_type', VARCHAR(6), index=True),
    Column('market_price', FLOAT(asdecimal=True), index=True),
    Column('target', FLOAT(asdecimal=True)),
    Column('stop', FLOAT(asdecimal=True)),
    Column('entry_price', FLOAT(asdecimal=True), index=True),
    Column('exit_price', FLOAT(asdecimal=True), index=True),
    Column('realized_pnl', FLOAT(asdecimal=True)),
    UniqueConstraint('algo', 'symbol', 'entry_time')
)

bars = Table(
    'bars', metadata,
    Column('id', INTEGER, primary_key=True),
    Column('datetime', DateTime(timezone=True), nullable=False, index=True),
    Column('symbol_id', ForeignKey('symbols.id'), nullable=False, index=True),
    Column('open', FLOAT(asdecimal=True)),
    Column('high', FLOAT(asdecimal=True)),
    Column('low', FLOAT(asdecimal=True)),
    Column('close', FLOAT(asdecimal=True)),
    Column('volume', INTEGER),
    UniqueConstraint('datetime', 'symbol_id')
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
