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

import asyncpg
import logging

from aqtlib import util
from sqlalchemy import MetaData, DateTime, create_engine
from sqlalchemy.sql.sqltypes import String
from sqlalchemy.dialects.postgresql import dialect as PostgresqlDialect

util.createLogger(__name__, logging.INFO)

__all__ = ['PG']


class NotInitializedError(Exception):
    pass


class StringLiteral(String):
    """
    Teach SA how to literalize various things.

    """
    def literal_processor(self, dialect):
        super_processor = super(StringLiteral, self).literal_processor(dialect)

        def process(value):
            if isinstance(value, int):
                return str(value)
            if not isinstance(value, str):
                value = str(value)
            result = super_processor(value)
            if isinstance(result, bytes):
                result = result.decode(dialect.encoding)
            return result
        return process


class LiteralDialect(PostgresqlDialect):
    """
    Default Dialect does not support in-place multirow inserts.

    """
    colspecs = {
        # prevent various encoding explosions
        String: StringLiteral,
        # teach SA about how to literalize a datetime
        DateTime: StringLiteral,
    }


class PG:
    """
    A wrapper around asyncpg for use with sqlalchemy core.

    """
    RequestTimeout = 0

    _dialect = LiteralDialect(
        paramstyle='pyformat', implicit_returning=True)

    __slot__ = ('_dsn', '_metadata', '_engine', '_pool', '_logger')

    def __init__(self, dsn: str, metadata: MetaData = None):
        self._logger = logging.getLogger(__name__)

        self._dsn = dsn
        self._metadata = metadata

        self._engine = create_engine(self._dsn)
        self._create_tables()

        self._pool = None

    def __repr__(self) -> str:
        return self._dsn

    __str__ = __repr__

    @property
    def pool(self):
        if not self._pool:
            raise NotInitializedError('pg.init_pool() needs to be called '
                                      'before you can make queries')
        else:
            return self._pool

    # @property
    # def engine(self):
    #     if self._engine is None:
    #         self._engine = create_engine(self._dsn)
    #     return self._engine

    # ---------------------------------------
    async def init_pool(self):
        # connect to the pool
        self._pool = await asyncpg.create_pool(self._dsn)
        self._logger.debug("PostgreSQL connection pool connected.")

        return self

    async def close_pool(self):
        """
        Gracefully close all connections in the pool,
        it is advisable to use asyncio.wait_for() to set a timeout.
        """

        if not self._pool:
            self._logger.warning("Connection pool already closed!")
            return

        await self._pool.close()
        self._logger.debug("Connection pool closed...")

    def _create_tables(self):
        """
        Create tables, and bind metadata to the engine.

        """
        self._metadata.create_all(self._engine)
        self._logger.debug('The schema tables have been created.')

    def _drop_tables(self):
        self._metadata.drop_all(self._engine)
        self._logger.debug('The schema tables have been dropped.')

    # ---------------------------------------
    def _literalquery(self, query, literal_binds=True):
        """
        NOTE: This is entirely insecure. DO NOT execute the resulting strings.

        """
        import sqlalchemy.orm
        if isinstance(query, sqlalchemy.orm.Query):
            query = query.statement

        literal = query.compile(
            dialect=self._dialect,
            compile_kwargs={'literal_binds': literal_binds},
        ).string

        self._logger.debug(literal)
        return literal

    # ---------------------------------------
    async def fetchval(self, query, *args, column=0, timeout=None):
        if not isinstance(query, str):
            query = self._literalquery(query)
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def fetch(self, query, *args, timeout=None) -> list:
        if not isinstance(query, str):
            query = self._literalquery(query)
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query, *args, timeout=None):
        """
        Good for insert/update/delete calls.

        """
        if not isinstance(query, str):
            query = self._literalquery(query)
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def execute(self, query, *args, timeout=None):
        if not isinstance(query, str):
            query = self._literalquery(query)
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)
