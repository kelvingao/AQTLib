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


def createLogger(name, level=logging.DEBUG):
    """
    DEBUG:    Detialed information, typically of interest of only when diagnosing problems.
    INFO:     Confirmation that things are working as expected.
    WARNING:  An indicaiton that something unexpected happened, or indicative of some problem in the near future
            (e.g. 'disk space low'). The software is still working as expected.
    ERROR:    Due to a more serious problem, the software has not been able to perform some function.
    CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

    Return a logger with the given `name` and optional `level`
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))

    logger.addHandler(handler)
    logger.propagate = False

    return logger
