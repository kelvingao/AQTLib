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

__all__ = ['Object']


class Object:
    """
    Base object, with:
    * __slots__ to avoid typos;
    * A general constructor;
    * A general string representation;
    """
    __slots__ = ('__weakref__',)
    defaults: dict = {}

    def __init__(self, *args, **kwargs):
        """
        Attribute values can be given positionally or as keyword.
        If an attribute is not given it will take its value from the
        'defaults' class member. If an attribute is given both positionally
        and as keyword, the keyword wins.
        """
        defaults = self.__class__.defaults
        d = {**defaults, **dict(zip(defaults, args)), **kwargs}
        for k, v in d.items():
            setattr(self, k, v)

    def __repr__(self):
        clsName = self.__class__.__qualname__
        kwargs = ', '.join(f'{k}={v!r}' for k, v in self.nonDefaults().items())
        return f'{clsName}({kwargs})'

    __str__ = __repr__

    def dict(self):
        """
        Return key-value pairs as a dictionary.
        """
        return {k: getattr(self, k) for k in self.__class__.defaults}

    def update(self, **kwargs):
        """
        Update key values.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def nonDefaults(self):
        """
        Get a dictionary of all attributes that differ from the default.
        """
        nonDefaults = {}
        for k, d in self.__class__.defaults.items():
            v = getattr(self, k)
            if v != d and (v == v or d == d):  # tests for NaN too
                nonDefaults[k] = v
        return nonDefaults
