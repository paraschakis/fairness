# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 00:17:41 2018

@author: ae0670

Taken from StackOverflow: https://stackoverflow.com/questions/48602709/fastest-way-to-find-unique-combinations-of-list
"""


def combinations(pool, r):
    """ Return an iterator over all distinct r-length
    combinations taken from a pool of values that
    may contain duplicates.

    Unlike itertools.combinations(), element uniqueness
    is determined by value rather than by position.

    """
    if r:
        seen = set()
        for i, item in enumerate(pool):
            if item not in seen:
                seen.add(item)
                for tail in combinations(pool[i+1:], r-1):
                    yield (item,) + tail
    else:
        yield ()

if __name__ == '__main__':
    from itertools import combinations

    pool = 'ABRACADABRA'
    for r in range(len(pool) + 1):
        assert set(uniq_comb(pool, r)) == set(combinations(pool, r))
        assert dict.fromkeys(uniq_comb(pool, r)) == dict.fromkeys(combinations(pool, r))