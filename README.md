# blurhash-py

[![PyPI - Version](https://img.shields.io/pypi/v/blurhash-py)](https://pypi.org/project/blurhash-py/)

A Python and rust library for encoding and decoding blurhash strings where most of the code is
ported from the original [blurhash](https://github.com/woltapp/blurhash) library.

Some of the optimizations are inspired by [this blog post](https://uploadcare.com/blog/faster-blurhash/).

Comparing with the original (C-based) implementation:

```python
In [1]: import blurhash as b1

In [2]: import blurhash_py as b2

In [3]: fname = './mio.png'

In [4]: %timeit b1.encode(fname, 4, 4)
388 ms ± 414 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [5]: %timeit b2.encode(fname, 4, 4)
75.7 ms ± 349 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Installation

```bash
pip install blurhash-py
# or to test out interactively
uvx --with blurhash-py ipython
```
