# Updating the code from Python 2 to Python 3

## `print`

The first and most obvious difference is that in Python 3 `print` takes parentheses.
This means that every

```
print "stuff", 1
```

had to be replaced with

```
print("stuff", 1)
```

This is mostly tedious.

## tuple unpacking

<a href="https://www.python.org/dev/peps/pep-3113/">PEP-3113</a> eliminates
tuple unpacking in certain places. In particular, that means that code like

```
key=lambda (a, b): b
```

has to be replaced with

```
key=lambda pair: pair[1]
```

## `.iteritems()`

In Python 3, laziness is the order of the day. In particular, `dict`-like
objects no longer have `.iteritems()` properties, so those all have to be replaced
with `.items()`

## binary mode for CSVs

In Python 2 it was best practice to open CSV files in binary mode to
make sure you dealt properly with Windows line endings:

```
f = open("some.csv", "rb")
```

In Python 3 that doesn't work for various reasons having to do with raw bytes
and string encodings. Instead you need to open them in text mode and
specify the line ending types:

```
f = open("some.csv", 'r', encoding='utf8', newline='')
```

## `reduce`

Guido doesn't like `reduce`, so in Python 3 it's hidden in `functools`. So any code
that uses it needs to add a

```
from functools import reduce
```
