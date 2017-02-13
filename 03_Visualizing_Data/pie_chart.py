#!/usr/bin/env python3

"""
Pie chart

This example exists in the original `visualizing_data.py` file
in the book repository, but does not appear in the book itself.
"""

import matplotlib.pyplot as plt


def make_chart_pie_chart():

    plt.pie([0.95, 0.05], labels=["Uses pie charts", "Knows better"])

    # make sure pie is a circle and not an oval
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    make_chart_pie_chart()
