#!/usr/bin/env python3

"""
Figure 3-8. A scatterplot with uncomparable axes
Figure 3-9. The same scatterplot with equal axes
"""

import matplotlib.pyplot as plt


def make_chart_scatter_plot_axes(equal_axes=False):

    test_1_grades = [ 99, 90, 85, 97, 80]
    test_2_grades = [100, 85, 60, 90, 70]

    plt.scatter(test_1_grades, test_2_grades)
    plt.xlabel("test 1 grade")
    plt.ylabel("test 2 grade")

    if equal_axes:
        plt.title("Axes Are Comparable")
        plt.axis("equal")
    else:
        plt.title("Axes Aren't Comparable")

    plt.show()


if __name__ == "__main__":
    make_chart_scatter_plot_axes(equal_axes=False)
    make_chart_scatter_plot_axes(equal_axes=True)
