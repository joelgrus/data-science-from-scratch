#!/usr/bin/env python3

"""
Figure 3-4. A chart with a misleading y-axis
Figure 3-5. The same chart with a nonmisleading y-axis
"""

import matplotlib.pyplot as plt

def make_chart_misleading_y_axis(mislead=True):

    mentions = [500, 505]
    years = [2013, 2014]

    plt.bar([2012.6, 2013.6], mentions, 0.8)
    plt.xticks(years)
    plt.ylabel("# of times I heard someone say 'data science'")

    # if you don't do this, matplotlib will label the x-axis 0, 1
    # and then add a +2.013e3 off in the corner (bad matplotlib!)
    plt.ticklabel_format(useOffset=False)

    if mislead:
        # misleading y-axis only shows the part above 500
        plt.axis([2012.5,2014.5,499,506])
        plt.title("Look at the 'Huge' Increase!")
    else:
        plt.axis([2012.5,2014.5,0,550])
        plt.title("Not So Huge Anymore.")
    plt.show()


if __name__ == "__main__":
    make_chart_misleading_y_axis(mislead=True)
    make_chart_misleading_y_axis(mislead=False)
