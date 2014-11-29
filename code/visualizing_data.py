import matplotlib.pyplot as plt
from collections import Counter

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

def make_chart_simple_line_chart(plt):

    # create a line chart, years on x-axis, gdp on y-axis
    plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
    # add a label to the y-axis
    plt.ylabel("Nominal GDP, Billions of $")
    plt.show()


movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

def make_chart_simple_bar_chart(plt):
    xs = range(len(movies)) # [0, 1, 2, 3, 4]

    # bars with left x-coordinates [xs], heights [num_oscars]
    plt.bar(xs, num_oscars)
    plt.ylabel("# of Academy Awards")

    # label x-axis with movie names
    plt.xticks([x + 0.5 for x in xs], movies)
    plt.show()

grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
decile = lambda grade: grade // 10 * 10 
histogram = Counter(decile(grade) for grade in grades)

def make_chart_histogram(plt):
    plt.bar([x - 4 for x in histogram.keys()], # shift each bar to the left by 4
            histogram.values(),                # give each bar its correct height
            8)                                 # give each bar a width of 8
    plt.axis([-5, 105, 0, 5])                  # x-axis from -5 to 105,
                                               # y-axis from 0 to 5
    plt.xticks([10 * i for i in range(11)])    # x-axis labels at 0, 10, ..., 100
    plt.show()

mentions = [500, 505]
years2 = [2013, 2014]

def make_chart_misleading_y_axis(plt, mislead=True):
    plt.bar(years2, mentions, 0.8)
    plt.ylabel("# of times I heard someone say 'data science'")
    plt.xticks([year + 0.35 for year in years], map(str,years2))

    if mislead:
        # misleading y-axis only shows the part above 500
        plt.axis([2013,2015,499,506])
        plt.title("look at the 'huge' increase!")
    else:
        plt.axis([2013,2015,0,550])
        plt.title("not so huge anymore.")       
    plt.show()

variance     = [1,2,4,8,16,32,64,128,256]
bias_squared = [256,128,64,32,16,8,4,2,1]
total_error  = [x + y for x, y in zip(variance, bias_squared)]

def make_chart_several_line_charts(plt):
    xs = range(len(variance))

    # we can make multiple calls to plt.plot 
    # to show multiple series on the same chart
    plt.plot(xs, variance,     'g-',  label='variance')    # green solid line
    plt.plot(xs, bias_squared, 'r-.', label='bias^2')      # red dot-dashed line
    plt.plot(xs, total_error,  'b:',  label='total error') # blue dotted line

    # because we've assigned labels to each series
    # we can get a legend for free
    # loc=9 means "top center"
    plt.legend(loc=9)
    plt.xlabel("model complexity")
    plt.show()

heights = [ 70,  65,  72,  63,  71,  64,  60,  64,  67]
weights = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

def make_chart_scatter_plot(plt):

    plt.scatter(heights, weights)

    # label each point
    for label, height, weight in zip(labels, heights, weights):
        plt.annotate(label, 
            xy=(height, weight),        # put the label with its point
            xytext=(5, -5),             # but slightly offset
            textcoords='offset points')
        
    plt.xlabel("height (inches)")
    plt.ylabel("weight (pounds)")
    plt.show() 

test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

def make_chart_scatterplot_axes(plt, equal_axes=False):
    plt.scatter(test_1_grades, test_2_grades)
    plt.xlabel("test 1 grade")
    plt.ylabel("test 2 grade")

    if equal_axes:
        plt.title("axes are comparable")
        plt.axis("equal")
    else:
        plt.title("axes aren't comparable")

    plt.show()

def make_chart_pie_chart(plt):

    plt.pie([0.95, 0.05], labels=["Uses pie charts", "Knows better"])

    # make sure pie is a circle and not an oval
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":

    make_chart_simple_line_chart(plt)

    make_chart_simple_bar_chart(plt)

    make_chart_histogram(plt)

    make_chart_misleading_y_axis(plt, mislead=True)

    make_chart_misleading_y_axis(plt, mislead=False)

    make_chart_several_line_charts(plt)

    make_chart_scatterplot_axes(plt, equal_axes=False)

    make_chart_scatterplot_axes(plt, equal_axes=True)

    make_chart_pie_chart(plt)
