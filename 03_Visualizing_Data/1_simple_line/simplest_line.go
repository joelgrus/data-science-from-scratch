package main

import (
	"strings"

	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
)

func main() {
	years := strings.Fields("1950 1960 1970 1980 1990 2000 2010")
	gdp := []float64{300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3}

	xys := make(plotter.XYs, len(gdp))
	for i, amount := range gdp {
		xys[i].X = float64(i)
		xys[i].Y = amount
	}

	p, err := plot.New()
	check(err)

	p.Title.Text = "Nominal GDP"
	p.X.Label.Text = "Years"
	p.NominalX(years...)
	p.Y.Label.Text = "Billions of $"

	line, err := plotter.NewLine(xys)
	check(err)
	p.Add(line)

	// Save the plot to a PNG file.
	err = p.Save(14*vg.Centimeter, 10*vg.Centimeter, "simplest_line.png")
	check(err)
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}
