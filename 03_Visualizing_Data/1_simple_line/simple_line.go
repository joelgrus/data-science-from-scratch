package main

import (
	"math/rand"
	"strings"

	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
)

func main() {
	years := strings.Fields("1950 1960 1970 1980 1990 2000 2010")
	gdp := []float64{300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3}

	pts := make(plotter.XYs, len(gdp))
	for i, amount := range gdp {
		pts[i].X = float64(i)
		pts[i].Y = amount
	}

	p, err := plot.New()
	check(err)

	p.Title.Text = "Nominal GDP"
	p.X.Label.Text = "Years"
	p.NominalX(years...)
	p.Y.Label.Text = "Billions of $"

	line, err := plotter.NewLine(pts)
	check(err)
	p.Add(line)

	// Save the plot to a PNG file.
	err = p.Save(4*vg.Inch, 4*vg.Inch, "simple_line.png")
	check(err)
}

// randomPoints returns some random x, y points.
func randomPoints(n int) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := range pts {
		if i == 0 {
			pts[i].X = rand.Float64()
		} else {
			pts[i].X = pts[i-1].X + rand.Float64()
		}
		pts[i].Y = pts[i].X + 10*rand.Float64()
	}
	return pts
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}
