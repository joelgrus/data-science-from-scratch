package main

import (
	"html/template"
	"os"

	"github.com/go-template/nvd3-go-template/browser"
)

var t *template.Template
type Data_type struct {
	X	int
	Y	float64
}

var data []Data_type

func init() {
	t = template.Must(template.ParseFiles("templates/linechart.html",
		"templates/linechartjs.tmpl"))
	data = []Data_type {
		Data_type {X: 1950, Y: 300.2},
		Data_type {X: 1950, Y: 300.2},
		Data_type {X: 1960, Y: 543.3},
		Data_type {X: 1970, Y: 1075.9},
		Data_type {X: 1980, Y: 2862.5},
		Data_type {X: 1990, Y: 5979.6},
		Data_type {X: 2000, Y: 10289.7},
		Data_type {X: 2010, Y: 14958.3},
	}
}

func render(path string) {
	var file = createFile(path)
	t.ExecuteTemplate(file, "linechart.html", data)
	defer file.Close()
}

func createFile(path string) (f *os.File) {
	f, err := os.Create(path)
	if err != nil {
    panic(err)
	}
	return f
}

func deleteDirOutput() {
		os.Remove("output")
}

func main() {
	deleteDirOutput()

	path := "output/" + os.Args[1]
	render(path)
	browser.Open(path)
}
