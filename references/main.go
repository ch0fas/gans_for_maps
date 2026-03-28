package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path"
	"strings"
)

func download_images(file *os.File) {
	// Opening CSV Reader
	num_counter := 0
	parts := strings.Split(file.Name(), "/")
	class_name := strings.ReplaceAll(parts[len(parts)-1], ".csv", "")
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	data, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}
	for _, row := range data {
		for _, col := range row {
			if strings.Contains(col, "FAVICON") == false && strings.Contains(col, "http") {
				//fmt.Printf(col)
				num_counter += 1
				temp_file_name := fmt.Sprintf("../data/raw/fantasy_images/%s_%d.jpg", class_name, num_counter)
				img, _ := os.Create(temp_file_name)
				defer img.Close()

				resp, err := http.Get(col)
				if err != nil {
					panic(err)
				}
				defer resp.Body.Close()

				b, _ := io.Copy(img, resp.Body)
				fmt.Println("File size: ", b)
			}
		}
	}
}

func listFiles(dir string) []string {
	root := os.DirFS(dir)

	csvFiles, err := fs.Glob(root, "*.csv")

	if err != nil {
		log.Fatal(err)
	}

	var files []string
	for _, v := range csvFiles {
		files = append(files, path.Join(dir, v))
	}
	return files
}

func main() {
	// Basic early operations
	dir := "../data/raw/fantasy_csvs"
	files := listFiles(dir)
	for _, v := range files {
		file, err := os.Open(v)
		fmt.Printf("Now, we are doing %s\n", file.Name())
		if err != nil {
			panic(err)
		}
		download_images(file)

	}
	// file, err := os.Open("urls.csv")
	// if err != nil {
	// 	panic(err)
	// }

}
