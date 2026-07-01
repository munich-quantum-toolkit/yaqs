#!/bin/bash
cd /Users/maximilianfrohlich/Desktop/GitHub/yaqs/manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
