#!/usr/bin/make

all:
	latexmk -pdf main.tex
	mv main.pdf ICR_Literature_Survey_8040_8077_141708001_5075_T5.pdf
	latexmk -pdf main_abstract.tex
	mv main_abstract.pdf abstract.pdf
	latexmk -pdf main_survey.tex
	mv main_survey.pdf survey.pdf
	latexmk -pdf main_references.tex
	mv main_references.pdf references.pdf

clean:
	latexmk -C -pdf main.tex
	latexmk -C -pdf main_abstract.tex
	latexmk -C -pdf main_survey.tex
	latexmk -C -pdf main_references.tex


.PHONY:
	all
.PHONY:
	clean
