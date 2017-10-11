
document = dome_calibration
bibliography = bibliography

${document}.pdf : ${document}_flat.tex
	pdflatex ${document}_flat.tex
	pdflatex ${document}_flat.tex
	mv ${document}_flat.pdf ${document}.pdf

${document}_flat.tex : ${document}.tex ${bibliography}.bbl
	latexpand --keep-comments --expand-bbl ${bibliography}.bbl ${document}.tex \
		> ${document}_flat.tex

${bibliography}.bbl : ${bibliography}.tex ${bibliography}.bib
	pdflatex ${bibliography}.tex
	bibtex ${bibliography}.aux

