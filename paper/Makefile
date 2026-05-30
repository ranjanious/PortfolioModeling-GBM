TEX     = main
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all clean view

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib
	$(LATEXMK) $(TEX).tex

view: $(TEX).pdf
	@if command -v xdg-open >/dev/null 2>&1; then xdg-open $(TEX).pdf; \
	elif command -v open >/dev/null 2>&1; then open $(TEX).pdf; \
	else echo "Open $(TEX).pdf manually."; fi

clean:
	latexmk -C
	rm -f $(TEX).bbl $(TEX).run.xml
