TECTONIC ?= tectonic
.PHONY: all
all: main.pdf
main.pdf: main.tex
	$(TECTONIC) --keep-logs --keep-intermediates main.tex
