# Minimal Makefile for Sphinx documentation.
#
# Usage:
#   make html        — one-shot build into _build/html
#   make fast        — skip autodoc2 (fast iteration on prose changes)
#   make serve       — live-reload server on :8000
#   make clean       — remove _build/
#   make linkcheck   — validate external links

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html fast serve clean linkcheck

help:
	@echo "make html       — build the docs into _build/html"
	@echo "make fast       — build without autodoc2 (faster for prose edits)"
	@echo "make serve      — live-reload server (requires sphinx-autobuild)"
	@echo "make clean      — remove _build/"
	@echo "make linkcheck  — validate external links"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

fast:
	SKIP_AUTODOC=true $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

serve:
	SKIP_AUTODOC=true sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --port 8000 --open-browser $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
