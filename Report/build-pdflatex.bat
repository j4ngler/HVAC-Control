@echo off
setlocal EnableExtensions
pushd "%~dp0"

echo Cleaning LaTeX auxiliary files...
if exist report.aux del /Q "report.aux"
if exist report.log del /Q "report.log"
if exist report.toc del /Q "report.toc"
if exist report.lof del /Q "report.lof"
if exist report.lot del /Q "report.lot"
if exist report.out del /Q "report.out"
if exist report.bbl del /Q "report.bbl"
if exist report.blg del /Q "report.blg"
if exist report.synctex.gz del /Q "report.synctex.gz"
echo Done cleaning.
echo.

chcp 65001 >nul 2>&1
echo Building LaTeX document (pass 1)...
pdflatex -synctex=1 -interaction=nonstopmode -file-line-error report.tex
echo.
echo Building LaTeX document (pass 2 - TOC and refs)...
pdflatex -synctex=1 -interaction=nonstopmode -file-line-error report.tex
echo.
echo Done!
popd
endlocal
