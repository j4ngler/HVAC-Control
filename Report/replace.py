import sys

file_path = r"d:\Python\HVAC-Control\Report\report.tex"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(r"\documentclass[12pt, a4paper]{article}", r"\documentclass[12pt, a4paper]{report}")

content = content.replace(r"\section{", r"\chapter{")
content = content.replace(r"\subsection{", r"\section{")
content = content.replace(r"\subsubsection{", r"\subsection{")

content = content.replace(r"\textbf{Mục ", r"\textbf{Chương ")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated successfully")
