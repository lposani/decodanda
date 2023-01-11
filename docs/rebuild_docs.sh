rm -f source/decodanda.* source/decodanda.rst source/modules.rst
sphinx-apidoc -o ./source/ ../decodanda
make html
open build/html/index.html
