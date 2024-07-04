source /Users/lorenzo/Dropbox/Research/Decodanda/venv/bin/python/activate
rm -f source/decodanda.* source/decodanda.rst
sphinx-apidoc -o ./source/ ../decodanda
make clean
make html
open build/html/index.html
