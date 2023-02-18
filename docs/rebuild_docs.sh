source ~/Dropbox/Research/Other\ stuff/PythonEnvs/decodanda/bin/activate
rm -f source/decodanda.* source/decodanda.rst
sphinx-apidoc -o ./source/ ../decodanda
make clean
make html
open build/html/index.html
