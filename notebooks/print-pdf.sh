#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "Usage: sh $0 <token>"
    exit
fi

token=$1

for nb in *.ipynb; do
    # decktape rise "http://127.0.0.1:8888/notebooks/5. Filling Missing Values in Traffic Data.ipynb?token=3ef6c0bf29edafe31d5bbf68abdb097a4fa86740a1426550" -s 1680x900 "5. Filling Missing Values in Traffic Data" --chrome-arg=--disable-web-security --chrome-arg=--disable-dev-shm-usage

    out=`echo $nb | sed 's/\(.*\.\)ipynb/\1pdf/'`
    out="../pdfs/$out"
    if [ -f "$out" ]; then
        echo "File \"$out\" already exists"
    else
        decktape rise "http://127.0.0.1:8888/notebooks/$nb?token=$token" -s 1440x900 "$out" --chrome-arg=--disable-web-security --chrome-arg=--disable-dev-shm-usage --chrome-path=/usr/bin/chromium
    fi
done
