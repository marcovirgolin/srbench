#!/bin/bash
# install gpg

# remove directory if it exists
if [ -d "gpg" ]; then 
    rm -rf gpg
fi

git clone  https://github.com/marcovirgolin/gpg
cd gpg
#fix version

make 