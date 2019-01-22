docker run -d --name embed  -v $PWD:/starspace/mount nandanrao/starspace /bin/bash -c "Starspace/embed_doc mount/${1} < mount/${2} | tail -n +5 > mount/${3}"
