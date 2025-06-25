mkdir -p data

wget -P data --no-clobber http://konect.cc/files/download.tsv.wikipedia_link_mi.tar.bz2
tar -xvjf data/download.tsv.wikipedia_link_mi.tar.bz2 -C data/

wget -P data --no-clobber http://konect.cc/files/download.tsv.arenas-jazz.tar.bz2
tar -xvjf data/download.tsv.arenas-jazz.tar.bz2 -C data/
