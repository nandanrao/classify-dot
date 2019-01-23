docker run -d -v $PWD:/starspace/mount --name docspace nandanrao/starspace Starspace/starspace train -trainFile mount/${1} -model mount/${2} -dim 200 -ngrams 1 -minCount 20 -trainMode 6 -negSearchLimit 100 -fileFormat labelDoc -thread 32 -epoch 10  -saveEveryEpoch 1