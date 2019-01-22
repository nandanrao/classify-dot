docker run --user root -e UID=5000 --rm -v /home/nandanrao:/home/jovyan/work jupyter/datascience-notebook jupyter nbconvert work/${1} --template work/report.tpl
