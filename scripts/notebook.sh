docker run --env-file .env -d --user root -e NB_UID=$UID --name notebook -v $PWD:/home/jovyan/work -p 8888:8888 nandanrao/starspace-notebook
