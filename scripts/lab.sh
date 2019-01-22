docker run --env-file .env -d --user root -e NB_UID=5000 --name lab -v /home/nandanrao:/home/jovyan/work -p 8888:8888 nandanrao/starspace-notebook start.sh jupyter lab
