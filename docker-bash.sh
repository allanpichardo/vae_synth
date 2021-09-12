docker run -u $(id -u):$(id -g) --runtime=nvidia -it --rm --gpus all --env-file .env -p 6006:6006 -v $PWD:/tmp -w /tmp mhz/tensorflow:latest bash
