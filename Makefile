IMAGE_NAME=mi-imagen:latest
CONTAINER_NAME=adaboost-container
PORT=8000
TEAM_NAME=equipo_demo # Cambiar por el identificador del equipo

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(IMAGE_NAME)

status:
	docker ps -a --filter name=$(CONTAINER_NAME)

stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

clean:
	docker system prune -f

package:
	tar -czf $(TEAM_NAME).tar.gz --exclude='*.tar.gz' --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' .