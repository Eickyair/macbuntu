IMAGE_NAME=adaboost-api:latest
CONTAINER_NAME=adaboost-container-api
PORT=8000
TEAM_NAME=equipo_macbuntu

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
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)
	docker image rm $(IMAGE_NAME)

package:
	tar -czf $(TEAM_NAME).tar.gz --exclude='$(TEAM_NAME).tar.gz' --exclude='*.tar.gz' --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='env' .

unpack:
	tar -xzf $(TEAM_NAME).tar.gz

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload