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

	@echo "Packaging application..."
	tar -czf ../$(TEAM_NAME).tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' --exclude='env' --exclude='.vscode' .
	mv ../$(TEAM_NAME).tar.gz .

unpack:
	tar -xzvf ./$(TEAM_NAME).tar.gz
test:
	python ./test_requests.py