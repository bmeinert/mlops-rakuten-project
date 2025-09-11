# The purpose of the Makerfile is to automate the setup and configuration 
# of a machine learning project using DVC (Data Version Control) and Docker Compose.
# It includes tasks for installing dependencies, initializing Git and DVC, 
# building Docker images, running pipelines, and tracking changes.

.PHONY: uv set-dvc init-git init-dvc build run track dvc-docto all

# uv installs the uv tool, which is a command-line utility for managing DVC pipelines.
# It is installed using a shell script from the Astral website. 
uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Set-dvc sets up the DVC remote storage configuration for the project.
# It adds a remote storage location (S3 bucket) and configures the access credentials.
set-dvc:
	dvc remote add origin s3://dvc
	dvc remote modify origin endpointurl https://dagshub.com/BeritM/mlops-rakuten-project.s3
	dvc remote modify origin --local access_key_id $(DVC_USER)
	dvc remote modify origin --local secret_access_key $(DVC_PASSWORD)
	dvc remote default origin

# init-git initializes a new Git repository and commits the initial project structure
# and pipeline definition files. It also creates a .gitignore file to exclude unnecessary files.
init-git:
	git init
	git add .gitignore Dockerfile docker-compose-dvc.yml dvc.yaml src/
	git commit -m "Initial project structure and pipeline definition"

# Initialize DVC in the current directory and commit the configuration files
init-dvc:
	dvc init
	git add .dvc/config .dvc/.gitignore
	git commit -m "Initialize DVC"

# Build Docker images using Docker Compose
build:
	docker compose -f docker-compose-dvc.yml build

# run the DVC pipeline using the dvc-runner service
run:
	docker compose run --rm dvc-runner dvc repro

# track changes in the repository and push them to the remote DVC storage
track:
	git add .
	git commit -m "updating the run"
	dvc commit
	git push
	dvc push

# dvc-docto runs the DVC doctor command to check the health of the DVC setup
# and configuration. It is useful for diagnosing issues with DVC pipelines.
dvc-docto:
	docker compose run --rm dvc-runner dvc doctor

# all is a target that runs all the tasks in the Makefile in sequence.
# It initializes Git and DVC, builds the Docker images, and runs the DVC pipeline.
all: init-git init-dvc build run
