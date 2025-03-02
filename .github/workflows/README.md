# Info about how to use Github Actions

## Adding data via a pull request from fork

The main github action here is: NewDataPipeline.yml and it does two things:


The new data pipeline relies on this workflow file to handle information from an incoming branch in a fork. It will first create a new branch and then open a pull request with the following setup:
    
1. First, create a new branch on this repository identical to the incoming branch from fork
2. Then, open a pull request with the following setup:
       
    * The base of the new pull request will match the base of the fork's pull request.
    * Example: If the fork is targeting the `main` branch with added files, this workflow creates a new branch in this repository and opens a pull request to the `main` branch of this repository.

3. Then the AddData.yml workflow will be started with the info about relevant branches. This will go through the steps for adding experiment data first then the steps for adding simulation data. If no experiment data is present then those steps will be skipped. The scripts for adding experiment data and simulation data is in `Scripts/WorkflowScripts`. 
The automated steps for sim data is described [here](https://nmrlipids.github.io/addingData.html).
The automated steps for experiment data is described [here](https://nmrlipids.github.io/addingExpData.html)
   * Experiment addition steps automated: 3,4,5
   * Via pull request simulation addition steps automated: 4,5,6
   * Via website simulation addition steps automated: 3,4,5,6,7






## Recomputing Data

You can initiate recomputes of simulation data using the workflow file: `RecomputeDatabank.yml`. The following options allow you to specify which folders to recompute and control the distribution of the workload. It will work with indexes of the folders as given by the method `sorted_databank()` which sorts the Databank based on the system IDs. This makes the indexing consistent. For each specified index it will then recompute the JSON files and push to the branch specified as the workflow input: `working_branch_name`

- **Specify Folder index Range**: Use the starting and ending index to define which folders to recompute.
  - **All Folders**: Set the starting index to `0` and the ending index to `-1` (negative one) to recompute all folders.
  
- **Set Number of Runners**: Define the number of runners to split the recompute job. Each runner will process its portion of the folders and commit the results to a specified branch.
  - For example, if there are 100 folders and you specify 5 runners, each runner will handle 20 folders.

This setup enables parallel processing, with each runner working on a distinct subset of folders.


## Docker Image Builder Workflow

The workflow file, `BuildDockerImages.yml`, allows you to rebuild and push specific Docker images to Docker Hub. The purpose is to make rebuilding the dockerimage easy when that is needed, for instance with major updates to dependencies, like MDanalsysis. 
#### Triggering the Workflow

The workflow can be triggered manually with the `workflow_dispatch` event, allowing you to specify which Docker image to rebuild.

#### Inputs

- `docker_image_name` (required): The name of the Docker image you want to rebuild and push. This should match the image names in the repository (e.g., `core`, `util`).

- `gromacs_version` (default is 2024.4): The version of gromacs used in the build

- `default branch` (default is main): This is the branch the docker container starts from. 

#### Steps in the Workflow

1. **Log in to Docker Hub**:
   - Uses Docker Hub credentials (stored as `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets) to authenticate before building and pushing images. These need to be added for this workflow to be functional. Instructions to add secrets to a github repository can be found [here](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions?tool=webui#creating-secrets-for-a-repository)

2. **Build Docker Image**:
   - Builds the specified Docker image based on the input provided. The image is tagged using the `nmrlipids/` prefix and the `latest` tag by default.

3. **Push Docker Image to Docker Hub**:
   - Pushes the tagged image to Docker Hub under the `nmrlipids/` namespace.

#### Example Usage

To manually trigger the workflow, go to the **Actions** tab in your GitHub repository, select `BuildDockerImages`, and click **Run workflow**. Enter the `docker_image_name` (e.g., `core`) to specify which image to rebuild.

#### Repository Structure

Dockerfiles are located in the `Docker` folder at the root of the repository, with names matching the Docker image names:
- `core` for the `nmrlipids/core` Docker image
- `utils` for the `nmrlipids/utils` Docker image


### Docker image details:
The main dockerimage contains software needed to run methods from the repository as well as working with github. It also contains the repository itself within the `Databank` directory which is the default folder for the container. Commands sent directly the container can therefore interact with the repository directly without changing folder. 
Some work done by github actions uses different scripts contained within the repository. For the scripts we have two directories:

`WorkflowScripts` and the subdirectory: `DockerRunInstructions`

`WorkflowScripts`: scripts within this folder do individual tasks like creating a new branch or configuring the Git config. These scripts are then used in different ways within the instructions found in `DockerRunInstructions`. Different tasks done by the runners require different combinations of smaller tasks which is why this organisation of folders was used. With this setup only one command needs to be sent to the docker container within the workflow files directly. 




