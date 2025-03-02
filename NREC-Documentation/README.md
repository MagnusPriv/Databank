# Setting Up a New GitHub Runner on NREC

This guide outlines the steps to set up a new GitHub runner on NREC using a provided image. 

## Launching a New Instance

For detailed instructions, refer to the [NREC guide](https://uh-iaas.readthedocs.io/create-linux-machine.html).

1. **Uploading provided Runner Image**
    - Navigato to `Images` under the `Compute` tab and click `Create Image`
    - Fill out Image Name 
    - Select the Image Source
    - Choose format: Raw. This has been tested previously and matches the Snapshot Format
    - Click `Create Image` which starts the upload of the image and takes some time. 

2. **Launch Instance**:
   - Navigate to the **Instances** tab on the left side on the NREC web interface and click **Launch Instance** on the right. 
   
   - From the **Source** tab, use "Select Boot Source" and choose the uploaded image from step 1.
   - Select a flavor that is medium or larger.
   - Add IPv6 under Networks.
   - Follow the [SSH guide](https://uh-iaas.readthedocs.io/create-linux-machine.html#ssh-key-pair) to add your SSH key for access.
   - Add a security group that supports SSH.
   - Click **Launch Instance**.

3. **Update the software**:
   You can connect to the instance using SSH with the IP listed on the site. After connecting it's good practice to update the software. 
   Updating the Linux packages can be done with the commands:
    
    ```sudo apt update```

    ```sudo apt upgrade```

## Mounting the SSD Volume

1. **Create Volume**:
   
   Click Volumes on NREC. It’s practical to create a snapshot of a base volume which contains a docker folder if it does not already exist. This can be done by first following the steps to create a volume and then creating a Docker directory at the root of the Volume. 
   
   Creating the volume can be done by following the [guide](https://uh-iaas.readthedocs.io/manage-volumes.html)

2. **Mount the volume from online interface**

    Follow the steps under `Attach`in the [guide](https://uh-iaas.readthedocs.io/manage-volumes.html)

2. **Mount the Volume**:

    Once it’s mounted via the web interface it also needs to be mounted via a command:
    
    ```sudo mount /dev/sdb /persistent01```

    ***Note***: If this command does not work it can be due to the name of the Attachement Point being changed. Look under `Volumes`on Web Interface and check the `Attached To` column. 
    For instance, if the attachement point reads: '/dev/vdb on Github Runner 2' then the command ```sudo mount /dev/sdb /persistent01``` needs to be changed to ```sudo mount /dev/vdb /persistent01```.


3. **Create a Docker Folder**:
   Once mounted, create a Docker folder from the root of the volume:
   ```sudo mkdir /persistent01/docker```
   After this is done you can save the Docker Volume as a snapshot. Later volumes can then be created with the Docker exisisting from start. 


## Configuring the Runner

1. **Start Docker**:
   
   ```sudo systemctl start docker```

2. **Navigate to the Actions Runner Directory**:

   ```cd actions-runner```

3. **Edit Runner Setup Script**:
   Open the `runnersetup.sh` script:

   ```nano runnersetup.sh``` 

   Set the following variables:
   - `GITHUB_OWNER`: The owner of your GitHub repository. 
   - `GITHUB_REPOSITORY`: Leave this as `Databank`if the repository name isn't changed.
   
   This will be used to create the URL to the correct repository. Example: If the URL to a repository is `https://github.com/MagnusSletten/Databank` the GITHUB_OWNER is `MagnusSletten`and the GITHUB_REPOSITORY is `Databank`

   Save and exit:
   - Press `Ctrl+O`, then `Enter` to save.
   - Press `Ctrl+X` to exit.

4. **Run the Setup Script**:

   `./runnersetup.sh {your_github_personal_access_token}`

   Here the runnersetup script is being run with the github token passed as a parameter. If a token is not already created you can find instructions here with the [Github official guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)


5. **Start the Runner**:

   ```./run.sh```


Additional details are covered in the general [NREC documentation](https://uh-iaas.readthedocs.io/).

For efficient deployment of several runners it can be wise to save a new snapshot of the runner once the  runnersetup.sh script is correctly pointing to your repository. 