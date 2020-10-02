# Azure Machine Learning Starter Workshop (Coding approach focus)

Based on the content from the microsoft learning repository here: [https://github.com/MicrosoftLearning/DP100.](https://github.com/MicrosoftLearning/DP100)
The full lab is designed to accompany the course [DP-100 Designing and Implementing a Data Science Solution on Azure] (https://docs.microsoft.com/en-us/learn/certifications/courses/dp-100t01) and take you through all technologies needed for the exam. 

This version in this repo is a selection of the exercises & instructions for the purpose of a more concise workshop you can follow alone or as a group in session of about 1 hour (depending on your background and familiarity with the tools of course). 

The content covered in this repo:
1. Azure Machine Learning Environment
* Deployment of the AML workspace and initial necessary components like compute. 
* Clone the repository and launch the jupyter interface where you will work with notebooks
2. Training a model in an experiment
* Topics: tracking and logging with the Experiment functionality, compute target, running as a script.
3. Model registration and deployment, real-time inference. 
* Deplying the model to an ACI and making a call for a prediction. 


# 1: Azure Machine Learning Environment
Here you will create the Azure Machine Learning workspace that you will use throughout the rest of this course.

### 1.1 In the Azure portal
As its name suggests, a workspace is a centralized place to manage all of the Azure ML assets you need to work on a machine learning project.
1. In the Azure portal, create a new Machine Learning resource, specifying a unique workspace name and creating a new resource group in the region nearest your location.
2. When the workspace and its associated resources have been created, view the workspace in the portal.

You can manage workspace assets in the Azure portal, but for data scientists, this tool contains lots of irrelevant information and links that relate to managing general Azure resources. An alternative, Azure ML-specific web interface for managing workspaces is available.

> Note: The web-based interface for Azure ML is named Azure Machine Learning studio, which you may find confusing as there is also a free Azure Machine Learning Studio product for creating machine learning models using a visual designer. A more scalable version of this visual designer is included in the new studio interface.

### 1.2 AML Studio
1. In the Azure portal blade for your Azure Machine Learning workspace, click the link to launch Azure Machine Learning studio; or alternatively, in a new browser tab, open https://ml.azure.com. If prompted, sign in using the Microsoft account you used in the previous task and select your Azure subscription and workspace.
3. View the Azure Machine Learning studio interface for your workspace - you can manage all of the assets in your workspace from here.

One of the benefits of Azure Machine Learning is the ability to create cloud-based compute on which you can run experiments and training scripts at scale.

### 1.3 Compute
1. In the Azure Machine Learning studio web interface for your workspace, view the **Compute** page. This is where you'll manage all the compute targets for your data science activities.
2. On the **Compute Instances** tab, add a new compute instance with the following settings. You'll use this as a workstation from which to test your model:
    - **Compute name**: *enter a unique name*
    - **Virtual Machine type**: CPU
    - **Virtual Machine size**: Standard_DS11_v2
3. While the compute instance is being created, switch to the **Compute Clusters** tab, and add a new compute cluster with the following settings. You'll use this to train a machine learning model:
    - **Compute name**: *enter a unique name*
    - **Virtual Machine type**: CPU
    - **Virtual Machine priority**: Dedicated
    - **Virtual Machine size**: Standard_DS11_v2
    - **Minimum number of nodes**: 0
    - **Maximum number of nodes**: 2
    - **Idle seconds before scale down**: 120
4. Note the **Inference Clusters** tab. This is where you can create and manage compute targets on which to deploy your trained models as web services for client applications to consume.
5. Note the **Attached Compute** tab. This is where you could attach a virtual machine or Databricks cluster that exists outside of your workspace.
    > **Note**: You'll explore compute targets in more detail later in the course.
    
### 1.4 Dataset

Now that you have some compute resources that you can use to process data, you'll need a way to store and ingest the data to be processed.

1. In the *Studio* interface, view the **Datastores** page. Your Azure ML workspace already includes two datastores based on the Azure Storage account that was created along with the workspace. These are used to store notebooks, configuration files, and data.

   > **Note**: In a real-world environment, you'd likely add custom datastores that reference your business data stores - for example, Azure blob containers, Azure Data Lakes, Azure SQL Databases, and so on. You'll explore this later in the course.

2. In the *Studio* interface, view the **Datasets** page. Datasets represent specific data files or tables that you plan to work with in Azure ML.
3. Create a new dataset from web files, using the following settings:
    * **Basic Info**:
        * **Web URL**: https://aka.ms/diabetes-data
        * **Name**: diabetes dataset (*be careful to match the case and spacing*)
        * **Dataset type**: Tabular
        * **Description**: Diabetes data
    * **Settings and preview**:
        * **File format**: Delimited
        * **Delimiter**: Comma
        * **Encoding**: UTF-8
        * **Column headers**: Use headers from first file
        * **Skip rows**: None
    * **Schema**:
        * Include all columns other than **Path**
        * Review the automatically detected types
    * **Confirm details**:
        * Do not profile the dataset after creation
4. After the dataset has been created, open it and view the **Explore** page to see a sample of the data. This data represents details from patients who have been tested for diabetes, and you will use it in many of the subsequent labs in this course.

    > **Note**: You can optionally generate a *profile* of the dataset to see more details. You'll explore datasets in more detail later in the course.
    
### 1.5 Notebook environment

1. On the **Compute** page for your workspace, view the **Compute Instances** tab (and if necessary, click **Refresh** periodically until the compute instance you created in the previous step has started)
2. Click your compute instance's **Jupyter** link to open Jupyter Notebooks in a new tab. If prompted, sign in using the Microsoft account associated with your Azure subscription.
3. In the notebook environment, create a new **Terminal**. This will open a new tab with a command shell.
4. The Azure Machine Learning SDK is already installed in the compute instance image, but it's worth ensuring you have the latest version, with the optional packages you'll need in this course; so enter the following command to update the SDK packages:

    ```bash
    pip install --upgrade azureml-sdk[notebooks,automl,explain]
    ```

    You may see some warnings as the package dependencies are installed. You can ignore these.

    > **More Information**: For more details about installing the Azure ML SDK and its optional components, see the [Azure ML SDK Documentation](https://docs.microsoft.com/python/api/overview/azure/ml/install?view=azure-ml-py).

5. Next, run the following commands to change the current directory to the **Users** directory (if multiple people work on the same workspace navigate to the folder with your username), and retrieve the notebooks you will use in the labs for this course:

    ```bash
    cd Users
    git clone https://github.com/MicrosoftLearning/DP100
    ```

6. After the command has completed, close the terminal tab and view the home page in your Jupyter notebook file explorer. Then open the **Users** folder - it should contain an **DP100** folder, containing the files you will use in the rest of this lab.
7. In the **Users/DP100** folder, open the **01B - Intro to the Azure ML SDK.ipynb** notebook. Then read the notes in the notebook, running each code cell in turn.
8. When you have finished running the code in the notebook, on the **File** menu, click **Close and Halt** to close it and shut down its Python kernel. Then close all Jupyter browser tabs.
9. If you're finished working with Azure Machine Learning for the day, in Azure Machine Learning studio, on the **Compute** page, select your compute instance and click **Stop** to shut it down. Otherwise, leave it running for the next lab.
