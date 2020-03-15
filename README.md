<a id="top"></a>
# MLRun
<!-- SLSL TODO: Check for locations where I might have changed "workflow" in
  the code to "pipeline". I already checked the README files. NOWNOWNOW -->

[![CircleCI](https://circleci.com/gh/mlrun/mlrun/tree/development.svg?style=svg)](https://circleci.com/gh/mlrun/mlrun/tree/development)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version fury.io](https://badge.fury.io/py/mlrun.svg)](https://pypi.python.org/pypi/mlrun/)
[![Documentation](https://readthedocs.org/projects/mlrun/badge/?version=latest)](https://mlrun.readthedocs.io/en/latest/?badge=latest)
<!-- SLSL-TODO: Split into multiple pages in a separate commit. -->
<!-- SLSL-TODO: Check all my internal comments, make necessary edits, and
    remove the comments before submitting a PR. -->
<!-- SLSL: The "Documentation" tag links to readthedocs doc.
  1. The current link leads to an empty page:
     https://readthedocs.org/projects/mlrun/badge/?version=latest
  2. https://readthedocs.org/projects/mlrun/ hast "latest" and "stable" links:
      https://mlrun.readthedocs.io/en/latest/
      https://mlrun.readthedocs.io/en/stable/
    Both versions have links to multiple doc files.
  3. I think we should at least remove the "Documentation" doc until we update
    the generated documentation and ensure that all relevant contributors know
    which source files to update to keep the docs up to date.
  4. There's an empty docs/contents.rst file?
  5. I think we need to have a short overview of the purpose of each package +
     verify that the packages' order in the generated TOC makes sense.
  6. Many of the API methods are missing documentation, especially for the "db"
     package, which doesn't have any doc text.
-->

MLRun is a generic and convenient mechanism for data scientists and software developers to describe and run tasks related to machine learning (ML) in various, scalable runtime environments and ML pipelines while automatically tracking executed code, metadata, inputs, and outputs.
MLRun integrates with the [Nuclio](https://nuclio.io/) serverless project and with [Kubeflow Pipelines](https://github.com/kubeflow/pipelines).

MLRun features a Python package (`mlrun`), a command-line interface (`mlrun`), and a graphical user interface (the MLRun dashboard).
<!-- SLSL: I added this. Is the MLRun library/package a "client" package or
  client & server? This affects also my addition to the docs/api.rst file for
  the readthedocs doc generation. NOWNOW-RND -->

#### In This Document
- [General Concept and Motivation](#concepts-n-motivation)
- [Installation](#installation)
- [Examples and Tutorial Notebooks](#examples-n-tutorial-notebooks)
- [Quick-Start Tutorial &mdash; Architecture and Usage Guidelines](#qs-tutorial)

<a id="concepts-n-motivation"></a>
## General Concept and Motivation
- [The Challenge](#the-challenge)
- [The MLRun Vision](#the-vision)

<a id="the-challenge"></a>
### The Challenge

As an ML developer or data scientist, you typically want to write code in your preferred local development environment (IDE) or web notebook, and then run the same code on a larger cluster using scale-out containers or functions.
When you determine that the code is ready, you or someone else need to transfer the code to an automated ML workflow pipeline (for example, using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart/)).
This pipeline should be secure and include capabilities such as logging and monitoring, as well as allow adjustments to relevant components and easy redeployment.

However, the implementation is challenging: various environments (**"runtimes"**) use different configurations, parameters, and data sources.
In addition, different frameworks and platforms are used to focus on different stages of the development life cycle.
This leads to constant development and DevOps/MLOps work.

Furthermore, as your project scales, you need greater computation power or GPUs, and you need to access large-scale data sets.
This cant work on laptops; you need a way to seamlessly run your code on a remote cluster and automatically scale it out.

<a id="the-vision"></a>
### The MLRun Vision

When running experiments, you should ideally be able to record and version your code, configuration, outputs, and associated inputs (lineage), so you can easily reproduce and explain your results.
The fact that you probably need to use different types of storage (such as files and AWS S3 buckets) and various databases, further complicates the implementation.

Wouldn't it be great if you could write the code once, using your preferred development environment and simple "local" semantics, and then run it as-is on different platforms?
Imagine a layer that automates the build process, execution, data movement, scaling, versioning, parameterization, outputs tracking, and more.
A world of easily developed, published, or consumed data/ML "functions" that can be used to form complex and large-scale ML pipelines.

In addition, imaging a marketplace of ML functions, which includes both open-source templates and your internally developed functions, to support code reuse across projects and companies and thus further accelerate your work.

<b>This is the goal of MLRun.</b>

> **Note:** The code is in early development stages and is provided as a reference.
> The hope is to foster wide industry collaboration and make all the resources pluggable, so that developers can code to one API and use various open-source projects or commercial products.
<!-- SLSL: I don't think that for platform v2.8 we can say that MLRun is in
  early development stages, as most our the tutorial demos in this version use
  MLRun and I believe we plan to officially support it (and not as Tech Preview)
  (Adi/Haviv?). NOWNOW-RND/ADI -->

[Back to top](#top)

<a id="installation"></a>
## Installation

Run the following command from your Python development environment (such as Jupyter Notebook) to install the MLRun package (`mlrun`), which includes a Python API library and the `mlrun` command-line interface (CLI):
```python
pip install mlrun
```
<!-- SLSL: I edited the description. NOWNOW-RND
-->

MLRun requires separate containers for the API and the dashboard (UI).
You can also select to use the pre-baked JupyterLab image.
<!-- SLSL: What's the "pre-baked JupyterLab image" and does it create two contains? NOWNOW-RND -->

To install and run MLRun locally using Docker or Kubernetes, see the instructions in [**hack/local/README.md**](hack/local/README.md).

<a id="installation-iguazio-platform"></a>
### Installation on the Iguazio Data Science Platform

To install MLRun on an instance of the Iguazio Data Science Platform (**"the platform"**) &mdash;

1. Create a copy of the [**hack/mlrun-all.yaml**](hack/mlrun-all.yaml) configuration file; you can also rename your copy.
    You can fetch the file from GitHub by running the following from a command line:
    ```sh
    curl -O https://raw.githubusercontent.com/mlrun/mlrun/master/hack/mlrun-all.yaml
    ```
    <!-- [c-mlrun-versions] TODO: When there are MLRun version tags, instruct
      to replace `master` with the version tag for the MLRun version supported
      for the current platform version. -->

2. Edit the configuration file to match your environment and desired configuration.
    The following is required:

    - Replace all `<...>` placeholders in the file.
        Be sure to replace `<access key>` with a valid platform access key and `<default Docker registry URL>` with the URL of the default Docker Registry service of your platform cluster.

        > **Note:** In platform cloud deployments, the URL of the default Docker Registry service is `docker-registry.default-tenant.<cluster DNS>:80`.
        > Note the port number (80), which indicates a local on-cluster registry (unlike the default Docker port number).
    - Uncomment the `volumes` and the`mlrun-api` container's `volumeMounts` configurations to add a volume for persisting data in the platform's data store (using the `v3io` data mount).
    - Ensure that the value of the `V3IO_USERNAME` environment variable (`env`) and the `volumes.subPath` field are set to the name of a platform user with MLRun admin privileges (default: "admin").

3. When you're ready, install MLRun by running the following from a platform command-line shell; replace `<namespace>` with your cluster's Kubernetes namespace, and `<configuration file>` with the path to your edited configuration file:
    ```sh
    kubectl apply -n <namespace> -f <configuration file>
    ```
  <!-- SLSL: Q: How do they know the k8s namespace of the cluster?  NOWNOW-RND
  -->

<!-- SLSL: NOWNOWNOW-RND
  I separated the code examples into two blocks, moved the comments
  outside of the code, and edited them, and created enumerated steps.
  
  I added a requirement to add a persistent-data mount and to change the
  "Admin" configurations if needed.
  
  I made related edits to the comments in the configuration files.
  (8.3.20) Haviv told me that
  (a) There's no need / there soon won't be a need to use the configuration
      files on the platform?!
  (b) We should refer to the MLRun service user, as special privileges are
      needed that not any every running user who can use the single MLRun
      service will have. But I think that this doesn't align with the current
      use of $V3IO_USERNAME, which is the platform's running-user envar + I
      don't think we can refer to the running user/owner of the MLRun service
      because if it's a single shared tenant-wide service it doesn't have a
      running user?!

  I referred to running kubectl from a platform command-line shell, although
  I think it's also possible to run it by connecting to the platform remotely.
 
  I'm not sure the instructions are specific to the platform, except for the
  specific explanations relating to the platform info to configure?
  
  Also, I edited the volumeMounts comment in mlrun-all.yaml and mlrunapi.yaml
  and I changed the indentation level for this configuration in both files to
  make it a direct child of `containers` and not of `args` - which didn't make
  sense, as it's assigned `[]` (`args: []`); note that in mlrun-local.yaml, the
  volumeMounts configuration isn't commented out and it's on the same level as
  `args`, as a direct child of `containers`.

  Q: How do they know the k8s namespace of the cluster?
-->

[Back to top](#top)

<a id="examples-n-tutorial-notebooks"></a>
## Examples and Tutorial Notebooks

MLRun has many code examples and tutorial Jupyter notebooks with embedded documentation, ranging from examples of basic tasks to full end-to-end use-case applications, including the following; note that some of the examples are found in other mlrun GitHub repositories:

- Learn MLRun basics &mdash; [**examples/mlrun_basics.ipynb**](examples/mlrun_basics.ipynb)
- Convert local runs to Kubernetes jobs and create automated pipelines in a single notebook &mdash; [**examples/mlrun_jobs.ipynb**](examples/mlrun_jobs.ipynb)
- End-to-end XGBoost pipeline, including data ingestion, model training, verification, and deployment &mdash; [**demo-xgb-project**](https://github.com/mlrun/demo-xgb-project) repo
- MLRun with scale-out runtimes &mdash;
  - Distributed TensorFlow with Horovod and MPIJob &mdash; [**examples/mlrun_mpijob_classify.ipynb**](examples/mlrun_mpijob_classify.ipynb)
  - Serverless model serving with Nuclio &mdash; [**examples/xgb_serving.ipynb**](examples/xgb_serving.ipynb)
  - Dask &mdash; [**examples/mlrun_dask.ipynb**](examples/mlrun_dask.ipynb)
  - Spark &mdash; [**examples/mlrun_sparkk8s.ipynb**](examples/mlrun_sparkk8s.ipynb)
- MLRun projects &mdash;
  - Load a project from a remote Git location and run pipelines &mdash; [**examples/load-project.ipynb**](examples/load-project.ipynb)
  - Create a new project, functions, and pipelines, and upload to Git &mdash; [**examples/new-project.ipynb**](examples/new-project.ipynb)
- Import and export functions using files or Git &mdash; [**examples/mlrun_export_import.ipynb**](examples/mlrun_export_import.ipynb)
- Query the MLRun DB &mdash; [**examples/mlrun_db.ipynb**](examples/mlrun_db.ipynb)

<a id="additional-examples"></a>
### Additional Examples

- Deep-learning pipeline (full end-to-end application), including data collection and labeling, model training and serving, and implementation of an automated workflow &mdash; [mlrun/demo-image-classification](https://github.com/mlrun/demo-image-classification) repo
- Additional end-to-end use-case applications &mdash; see the [mlrun/demos](https://github.com/mlrun/demos) repo
- MLRun Functions Library (work in progress) is in [mlrun/functions repo](https://github.com/mlrun/functions)

[Back to top](#top)

<a id="qs-tutorial"></a>
## Quick-Start Tutorial &mdash; Architecture and Usage Guidelines
<!-- TODO: Move this to a separate docs/quick-start.md file, add an opening
  paragraph, update the heading levels, add a `top` anchor, and remove the
  "Back to quick-start TOC" links (leaving only the "Back to top" links). -->

- [Basic Components](#basic-components)
- [Managed and Portable Execution ](#managed-and-portable-execution)
- [Automated Code Deployment and Containerization](#auto-parameterization-artifact-tracking-n-logging)
- [Using Hyperparameters for Job Scaling](#using-hyperparameters-for-job-scaling)
- [Automated Code Deployment and Containerization](#auto-code-deployment-n-containerization)
- [Build and run function from a remote IDE using the CLI](examples/remote.md)
- [Running an ML Workflow with Kubeflow Pipelines](#run-ml-workflow-w-kubeflow-pipelines)
- [Viewing Run Information and Artifacts](#view-run-info-n-artifacts)
- [The MLRun Dashboard](#mlrun-ui)
- [MLRun Database Methods](#mlrun-db-methods)
- [Additional Information and Examples](#additional-info-n-examples)
  - [Replacing Runtime Context Parameters from the CLI](#replace-runtime-context-param-from-cli)
  - [Using Remote Function Code](#using-remote-function-code)
- [Running an MLRun Database/API Service](#run-mlrun-db-service)
  - [Using Docker](#run-mlrun-service-docker)
  - [Using the MLRun CLI](#run-mlrun-service-cli)

<a id="basic-components"></a>
### Basic Components
<!-- SLSL: I moved the following from "Managed and Portable Execution" to
  a separate section.
  I edited descriptions.
  I'm not sure about the task/job terminology distinction?
  NOWNOW-RND -->

MLRun has the following main components, which are usually grouped into **"projects"**:

- <a id="def-function"></a>**Function** &mdash; a software package with one or more methods and runtime-specific attributes (such as image, command, arguments, and environment).
    A function can run one or more runs or tasks, it can be created from templates, and it can be stored in a versioned database.
    <!-- SLSL:
    (a) I don't quite understand the runtime attributes part, therefore I only
        slightly rephrased it. NOWNOWNOW-RND
    (b) I believe the templates and versioning referred to the function
        and not the executed tasks/runs, and I rephrased accordingly. NOWNOW-RND
    -->
- <a id="def-task"></a>**Task** &mdash; defines the parameters, inputs, and outputs of a logical job or task to execute.
    A task can be created from a template, and can run over different runtimes or functions.
    <!-- SLSL: I changed "over" to "on" (especially after I saw that the Run
      description refers to running a task "on" a function), but I'm still not
      sure what this means, specifically with regards to running on a function?
      NOWNOWNOW-RND -->
- <a id="def-run"></a>**Run** &mdash; contains information about an execute task, including the execution status, the run results, and all attributes of the executed task.
    <!-- SLSL: Verify my editing, including removing the specific reference to
      running a task on a function.
      Do the task "attributes" include the inputs/parameter values?
      What about info about the location of run artifacts (not sure if it
      should be input/output artifacts because on 8.3.20 Yaron said that there
      were changes to related params/flags to refer to an artifacts path
      without an inputs/outputs distinction)?
      NOWNOWNOW-RND -->
- <a id="def-artifact"></a>**Artifact** &mdash; versioned data artifacts (such as files, objects, data sets, and models) that are produced or consumed by functions, runs, and workflows.
- <a id="def-workflow"></a>**Workflow** &mdash; defines a functions pipeline or directed acyclic graph (DAG) to execute using Kubeflow Pipelines.
  <!-- SLSL: Verify my editing, including that both "pipeline" and the DAG
    refer to functions. NOWNOWNOW-RND -->

<a id="managed-and-portable-execution"></a>
### Managed and Portable Execution

MLRun supports various types of **"runtimes"** &mdash; computation frameworks such as local, Kubernetes job, Dask, Nuclio, Spark, or MPI job (Horovod).
Runtimes may support parallelism and clustering to distribute the work among multiple workers (processes/containers).

The following code example creates a task that defines a run specification &mdash; including the run parameters, inputs, and secrets.
You run the task on a "job" function, and print the result output (in this case, the "model" artifact) or watch the run's progress.
For more information and examples, see the [**examples/mlrun_basics.ipynb**](examples/mlrun_basics.ipynb) notebook.
```python
# Create a task and set its attributes
task = NewTask(handler=handler, name='demo', params={'p1': 5})
task.with_secrets('file', 'secrets.txt').set_label('type', 'demo')

run = new_function(command='myfile.py', kind='job').run(task)
run.logs(watch=True)
run.show()
print(run.artifact('model'))
```

You can run the same [task](#def-task) on different functions &mdash; enabling code portability, re-use, and AutoML &mdash; and you can also use the same [function](#def-function) to run different tasks or parameter combinations with minimal coding effort.

Moving from local notebook execution to remote execution &mdash; such as running a container job, a scaled-out framework, or an automated workflow engine like Kubeflow &mdash; is seamless: just swap the runtime function or wire functions in a graph.
Continuous build integration and deployment (CI/CD) steps can also be configured as part of the workflow, using the `deploy_step` function method.
<!-- SLSL: I removed "; [see this tutorial for details]()" at the end of the
  first sentence, as it didn't link to any tutorial.
  I also made other edits, including changing "runtime/function" to "runtime
  function"?
  I'm not sure what "wire functions in a graph" means?
  "`.deploy_step()` function methods" > "`deploy_step` function method".
  NOWNOW-RND -->

Functions (function objects) can be created using one of three methods:

- **`new_function`** &mdash; creates a function "from scratch" or from another function.
- **`code_to_function`** &mdash; creates a function from source code, a source-code URL, or a web notebook.
- **`import_function`** &mdash; imports a function from a local or remote YAML function-configuration file or from a function object in the MLRun database (using a DB address of the format `db://<project>/<name>[:<tag>]`).
<!-- SLSL: Confirm my edits, specifically for `import_function`. (The original
  `import_function` doc was "functions are imported from a local/remote YAML
  file or from the function DB (prefix: `db://<project>/<name>[:tag]`)".)
  I thought of adding "[...]" at the end of the URL because of the use of the
  "prefix" terminology in the original doc, but from the uses and embedded and
  NB doc in the current code I don't see such paths? NOWNOW-RND -->

You can use the `save` function method to save a function object in the MLRun database, or the `export` method to save a YAML function-configuration function to your preferred local or remote location.
For function-method details and examples, see the embedded documentation/help text.
<!-- SLSL: Why don't we include the function methods in the generated reference?
  Is it because it's runtime-specific? (It's not a "module" ...?) NOWNOW-RND -->

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="auto-parameterization-artifact-tracking-n-logging"></a>
### Automated Parameterization, Artifact Tracking, and Logging

After running a job, you need to be able to track it, including viewing the run parameters, inputs, and outputs.
To support this, MLRun introduces a concept of a runtime **"context"**: the code can be set up to get parameters and inputs from the context, as well as log run outputs, artifacts, tags, and time-series metrics in the context.
<!-- SLSL: I replaced the terminology "ML context" with "runtime context",
  which was already used elsewhere. NOWNOW-RND -->

<a id="auto-parameterization-artifact-tracking-n-logging-example"></a>
#### Example
<!-- SLSL: I edited to create a single Example section and edit the texts.
  I also edited the code comments and some line breaks (for PEP8) and prepared
  similar updates in the code (in the mlrun/demo-xgboost repo) that will be
  committed in a separate PR. NOWNOW-RND -->

The following code example from the [**train-xgboost.ipynb**](https://github.com/mlrun/demo-xgb-project/blob/master/notebooks/train-xgboost.ipynb) notebook of the MLRun XGBoost demo (**demo-xgboost**) defines two functions:
the `iris_generator` function loads the Iris data set and saves it to the function's context object; the `xgb_train` function uses XGBoost to train an ML model on a data set and saves the log results in the function's context:

```python
import xgboost as xgb
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from mlrun.artifacts import TableArtifact, PlotArtifact
import pandas as pd


def iris_generator(context):
    iris = load_iris()
    iris_dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_labels = pd.DataFrame(data=iris.target, columns=['label'])
    iris_dataset = pd.concat([iris_dataset, iris_labels], axis=1)
    context.logger.info('Saving Iris data set to "{}"'.format(context.out_path))
    context.log_artifact(TableArtifact('iris_dataset', df=iris_dataset))


def xgb_train(context,
              dataset='',
              model_name='model.bst',
              max_depth=6,
              num_class=10,
              eta=0.2,
              gamma=0.1,
              steps=20):

    df = pd.read_csv(dataset)
    X = df.drop(['label'], axis=1)
    y = df['label']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    # Get parameters from event
    param = {"max_depth": max_depth,
             "eta": eta, "nthread": 4,
             "num_class": num_class,
             "gamma": gamma,
             "objective": "multi:softprob"}

    xgb_model = xgb.train(param, dtrain, steps)

    preds = xgb_model.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    context.log_result('accuracy', float(accuracy_score(Y_test, best_preds)))
    context.log_artifact('model', body=bytes(xgb_model.save_raw()),
                         local_path=model_name, labels={'framework': 'xgboost'})
```

The example training function can be executed locally with parameters, and the run results and artifacts can be logged automatically into a database by using a single command, as demonstrated in the following example; the example sets the function's `eta` parameter:
```python
train_run = new_function().run(handler=xgb_train).with_params(eta=0.3)
```

Alternatively, you can replace the function with a serverless runtime to run the same code on a remote cluster, which could result in a ~10x performance boost.
You can find examples for different runtimes &mdash; such as a Kubernetes job, Nuclio, Dask, Spark, or an MPI job &mdash; in the MLRun [**examples**](examples) directory.
<!-- SLSL: Edited. I added "remote" before "cluster"? NOWNOW-RND -->

If you run your code from the `main` function, you can get the runtime context by calling the `get_or_create_ctx` method, as demonstrated in the following code from the MLRun [**training.py**](examples/training.py) example application.
The code also demonstrates how you can use the context object to read and write execution metadata, parameters, secrets, inputs, and outputs:
<!-- SLSL: Edited.
  I replaced the ref to the horovod-training/py example with a ref to the
  training.py example, as the example is from the latter example.
  I made some edits to the example comments and line breaks (for PEP8) and
  prepared similar edits in the examples/training.py example, to be committed
  as part of a separate PR.
  NOTE: I removed the comment to add params or use default, as it wasn't
  followed by any example; instead, I edited the prints comment also to refer
  to accessing context parameter values (which are passed to the my_job()
  function from the runtime context as separate parameters in addition to a
  context parameter - see the `if __name__ == "__main__"`` portion of the
  training.py file. And I added a line break in the code for PEP8.
  I also added "TODO:" to the comment to run some code, and rephrased.
  NOWNOWNOW-RND -->

```python
from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact, TableArtifact


def my_job(context, p1=1, p2='x'):
    # Load the MLRun runtime context. The context is set by the runtime
    # framework - for example, Kubeflow.

    # Access runtime-context information - input metadata, parameter values,
    # authentication secret (access key), and input artifacts (files)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(context.get_input('infile.txt', 'infile.txt')
          .get()))

    # TODO: Run some useful code, such as ML training or data preparation.

    # Log scalar result values (job-result metrics)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)
    context.set_label('framework', 'sklearn')

    # Log various types of artifacts (file, web page, table), which will be
    # versioned and visible on the MLRun dashboard
    context.log_artifact('model', body=b'abc is 123', local_path='model.txt', labels={'framework': 'xgboost'})
    context.log_artifact('html_result', body=b'<b> Some HTML <b>', local_path='result.html')
    context.log_artifact(TableArtifact('dataset', '1,2,3\n4,5,6\n', visible=True,
                                        header=['A', 'B', 'C']), local_path='dataset.csv')

    # Create a chart output, which will be visible in the Kubeflow Pipelines UI
    chart = ChartArtifact('chart')
    chart.labels = {'type': 'roc'}
    chart.header = ['Epoch', 'Accuracy', 'Loss']
    for i in range(1, 8):
        chart.add_row([i, i/20+0.75, 0.30-i/20])
    context.log_artifact(chart)


if __name__ == "__main__":
    context = get_or_create_ctx('train')
    p1 = context.get_param('p1', 1)
    p2 = context.get_param('p2', 'a-string')
    my_job(context, p1, p2)
```

The example **training.py** application can be invoked as a local task, as demonstrated in the following code from the MLRun [**mlrun_basics.ipynb**](examples/mlrun_basics.ipynb) example notebook:
```python
run = run_local(task, command='training.py')
```
Alternatively, you can invoke the application by using the `mlrun` CLI; edit the parameters, inputs, and/or secret information, as needed:
```sh
mlrun run --name train -p p2=5 -i infile.txt=s3://my-bucket/infile.txt -s file=secrets.txt training.py
```

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="using-hyperparameters-for-job-scaling"></a>
### Using Hyperparameters for Job Scaling

Data science involves long computation times and data-intensive tasks.
To ensure efficiency and scalability, you need to implement parallelism whenever possible.
MLRun supports this by using two mechanisms:

1. Clustering &mdash; run the code on a distributed processing engine (such as Dask, Spark, or Horovod).
2. Load-balancing/partitioning &mdash; split (partition) the work across multiple workers.

MLRun functions and tasks can accept hyperparameters or parameter lists, deploy many parallel workers, and partition the work among the deployed workers.
The parallelism implementation is left to the runtime.
Each runtime may have its own method of concurrent tasks execution.
For example, the Nuclio serverless engine manages many micro threads in the same process, which can run multiple tasks in parallel.
In a containerized system like Kubernetes, you can launch multiple containers, each processing a different task.

MLRun supports parallelism.
For example, the following code demonstrates how to use hyperparameters to run the XGBoost model-training task from the example in the previous section (`xgb_train`) with different parameter combinations:
```python
    parameters = {
         "eta":       [0.05, 0.10, 0.20, 0.30],
         "max_depth": [3, 4, 5, 6, 8, 10],
         "gamma":     [0.0, 0.1, 0.2, 0.3],
         }

    task = NewTask(handler=xgb_train, out_path='/User/mlrun/data').with_hyper_params(parameters, 'max.accuracy')
    run = run_local(task)
```
<!-- SLSL: I would rename `with_hyper_params` to `with_hyperparams`, as we went
  with the "hyperparameters" spelling in the docs. NOWNOW-RND -->
<!-- SLSL: The example uses a hardcoded local platform `/User/...` path without
  mentioning this anywhere in the doc + it would have been better to use
  os.path.join(). mlrun_basics.ipynb defines an artifact_path variable - 
  `artifact_path = path.join(out, '{{run.uid}}')` - and then uses it in the
  NewTask command (`artifact_path=artifact_path` in the NewTask function call)
  and in the CLI equivalent (`{artifact_path}`).
  Also, the original README CLI equivalent below didn't set the out-path flag,
  even though it's supposedly an equivalent of the NewTask command above. => I
  added the out-path CLI option, similar to what's done in the
  mlrun_basics.ipynb example (except that it's done by using a variable).
  For now, I didn't change the output path in the doc or add explanations.
  NOWNOWNOW-RND -->

This code demonstrates how to instruct MLRun to run the same task while choosing the parameters from multiple lists (grid search).
MLRun then records all the runs, but marks only the run with minimal loss as the selected result.
For parallelism, it would be better to use runtimes like Dask, Nuclio, or jobs.

Alternatively, you can run a similar task (with hyperparameters) by using the MLRun CLI (`mlrun`):
```sh
mlrun run --name train_hyper -x p1="[3,7,5]" -x p2="[5,2,9]" --out-path '/User/mlrun/data' training.py
```

You can also use a parameters file if you want to control the parameter combinations or if the parameters are more complex.
The following code from the example **mlrun_basics** notebook demonstrates how to run a task that uses a CSV parameters file (**params.csv** in the current directory):
```python
    task = NewTask(handler=xgb_train).with_param_file('params.csv', 'max.accuracy')
    run = run_local(task)
```
<!-- SLSL: Is the `with_param_file()` `param_file` parameter a local path to
  a parameters file - in this example, a params.csv file in the current
  directory? Can the function also receive a URL for a remote params file?
  NOWNOW-RND -->

> **Note:** Parameter lists can be used in various ways.
> For example, you can pass multiple parameter files and use multiple workers to process the files simultaneously instead of one at a time.
<!-- SLSL: I rephrased, including to replace the "list" terminology, because I
  think the `NewTask` `with_param_file` method accepts the path/name of a single
  parameters file and the reference here is probably to calling this method
  multiple times (although I still don't know how you can configure different
  workers for each params file)? NOWNOW-RND -->

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="auto-code-deployment-n-containerization"></a>
### Automated Code Deployment and Containerization

MLRun adopts Nuclio serverless technologies for automatically packaging code and building containers.
This enables you to provide code with some package requirements and let MLRun build and deploy your software.

To build or deploy a function, all you need is to call the function's `deploy` method, which initiates a build or deployment job.
Deployment jobs can be incorporated in pipelines just like regular jobs (using the `deploy_step` method of the function or Kubernetes-job runtime), thus enabling full automation and CI/CD.
<!-- SLSL: I Added the reference to the function or Kubernetes-job runtime.
  `deploy_step` is defined as a `RemoteRuntime` method in
  mlrun/runtimes/function.py and as a `KubjobRuntime` method in
  mlrun/runtimes/kubejob.py. NOWNOW-RND -->

A functions can be built from source code or from a function specification, web notebook, Git repo, or TAR archive.

A function can also be built by using the `mlrun` CLI and providing it with the path to a YAML function-configuration file.
You can generate such a file by using the `to_yaml` or `export` function method.
For example, the following CLI code builds a function from a **function.yaml** file in the current directory:
```sh
mlrun build function.yaml
```
Following is an example **function.yaml** configuration file:
```yaml
kind: job
metadata:
  name: remote-git-test
  project: default
  tag: latest
spec:
  command: 'myfunc.py'
  args: []
  image_pull_policy: Always
  build:
    commands: ['pip install pandas']
    base_image: mlrun/mlrun:dev
    source: git://github.com/mlrun/ci-demo.git
```

For more examples of building and running functions remotely using the MLRun CLI (`mlrun`), see the [**remote**](examples/remote.md) example.

You can also convert your web notebook to a containerized job, as demonstrated in the following sample code; for a similar example with more details, see the [**mlrun_jobs.ipynb**](examples/mlrun_jobs.ipynb) example:

```python
# Create an ML function from the your notebook; use the `v3io` mount to attach
# the function to the data store of the Iguazio Data Science Platform
fn = code_to_function(kind='job').apply(mount_v3io())

# Prepare an image from the dependencies to allow updating the code and
# parameters per run without the need to build a new image
fn.build(image='mlrun/nuctest:latest')
```

[Back to top](#top)

<a id="run-ml-workflow-w-kubeflow-pipelines"></a>
### Running an ML Workflow with Kubeflow Pipelines

ML pipeline execution with MLRun is similar to CLI execution.
A pipeline is created by running an MLRun workflow.
MLRun automatically saves outputs and artifacts in a way that is visible to [Kubeflow Pipelines](https://github.com/kubeflow/pipelines), and allows interconnecting steps.

For an example of a full ML pipeline that's implemented in a web notebook, see the XGBoost MLRun demo ([**demo-xgb-project**](https://github.com/mlrun/demo-xgb-project)).
The  [**train-xgboost.ipynb**](https://github.com/mlrun/demo-xgb-project/blob/master/notebooks/train-xgboost.ipynb) demo notebook includes the following code for creating an XGBoost ML-training workflow pipeline:
```python
@dsl.pipeline(
    name='My XGBoost training pipeline',
    description='Shows how to use mlrun.'
)
def xgb_pipeline(
   eta = [0.1, 0.2, 0.3], gamma = [0.1, 0.2, 0.3]
):

    ingest = xgbfn.as_step(name='ingest_iris', handler='iris_generator',
                          outputs=['iris_dataset'])


    train = xgbfn.as_step(name='xgb_train', handler='xgb_train',
                          hyperparams = {'eta': eta, 'gamma': gamma},
                          selector='max.accuracy',
                          inputs = {'dataset': ingest.outputs['iris_dataset']},
                          outputs=['model'])


    plot = xgbfn.as_step(name='plot', handler='plot_iter',
                         inputs={'iterations': train.outputs['iteration_results']},
                         outputs=['iris_dataset'])

    # Deploy the model-serving function with inputs from the training stage
    deploy = srvfn.deploy_step(project = 'iris', models={'iris_v1': train.outputs['model']})
```

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="view-run-info-n-artifacts"></a>
### Viewing Run Information and Artifacts

When you configure an MLRun database, the results and artifacts from each run are recorded and can be viewed from the MLRun dashboard or by using various MLRun database methods from your code.
For more information, see [The MLRun Dashboard](#mlrun-ui) and [MLRun Database Methods](#mlrun-db-methods).

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="mlrun-ui"></a>
### The MLRun Dashboard

The MLRun dashboard is a graphical user interface (GUI) for working with MLRun.
<!-- SLSL: TODO: Add more info after I establish what can be done from the UI.
  Can they also run jobs, or just view results? Can they analyze results? etc.
-->

> **Note:** The UI requires an MLRun database/API service; see the Kubernetes YAML files in the [**hack**](hack) directory.
<!-- SLSL: How are the hack YAML files related to running an MLRun API service? -->

<br><p align="center"><img src="mlrunui.png" width="800"/></p><br>

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="mlrun-db-methods"></a>
### MLRun Database Methods

If you configured an MLRun database, you can use the `get_run_db` DB method to get an MLRun DB object.
Then, use the DB object's `connect` method to connect to the MLRun database or API service, and use additional methods to perform different operations &mdash; such as `list_artifacts` to list run artifacts or `del_runs` to delete completed runs.
For more information and examples, see the [**mlrun_db.ipynb**](examples/mlrun_db.ipynb) example notebook, which includes the following sample DB method calls:
```python
from mlrun import get_run_db

# Get an MLRun DB object and connect to an MLRun database/API service.
# Specify the DB path (for example, './' for the current directory) or
# the API URL ('http://mlrun-api:8080' for the default configuration).
db = get_run_db('./').connect()

# List all runs
db.list_runs('').show()

# List all artifacts for version 'latest' (default)
db.list_artifacts('', tag='').show()

# Check different artifact versions
db.list_artifacts('ch', tag='*').show()

# Delete completed runs
db.del_runs(state='completed')
```
<!-- SLSL: I edited the comments here and in examples/mlrun_db.ipynb.
  For the connect() method, the original README comment was
  "Connect to a local file DB" while the original example NB comment was
  "specify the DB path (use 'http://mlrun-api:8080' for api service)".
  I changed the NB comment to 
    "# Connect to an MLRun database/API service
    "# Specify the DB path (for example, './' for the current directory) or\n"
    "# the API URL ('http://mlrun-api:8080' for the default configuration).\n",
  In other examples, I see 'http://mlrun-api:8080' set as the API URL (e.g.,
  using `mlconf.dbpath = 'http://mlrun-api:8080'`), but not in mlrun_db.ipynb
  nor do I see a specific configuration file reference. I'm assuming there's a
  default configuration file/a default embedded basic configuration; all the
  mlrun configuration files seem to set "mlrun-api" as the name of the MLRun
  service and 8080 as the port number, although my assumption is that this is
  configurable (and the READMEs also refer to the option of changing the port
  number).
  Also, there seems to be many more "db" methods, although none of the methods
  are currently documented - see also the generated Python doc at
  https://mlrun.readthedocs.io/en/latest/mlrun.db.html. 
  NOWNOW-RND -->

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="additional-info-n-examples"></a>
### Additional Information and Examples

- [Replacing Runtime Context Parameters from the CLI](#replace-runtime-context-param-from-cli)
- [Using Remote Function Code](#using-remote-function-code)
  - [Function Deployment](#using-remote-function-code-function-deployment)
- [Running the MLRun Database/API Service](#run-mlrun-db-service)

<a id="replace-runtime-context-param-from-cli"></a>
#### Replacing Runtime Context Parameters from the CLI

```sh
mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt training.py
```

When running this sample command &mdash;
- The value of parameter `p1` is set to `5` (overwriting the current value).
- The file **infile.txt** is downloaded from a remote AWS S3 bucket.
- The credentials for the S3 downloaded are retrieved from the **secrets.txt** file.

<a id="using-remote-function-code"></a>
#### Using Remote Function Code

The same code can be implemented as a remote HTTP endpoint &mdash; for example, by using [Nuclio serverless functions](https://github.com/nuclio/nuclio).

For example, the same code can be wrapped within a Nuclio handler and executed remotely by using the same CLI command.

<a id="using-remote-function-code-function-deployment"></a>
##### Function Deployment

To deploy the function into a cluster, you can run the following commands.

> **Note: You must first install the [`nuclio-jupyter`](https://github.com/nuclio/nuclio-jupyter) package for using Nuclio from Jupyter Notebook.

```python
# Create the function from the notebook code and annotations; add volumes and
# an HTTP trigger with multiple workers for parallel execution
fn = code_to_function('xgb_train', runtime='nuclio:mlrun')
fn.apply(mount_v3io()).with_http(workers=32)

run = fn.run(task, handler='xgb_train')
```

To execute the code remotely, just substitute the file name with the function URL' replace `<function endpoint>` with your remote function endpoint:
<!-- SLSL: I added the replace text, but they would probably also need to
  replace the S3 bucket URL, file name, and parameter values, as indicated
  elsewhere. => TODO NOWNOW -->
```sh
mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt http://<function-endpoint>
```

[Back to top](#top) / [Back to quick-start TOC](#qs-tutorial)

<a id="run-mlrun-db-service"></a>
### Running an MLRun Database/API Service
<!-- SLSL: TODO: Use "MLRun service" and explain in relevant initial locations
  that it's a database and API service. -->

The MLRun database/API service includes an MLRun database and HTTP web API.
You can run an instance of this service in either of the following ways:
- [Using Docker](#run-mlrun-service-docker)
- [Using the MLRun CLI](#run-mlrun-service-cli)

<a id="run-mlrun-service-docker"></a>
#### Using Docker to Run an MLRun Service

Use the following command to run the MLRun database/API service using Docker; replace `<DB path>` with the path to the MLRun database"
```sh
docker run -p8080:8080 -v <DB path>:/mlrun/db
```
<!-- SLSL: NOWNOWNOW
  I added the preceding text and replaced `/path/to/db` in the command
  with <DB path>.
  (1) Should they always add `:/mlrun/db` to the path, as currently implied?
  (2) Can they also set a service URL rather than a local `/...` path?
  (3) We mention below the option of setting the `MLRUN_httpdb__port` envar to
      change the port. ->
      (a) Does the port number in the `docker` command have to be 8080 and can
          only be changed via the envar or is the above just an example?
      (b) If they can also set a different port number in the `docker` command,
          we need replace 8080 with a placeholder and edit the related text. =>
          Should both 8080 instances be replaced or just the second one because
          the port parameter is named `p8080`?
      (c) Can they also set the MLRun DB path/service URL by using the
          `MLRUN_httpdb__dirpath` environment variable instead of setting it
          in the `docker` command usihng the `-v` option?
-->

You can set the `MLRUN_httpdb__port` environment variable to change the port.

<a id="run-mlrun-service-cli"></a>
#### Using MLRun CLI to Run an MLRun Service

Use the `mlrun` CLI's `db` command to run an MLRun database/API service:
<!-- SLSL: NOWNOW
- Here and in most (if not all) CLI examples in the MLRun doc we use
  `<cmd> <options>`, but what seems to be the auto-generated CLI usage output
  that I saw in the examples/mlrun_basics.ipynb NB places [OPTIONS] first:
  `mlrun [OPTIONS] COMMAND [ARGS]...`. I suspect both variations work (this is
  also my recollection from the TSDB CLI). I consulted Haviv on Slack, but he
  hadn't replied. (8.3.20) I consulted Ilan and he agreed that the doc syntax
  of placing the command first seems to make more sense but said to confirm
  with Haviv. Note that Yasha also told me that this is the common syntax.
- Should we mention here and/or in the CLI help txt (in __main__.py) the option
  of setting the MLRUN_httpdb__port and/or MLRUN_httpdb__dirpath environment
  variables instead of using the command options / to set the default values?
  (The README already mentions the port envar in the Docker execution section.)
-->
```sh
mlrun db [OPTIONS]
```

To see the supported options, run `mlrun db --help`:
```
Options:
  -p, --port INTEGER  Port to listen on
  -d, --dirpath TEXT  Path to the MLRun DB/API service directory
```
<!-- SLSL: TODO: Confirm my CLI help-text changes in __main__.py and in the
  examples/mlrun_basics.ipynb CLI cell output with the help text.
  `dirpath` changed from "api/db path".
  NOWNOW -->

