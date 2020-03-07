# MLRun

[![CircleCI](https://circleci.com/gh/mlrun/mlrun/tree/development.svg?style=svg)](https://circleci.com/gh/mlrun/mlrun/tree/development)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version fury.io](https://badge.fury.io/py/mlrun.svg)](https://pypi.python.org/pypi/mlrun/)
[![Documentation](https://readthedocs.org/projects/mlrun/badge/?version=latest)](https://mlrun.readthedocs.io/en/latest/?badge=latest)
<!-- SLSL TODO: Split into multiple pages in a separate commit. -->

MLRun is a generic and convenient mechanism for data scientists and software developers to describe and run tasks related to machine learning (ML) in various scalable runtime environments and ML pipelines while automatically tracking executed code, metadata, inputs, and outputs of.
MLRun integrates with the [Nuclio](https://nuclio.io/) serverless project and with [Kubeflow Pipelines](https://github.com/kubeflow/pipelines).

#### In This Document
- [General Concept and Motivation](#concepts-n-motvation)
- [Architecture and Quick-Start Tutorial](#arch-n-qs-tutorial)
  - [Managed and Portable Execution ](#managed-and-portable-execution)
  - [Using Hyperparameters for Job Scaling](#using-hyperparameters-for-job-scaling)
  - [Automated Parameterization, Artifact Tracking, and Logging](#auto-parameterization-artifact-tracking-n-logging)
  - [Automated Code Deployment and Containerization](#auto-code-deployment-n-containerization)
  - [Running an MLRun ML Pipeline with Kubeflow Pipelines](#run-mlrun-ml-pipeline-w-kubeflow-pipelines)
  - [Viewing the Run Results](#view-run-results)
- [The MLRun Dashboard](#mlrun-ui)
- [Additional Information and Examples](#additional-info-n-examples)
  - [Replacing Runtime Context Parameters from the CLI](#replace-runtime-context-param-from-cli)
  - [Using Remote Function Code](#using-remote-function-code)
- [Running the MLRun Database/API Service](#run-mlrun-db-service)

<a id="concepts-n-motvation"></a>
## General Concept and Motivation
## [Installation](#installation)

A developer or data-scientist writes code in a local IDE or notebook, then he would like to run the same code on a larger cluster using scale-out containers or functions, once the code is ready he or another developer need to transfer the code into an automated ML workflow (for example, using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-quickstart/)), add logging, monitoring, security, etc.

The various environments ("runtimes") use different configurations, parameters, and data sources.
In addition, different frameworks and platforms are used to focus on different stages in the life cycle.
This leads to constant development and DevOps/MLOps work.

As your project scales, you need greater computation power or GPUs, and you need to access large-scale data sets, this cant work on laptops, you need a way to seamlessly run your code on a remote cluster and automatically scale it out.

When running experiments, you should ideally be able to record/version all the code, configuration, outputs, and associated inputs (lineage), so you can easily reproduce or explain your results.
The fact that you use different forms of storage (files, S3, etc.) and databases doesn't make it easy.

Many of those code and ML functions can be reused across projects and companies.
Having a function marketplace that comprises highly tuned open-source templates alongside your internally developed functions can further accelerate your work.

Wouldn't it be great if you could write the code once, using simple "local" semantics, and then run it as-is on various platforms?
Imagine a layer that automates the build process, execution, data movement, scaling, versioning, parameterization, outputs tracking, etc.
A world of easily developed, published or consumed data/ML "functions" that can be used to form complex and large scale ML pipelines.

<b>This is the goal of this package!</b>

The code is in early development stages and provided as a reference.
The hope is to foster wide industry collaboration and make all the resources pluggable, so that developers can code to one API and use various open-source projects or commercial products.

<a id="installation"></a>
## Installation

Run `pip install mlrun` to get the library and CLI.

MLRun requires two containers (for the API and the dashboard), you can also use the pre-baked Jupyter lab image.

To run MLRun using Docker or Kubernetes, see the related [instructions page](hack/local/README.md).

For installation on Iguazio Data Science Platform clusters, use [this YAML file](hack/mlrun-all.yaml); remember to set the access-key and default registry URL for your cluster.
For example:

```
curl -O https://raw.githubusercontent.com/mlrun/mlrun/master/hack/mlrun-all.yaml
# as a minimum replace the <set default registry url> and <access-key> with real values
# in iguazio cloud the default registry url is: docker-registry.default-tenant.<cluster-dns>:80
# Note: must suffix :80 for local registry!!! (docker defaults to another port)

kubectl apply -n <namespace> -f <updated-yaml-file>
```

#### Examples and Notebooks
- [Learn MLRun basics](examples/mlrun_basics.ipynb)
- [From local runs to Kubernetes jobs, and automated pipelines in a single Notebook](examples/mlrun_jobs.ipynb)
- [Create an end to end XGBoost pipeline: ingest, train, verify, deploy](https://github.com/mlrun/demo-xgb-project)
- Examples for MLRun with scale-out runtimes
  * [Distributed TensorFlow (Horovod and MpiJob)](examples/mlrun_mpijob_classify.ipynb)
  * [Nuclio-serving (Serverless model serving)](examples/xgb_serving.ipynb)
  * [Dask](examples/mlrun_dask.ipynb)
  * [Spark](examples/mlrun_sparkk8s.ipynb)
- MLRun Projects
  * [Load a project from remote Git and run pipelines](examples/load-project.ipynb)
  * [Create a new project + functions + pipelines and upload to Git](examples/new-project.ipynb)
- [Importing and exporting functions using files or git](examples/mlrun_export_import.ipynb)
- [Query the MLRun DB](examples/mlrun_db.ipynb)

#### Additional Examples

- Complete demos can be found in [mlrun/demos repo](https://github.com/mlrun/demos)
  * [Deep learning pipeline](https://github.com/mlrun/demos/blob/master/image_classification/README.md) (data collection, labeling, training, serving + automated workflow)
- MLRun Functions Library (work in progress) is in [mlrun/functions repo](https://github.com/mlrun/functions)

<a id="arch-n-qs-tutorial"></a>
## Architecture and Quick-Start Tutorial

- [Managed and Portable Execution ](#managed-and-portable-execution)
- [Automated Code Deployment and Containerization](#auto-parameterization-artifact-tracking-n-logging)
- [Using Hyperparameters for Job Scaling](#using-hyperparameters-for-job-scaling)
- [Automated Code Deployment and Containerization](#auto-code-deployment-n-containerization)
- [Build and run function from a remote IDE using the CLI](examples/remote.md)
- [Running an MLRun ML Pipeline with Kubeflow Pipelines](#run-mlrun-ml-pipeline-w-kubeflow-pipelines)
- [Viewing the Run Results](#view-run-results)
  - [Using the MLRun Dashboard (UI)](#get-run-results-mlrun-ui)
  - [Using the db Method](#get-run-results-db-medhod)

<a id="managed-and-portable-execution"></a>
### Managed and Portable Execution

MLRun has a few main components, which are usually grouped into "projects":

- **Function** &mdash; a software package with one or more methods and a bunch of `runtime` specific attributes (for example, image, command, args, environment, ...). function can run one or many runs/tasks, they can be created from templates, and be stored in a versioned database.
- **Task** &mdash; define the desired parameters, inputs, outputs of a job/task.
Task can be created from a template and run over different `runtimes` or `functions`.
- **Run** &mdash; is the result of running a Task on a Function, it has all the attributes of a Task + the execution status and results.
- **Artifact** &mdash; versioned files, objects, data sets, models, etc. which are produced or consumed by functions/runs/workflows.
- **Workflow** &mdash; defines a pipeline/graph (DAG) of functions (using Kubeflow Pipelines)

MLRun support multiple "runtimes" &mdash; computation frameworks &mdash; such as local, Kubernetes job, Dask, Nuclio, Spark, or MPI job (Horovod).
Runtimes may support parallelism and clustering to distribute the work among processes/containers.

Example
```python
# Create a task and set its attributes
task = NewTask(handler=handler, name='demo', params={'p1': 5})
task.with_secrets('file', 'secrets.txt').set_label('type', 'demo')

run = new_function(command='myfile.py', kind='job').run(task)
run.logs(watch=True)
run.show()
print(run.artifact('model'))
```

In this example, the task defines your run spec (parameters, inputs, secrets, etc.).
You run the task on a **"job"** function, and print the result output (in this case, the **"model"** artifact) or watch the progress of that run.
See the [docs and example notebook](examples/mlrun_basics.ipynb).

You can run the same **"task"** on different functions &mdash; enabling code portability, re-use, and AutoML &mdash; or you can use the same **"function"** to run different tasks or parameter combinations with minimal coding effort.

Moving from run on a local notebook, to running in a container job, a scaled-out framework, or an automated workflow engine like Kubeflow is seamless, just swap the runtime/function or wire functions in a graph; [see this tutorial for details]().
CI/CD steps (build, deploy) can also be specified as part of the workflow (using `.deploy_step()` function methods).

Functions can be created using one of three methods:

- `new_function` &mdash; create a function object from scratch or another function.
- `code_to_function` &mdash; functions are created from source code, source URL or notebook.
- `import_function` &mdash; functions are imported from a local/remote YAML file or from the function DB (prefix: `db://<project>/<name>[:<tag>]`).

`function.save(tag="")` (store in db) and `function.export(target-path)` (store yaml) can be used to save functions.

See each function doc/help and examples for details.

<a id="auto-parameterization-artifact-tracking-n-logging"></a>
### Automated Parameterization, Artifact Tracking, and Logging

After running a job, you need to track the run, it inputs, parameters, and outputs.
MLRun introduces a concept of an ML **"context"**: the code can be instrumented to get parameters and inputs from the context, as well as log outputs, artifacts, tags, and time-series metrics.

<b>Example XGBoost training function</b>

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
    context.logger.info('saving iris dataframe to {}'.format(context.out_path))
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

This example function can be executed locally with parameters (for example, `eta` or `gamma`).
The results and artifacts can be logged automatically into a database by using a single command:
```
train_run = new_function().run(handler=xgb_train).with_params(eta=0.3)
```

You can swap the function with a serverless runtime and the same will run on a cluster.<br>
This can result in 10x performance boost.

The [**examples**](examples) directory contains more examples, using different runtimes &mdash; such as a Kubernetes job, Nuclio, Dask, Spark, or an MPI job.

If your run your code from `main`, you can get the runtime context by calling the `get_or_create_ctx` method.

The following example demonstrates how you can use the context object in various ways to read and write metadata, secrets, inputs, or outputs.
For more details, see the [**horovod-training.py**](examples/horovod-training.py) example.

<b>Example: obtaining and using the context object</b>

```python
from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact, TableArtifact


def my_job(context, p1=1, p2='x'):
    # Load the MLRun runtime context; the context is set by the runtime
    # framework - for example, Kubeflow

    # Get parameters from the runtime context (or use defaults)

    # Access input metadata, values, files, and secrets (passwords)
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(context.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(context.get_input('infile.txt', 'infile.txt').get()))

    # Run some useful code - for example, ML training or data preparation

    # Log scalar result values (job-result metrics)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)
    context.set_label('framework', 'sklearn')

    # Log various types of artifacts (file, web page, table), which will be
    # versioned and visible in the MLRun dashboard
    context.log_artifact('model', body=b'abc is 123', local_path='model.txt', labels={'framework': 'xgboost'})
    context.log_artifact('html_result', body=b'<b> Some HTML <b>', local_path='result.html')
    context.log_artifact(TableArtifact('dataset', '1,2,3\n4,5,6\n', visible=True,
                                        header=['A', 'B', 'C']), local_path='dataset.csv')

    # Create a chart output, which will be visible in the pipelines UI
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

This code sample can be invoked with the following code:
```python
run = run_local(task, command='training.py')
```
Alternatively, it can be invoked by using the `mlrun` CLI; edit the parameters and the S3 path in the input data, as needed command, and ensure that there's a **secrets.txt** file with the required S3 download credentials:
```sh
mlrun run --name train -p p2=5 -i infile.txt=s3://my-bucket/infile.txt -s file=secrets.txt training.py
```

<a id="using-hyperparameters-for-job-scaling"></a>
### Using Hyperparameters for Job Scaling

Data science involves long-running compute and data-intensive tasks.
To gain efficiency, you need to implement parallelism whenever possible.
MLRun delivers scalability using two mechanisms:

1. Clustering &mdash; run the code on a distributed processing engined (such as Dask, Spark, or Horovod).
2. Load-balancing/partitioning &mdash; partition the work to multiple workers.

MLRun can accept hyperparameters or parameter lists, deploy many parallel workers, and partition the work among those.
The parallelism implementation is left to the runtime.
Each runtime may have its own way of running tasks concurrently.
For example, the Nuclio serverless engine manages many micro threads in the same process, which can run multiple tasks in parallel.
In a containerized system like Kubernetes, you can launch multiple containers each processing a different task.

MLRun supports parallelism.
You can run many parameter combinations for the previous `xgboost` function by using hyperparameters:

```python
    parameters = {
         "eta":       [0.05, 0.10, 0.20, 0.30],
         "max_depth": [3, 4, 5, 6, 8, 10],
         "gamma":     [0.0, 0.1, 0.2, 0.3],
         }

    task = NewTask(handler=xgb_train, out_path='/User/mlrun/data').with_hyper_params(parameters, 'max.accuracy')
    run = run_local(task)
```

This code demonstrates how to tell MLRun to run the same task while choosing the parameters from multiple lists (grid search).
MLRun will record all the runs, but mark only the run with minimal loss as the selected result.
For parallelism, it would be better to use runtimes like Dask, Nuclio, or jobs.

The same logic and also be executed by using the MLRun CLI (`mlrun`):
```sh
mlrun run --name train_hyper -x p1="[3,7,5]" -x p2="[5,2,9]" training.py
```

You can use a parameters file if you want to control the parameter combinations or if the parameters are more complex.
The following code demonstrates how to configure a CSV parameters file:

```python
    task = NewTask(handler=xgb_train).with_param_file('params.csv', 'max.accuracy')
    run = run_local(task)
```

> **Note:** Parameter lists can be used for various tasks.
> Another example is to pass a list of files and have multiple workers process the files simultaneously instead of one at a time.

<a id="auto-code-deployment-n-containerization"></a>
### Automated Code Deployment and Containerization

MLRun adopts Nuclio serverless technologies for automatically packaging code and building containers.
This way, you can specify code with some package requirements and let the system build and deploy your software.

Building and deploying a function is as easy as typing `function.deploy(...)`.
This initiates a build or deployment job.
Deployment jobs can be incorporated in pipelines just like regular jobs (using the `.deploy_step` method), enabling full automation and CI/CD.

Functions can be built from source code, function specs, notebooks, Git repos, or TAR archives.

Build can also be done using the `mlrun` CLI, by providing the CLI with the path to a YAML function configuration file; you can generate such a file by using the `function.to_yaml` or `function.export` methods.

For example, the following CLI code refers to a **function.yaml** file:
```sh
mlrun build function.yaml
```

Following is an example **function.yaml** file:
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

For more examples of building and running functions from remotely using the MLRun CLI, see the [**remote**](examples/remote.md) example.

You can also convert your notebook into a containerized job, as demonstrated in the following sample code.
For details, see the [**mlrun_jobs.ipynb**](examples/mlrun_jobs.ipynb) example.

```python
# Create an ML function from the notebook; store data in the data store of the
# Iguazio Data Science Platform (v3io mount)
fn = code_to_function(kind='job').apply(mount_v3io())

# Prepare an image from the dependencies, to avoid building it on each run
fn.build(image='mlrun/nuctest:latest')
```

<a id="run-mlrun-ml-pipeline-w-kubeflow-pipelines"></a>
## Running an MLRun ML Pipeline with Kubeflow Pipelines

ML pipeline execution in MLRun is similar to CLI execution.
MLRun automatically saves outputs and artifacts in a way that is visible to [Kubeflow Pipelines](https://github.com/kubeflow/pipelines), and allows interconnecting steps.

For an example of a full ML pipeline that's implemented in a Jupyter notebook, see the MLRun [demo-xgb-project](https://github.com/mlrun/demo-xgb-project) repository.
The  [**train-xgboost.ipynb**](https://github.com/mlrun/demo-xgb-project/blob/master/notebooks/train-xgboost.ipynb) demo notebook includes the following code:
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

<a id="view-run-results"></a>
### Viewing the Run Results

You can view MLRun execution results from the graphical [MLRun dashboard](#mlrun-ui).

If you configured an MLRun database (`rundb`), the results and artifacts from each run are recorded.

You can use various `db` methods; see the [example notebook](examples/mlrun_db.ipynb).

```python
from mlrun import get_run_db

# Connect to a local file DB
db = get_run_db('./').connect()

# List all runs
db.list_runs('').show()

# List all artifact for version "latest"
db.list_artifacts('', tag='').show()

# Check different artifact versions
db.list_artifacts('ch', tag='*').show()

# Delete completed runs
db.del_runs(state='completed')
```

<a id="mlrun-ui"></a>
## The MLRun Dashboard

The MLRun dashboard is a graphical user interface (GUI) for working with MLRun.
<!-- SLSL: TODO: Add more info after I establish what can be done from the UI.
  Can they also run jobs, or just view results? Can they analyze results? etc.
-->

> **Note:** The UI requires an MLRun database/API service; see the Kubernetes YAML files in the [**hack**](hack) directory.
<!-- SLSL: How are the hack YAML files related to running an MLRun API service? -->

<br><p align="center"><img src="mlrunui.png" width="800"/></p><br>

<a id="additional-info-n-examples"></a>
## Additional Information and Examples

- [Replacing Runtime Context Parameters from the CLI](#replace-runtime-context-param-from-cli)
- [Using Remote Function Code](#using-remote-function-code)
  - [Function Deployment](#using-remote-function-code-function-deployment)
- [Running the MLRun Database/API Service](#run-mlrun-db-service)

<a id="replace-runtime-context-param-from-cli"></a>
### Replacing Runtime Context Parameters from the CLI

```sh
python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt training.py
```

When running this sample command &mdash;
- The value of parameter `p1` is set to `5` (overwriting the current value).
- The file **infile.txt** is downloaded from a remote AWS S3 bucket.
- The credentials for the S3 downloaded are retrieved from the **secrets.txt** file.

<a id="using-remote-function-code"></a>
### Using Remote Function Code

The same code can be implemented as a remote HTTP endpoint &mdash; for example, by using [Nuclio serverless functions](https://github.com/nuclio/nuclio).

For example, the same code can be wrapped within a Nuclio handler and executed remotely by using the same CLI command.

<a id="using-remote-function-code-function-deployment"></a>
#### Function Deployment

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
python -m mlrun run -p p1=5 -s file=secrets.txt -i infile.txt=s3://mybucket/infile.txt http://<function-endpoint>
```

<a id="run-mlrun-db-service"></a>
### Running the MLRun Database/API Service

The MLRun database/API service includes the MLRun database and HTTP web API.
You can run this service in either of the following ways:
- Using [Docker](#run-mlrun-service-docker)
- Using the [CLI](#run-mlrun-service-cli)

<a id="run-mlrun-service-docker"></a>
#### Docker

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
#### Command line

Use the `mlrun` CLI's `db` command to run an MLRun database/API service:
<!-- SLSL: NOWNOW
- I switched the order to place the options before the command, per
  the CLI usage instructions that I saw in a cell output in the examples/
  mlrun_basics.ipynb NB, which also makes sense for CLI syntax, even though I
  suspect that both variations work (like in the TSDB CLI).
  (Yasha though it should be `mlrun <command> [OPTIONS]`.)
- Should we mention here and/or in the CLI help txt (in __main__.py) the option
  of setting the MLRUN_httpdb__port and/or MLRUN_httpdb__dirpath environment
  variables instead of using the command options / to set the default values?
  (The README already mentions the port envar in the Docker execution section.)
-->
```sh
mlrun [OPTIONS] db
```

To see the supported options, run `mlrun --help db`:
```
Options:
  -p, --port INTEGER  Port to listen on
  -d, --dirpath TEXT  Path to the MLRun DB/API service directory
```
<!-- SLSL: TODO: Confirm my CLI help-text changes in __main__.py and in the
  examples/mlrun_basics.ipynb CLI cell output with the help text.
  `dirpath` changed from "api/db path".
  NOWNOW -->

