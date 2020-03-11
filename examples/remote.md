# Using MLRun from a Remote Client

This tutorial explains how to use MLRun from a local development environment (IDE) to run jobs on a remote cluster.

#### In This Document

- [Prerequisites](#prerequisites)
- [CLI Commands](#cli-commands)
  - [The `build` Command](#cli-cmd-build)
  - [The `run` Command](#cli-cmd-run)
- [Building and Running a Function from a Git Repository](#git-func)
- [Using a Sources Archive](#sources-archive)

<a id="prerequisites"></a>
## Prerequisites

Before you begin, do the following:

1. Install MLRun locally.
    You can do this by running the following from a command line:
    ```sh
    pip install mlrun
    ```
2. Ensure that you have remote access to your MLRun service (i.e., to the service's NodePort on the remote Kubernetes cluster).
3. Set environment variables to define your MLRun configuration.
    As a minimum requirement &mdash;

    - Set `MLRUN_DBPATH` to the URL of the remote MLRun database/API service; replace the `<...>` placeholders to identify your remote target:
      ```sh
      MLRUN_DBPATH=http://<cluster IP>:<port>
      ```
    - If the remote service is on an instance of the Iguazio Data Science Platform (**"the platform"**), set the following environment variables as well; replace the `<...>` placeholders with the information for your specific platform cluster:
      ```sh
      V3IO_USERNAME=<username of the remote MLRun service owner>
      V3IO_API=<API endpoint of the web-APIs service endpoint; e.g., "webapi.default-tenant.app.mycluster.iguazio.com">
      V3IO_ACCESS_KEY=<access key>
      ```

<a id="cli-commands"></a>
## CLI Commands

Use the following MLRun CLI (`mlrun`) commands to build and run MLRun functions:

- [`build`](#cli-cmd-build)
- [`run`](#cli-cmd-run)

<a id="cli-cmd-build"></a>
### The `build` Command

Use the `build` CLI command to build all the function dependencies from the function specification into a function container (Docker image).
This command supports many options, including the following; for the full list, run `mlrun build --help`:

```sh
  --name TEXT            Function name
  --project TEXT         Project name
  --tag TEXT             Function tag
  -i, --image TEXT       Target image path
  -s, --source TEXT      Path/URL of the function source code - a PY file, or a directory to archive
                         when using the -a|--archive option (default: './')
  -b, --base-image TEXT  Base docker image
  -c, --command TEXT     Build commands; for example, '-c pip install pandas'
  --secret-name TEXT     Name of a container-registry secret
  -a, --archive TEXT     Path to a TAR archive file to create from the function sources (see -s|--source)
                         and extract to the function container during the build
  --silent               Don't show build logs
  --with-mlrun           Add the MLRun package ('mlrun')
```

> **Note:** For information about using the `-a|--archive` option to create a function-sources archive, see [Using a Sources Archive](#sources-archive) later in this tutorial.

<a id="cli-cmd-run"></a>
### The `run` Command

Use the `run` CLI command to execute a task by using a local or remote function.
This command supports many options, including the following; for the full list, run `mlrun run --help`:

```sh
  -p, --param key=val    Parameter name and value tuples; for example, `-p x=37 -p y='text'`
  -i, --inputs key=path  Path/URL for getting input artifacts
  --in-path TEXT         Default directory path/URL for retrieving input artifacts
  --out-path TEXT        Default directory path/URL for storing output artifacts
  -s, --secrets TEXT     Secrets, either as `file=<filename>` or `env=<ENVAR>,...`
  --name TEXT            Run name
  --project TEXT         Project name or ID
  -f, --func-url TEXT    Path/URL of a YAML function-configuration file, or db://<project>/<name>[:tag] for a DB function object
  --task TEXT            Path/URL of a YAML task-configuration file
  --handler TEXT         Invoke the function handler inside the code file
```

<a id="git-func"></a>
## Building and Running a Function from a Git Repository

To build and run a function from a Git repository, start out by adding a YAML function-configuration file in your local environment; this file should describe the function and define its specification.
For example, create a **myfunc.yaml** file with the following content in your working directory:
```yaml
kind: job
metadata:
  name: remote-demo1
  project: ''
spec:
  command: 'examples/training.py'
  args: []
  image: .mlrun/func-default-remote-demo-ps-latest
  image_pull_policy: Always
  build:
    #commands: ['pip install pandas']
    base_image: mlrun/mlrun:dev
    source: git://github.com/mlrun/mlrun
```

Then, run the following CLI command from the command line and pass the path to your local function-configuration file as an argument to build the function's container image according to the configured requirements.
For example, the following command builds the function using the **myfunc.yaml** file from the current directory:
```sh
mlrun build myfunc.yaml
```

When the build completes, you can use the `run` CLI command to run the function.
Set the `-f` option to the path to the local function-configuration file and pass relevant parameters.
For example:
```sh
mlrun run -f myfunc.yaml -w -p p1=3
```

You can also try the following function-configuration example, which is based on the MLRun CI demo:
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

<a id="sources-archive"></a>
## Using a Sources Archive

The `-a|--archive` option of the CLI [`build`](#cli-cmd-build) command enables you to define a remote object-path location &mdash; such as an AWS S3 bucket or an Iguazio Data Science Platform ("platform") data path (using the v3io mount) &mdash; for storing TAR archive files with all the required code dependencies.
You can also set such an archive path by using the `MLRUN_DEFAULT_ARCHIVE` environment variable.
When an archive path is provided, the remote builder extracts (untars) all of the archive files into the working directory of the function container.
<!-- [IntInfo] MLRUN_DEFAULT_ARCHIVE is referenced in the code using
  `mlconf.default_archive` when using `from .config import config as mlconf`.
-->

To use this option, first create a local function-configuration file.
For example, you can create a **function.yaml** file in your working directory with the following content; the specification describes the environment to use, defines a Python base image, adds several packages, and defines **examples/training.py** as the application to execute on `run` commands:
```yaml
kind: job
metadata:
  name: remote-demo4
  project: ''
spec:
  command: 'examples/training.py'
  args: []
  image_pull_policy: Always
  build:
    commands: ['pip install mlrun pandas']
    base_image: python:3.6-jessie
```

Next, run the following MLRun CLI command to build the function; replace the `<...>` placeholders to match your configuration:
```sh
mlrun build <path to function-configuration file> -a <path to archive> [-s <path/URL of function-code sources>]
```
> **Note:**
> - `.` is a shorthand for a **function.yaml** configuration file in the local working directory.
> - The `-a|--archive` option is used to instruct MLRun to create an archive file from the function-code sources at the location specified by the `-s|--sources` option.
> The default sources location is the local working directory (`./`).
For example, the following command sets the target archive path to `v3io:///users/$V3IO_USERNAME/tars` &mdash; a **tars** directory within the MLRun user directory (`$V3IO_USERNAME`) in the "users" data container of a platform cluster that's accessed via the `v3io` data mount:

```sh
mlrun build . -a v3io:///users/$V3IO_USERNAME/tars
```

After the function build completes, you can run the function with some parameters.
For example:
```sh
mlrun run -f . -w -p p1=3
```

