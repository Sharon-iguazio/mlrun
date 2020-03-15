#!/usr/bin/env python

# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
from ast import literal_eval
from base64 import b64decode, b64encode
from os import environ, path
from pprint import pprint
from subprocess import Popen
from sys import executable

import click
import yaml

from tabulate import tabulate

from mlrun import load_project
from . import get_version
from .config import config as mlconf
from .builder import upload_tarball
from .datastore import get_object
from .db import get_run_db
from .k8s_utils import K8sHelper
from .model import RunTemplate
from .run import new_function, import_function_to_dict, import_function
from .runtimes import RemoteRuntime, RunError
from .utils import (list2dict, logger, run_keys, update_in, get_in,
                    parse_function_uri, dict_to_yaml)


@click.group()
def main():
    pass


# `run` Command
@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("url", type=str, required=False)
@click.option('--param', '-p', default='', multiple=True,
              help="Parameter name and value tuples; for example,"
              "`-p x=37 -p y='text'`")
@click.option('--inputs', '-i', multiple=True,
              help="Input artifact of the format <name>='<path>'; for example, '
              '`-i infile.txt=s3://mybucket/infile.txt`;"
              "                       if the path string is empty, the "
              "default input path is used (see --in-path)")
@click.option('--outputs', '-o', multiple=True,
              help='Output artifact (Kubeflow Pipelines run result)')
              # SLSL: Rephrased. I wanted to add an example like I did for
              # --inputs/-i, but I didn't find any in the mlrun repo.
              # When is the user expected to pass an output artifact to the CLI?
              # (Note that this wasn't included in the main CLI command examples
              # in examples/remote.md, unlike the inputs flag.) NOWNOW-RND
@click.option('--in-path',
              help='Base directory path/URL for storing input artifacts')
@click.option('--out-path',
              help='Base directory path/URL for storing output artifacts')
              # SLSL: I rephrased the in-path and out-path flags doc from
              # "default input path/url (prefix) for artifact" and a similar
              # output doc. I understood from Yaron that "prefix" refers to
              # this being the root directory path for storing artifacts.
              # I wasn't sure about the "default" indication, but it didn't seem
              # to me that the user can override it after setting it in the CLI
              # command and I suspect it was mean to convey a "base" dir path as
              # well, so I removed it in my version. TODO: Verify. NOWNOW-RND
              # Remember to make any edits also in examples/remote.md.
@click.option('--secrets', '-s', multiple=True,
              help='Secrets, either as `file=<filename>` or `env=<ENVAR>,...`;'
              ' for example, `-s file=secrets.txt`')
              # SLSL: Edited - confirm. Secrets for what? NOWNOW-RND
              # (I asked Haviv about this on Slack on 8.3.20.)
              # Remember to also edit in examples/remote.md.
@click.option('--uid',
              help='Unique run ID')
@click.option('--name',
              help='Run name')
              # SLSL: The original `run` help output in examples/remote.md as
              # "optional run name", but there was no optional indication here
              # and as I think the correct way to mark optional options is with
              # [Optional] at the start, as we did elsewhere, but we don't do
              # this for other optional `run` options, I didn't add it.
              # NOWNOW-RND
@click.option('--workflow',
              help='Workflow name or ID')
@click.option('--project',
              help='Project name or ID')
@click.option('--db', default='',
              help='DB path or DB/API service URL for saving run information')
@click.option('--runtime', '-r', default='',
              help='Function-spec dictionary, for pipeline usage')
@click.option('--kfp', is_flag=True,
              help='Running inside Kubeflow Pipelines; DO NOT USE')
@click.option('--hyperparam', '-x', default='', multiple=True,
              help='Hyperparameters (will expand to multiple tasks); '
              'for example, `--hyperparam p2=[1,2,3]`')
@click.option('--param-file', default='',
              help='Path to a CSV run-parameters/hyperparameters file')
@click.option('--selector', default='',
              help='How to select the best result from a list; for example, '
              'max.accuracy')
@click.option('--func-url', '-f', default='',
              help='Path/URL of a YAML function-configuration file, or '
              'db://<project>/<name>[:tag] for a DB function object')
@click.option('--task', default='',
              help='Path/URL of a YAML task-configuration file')
@click.option('--handler', default='',
              help='Invoke the function handler inside the code file')
@click.option('--mode',
              help='Special run mode: "noctx" | "pass"')
@click.option('--schedule', help='cron schedule')
@click.option('--from-env', is_flag=True, help='read the spec from the env var')
@click.option('--dump', is_flag=True, help='dump run results as YAML')
@click.option('--image', default='', help='container image')
@click.option('--workdir', default='', help='run working directory')
@click.option('--watch', '-w', is_flag=True, help='watch/tail run log')
@click.argument('run_args', nargs=-1, type=click.UNPROCESSED)
def run(url, param, inputs, outputs, in_path, out_path, secrets, uid,
        name, workflow, project, db, runtime, kfp, hyperparam, param_file,
        selector, func_url, task, handler, mode, schedule, from_env, dump,
        image, workdir, watch, run_args):
    """Executes a task and injects task parameters."""

    out_path = out_path or environ.get('MLRUN_ARTIFACT_PATH')
    config = environ.get('MLRUN_EXEC_CONFIG')
    if from_env and config:
        config = json.loads(config)
        runobj = RunTemplate.from_dict(config)
    elif task:
        obj = get_object(task)
        task = yaml.load(obj, Loader=yaml.FullLoader)
        runobj = RunTemplate.from_dict(task)
    else:
        runobj = RunTemplate()

    set_item(runobj.metadata, uid, 'uid')
    set_item(runobj.metadata, name, 'name')
    set_item(runobj.metadata, project, 'project')

    if workflow:
        runobj.metadata.labels['workflow'] = workflow

    if db:
        mlconf.dbpath = db

    if func_url:
        if func_url.startswith('db://'):
            func_url = func_url[5:]
            project, name, tag = parse_function_uri(func_url)
            mldb = get_run_db(mlconf.dbpath).connect()
            runtime = mldb.get_function(name, project, tag)
        else:
            func_url = 'function.yaml' if func_url == '.' else func_url
            runtime = import_function_to_dict(func_url, {})
        kind = get_in(runtime, 'kind', '')
        if kind not in ['', 'local', 'dask'] and url:
            if path.isfile(url) and url.endswith('.py'):
                with open(url) as fp:
                    body = fp.read()
                based = b64encode(body.encode('utf-8')).decode('utf-8')
                logger.info('Packing code at {}'.format(url))
                update_in(runtime, 'spec.build.functionSourceCode', based)
                url = ''
                update_in(runtime, 'spec.command', '')
    elif runtime:
        runtime = py_eval(runtime)
        if not isinstance(runtime, dict):
            print('runtime parameter must be a dict, not {}'.format(type(runtime)))
            exit(1)
    else:
        runtime = {}

    code = environ.get('MLRUN_EXEC_CODE')
    if get_in(runtime, 'kind', '') == 'dask':
        code = get_in(runtime, 'spec.build.functionSourceCode', code)
    if from_env and code:
        code = b64decode(code).decode('utf-8')
        if kfp:
            print('code:\n{}\n'.format(code))
        with open('main.py', 'w') as fp:
            fp.write(code)
        url = url or 'main.py'

    if url:
        update_in(runtime, 'spec.command', url)
    if run_args:
        update_in(runtime, 'spec.args', list(run_args))
    if image:
        update_in(runtime, 'spec.image', image)
    set_item(runobj.spec, handler, 'handler')
    set_item(runobj.spec, param, 'parameters', fill_params(param))
    set_item(runobj.spec, hyperparam, 'hyperparams', fill_params(hyperparam))
    set_item(runobj.spec, param_file, 'param_file')
    set_item(runobj.spec, selector, 'selector')

    set_item(runobj.spec, inputs, run_keys.inputs, list2dict(inputs))
    set_item(runobj.spec, in_path, run_keys.input_path)
    set_item(runobj.spec, out_path, run_keys.output_path)
    set_item(runobj.spec, outputs, run_keys.outputs, list(outputs))
    set_item(runobj.spec, secrets, run_keys.secrets, line2keylist(secrets, 'kind', 'source'))

    if kfp:
        print('MLRun version: {}'.format(get_version()))
        print('Runtime:')
        pprint(runtime)
        print('Run:')
        pprint(runobj.to_dict())

    try:
        update_in(runtime, 'metadata.name', name, replace=False)
        fn = new_function(runtime=runtime, kfp=kfp, mode=mode)
        if workdir:
            fn.spec.workdir = workdir
        fn.is_child = from_env and not kfp
        resp = fn.run(runobj, watch=watch, schedule=schedule)
        if resp and dump:
            print(resp.to_yaml())
    except RunError as err:
        print('runtime error: {}'.format(err))
        exit(1)


# `build` Command
@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("func_url", type=str, required=False)
@click.option('--name',
              help='Function name')
@click.option('--project',
              help='Project name')
@click.option('--tag', default='',
              help='Function tag')
@click.option('--image', '-i',
              help='Target image path')
@click.option('--source', '-s', default='',
              help="Path/URL of the function source code - a PY file, or a '
              'directory to archive\n                         '
              'when using the -a|--archive option (default: './')")
              # SLSL: Edited, here and in examples/remote.md.
              # TODO: Verify the use of "\n" and the spaces at the start of the
              # 2nd outline line. 
              # NOWNOW-RND
@click.option('--base-image', '-b', help='Base Docker image')
@click.option('--command', '-c', default='', multiple=True,
              help="Build commands; for example, "
              "'-c pip install pandas'")
@click.option('--secret-name', default='',
              help='Name of a container-registry secret')
@click.option('--archive', '-a', default='',
              help='  -a, --archive TEXT     Path/URL of a target '
              'function-sources archive directory: as part of the\n'
              '                       build, the function sources (see '
              '-s|--source) are archived into a\n                       '
              'TAR file and then extracted into the archive directory'
              # SLSL: Edited, here and in examples/remote.md. NOWNOW-RND
@click.option('--silent', is_flag=True,
              help="Don't show build logs")
@click.option('--with-mlrun', is_flag=True,
              help='Add the MLRun package ("mlrun")')
              # SLSL: What does it mean to add the mlrun package? NOWNOW-RND?
@click.option('--db', default='',
              help='Save run results to path or DB URL')
@click.option('--runtime', '-r', default='',
              help='Function spec dictionary, for pipeline usage')
@click.option('--kfp', is_flag=True,
              help='Running inside Kubeflow Pipelines; DO NOT USE')
@click.option('--skip', is_flag=True,
              help='Skip if already deployed')
def build(func_url, name, project, tag, image, source, base_image, command,
          secret_name, archive, silent, with_mlrun, db, runtime, kfp, skip):
    """Builds a container image from code and requirements."""

    if runtime:
        runtime = py_eval(runtime)
        if not isinstance(runtime, dict):
            print('runtime parameter must be a dict, not {}'.format(type(runtime)))
            exit(1)
        if kfp:
            print('Runtime:')
            pprint(runtime)
        func = new_function(runtime=runtime)
    elif func_url.startswith('db://'):
        func_url = func_url[5:]
        project, name, tag = parse_function_uri(func_url)
        func = import_function(func_url, db=db)
    elif func_url:
        func_url = 'function.yaml' if func_url == '.' else func_url
        func = import_function(func_url, db=db)
    else:
        print('please specify the function path or url')
        exit(1)

    meta = func.metadata
    meta.project = project or meta.project or mlconf.default_project
    meta.name = name or meta.name
    meta.tag = tag or meta.tag

    b = func.spec.build
    if func.kind not in ['', 'local']:
        b.base_image = base_image or b.base_image
        b.commands = list(command) or b.commands
        b.image = image or b.image
        b.secret = secret_name or b.secret

    if source.endswith('.py'):
        if not path.isfile(source):
            print("Source file doesn't exist ({})".format(source))
            exit(1)
        with open(source) as fp:
            body = fp.read()
        based = b64encode(body.encode('utf-8')).decode('utf-8')
        logger.info('Packing code at "{}"'.format(source))
        b.functionSourceCode = based
        func.spec.command = ''
    else:
        b.source = source or b.source
        # TODO: Upload stuff

    archive = archive or mlconf.default_archive
    if archive:
        src = b.source or './'
        logger.info('Uploading data from {} to {}'.format(src, archive))
        target = archive if archive.endswith('/') else archive + '/'
        target += 'src-{}-{}-{}.tar.gz'.format(meta.project, meta.name,
                                               meta.tag or 'latest')
        upload_tarball(src, target)
        # TODO: Replace function.yaml inside the tar
        b.source = target

    if hasattr(func, 'deploy'):
        logger.info('Remote deployment started')
        try:
            func.deploy(with_mlrun=with_mlrun, watch=not silent,
                        is_kfp=kfp, skip_deployed=skip)
        except Exception as err:
            print('Deployment error: {}'.format(err))
            exit(1)

        if kfp:
            state = func.status.state
            image = func.spec.image
            with open('/tmp/state', 'w') as fp:
                fp.write(state)
            full_image = func.full_image_path(image) or ''
            with open('/tmp/image', 'w') as fp:
                fp.write(full_image)
            print('Function built, state="{}" image="{}"'
                  .format(state, full_image))
    else:
        print("Function doesn't have a `deploy` method")
        exit(1)


# `deploy` Command
@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("spec", type=str, required=False)
@click.option('--source', '-s', default='',
              help='Location/URL of the sources to deploy')
@click.option('--dashboard', '-d', default='',
              help='Nuclio dashboard URL')
@click.option('--project', '-p', default='',
              help='Name of a container-registry secret')
              # SLSL: How is the description related to the flag name?!
              # Perhaps it was copied by mistake from the --secret-name flag?
              # (I've now edited the description in both locations and the
              # copy of the --secret-name doc in examples/remote.md.)
              # NOWNOWNOW-RND
@click.option('--model', '-m', multiple=True,
              help='Input artifact')
              # SLSL: I changed to "artifacts" (plural).
              # Why is the flag named --model?
              # NOWNOW-RND
@click.option('--kind', '-k', default=None,
              help='Runtime sub kind')
@click.option('--tag', default='',
              help='Version tag')
@click.option('--env', '-e', multiple=True,
              help='Environment variables')
@click.option('--verbose', is_flag=True,
              help='Verbose log')
def deploy(spec, source, dashboard, project, model, tag, kind, env, verbose):
    """Deploys a model or function."""
    if spec:
        runtime = py_eval(spec)
    else:
        runtime = {}
    if not isinstance(runtime, dict):
        print('Runtime parameter must be of type dict, not {}'
              .format(type(runtime)))
        exit(1)

    f = RemoteRuntime.from_dict(runtime)
    f.spec.source = source
    if kind:
        f.spec.function_kind = kind
    if env:
        for k, v in list2dict(env).items():
            f.set_env(k, v)
    f.verbose = verbose
    if model:
        models = list2dict(model)
        for k, v in models.items():
            f.add_model(k, v)

    try:
        addr = f.deploy(dashboard=dashboard, project=project, tag=tag, kind=kind)
    except Exception as err:
        print('deploy error: {}'.format(err))
        exit(1)

    print('Function deployed, address={}'.format(addr))
    with open('/tmp/output', 'w') as fp:
        fp.write(addr)


# `watch` Command
@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("pod", type=str)
@click.option('--namespace', '-n',
              help='Kubernetes namespace')
@click.option('--timeout', '-t', default=600, show_default=True,
              help='Timeout period, in seconds')
def watch(pod, namespace, timeout):
    """Reads current or previous task (pod) logs."""
    k8s = K8sHelper(namespace)
    status = k8s.watch(pod, namespace, timeout)
    print('Last status of pod {} is "{}"'.format(pod, status))


# `get` Command
@main.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('kind', type=str)
@click.argument('name', type=str, default='', required=False)
@click.option('--selector', '-s', default='',
              help='Label selector')
@click.option('--namespace', '-n',
              help='Kubernetes namespace')
@click.option('--uid',
              help='Unique ID')
              # SLSL: UID of what? The project?
              # Note that other --project flags are described as project
              # name/ID, but the `watch` --project flag below is described only
              # as "Project name"?
              # NOWNOW-RND
@click.option('--project',
              help='Project name')
@click.option('--tag', '-t', default='',
              help='Artifacts/function tag')
@click.option('--db',
              help='Path to the MLRun DB/API service')

@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def get(kind, name, selector, namespace, uid, project, tag, db, extra_args):
    """Lists/gets one or more objects for a specific kind (class)."""

    if kind.startswith('po'):
        k8s = K8sHelper(namespace)
        if name:
            resp = k8s.get_pod(name, namespace)
            print(resp)
            return

        items = k8s.list_pods(namespace, selector)
        print('{:10} {:16} {:8} {}'.format('state', 'started', 'type', 'name'))
        for i in items:
            task = i.metadata.labels.get('mlrun/class', '')
            if task:
                name = i.metadata.name
                state = i.status.phase
                start = ''
                if i.status.start_time:
                    start = i.status.start_time.strftime("%b %d %H:%M:%S")
                print('{:10} {:16} {:8} {}'.format(state, start, task, name))
    elif kind.startswith('run'):
        mldb = get_run_db(db or mlconf.dbpath).connect()
        if name:
            run = mldb.read_run(name, project=project)
            print(dict_to_yaml(run))
            return

        runs = mldb.list_runs(uid=uid, project=project)
        df = runs.to_df()[['name', 'uid', 'iter', 'start', 'state', 'parameters', 'results']]
        #df['uid'] = df['uid'].apply(lambda x: '..{}'.format(x[-6:]))
        df['start'] = df['start'].apply(time_str)
        df['parameters'] = df['parameters'].apply(dict_to_str)
        df['results'] = df['results'].apply(dict_to_str)
        print(tabulate(df, headers='keys'))

    elif kind.startswith('art'):
        mldb = get_run_db(db or mlconf.dbpath).connect()
        artifacts = mldb.list_artifacts(name, project=project, tag=tag)
        df = artifacts.to_df()[['tree', 'key', 'iter', 'kind', 'path', 'hash', 'updated']]
        df['tree'] = df['tree'].apply(lambda x: '..{}'.format(x[-8:]))
        df['hash'] = df['hash'].apply(lambda x: '..{}'.format(x[-6:]))
        print(tabulate(df, headers='keys'))

    elif kind.startswith('func'):
        mldb = get_run_db(db or mlconf.dbpath).connect()
        if name:
            f = mldb.get_function(name, project=project, tag=tag)
            print(dict_to_yaml(f))
            return

        functions = mldb.list_functions(name, project=project)
        lines = []
        headers = ['kind', 'state', 'name:tag', 'hash']
        for f in functions:
            line = [
                get_in(f, 'kind', ''),
                get_in(f, 'status.state', ''),
                '{}:{}'.format(get_in(f, 'metadata.name'), get_in(f, 'metadata.tag', '')),
                get_in(f, 'metadata.hash', ''),
            ]
            lines.append(line)
        print(tabulate(lines, headers=headers))
    else:
        print('Currently, only get pods | runs | artifacts | func [name] are supported.')


# `db` Command
@main.command()
@click.option('--port', '-p', help='Port to listen on', type=int)
@click.option('--dirpath', '-d',
              help='Path to the MLRun DB/API service directory')
def db(port, dirpath):
    """Runs an MLRun database/HTTP API service."""
    env = environ.copy()
    if port is not None:
        env['MLRUN_httpdb__port'] = str(port)
    if dirpath is not None:
        env['MLRUN_httpdb__dirpath'] = dirpath

    cmd = [executable, '-m', 'mlrun.db.httpd']
    child = Popen(cmd, env=env)
    returncode = child.wait()
    if returncode != 0:
        raise SystemExit(returncode)
# SLSL: NOWNOW I edited the description from "Run HTTP api/database server".


# `version` Command
@main.command()
def version():
    """Displays the MLRun version."""

    print('MLRun version: {}'.format(get_version()))


# `logs` Command
@main.command()
@click.argument('uid', type=str)
@click.option('--project', '-p',
              help='Project name')
@click.option('--offset', type=int, default=0,
              help='Byte offset')
@click.option('--db',
              help='Path or URL of the MLRun database/API service')
@click.option('--watch', '-w', is_flag=True, help='watch/follow log')
def logs(uid, project, offset, db, watch):
    """Gets or displays task logs."""

    mldb = get_run_db(db or mlconf.dbpath).connect()
    if mldb.kind == 'http':
        state = mldb.watch_log(uid, project, watch=watch, offset=offset)
    else:
        state, text = mldb.get_log(uid, project, offset=offset)
        if text:
            print(text.decode())

    if state:
        print('final state: {}'.format(state))


# `project` Command
@main.command()
@click.argument('context', type=str)
@click.option('--name', '-n',
              help='project name')
@click.option('--url', '-u',
              help='Remote Git or archive URL')
@click.option('--run', '-r',
              help='Name of the run-workflow PY file')
@click.option('--arguments', '-a',
              help='Arguments dictionary')
@click.option('--artifact-path', '-p',
              help='Path for storing output artifacts')
@click.option('--namespace',
              help='Kubernetes namespace')
@click.option('--db',
              help='Path or URL of the MLRun database/API service')
@click.option('--init-git', is_flag=True,
              help='Git initialization context for new projects')
@click.option('--clone', '-c', is_flag=True,
              help='Force overriding/cloning of the context directory')
@click.option('--sync', is_flag=True,
              help='Synchronize functions into the DB')
              # SLSL: What does "sync functions into db" means?! NOWNOW-RND
@click.option('--dirty', '-d', is_flag=True,
              help='Allow using Git files with uncommitted changes')
def project(context, name, url, run, arguments, artifact_path,
            namespace, db, init_git, clone, sync, dirty):
    """Loads and/or runs an MLRun project."""
    if db:
        mlconf.dbpath = db

    proj = load_project(context, url, name, init_git=init_git, clone=clone)
    print('Loading project {}{} into {}:\n'.format(
        proj.name, ' from ' + url if url else '', context))
    print(proj.to_yaml())

    if run:
        workflow_path = None
        if run.endswith('.py'):
            workflow_path = run
            run = None

        args=None
        if arguments:
            try:
                args = literal_eval(arguments)
            except (SyntaxError, ValueError):
                print('arguments ({}) must be a dict object/str'
                      .format(arguments))
                exit(1)

        print('Running workflow "{}", file "{}"'.format(run, workflow_path))
        run = proj.run(run, workflow_path, arguments=args,
                       artifact_path=artifact_path, namespace=namespace,
                       sync=sync, dirty=dirty)
        print('run id: {}'.format(run))

    elif sync:
        print('Saving project functions to DB ...')
        proj.sync_functions(save=True)


# `clean` Command
@main.command()
@click.option('--api',
              help='Path to the MLRun DB/API service')
@click.option('--namespace', '-n', help='kubernetes namespace')
@click.option('--pending', '-p', is_flag=True,
              help='Clean pending pods as well')
@click.option('--running', '-r', is_flag=True,
              help='clean running pods as well')
def clean(api, namespace, pending, running):
    """Cleans completed or failed pods/jobs."""
    k8s = K8sHelper(namespace)
    #mldb = get_run_db(db or mlconf.dbpath).connect()
    items = k8s.list_pods(namespace)
    states = ['Succeeded', 'Failed']
    if pending:
        states.append('Pending')
    if running:
        states.append('Running')
    print('{:10} {:16} {:8} {}'.format('state', 'started', 'type', 'name'))
    for i in items:
        task = i.metadata.labels.get('mlrun/class', '')
        state = i.status.phase
        # TODO: Clean MPI, Spark, ... jobs (+ CRDs)
        if task and task in ['build', 'job', 'dask'] and state in states:
            name = i.metadata.name
            start = ''
            if i.status.start_time:
                start = i.status.start_time.strftime("%b %d %H:%M:%S")
            print('{:10} {:16} {:8} {}'.format(state, start, task, name))
            k8s.del_pod(name)


# `config` Command
@main.command(name='config')
def show_config():
    """Displays an MLRun YAML configuration configuration."""
    print(mlconf.dump_yaml())


## Internal Functions


def fill_params(params):
    params_dict = {}
    for param in params:
        i = param.find('=')
        if i == -1:
            continue
        key, value = param[:i].strip(), param[i + 1:].strip()
        if key is None:
            raise ValueError(
                'Cannot find parameter key in line ({})'.format(param))
        params_dict[key] = py_eval(value)
    return params_dict


def py_eval(data):
    try:
        value = literal_eval(data)
        return value
    except (SyntaxError, ValueError):
        return data


def set_item(obj, item, key, value=None):
    if item:
        if value:
            setattr(obj, key, value)
        else:
            setattr(obj, key, item)


def line2keylist(lines: list, keyname='key', valname='path'):
    out = []
    for line in lines:
        i = line.find('=')
        if i == -1:
            raise ValueError('cannot find "=" in line ({}={})'.format(keyname, valname))
        key, value = line[:i].strip(), line[i + 1:].strip()
        if key is None:
            raise ValueError('cannot find key in line ({}={})'.format(keyname, valname))
        value = path.expandvars(value)
        out += [{keyname: key, valname: value}]
    return out


def time_str(x):
    try:
        return x.strftime("%b %d %H:%M:%S")
    except ValueError:
        return ''


def dict_to_str(struct: dict):
    if not struct:
        return []
    return ','.join(['{}={}'.format(k, v) for k, v in struct.items()])


# Execute the main function
if __name__ == "__main__":
    main()

