# To run this function with Nuclio, you must do the following:
# - Set the Python base image. For example:
#     python:3.6-jessie
# - Add installation of the MLRun package (`mlrun`) to the build commands:
#     pip install mlrun

from mlrun import get_or_create_ctx
import time


def handler(context, event):
    ctx = get_or_create_ctx('myfunc', event=event)
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    context.logger.info(
        f'Run: {ctx.name} uid={ctx.uid}:{ctx.iteration} Params: p1={p1}, p2={p2}')

    time.sleep(1)

    # Log scalar values (Kubeflow Pipelines metrics)
    ctx.log_result('accuracy', p1 * 2)
    ctx.log_result('latency', p1 * 3)

    # Log various types of artifacts and set dashboard (UI) viewers
    ctx.log_artifact('test', body=b'abc is 123', local_path='test.txt')
    ctx.log_artifact('test_html', body=b'<b> Some HTML <b>', format='html')

    context.logger.info('run complete!')
    return ctx.to_json()
