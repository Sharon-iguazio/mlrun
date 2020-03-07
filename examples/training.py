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
