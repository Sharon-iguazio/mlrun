from mlrun import get_or_create_ctx
from mlrun.artifacts import TableArtifact, ChartArtifact

def my_func(ctx):
    p1 = ctx.get_param('p1', 1)
    p2 = ctx.get_param('p2', 'a-string')

    print(f'Run: {ctx.name} (uid={ctx.uid})')
    print(f'Params: p1={p1}, p2={p2}')
    print('accesskey = {}'.format(ctx.get_secret('ACCESS_KEY')))
    print('file\n{}\n'.format(ctx.get_object('infile.txt').get()))

    ctx.log_output('accuracy', p1 * 2)
    ctx.log_output('latency', p1 * 3)

    ctx.log_artifact('test.txt', body=b'abc is 123')
    ctx.log_artifact('test.html', body=b'<b> Some HTML <b>', viewer='html')

    table = TableArtifact('tbl.csv', '1,2,3\n4,5,6\n',
                          viewer='table', header=['A', 'B', 'C'])
    ctx.log_artifact(table)

    chart = ChartArtifact('chart.html')
    chart.header = ['Hour','One', 'Two']
    for i in range(1,4):
        chart.add_row([i, 1+2, 2*i])
    ctx.log_artifact(chart)

if __name__ == "__main__":
    ex = get_or_create_ctx('mytask')
    my_func(ex)
    ex.commit('aa')