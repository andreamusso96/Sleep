import plotly.graph_objs as go
from plotly.subplots import make_subplots

if __name__ == '__main__':
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import numpy as np

    # Generate some random data
    np.random.seed(0)
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x) + np.random.rand(100) - 0.5
    y2 = np.cos(x) + np.random.rand(100) - 0.5
    y3 = np.tan(x) + np.random.rand(100) - 0.5

    # Create the base for 3 subplots
    fig = make_subplots(rows=3, cols=1)

    # Add main subplots
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='plot 1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='plot 2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='plot 3'), row=3, col=1)

    # Add inset plots
    inset1 = go.Scatter(x=x, y=y1, mode='lines', showlegend=False, xaxis='x4', yaxis='y4')
    inset2 = go.Scatter(x=x, y=y2, mode='lines', showlegend=False, xaxis='x5', yaxis='y5')
    inset3 = go.Scatter(x=x, y=y3, mode='lines', showlegend=False, xaxis='x6', yaxis='y6')

    # Update layout for the insets
    fig.update_layout(
        xaxis4=dict(domain=[0.7, 1], anchor="y4", range=[0, 2 * np.pi]),
        yaxis4=dict(domain=[0, 0.2], anchor="x4", range=[-1.5, 1.5]),
        xaxis5=dict(domain=[0.7, 1], anchor="y5", range=[0, 2 * np.pi]),
        yaxis5=dict(domain=[0, 0.2], anchor="x5", range=[-1.5, 1.5]),
        xaxis6=dict(domain=[0.7, 1], anchor="y6", range=[0, 2 * np.pi]),
        yaxis6=dict(domain=[0, 0.2], anchor="x6", range=[-10, 10]),
    )

    fig.add_trace(inset1, row=1, col=1)
    fig.add_trace(inset2, row=2, col=1)
    fig.add_trace(inset3, row=3, col=1)


    fig.show()
