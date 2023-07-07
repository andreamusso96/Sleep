from typing import List, Tuple, Union, Dict, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from FeatureSelection.Regression import Regression
from FeatureSelection.Controls import Control


class MultiRegressionPlotLayoutParameters:
    def __init__(self, nrows: int, ncols: int, width: int, height: int, vertical_spacing: float, horizontal_spacing: float,
                 font_size: int, line_width: int, template: str, inset_font_size: int, inset_line_width: int, inset_bar_legend_names: List[str],
                 inset_bar_colors: Union[Dict[str, str], None], inset_size_x: float, inset_size_y: float,
                 inset_shift_x: float, inset_shift_y: float, legend_x: float, legend_y: float, skip: List[Tuple[int, int]]):
        self.nrows = nrows
        self.ncols = ncols
        self.width = width
        self.height = height
        self.vertical_spacing = vertical_spacing
        self.horizontal_spacing = horizontal_spacing
        self.font_size = font_size
        self.line_width = line_width
        self.template = template
        self.inset_font_size = inset_font_size
        self.inset_line_width = inset_line_width
        self.inset_bar_legend_names = inset_bar_legend_names
        self.inset_bar_colors = inset_bar_colors if inset_bar_colors is not None else {inset_bar_legend_names[i]: px.colors.qualitative.Plotly[i] for i in range(len(inset_bar_legend_names))}
        self.inset_size_x = inset_size_x
        self.inset_size_y = inset_size_y
        self.inset_shift_x = inset_shift_x
        self.inset_shift_y = inset_shift_y
        self.legend_x = legend_x
        self.legend_y = legend_y
        self.skip = skip


class AxisManager:
    def __init__(self, fig: go.Figure, params: MultiRegressionPlotLayoutParameters):
        self.fig = fig
        self.params = params

    def axis(self, inset: bool, row: int, col: int, is_xaxis: bool = None, title: str = '', **kwargs) -> Dict[str, Any]:
        if inset:
            assert is_xaxis is not None, 'You must specify if inset axis is x or y axis'
            domain = self._inset_domain(row=row, col=col, is_xaxis=is_xaxis)
            axis = self._inset_axis(axis_number=self.inset_axis_number(row=row, col=col), is_xaxis=is_xaxis, title=title, domain=list(domain), **kwargs)
        else:
            axis = self._main_axis(title=title, **kwargs)
        return axis

    def main_axis_number(self, row: int, col: int) -> int:
        main_axis_number = (row - 1) * self.params.ncols + col
        return main_axis_number

    def inset_axis_number(self, row: int, col: int) -> int:
        main_axis_number = self.main_axis_number(row=row, col=col)
        inset_axis_number = self.params.ncols * self.params.nrows + main_axis_number
        return inset_axis_number

    def _main_axis(self, title: str,  **kwargs) -> Dict[str, Any]:
        axis = dict(title=dict(text=title), linecolor='black', linewidth=self.params.line_width, **kwargs)
        return axis

    def _inset_axis(self, axis_number: int, title: str, domain: List[float], is_xaxis, **kwargs) -> Dict[str, Any]:
        anchor = f'y{axis_number}' if is_xaxis else f'x{axis_number}'
        inset_axis = dict(title=dict(text=title, font=dict(size=self.params.inset_font_size)),
                          tickfont=dict(size=self.params.inset_font_size), domain=domain, anchor=anchor,
                          showgrid=False, showline=True, linecolor='black', linewidth=self.params.inset_line_width, **kwargs)
        return inset_axis

    def _inset_domain(self, row: int, col: int, is_xaxis: bool) -> Tuple[float, float]:
        main_axis_number = self.main_axis_number(row=row, col=col)
        if is_xaxis:
            bottom_right_corner_main_axis = self.fig.layout[f'xaxis{main_axis_number}']['domain'][1]
            inset_right_corner = bottom_right_corner_main_axis - self.params.inset_shift_x
            inset_left_corner = inset_right_corner - self.params.inset_size_x
            inset_domain = (inset_left_corner, inset_right_corner)
        else:
            bottom_right_corner_main_axis = self.fig.layout[f'yaxis{main_axis_number}']['domain'][0]
            inset_bottom_corner = bottom_right_corner_main_axis + self.params.inset_shift_y
            inset_top_corner = inset_bottom_corner + self.params.inset_size_y
            inset_domain = (inset_bottom_corner, inset_top_corner)

        return inset_domain


class RegressionSubplot:
    def __init__(self, regression: Regression, controls: List[Control], xtitle: str, ytitle: str):
        self.regression = regression
        self.controls = controls
        self.xtitle = xtitle
        self.ytitle = ytitle
        self.row = None
        self.col = None

    def add_subplot(self, fig: go.Figure, axis_manager: AxisManager, colors: Dict[str, str]) -> Dict[str, Any]:
        raise NotImplementedError

    def _add_subplot(self, fig: go.Figure, axis_manager: AxisManager, trace_main1, trace_main2, trace_inset) -> Dict[str, Any]:
        fig.add_trace(trace_main1, row=self.row, col=self.col)
        fig.add_trace(trace_main2, row=self.row, col=self.col)
        fig.add_trace(trace_inset)
        axis_layout = self._subplot_axis_layout(axis_manager=axis_manager, row=self.row, col=self.col,
                                                xtitle=self.xtitle, ytitle=self.ytitle)
        return axis_layout

    @staticmethod
    def _subplot_axis_layout(axis_manager: AxisManager, row: int, col: int, xtitle: str, ytitle: str) -> Dict[str, Any]:
        main_axis_number = axis_manager.main_axis_number(row=row, col=col)
        inset_axis_number = axis_manager.inset_axis_number(row=row, col=col)
        xaxis_main = axis_manager.axis(inset=False, row=row, col=col, title=xtitle, showline=True)
        yaxis_main = axis_manager.axis(inset=False, row=row, col=col, title=ytitle if col == 1 else '', nticks=5, showline=True if col == 1 else False)
        xaxis_inset = axis_manager.axis(inset=True, row=row, col=col, is_xaxis=True, title=' ')
        yaxis_inset = axis_manager.axis(inset=True, row=row, col=col, is_xaxis=False, title='Effect', nticks=3, side='left')
        axis_layout = {f'xaxis{main_axis_number}': xaxis_main, f'yaxis{main_axis_number}': yaxis_main,
                       f'xaxis{inset_axis_number}': xaxis_inset, f'yaxis{inset_axis_number}': yaxis_inset}
        return axis_layout


class RegressionSubplotScatter(RegressionSubplot):
    def __init__(self, regression: Regression, controls: List[Control], xtitle: str, ytitle: str):
        super().__init__(regression=regression, controls=controls, xtitle=xtitle, ytitle=ytitle)

    def add_subplot(self, fig: go.Figure, axis_manager: AxisManager, colors: Dict[str, str]) -> Dict[str, Any]:
        trace_bar_chart_inset, trace_scatter, trace_regression_line = self.regression.scatter_plot_with_controls(
            controls=self.controls, colors=colors,
            axis_number=axis_manager.inset_axis_number(row=self.row, col=self.col), show=False)
        return self._add_subplot(fig=fig, axis_manager=axis_manager, trace_main1=trace_scatter,
                                 trace_main2=trace_regression_line, trace_inset=trace_bar_chart_inset)


class RegressionSubplotHeatmap(RegressionSubplot):
    def __init__(self, regression: Regression, controls: List[Control], xtitle: str, ytitle: str):
        super().__init__(regression=regression, controls=controls, xtitle=xtitle, ytitle=ytitle)

    def add_subplot(self, fig: go.Figure, axis_manager: AxisManager, colors: Dict[str, str]) -> Dict[str, Any]:
        trace_bar_chart_inset, trace_heatmap, trace_mean_outcome_line = self.regression.heatmap_plot_with_controls(
            controls=self.controls, colors=colors,
            axis_number=axis_manager.inset_axis_number(row=self.row, col=self.col), show=False)
        return self._add_subplot(fig=fig, axis_manager=axis_manager, trace_main1=trace_heatmap,
                                 trace_main2=trace_mean_outcome_line, trace_inset=trace_bar_chart_inset)


class MultiRegressionPlot:
    def __init__(self, regression_subplots: List[RegressionSubplot], layout_parameters: MultiRegressionPlotLayoutParameters):
        self.regression_subplots = regression_subplots
        self.params = layout_parameters
        self.fig = make_subplots(rows=self.params.nrows, cols=self.params.ncols, shared_xaxes=False, shared_yaxes=True,
                                 vertical_spacing=self.params.vertical_spacing, horizontal_spacing=self.params.horizontal_spacing)
        self.axis_manager = AxisManager(fig=self.fig, params=self.params)

    def plot(self, show: bool = True):
        self._assign_rows_and_columns(regression_subplots=self.regression_subplots)
        subplot_axis_layouts = {}
        for subplot in self.regression_subplots:
            axis_layout = subplot.add_subplot(fig=self.fig, axis_manager=self.axis_manager, colors=self.params.inset_bar_colors)
            subplot_axis_layouts.update(axis_layout)

        self._figure_layout(subplot_axis_layouts=subplot_axis_layouts)
        if show:
            self.fig.show(renderer='browser')
        return self.fig

    def _assign_rows_and_columns(self, regression_subplots):
        row = 1
        col = 1
        for subplot in regression_subplots:
            if (row, col) in self.params.skip:
                col += 1

            if col > self.params.ncols:
                col = 1
                row += 1
            subplot.row = row
            subplot.col = col
            col += 1

    def _figure_layout(self, subplot_axis_layouts):
        self._add_legend()
        self._add_colorbar()
        layout = go.Layout(
            width=self.params.width,
            height=self.params.height,
            template=self.params.template,
            font=dict(size=self.params.font_size, color='black'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=self.params.legend_y, xanchor="left",
                        x=self.params.legend_x),
            **subplot_axis_layouts)
        self.fig.update_layout(layout)

    def _add_legend(self):
        for i, name in enumerate(self.params.inset_bar_legend_names):
            self.fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=list(self.params.inset_bar_colors.values())[i]), showlegend=True, name=name), row=1, col=1)

    def _add_colorbar(self):
        colorbar_trace = go.Scatter(x=[None], y=[None], mode='markers', marker=dict(colorscale='Greys', showscale=True, cmin=0, cmax=1, colorbar=dict(thickness=10, tickvals=[0, 1], ticktext=['Low', 'High'], outlinewidth=0, orientation='v')), showlegend=False)
        self.fig.add_trace(colorbar_trace, row=1, col=1)

