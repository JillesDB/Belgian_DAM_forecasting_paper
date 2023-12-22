import pandas as pd
import numpy as np
import scipy.stats as stat

from Additional_plots import formulas as fm

from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 14})
class Evaluation:

    def __init__(self, actual, forecast):
        self.actual = actual
        self.forecast = forecast

    def metrics(self):
        mse = fm.mse(self.actual, self.forecast)
        var_actual = fm.var(self.actual)
        var_forecast = fm.var(self.forecast)
        correl = stat.pearsonr(self.actual, self.forecast)[0]
        bias = fm.unconditional_bias(self.actual, self.forecast)
        conditional_bias_1 = fm.conditional_bias_1(self.actual, self.forecast)
        resolution = fm.resolution(self.actual, self.forecast)
        conditional_bias_2 = fm.conditional_bias_2(self.actual, self.forecast)
        discrimination = fm.discrimination(self.actual, self.forecast)

        dict = {'MSE': mse, 'Var(x)': var_actual,
                'Var(y)': var_forecast,
                'Corr': correl,
                'Bias': bias,
                'Conditional bias 1': conditional_bias_1,
                'Resolution': resolution,
                'Conditional bias 2': conditional_bias_2,
                'Discrimination': discrimination}

        pd.options.display.float_format = "{:,.3f}".format

        metrics = pd.DataFrame(dict, index=['Metrics'])

        return metrics

    def plot_joint(self, levels=20,xlabel='xlabel', ylabel='ylabel'):
        sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 300})
        sns.set_context('notebook')
        sns.set_style("ticks")
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats('retina')

        x = self.actual
        y = self.forecast
        print(x,y)
        xandy = np.append(x, y)
        lin = np.linspace(min(xandy) - np.average(xandy) * 0.4, max(xandy) + np.average(xandy) * 0.4, 1000)
        liny = lin

        diff = np.abs(x - y)

        plt.figure(figsize=(7, 7))
        sns.set_style("white")

        palette = sns.dark_palette("#79C", n_colors=levels, reverse=True, as_cmap=True)
        #palette = sns.color_palette("cividis", as_cmap=True)

        g = sns.jointplot(x=x, y=y, s=0, marginal_kws=dict(bins=1, color="white", alpha=0), )
        #g.plot_joint(sns.kdeplot, alpha=0.3, fill=False, zorder=2, levels=levels, linewidth=0.4)
        g.plot_joint(sns.scatterplot, s=3,hue=diff,  linewidths=0, alpha=0.6, palette=palette)
        sns.lineplot(x=lin, y=liny, linewidth=0.5, color='blue', zorder=3)
        g.ax_joint.set_xlabel('Actuals')
        g.ax_joint.set_ylabel('Forecast')
        g.ax_joint.set_xticks([-100,0,100, 200,300,400,500,600,700,800,900])
        g.ax_joint.set_yticks([-100,0,100, 200,300,400,500,600,700,800,900])

        plt.xlim(min(lin), max(lin))
        plt.ylim(min(lin), max(lin))
        plt.legend(title="Diff. x-y")
        g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)


    def plot_conditional(self, intervals=11, x_label="x given y", y_label='Forecast intervals', size=(10, 5)):
        sns.set(rc={"figure.figsize": size, "figure.dpi": 100, 'savefig.dpi': 300})
        # sns.set_context('notebook')
        sns.set_style("ticks")
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats('retina')
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        x = self.actual
        y = self.forecast

        intervals = fm.interval_calc(y, num_intervals=intervals)

        interval_df = fm.x_given_y_intervals(x, y, intervals)
        interval_df = interval_df.reindex(index=interval_df.index[::-1])
        interval_df[x_label] = interval_df['x given y']

        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(interval_df, row="y interval", hue="y interval", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, x_label, clip_on=False, fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, x_label, clip_on=False, color="w", lw=2)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="right", va="center", transform=ax.transAxes)

        g.map(label, x_label)

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)
        g.set_titles("")
        g.fig.suptitle(y_label, x=0.05, y=0.9, ha='left', size='medium', fontweight='bold')
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

# import pandas as pd
# import numpy as np
# import scipy.stats as stat
#
# from Plotting_Joris import formulas as fm
#
# from matplotlib import pyplot as plt
# import seaborn as sns
#
# import warnings
#
# warnings.filterwarnings("ignore")
#
#
# class Evaluation:
#
#     def __init__(self, actual, forecast):
#         self.actual = actual
#         self.forecast = forecast
#
#     def metrics(self):
#         mse = fm.mse(self.actual, self.forecast)
#         var_actual = fm.var(self.actual)
#         var_forecast = fm.var(self.forecast)
#         correl = stat.pearsonr(self.actual, self.forecast)[0]
#         bias = fm.unconditional_bias(self.actual, self.forecast)
#         conditional_bias_1 = fm.conditional_bias_1(self.actual, self.forecast)
#         resolution = fm.resolution(self.actual, self.forecast)
#         conditional_bias_2 = fm.conditional_bias_2(self.actual, self.forecast)
#         discrimination = fm.discrimination(self.actual, self.forecast)
#
#         dict = {'MSE': mse, 'Var(x)': var_actual,
#                 'Var(y)': var_forecast,
#                 'Corr': correl,
#                 'Bias': bias,
#                 'Conditional bias 1': conditional_bias_1,
#                 'Resolution': resolution,
#                 'Conditional bias 2': conditional_bias_2,
#                 'Discrimination': discrimination}
#
#         pd.options.display.float_format = "{:,.3f}".format
#
#         metrics = pd.DataFrame(dict, index=['Metrics'])
#
#         return metrics
#
#     def plot_joint(self, levels=20, xlabel='xlabel', ylabel='ylabel'):
#         sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 300})
#         sns.set_context('notebook')
#         sns.set_style("ticks")
#         from IPython.display import set_matplotlib_formats
#         set_matplotlib_formats('retina')
#
#         x = self.actual
#         y = self.forecast
#         xandy = np.append(x, y)
#         lin = np.linspace(min(xandy) - np.average(xandy) * 0.4, max(xandy) + np.average(xandy) * 0.4, 1000)
#         liny = lin
#
#         diff = np.abs(x - y)
#
#         plt.figure(figsize=(7, 7))
#         sns.set_style("white")
#     #    palette = sns.color_palette("coolwarm", as_cmap=True)
#
#         palette = sns.dark_palette("#79C", n_colors=levels, reverse=True, as_cmap=True)
#
#         g = sns.jointplot(x=x, y=y, s=0, marginal_kws=dict(bins=levels, color="grey", alpha=0.8))
#         print('yay')
#         g.plot_joint(sns.kdeplot, alpha=0.3, fill=False, zorder=2, levels=levels, linewidths=0.4)
#         print('yay2')
#         g.plot_joint(sns.scatterplot, s=3, hue=diff, palette=palette, linewidths=0, alpha=0.6)
#         print('yay3')
#         sns.lineplot(x=lin, y=liny, linewidth=0.5, color='red', zorder=3)
#         print('yay4')
#         print(x, y, min(lin), max(lin))
#         g.ax_joint.set_xlabel('Actuals')
#         g.ax_joint.set_ylabel('Forecast')
#         plt.xlim(min(lin), max(lin))
#         plt.ylim(min(lin), max(lin))
#         handles, labels = g.ax_joint.get_legend_handles_labels()
#         print(handles,labels)
#         for artist in g.ax_joint.get_children():
#             print(artist, artist.get_label())
#         # Check if there are any non-underscored artists with labels before calling legend()
#         handles, labels = g.ax_joint.get_legend_handles_labels()
#         print(handles,labels)
#         if any(label.startswith('_') for label in labels):
#             print("No artists with labels found to put in legend.")
#             print("Ignored artists with labels:")
#             for handle, label in zip(handles, labels):
#                 if label.startswith('_'):
#                     print(label)
#         else:
#             plt.legend(title="Diff. x-y")
#
#
#     # def plot_joint(self, levels=20,xlabel = 'xlabel',ylabel = 'ylabel'):
#     #     sns.set(rc={"figure.dpi": 100, 'savefig.dpi': 300})
#     #     sns.set_context('notebook')
#     #     sns.set_style("ticks")
#     #     from IPython.display import set_matplotlib_formats
#     #     set_matplotlib_formats('retina')
#     #
#     #     x = self.actual
#     #     y = self.forecast
#     #     xandy = np.append(x, y)
#     #     lin = np.linspace(min(xandy) - np.average(xandy) * 0.4, max(xandy) + np.average(xandy) * 0.4, 1000)
#     #     liny = lin
#     #
#     #     diff = np.abs(x - y)
#     #
#     #     plt.figure(figsize=(7, 7))
#     #     sns.set_style("white")
#     #
#     #     palette = sns.dark_palette("#79C",n_colors=levels, reverse=True, as_cmap=True)
#     #     # palette = sns.color_palette("cividis", as_cmap=True)
#     #
#     #
#     #     g = sns.jointplot(x=x, y=y, s=0, marginal_kws=dict(bins=levels, color="grey", alpha=0.8) )
#     #     print('yay')
#     #     g.plot_joint(sns.kdeplot, alpha=0.3, fill=False, zorder=2, levels=levels, linewidths=0.4)
#     #     print('yay2')
#     #     g.plot_joint(sns.scatterplot, s=3, hue=diff, palette=palette, linewidths=0, alpha=0.6)
#     #     print('yay3')
#     #     sns.lineplot(x=lin, y=liny, linewidth=0.5, color='red', zorder=3)
#     #     print('yay4')
#     #     print(x,y,min(lin),max(lin))
#     #     g.ax_joint.set_xlabel('Actuals')
#     #     g.ax_joint.set_ylabel('Forecast')
#     #     plt.xlim(min(lin), max(lin))
#     #     plt.ylim(min(lin), max(lin))
#     #     plt.legend(title="Diff. x-y")
#     #  #   g.set_axis_labels(xlabel=xlabel,ylabel=ylabel)
#
#     def plot_conditional(self, intervals=11, x_label="x given y", y_label='Forecast intervals', size=(10, 5)):
#         sns.set(rc={"figure.figsize": size, "figure.dpi": 100, 'savefig.dpi': 300})
#         # sns.set_context('notebook')
#         sns.set_style("ticks")
#         from IPython.display import set_matplotlib_formats
#         set_matplotlib_formats('retina')
#         sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
#
#         x = self.actual
#         y = self.forecast
#
#         intervals = fm.interval_calc(y, num_intervals=intervals)
#
#         interval_df = fm.x_given_y_intervals(x, y, intervals)
#         interval_df = interval_df.reindex(index=interval_df.index[::-1])
#         interval_df[x_label] = interval_df['x given y']
#
#         pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
#         g = sns.FacetGrid(interval_df, row="y interval", hue="y interval", aspect=15, height=.5, palette=pal)
#
#         # Draw the densities in a few steps
#         g.map(sns.kdeplot, x_label, clip_on=False, fill=True, alpha=1, linewidth=1.5)
#         g.map(sns.kdeplot, x_label, clip_on=False, color="w", lw=2)
#
#         # passing color=None to refline() uses the hue mapping
#         g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
#
#         # Define and use a simple function to label the plot in axes coordinates
#         def label(x, color, label):
#             ax = plt.gca()
#             ax.text(0, .2, label, fontweight="bold", color=color,
#                     ha="right", va="center", transform=ax.transAxes)
#
#         g.map(label, x_label)
#
#         # Set the subplots to overlap
#         g.figure.subplots_adjust(hspace=-.25)
#         g.set_titles("")
#         g.fig.suptitle(y_label, x=0.0, y=0.98, ha='left', size='medium', fontweight='bold')
#         g.set(yticks=[], ylabel="")
#         g.despine(bottom=True, left=True)
