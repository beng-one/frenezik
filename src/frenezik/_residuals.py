"""|----- frenezik -----|"""

# Data preprocessing
import numpy as np
import pandas as pd
import scipy

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import Visual_Identity, Check, Compute
import statsmodels.api as sm

# Class to simulate residuals
class Residuals:

    """
    Objective:
    ----------
    This class is used to simulate and analyze residuals that follow a prior law.

    Parameters:
    ---------
    population_size --> (int) : population size wanted.

    Functions:
    ---------
    Residuals.Simulation : Function to simulate residuals of a population according to a prior law.
    Residuals.Statistics : Function to compute and show descriptive statistics of simulated residuals.
    Residuals.Visualization : Function to visualize the distribution of simulated residuals.
    """

    def __init__(self, population_size, random_seed):

        # Check of <<population_size>>
        Check.data_size_(obj=population_size, name='population_size')

        # Check of <<random_seed>>
        Check.random_seed_(obj=random_seed)

        self.population_size = population_size
        self.random_seed = random_seed

    # Function to simulate residuals of a population according to a prior law
    def Simulation(self, law="normal", law_parameters=None):

        """
        Objective:
        ----------
        Function to simulate residuals of a population according to a prior law.

        Parameters:
        ---------
        law --> (str) | default="normal" : law of residuals.
        law_parameters --> (list, np.ndarray) | default=[0,1] : parameters of the law of residuals.

        Output:
        -------
        residuals_simulated --> (np.ndarray) : simulated residuals.
        """

        # Check of <<law>>
        Check.law_(obj=law)

        # Check of <<law_parameters>>
        Check.law_parameters_(obj=law_parameters)

        if law_parameters is None:
            law_parameters = [0, 1]
        if law == "normal" and len(law_parameters)!= 2:
            raise ValueError("<<law parameters>> associated with normal law must contain 2 elements.")
        if law == "student" and len(law_parameters)!= 1:
            raise ValueError("<<law parameters>> associated with student law must contain 1 element.")
        if law == "pareto" and len(law_parameters)!= 1:
            raise ValueError("<<law parameters>> associated with pareto law must contain 1 element.")
        if law == "weibull" and len(law_parameters)!= 1:
            raise ValueError("<<law parameters>> associated with weibull law must contain 1 element.")

        # Seed
        np.random.seed(self.random_seed)

        # residuals simulation according to a prior law
        if law == "normal":
            residuals_simulated = np.random.normal(loc=law_parameters[0], scale=law_parameters[1], size=self.population_size)
        elif law == "student":
            residuals_simulated = np.random.standard_t(df=law_parameters[0], size=self.population_size)
        elif law == "pareto":
            residuals_simulated = np.random.pareto(a=law_parameters[0], size=self.population_size)
        elif law == "weibull":
            residuals_simulated = np.random.weibull(a=law_parameters[0], size=self.population_size)

        # Output of simulated residuals
        return residuals_simulated

    # Function to compute and show descriptive statistics of simulated residuals
    def Statistics(self, residuals_simulated, decimals=0):

        """
        Objective:
        ----------
        Function to compute and show descriptive statistics of simulated residuals.

        Parameters:
        ---------
        residuals_simulated --> (np.ndarray, list) : data of simulated residuals.
        decimals --> (int) | default=0 : Number of decimals.

        Output:
        -------
        summary_residuals_statistics --> (pd.DataFrame) : Resume of descriptive statistics of simulated residuals.
        """

        # Check of <<residuals_simulated>>
        Check.vector_(obj=residuals_simulated, name='residuals_simulated')

        # Check of <<decimals>>
        Check.decimals_(obj=decimals)

        # Compute of descriptive statistics
        self._Count_ = int(residuals_simulated.shape[0])
        self._Mean_ = round(residuals_simulated.mean(), decimals)
        self._Variance_ = round(residuals_simulated.var(), decimals)
        self._Q25_ = round(np.quantile(residuals_simulated, q=0.25), decimals)
        self._Q50_ = round(np.quantile(residuals_simulated, q=0.50), decimals)
        self._Q75_ = round(np.quantile(residuals_simulated, q=0.75), decimals)
        self._Min_ = round(residuals_simulated.min(), decimals)
        self._Max_ = round(residuals_simulated.max(), decimals)
        self._Skewness_ = round(scipy.stats.skew(residuals_simulated), decimals)
        self._Kurtosis_ = round(scipy.stats.kurtosis(residuals_simulated), decimals)

        # Resume of descriptive statistics in dataframe
        statistics_values = [self._Count_, self._Mean_, self._Variance_, self._Q25_, self._Q50_, self._Q75_, self._Min_, self._Max_, self._Skewness_, self._Kurtosis_]
        statistics_names = ["Count", "Mean", "Variance", "Q25", "Q50", "Q75", "Min", "Max", "Skewness", "Kurtosis"]
        self.summary_residuals_statistics = pd.DataFrame({
            "Name": statistics_names,
            "Value": statistics_values
        })

        # Output of resume of descriptive statistics
        return self.summary_residuals_statistics

    # Function to visualize the distribution of simulated residuals
    def Visualization(self, residuals_simulated, figure="histogram"):

        """
        Objective:
        ----------
        Function to visualize the distribution of simulated residuals.

        Parameters:
        ---------
        residuals_simulated --> (np.ndarray, list) : data of simulated residuals.
        figure --> (str) | default='histogram' : type of figure wanted among histogram, boxplot, qqplot.

        Output:
        -------
        histogram --> plot : histogram of simulated residuals.
        boxplot --> plot : boxplot of simulated residuals.
        qqplot --> plot : qqplot of simulated residuals.
        """

        # Check of <<residuals_simulated>>
        Check.vector_(obj=residuals_simulated, name='residuals_simulated')

        # Check of <<figure>>
        Check.string_(obj=figure, name='figure')

        # Histogram of simulated residuals
        if figure == "histogram":
            sns.histplot(residuals_simulated, stat="density", color=Visual_Identity.fill_violet_color, linestyle='-', edgecolor=Visual_Identity.edge_orange_color, alpha=1)
            sns.kdeplot(residuals_simulated, color=Visual_Identity.edge_orange_color)
            xmin, xmax, ymin, ymax = plt.axis()
            plt.title(f"Histogram of simulated residuals ")
            plt.text(xmax * (1-0.05),
                     ymax * (1-0.05),
                     f"$N$={self._Count_}\n"
                     f" $\\mu$={self._Mean_}\n"
                     f" $\\sigma^{2}$={self._Variance_}\n"
                     f" $skewness$={self._Skewness_}\n"
                     f" $kurtosis$={self._Kurtosis_}",
                     ha='right',
                     va='top')
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            return plt.plot()

        # Boxplot of simulated residuals
        elif figure == "boxplot":
            sns.boxplot(x=residuals_simulated, linecolor=Visual_Identity.edge_orange_color, color=Visual_Identity.fill_violet_color, linewidth=1)
            plt.title(f"Boxplot of simulated residuals")
            plt.xlabel('residuals')
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            return plt.plot()

        # QQplot of simulated residuals
        elif figure == "qqplot":
            pp = sm.ProbPlot(residuals_simulated, fit=True)
            qq = pp.qqplot(marker='.', markerfacecolor=Visual_Identity.fill_violet_color, markeredgecolor=Visual_Identity.fill_violet_color, alpha=1,
                           markersize=12)
            sm.qqline(qq.axes[0], line='45', color=Visual_Identity.edge_orange_color)
            plt.title("QQplot of simulated residuals")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            return plt.show()
        else:
            raise ValueError("The value of <<figure>> is not correct. <<figure>> must take a value among : 'histogram', 'boxplot', 'qqplot'.")