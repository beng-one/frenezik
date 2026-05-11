"""|----- frenezik -----|"""

# Data processing
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import Visual_Identity, Check, Compute

#  Class to simulate inputs and output data
class TargetPredictors:

    """
    Objective:
    ----------
    This class is used to simulate inputs data X and output data y of the population and make descriptive analysis.

    Parameters:
    ---------
    population_size --> (int) : population size wanted.
    true_coefficients --> (list, np.ndarray) : true parameters to estimate.

    Functions:
    ---------
    Residuals.Simulation : Function to simulate inputs data X and output data y of the population.
    Residuals.Sampling : Function to extract sample data from population data randomly.
    Residuals.Visualization_X_y : Function to visualize simulated data.
    """

    def __init__(self, population_size, true_coefficients):

        # Check of <<population_size>>
        Check.data_size_(obj=population_size, name='population_size')

        # Check of <<true_coefficients>>
        Check.vector_(obj=true_coefficients, name='true_coefficients')

        self.true_coefficients = true_coefficients
        self.population_size = population_size

    # Function to simulate inputs data X and output data y of the population
    def Simulation(self, residuals_simulated):

        """
        Objective:
        ----------
        Function to simulate inputs data X and output data y of the population.

        Parameters:
        ---------
        residuals_simulated --> (list, np.ndarray) : data of simulated residuals.

        Output:
        -------
        data_simulated_population --> (pd.DataFrame) : simulated data of the population.
        """

        # Check of <<residuals_simulated>>
        Check.vector_(obj=residuals_simulated, name='residuals_simulated')

        # Number of true coefficients
        number_coefficient = len(self.true_coefficients)

        # Simulation of inputs data X
        X = np.zeros(shape=(self.population_size, number_coefficient))
        for col in range(0, number_coefficient):
            gaussian_params = np.random.rand(1, 2)
            mu = gaussian_params[:, 0]
            sigma = gaussian_params[:, 1]
            X[:, col] = np.random.normal(mu, sigma, self.population_size)

        # Simulation of output data y
        self.true_coefficients = np.array(self.true_coefficients).reshape(1, number_coefficient)
        y = X @ self.true_coefficients.T + residuals_simulated.reshape(self.population_size, 1)

        # Merge of X and y data
        X_y_data = np.hstack((y, X))

        # Conversion to a dataframe
        X_y_col_names = ['y']
        for i in range(1, number_coefficient + 1):
            col_name = f"X_{i}"
            X_y_col_names.append(col_name)
        self.data_simulated_population = pd.DataFrame(data=X_y_data, columns=X_y_col_names)

        # Output of simulated data of the population
        return self.data_simulated_population

    # Function to extract sample data from population data randomly
    def Sampling(self, population_simulated, sample_size, random_seed):

        """
        Objective:
        ----------
        Function to extract sample data from population data randomly.

        Parameters:
        ---------
        population_simulated --> (pd.DataFrame) : simulated data of the population.
        sample_size --> (int) : size of the sample wanted.
        random_seed --> (float, None) : number that ensure the reproducibility of results

        Output:
        -------
        data_simulated_sample --> (pd.DataFrame) : simulated data of the sample.
        """

        # Check of <<population_simulated>>
        Check.dataframe_(obj=population_simulated, name='population_simulated')

        # Check of <<sample_size>>
        Check.data_size_(obj=sample_size, name='sample_size')

        # Check of <<random_seed>>
        Check.random_seed_(obj=random_seed)

        # Random sampling
        data_simulated_sample = population_simulated.sample(sample_size, random_state=random_seed)

        # Output simulated data of sample
        return data_simulated_sample

    # Function to visualize simulated data
    @staticmethod
    def Visualization_X_y(data_simulated, figure):

        """
        Objective:
        ----------
        Function to visualize simulated data.

        Parameters:
        ---------
        data_simulated --> (pd.DataFrame) : simulated data of X and y.
        figure --> (str) | default='lineplot' : type of figure wanted among lineplot, scatterplot, correlation_matrix.

        Output:
        -------
        lineplot --> plot : lineplot of X and y.
        scatterplot --> plot : Scatterplot between y and Y data.
        correlation_matrix --> dataframe : correlation matrix of simulated X and y.
        """

        # Check of <<data_simulated>>
        Check.dataframe_(obj=data_simulated, name='data_simulated')

        # Check of <<figure>>
        Check.string_(obj=figure, name='figure')

        # Customization of color palette
        cmaps = plt.colormaps
        variable_number = data_simulated.shape[1]
        color_index_inf = int(variable_number * (1 - 0.70))
        color_index_sup = int(variable_number * (1 + 0.30))
        colors = cmaps.get_cmap("magma").resampled(color_index_sup).colors

        # Lineplot
        if figure == "lineplot":
            fig, axs = plt.subplots()
            for i, col in enumerate(data_simulated.columns):
                axs.plot(data_simulated[col], label=col, linestyle="-", color=colors[color_index_inf+i], linewidth=1)
            plt.title(f"lineplot of data simulated")
            plt.xlabel(f"Numbers of observations")
            plt.ylabel(f"Values")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            return plt.show()

        # Scatterplot
        elif figure == "scatterplot":
            fig, axs = plt.subplots()
            for i, var in enumerate(data_simulated.columns):
                axs.plot(data_simulated[var], label=var, linestyle="dotted", color=colors[i], marker="")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            return plt.show()

        # Matrix of correlation
        elif figure == "correlation_matrix":
            data_simulated_corr = data_simulated.corr()
            sns.heatmap(data_simulated_corr, vmin=-1, vmax=1, cmap='magma_r', annot=True, cbar=True,
                        linewidths=1, annot_kws={'fontsize': 9})
            plt.title("Correlation matrix of simulated data")
            return plt.show()
        else:
            raise ValueError("The value of <<figure>> is not correct. <<figure>> must take a value among : 'lineplot', 'scatterplot', 'correlation_matrix'.")