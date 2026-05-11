"""|----- frenezik -----|"""

# Data preprocessing
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
from scipy.stats import alpha
from .utils import Visual_Identity, Check, Compute

# Regression models
from sklearn.linear_model import Ridge, Lasso

# Class to make a regularized regression
class Regularization:

    """
    Objective:
    ----------
    This class is used to model y in function of X by using a regularized regression.

    Parameters:
    ---------
    predictors --> (pd.DataFrame) : matrix of inputs data X.
    target --> (list, np.ndarray) : vector of output data y.
    alpha --> (list, np.ndarray, float, int) : penality factor.
    intercept | default--> (bool) : intercept of model.
    random_seed --> (float, None) : number that ensure the reproducibility of results.

    Functions:
    ---------
    Residuals.Penalized_Regression : Function to model y in function of X by using a regularized regression.
    Residuals.Visualization_Shrinking : Function to visualize the shrinkage of estimated coefficients in function of alpha parameter.
    Residuals.Predict : Function to compute prediction of y  and estimated residuals of regularized regression model.
    Residuals.Visualization_Residuals : Function pour visualize estimated residuals in function of estimated penalized coefficients.
    """

    def __init__(self, predictors, target, alpha, intercept, random_seed):

        # Check of <<predictors>>
        Check.dataframe_(obj=predictors, name='predictors')

        # Check of <<target>>
        Check.vector_(obj=target, name="target")

        # Check of <<alpha>>
        Check.alpha_s_(obj=alpha)

        # Check of <<intercept>>
        Check.intercept_s_(obj=intercept)

        # Check of <<random_seed>>
        Check.random_seed_(obj=random_seed)

        self.predictors = predictors
        self.target = target
        self.alpha = alpha
        self.intercept = intercept
        self.random_seed = random_seed

    # Function to model y in function of X by using a regularized regression
    def Penalized_Regression(self, model):

        """
        Objective:
        ----------
        Function to model y in function of X by using a regularized regression.
        The optimization algorithm used is LSQR and the data X and y are standardized with mean = True and sd = False for more stability.
        So : X_centered = X - X.mean() and y_centered = y - y.mean()

        Parameters:
        ---------
        model --> (str) : type of model wanted among Ridge and Lasso.

        Output:
        -------
        penalized_coefficients_table --> (pd.DataFrame) : resume of penalized coefficients in function of alpha.
        summary_ridge --> (pd.DataFrame) : summary of statistics of ridge regression.
        summary_lasso --> (pd.DataFrame) : summary of statistics of lasso regression.
        """

        # Check of <<model>>
        Check.string_(obj=model, name='model')

        # list of variables names
        predictors_names = list(self.predictors.columns)

        # Addition of intercept
        if self.intercept == True:
            predictors_names.insert(0, "Intercept")
        elif self.intercept == False:
            predictors_names = predictors_names
        else:
            raise ValueError(
                "The value of <<intercept>> is not correct. <<intercept>> must take a value among 'True' or 'False'.")

        # results of regularized regression with alpha scalar
        if isinstance(self.alpha, (float, int)):
            alpha_scalar = self.alpha
            result_model = Compute.summary(X_=self.predictors, y_=self.target, alpha_scalar_=alpha_scalar,
                                           intercept_=self.intercept, random_seed_=self.random_seed, model_=model)
            return result_model

        # summary results of penalized coefficients according to alpha
        else:
            penalized_coefficients_table = pd.DataFrame({
                'Variables': predictors_names,
            })

            # Regularized regression model
            for alpha_elem in self.alpha:

                # Ridge regression model if alpha==0 by default
                if alpha_elem == 0:
                    penalized_model = Ridge(alpha=alpha_elem, fit_intercept=self.intercept,
                                            random_state=self.random_seed)

                # Regularized regression model if alpha !=0
                else:
                    if model == 'ridge':
                        # Ridge regression model
                        penalized_model = Ridge(alpha=alpha_elem, fit_intercept=self.intercept,
                                                random_state=self.random_seed)
                    elif model == 'lasso':
                        # Lasso regression model
                        penalized_model = Lasso(alpha=alpha_elem, fit_intercept=self.intercept,
                                                random_state=self.random_seed)
                    else:
                        raise ValueError(
                            f"The value of <<model>> is not correct. <<model>> must take a value among 'ridge', 'lasso'.")

                # Intercept and estimated coefficients results
                penalized_model.fit(self.predictors, self.target)
                self.result_intercept = penalized_model.intercept_
                result_coefficients = penalized_model.coef_

                # Addition of intercept
                if self.intercept == True:
                    result_complete = np.insert(result_coefficients, 0, self.result_intercept)
                elif self.intercept == False:
                    result_complete = result_coefficients
                else:
                    raise ValueError(
                        "The value of <<intercept>> is not correct. <<intercept>> must take a value among 'True' or 'False'.")

                # update of dataframe
                alpha_elem_name = f"{alpha_elem}"
                penalized_coefficients_table[alpha_elem_name] = result_complete

            # output of dataframe
            return penalized_coefficients_table

    #  Function to visualize the shrinkage of estimated coefficients in function of alpha parameter
    def Visualization_Shrinking(self, penalized_coefficients_table, variables_selected, alpha_value_selected, figure):

        """
        Objective:
        ----------
        Function to visualize the shrinkage of estimated coefficients in function of alpha parameter.

        Parameters:
        ---------
        penalized_coefficients_table --> (pd.DataFrame) : resume of penalized coefficients in function of alpha.
        variables_selected --> (list, np.ndarray, None) : the pair of variables to select. If figure == 'curve', variable_selected is None.one.
        figure --> (str) : type de figure wanted among curve.
        alpha_value_selected ----> (int, str, None) : the value of penaly factor alpha to select. If figure == 'curve', alpha_value_selected is N

        Output:
        -------
        curve --> (plot) : the regularization path.
        """

        # Check of <<penalized_coefficients_table>>
        Check.dataframe_(obj=penalized_coefficients_table, name='penalized_coefficients_table')

        # Check of <<variables_selected>>
        Check.variable_selected_(obj=variables_selected, name='variables_selected')

        # Check of <<alpha_value>>
        Check.alpha_value_selected_(obj=alpha_value_selected)

        # Check of <<figure>>
        Check.string_(obj=figure, name='figure')

        # Regularization path
        if figure == "curve":

            # Unnecessary args
            variables_selected = None
            alpha_value_selected = None

            # Preprocessing of penalized coefficients table
            df = pd.DataFrame(penalized_coefficients_table)
            df.set_index(keys="Variables", drop=True, inplace=True)
            df_T = df.T

            # Customization of color palet
            variable_number = df_T.shape[1]
            color_index_inf = int(variable_number * (1 - 0.70))
            color_index_sup = int(variable_number * (1 + 0.30))
            cmaps = plt.colormaps
            colors = cmaps.get_cmap("magma").resampled(color_index_sup).colors

            # Visualization
            fig, axs = plt.subplots()
            for i, predictor in enumerate(df_T.columns):
                axs.plot(df_T.index, df_T[predictor], label=predictor, linestyle="-", alpha=1, linewidth=2, marker="*",
                         color=colors[color_index_inf + i])
            xmin, xmax, ymin, ymax = plt.axis()
            plt.axhline(y=0, xmin=0, xmax=1, color='gray', linestyle='--')
            plt.title('Regularization path')
            plt.xlabel("alpha")
            plt.ylabel("Coefficients")
            plt.grid(visible=True, alpha=0.25, linestyle='--')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            return plt.show()
        raise ValueError(
            "The value of <<figure>> is not correct. <<figure>> must take a value among : 'curve'.")

    # Function to compute predicted y and estimated residuals of regularized regression model
    def Predict(self, penalized_coefficients_table, alpha, target, predictors, purpose):

        """
        Objective:
        ----------
        Function to compute predicted y  and estimated residuals of regularized regression model.

        Parameters:
        ---------

        penalized_coefficients_table --> (pd.DataFrame) : resume of penalized coefficients in function of alpha.
        variables_selected --> (list, np.ndarray) : the pair of variables to select.
        alpha --> (int, str) : the value of penality factor alpha to select.
        predictors --> (np.ndarray, pd.DataFrame) : matrix of inputs data X.
        target --> (list, np.ndarray) : vector of output data y.
        purpose --> (np.ndarray) : type output wanted among target_predicted, residuals_predicted and residuals_sum_square.

        Output:
        -------
        target_predicted --> (np.ndarray) : predicted target.
        residual_predicted  --> (np.ndarray) : estimated residuals.
        residuals_sum_square --> (float, int) : residuals sum of square metric.
        """

        # Check of <<penalized_coefficients_table>>
        Check.dataframe_(obj=penalized_coefficients_table, name='penalized_coefficients_table')

        # Check of <<alpha>>
        Check.alpha_s_(obj=alpha)

        # Check of <<target>>
        Check.vector_(obj=target, name='target')

        # Check of <<predictors>>
        Check.dataframe_(obj=predictors, name='predictors')

        # Check of <<purpose>>
        Check.string_(obj=purpose, name='purpose')

        # data preparation
        beta_names = predictors.columns.tolist()
        X_mat = predictors.values
        n_obs = predictors.shape[0]
        k_var = predictors.shape[1]
        y_vect = target.ravel()
        alpha_str = f'{alpha}'

        # Intercept
        if 'Intercept' in penalized_coefficients_table['Variables'].tolist():
            filter_intercept = penalized_coefficients_table['Variables'].isin(["Intercept"])
            intercept_value = penalized_coefficients_table.loc[filter_intercept, alpha_str].values
            beta_np = penalized_coefficients_table.loc[1:, alpha_str].values
            beta_vect = np.reshape(beta_np, (k_var, 1))
        else:
            intercept_value = 0
            beta_np = penalized_coefficients_table.loc[:, alpha_str].values
            beta_vect = np.reshape(beta_np, (k_var, 1))

        # Equation of prediction
        y_pred = X_mat @ beta_vect + intercept_value * np.ones(shape=(n_obs, 1))
        y_pred = np.ravel(y_pred)

        # output of target predicted
        if purpose == "target_predicted":
            return np.ravel(y_pred)

        # output of estimated residuals
        elif purpose == "residuals_predicted":
            residuals = (y_vect - y_pred) ** 2
            return residuals

        # output of residuals sum of square
        elif purpose == "residuals_sum_square":
            residuals = (y_vect - y_pred) ** 2
            rss = np.sum(residuals)
            return rss

        # output error handling
        else:
            raise ValueError(
                "The value of <<purpose>> is not correct. <<purpose>> must take a value among : 'target_predicted', 'residuals_predicted', 'residuals_sum_square'.")

    # Function pour visualize estimated residuals in function of estimated penalized coefficients
    def Visualization_Residuals(self, list_beta_1, list_beta_2, penalized_coefficients_table, predictors, target, var_1,
                                var_2, intercept, figure):

        """
        Objective:
        ----------
        Function to visualize the relation between residuals and simulated penalized regression.

        Parameters:
        ---------
        list_beta_1 --> (list, np.ndarray) : vector of simulated coefficients linked to variable 1.
        list_beta_2 --> (list, np.ndarray) : vector of simulated coefficients linked to variable 2.
        predictors --> (np.ndarray, pd.DataFrame) : matrix of inputs data X.
        target --> (list, np.ndarray) : vector of output data y.
        var_1 --> (str) : variable 1 to select associated y-axis.
        var_2 --> (str) : variable 1 to select associated x-axis.
        figure --> (str) : type of figure wanted among contour_map, surface3D.

        Output:
        -------
        contour_map --> (plot) : contour plot to visualize the relation between  residual and simulated coefficients.
        surface3D --> (plot) : 3D surface to visualize the relation between residual and simulated coefficients.
        """

        # Check of <<list_beta_1>>
        Check.vector_(obj=list_beta_1, name='list_beta_1')

        # Check of <<list_beta_2>>
        Check.vector_(obj=list_beta_2, name='list_beta_2')

        # Check of <<predictors>>
        Check.dataframe_(obj=predictors, name='predictors')

        # Check of <<true_target>>
        Check.vector_(obj=target, name='target')

        # Check of <<penalized_coefficients_table>>
        Check.dataframe_(obj=penalized_coefficients_table, name='penalized_coefficients_table')

        # Check of <<var_1>>
        Check.string_(obj=var_1, name='var_1')

        # Check of <<var_2>>
        Check.string_(obj=var_2, name='var_2')

        if not {var_1, var_2}.issubset(set(predictors.columns.tolist())):
            raise ValueError(
                f"The value of <<{var_1}>> and/ or the value of <<{var_2}>> is not correct. <<{var_1}>> et <<{var_2}>> must take values among : {predictors.columns.tolist()}.")

        # Check of <<intercept>>
        Check.intercept_s_(obj=intercept)

        # Check of <<figure>>
        Check.string_(obj=figure, name='figure')

        col_list = predictors.columns.tolist()

        # Configuration des variables employées
        X_mat = predictors[[var_1, var_2]].values
        y_vect = target
        b1 = list_beta_1
        b2 = list_beta_2
        rss = np.zeros(shape=(len(list_beta_1), len(list_beta_2)))

        # Compute residuals sum of square
        j = 0
        for p in b1:
            i = 0
            for q in b2:
                rss[i, j] = Compute.residual_sum_square(beta_=[p, q], X_=X_mat, y_=y_vect, intercept_=intercept)
                i += 1
            j += 1
        B1, B2 = np.meshgrid(b1, b2)

        # Contour plot
        if figure == "contour_map":
            plt.contour(B1, B2, rss, levels=50, linewidths=0.5, colors='yellow')
            cp = plt.contourf(B1, B2, rss, levels=100, cmap='magma')
            plt.colorbar(cp, label="RSS")
            for enum, col in enumerate(penalized_coefficients_table.columns[1:]):
                list_penalized = penalized_coefficients_table.iloc[1:, enum + 1].values
                if col == 'true_coefficients':
                    plt.scatter(list_penalized[0], list_penalized[1], alpha=1, linestyle='--', color='green',
                                label="true_coefs")
                elif col == "0.0":
                    plt.scatter(list_penalized[0], list_penalized[1], alpha=1, linestyle='--', color='yellow',
                                label="mco_coefs")
                else:
                    plt.scatter(list_penalized[0], list_penalized[1], alpha=1, linestyle='--', color='blue',
                                label="penalized_coefs")
            plt.xlabel(f"$\\beta_{col_list.index(var_1) + 1}$")
            plt.ylabel(f"$\\beta_{col_list.index(var_2) + 1}$")
            plt.title("Contour plot of residuals sum of square")
            plt.grid(True, alpha=0.15)
            plt.show()

        # Surface 3D
        elif figure == "surface3D":
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_rotate_label(True)
            ax.view_init(30, 35)
            ax.plot_surface(B1, B2, rss, cmap="magma", alpha=0.7, antialiased=False)
            ax.plot_wireframe(B1, B2, rss, color='red', alpha=0.1)
            plt.xlabel(f"$\\beta_{col_list.index(var_1) + 1}$")
            plt.ylabel(f"$\\beta_{col_list.index(var_2) + 1}$")
            ax.set_zlabel("$Residuals$", rotation=90)
            ax.set_title("3D surface of residuals sum of square", size=15)
        else:
            raise ValueError(
                "The value of <<figure>> is not correct. <<figure>> must take a value among : contour_map, surface3D.")