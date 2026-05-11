"""|----- frenezik -----|"""

# Data processing
import numpy as np
import pandas as pd

# Regression models
from sklearn.linear_model import Ridge, Lasso
from scipy import stats

# Class to define visual identity of package
class Visual_Identity:

    '''
    Objective:
    ----------
    Class to define visual identity of frenezik package. These colors are compatible with << magma >> color palet.

    Parameters:
    -----------
    fill_violet_color --> (str) : first Color
    edge_orange_color --> (str) : second color.
    '''

    fill_violet_color = "#3d0261"
    edge_orange_color = "#fbba6d"

# Class for check
class Check:


    '''
    Objective:
    ----------
    Class for error handling with a focus on TypeError, ValueError.
     '''

    # Check of <<population_size>>
    @staticmethod
    def data_size_(obj, name):
        if not isinstance(obj, int):
            raise TypeError(f"The type of <<{name}>> is not correct. <<{name}>> must be of type 'int'.")
        if not obj >= 0:
            raise ValueError(f"The value of <<{name}>> is not correct. <<{name}>> must be greater than or equal to 0.")

    # Check of <<random_seed>>
    @staticmethod
    def random_seed_(obj):
        if not isinstance(obj, (int, type(None))):
            raise TypeError("The type of <<random_seed>> is not correct. <<random_seed>> must be of type 'float' or 'None'.")

    # Check of <<law>>
    @staticmethod
    def law_(obj):
        if not isinstance(obj, str):
            raise TypeError("The type of <<law>> is not correct. <<law>> must be of type 'str'.")
        if not obj in ("normal", "student", "pareto", "weibull"):
            raise ValueError("The value of <<law>> is not correct. <<law>> must take a value among : 'normal', 'student', 'pareto', 'weibull'.")

    # Check of <<law_parameters>>
    @staticmethod
    def law_parameters_(obj):
        if not isinstance(obj, (list, np.ndarray, None)):
            raise TypeError(
                "The type of <<law_parameters>> is not correct. <<law_parameters>> must be of type 'list', 'np.ndarray' or 'None'.")

    # Check of <<residuals_simulated>>
    @staticmethod
    def vector_(obj, name):
        if not isinstance(obj, np.ndarray):
            if isinstance(obj, list):
                obj = np.array(obj)
                return obj
            else:
                raise TypeError(
                    f"The type of <<{name}>> is not correct. <<{name}>> must be of type 'np.ndarray' or 'list'.")

    # Check of <<decimals>>
    @staticmethod
    def decimals_(obj):
        if not isinstance(obj, int):
            raise TypeError("The type of <<decimals>> is not correct. <<residuals_simulated>> must be of type 'int'.")
        if not obj >= 0:
            raise TypeError("The value of <<decimals>> is not correct. <<decimals>> must be greater than or equal to 0.")

    # Check of <<figure>>
    @staticmethod
    def string_(obj, name):
        if not isinstance(obj, str):
            raise TypeError(f"The type of <<{name}>> is not correct. <<{name}>> must be of type 'str'.")


    # Check of <<population_simulated>>
    @staticmethod
    def dataframe_(obj, name):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(
                f"The type of <<{name}>> is not correct. <<{name}>> must be of type 'pd.DataFrame'.")

    # Check of <<alpha>>
    @staticmethod
    def alpha_s_(obj):
        if not isinstance(obj, np.ndarray):

            if isinstance(obj, list):
                obj = np.array(obj)
                return obj
            else:
                if isinstance(obj, (float, int)):
                    if obj >= 0:
                        return obj
                    else:
                        raise ValueError(
                            "The value of <<alpha>> is not correct. <<alpha>> must be greater than or equal to 0.")
                else:
                    raise TypeError(
                        "The type of <<alpha>> is not correct. <<alpha>> must be of type 'float', 'np.ndarray' or 'list'.")

    # Check of <<intercept>>
    @staticmethod
    def intercept_s_(obj):
        if not isinstance(obj, (int, float)):
            raise TypeError("The type of <<intercept>> is not correct. <<intercept>> must be of type 'float' or 'int'.")


    # Check of <<variables_selected>>
    @staticmethod
    def variable_selected_(obj, name):
        if not isinstance(obj, (list, type(None))):
            raise TypeError(
                f"The type of <<{name}>> is not correct.  <<{name}>> must be of type 'list' or 'None'.")

    # Check of <<alpha_value>>
    @staticmethod
    def alpha_value_selected_(obj):
        if not isinstance(obj, (float, int, type(None))):
            raise TypeError(
                "The type of <<alpha_value_selected>> is not correct. <<alpha_value_selected>> must be of type float, 'int' or 'None'.")


# Class to compute recurrent operations
class Compute:

    # Function to show the results of penalized regression model
    @staticmethod
    def summary(X_, y_, alpha_scalar_, intercept_, random_seed_, model_):


        """
        Objective:
        ----------
        Function to model y in function of X by using a regularized regression and show results of statistics.

        Parameters:
        ---------
        X_ --> (pd.DataFrame) : matrix of inputs data X
        y_ --> (list, np.ndarray) : vector of output data y.
        alpha_scalar_  --> (int, float) : penality factor.
        intercept_ --> (bool) : intercept of model
        model_ --> (str) : type of model wanted among Ridge and Lasso.

        Output:
        -------
        summary_ridge --> (pd.DataFrame) : summary of statistics of ridge regression.
        summary_lasso --> (pd.DataFrame) : summary of statistics of lasso regression.
        """

        # Check of <<X_>>
        Check.dataframe_(obj=X_, name='X_')

        # Check of <<y_>>
        Check.vector_(obj=y_, name='y_')

        # Check of <<alpha_scalar_>>
        Check.alpha_s_(obj=alpha_scalar_)

        # Check of <<intercept_>>
        Check.intercept_s_(obj=intercept_)

        # Check of <<model_>>
        Check.string_(model_, 'model_')

        # N observations, K variables, Names of variables
        n_obs = X_.shape[0]
        k_var = X_.shape[1]
        predictors_names = X_.columns.tolist()
        X_mat = X_.values
        y_vect = y_.ravel()

        # Addition of intercept in predictors names
        if intercept_:
            predictors_names.insert(0, "Intercept")
        else:
            predictors_names = predictors_names

        # Ridge regression
        if alpha_scalar_ == 0 or model_ == 'ridge':

            # scaling of data
            X_mean = X_mat.mean(axis=0)
            y_mean = y_vect.mean()
            X_centered = X_mat - X_mean
            y_centered = y_vect - y_mean

            # X_c'X_c
            XtX = X_centered.T @ X_centered
            beta_ridge = np.linalg.solve(XtX + alpha_scalar_ * np.eye(k_var), X_centered.T @ y_centered)

            # W = (X'X + alpha * I)^(-1)
            W = np.linalg.inv(XtX + alpha_scalar_ * np.identity(k_var))

            # Intercept
            if intercept_:
                intercept_value = y_mean - X_mean @ beta_ridge
                beta_ridge_complete = np.insert(beta_ridge, 0, intercept_value)
            else:
                intercept_value = None
                beta_ridge_complete = beta_ridge

            # Predictions
            y_pred = X_centered @ beta_ridge
            residuals = y_centered - y_pred

            # Degree of freedom : df = trace(2H - HH') Estimated variance
            XtX_inv = np.linalg.inv(XtX + alpha_scalar_ * np.eye(k_var))
            H = X_centered @ XtX_inv @ X_centered.T
            df = n_obs - np.trace(2 * H - H @ H.T)
            rss = np.sum(residuals ** 2)
            sigma2_hat = rss / df

            # Variance-Covariance Matrix
            var_cov_beta = sigma2_hat * XtX_inv @ XtX @ XtX_inv
            se_beta = np.sqrt(np.diag(var_cov_beta))

            # Standard Error se
            if intercept_:
                se_intercept = np.sqrt(sigma2_hat * (
                            1 / n_obs + X_mean @ XtX_inv @ X_mean))  # SE(Intercept) = sqrt(σ² * (1/n + X̄' (X'X + λI)⁻¹ X̄))
                se_complete = np.insert(se_beta, 0, se_intercept)
            else:
                se_complete = se_beta

            # t-stats
            t_stat = np.abs(beta_ridge_complete) / se_complete

            # pvalue
            pval = 2 * (1 - stats.norm.cdf(t_stat))
            pval[pval < 2e-16] = 2e-16

            # summary results of ridge regression
            summary_rigde = pd.DataFrame({
                'Variables': predictors_names,
                'Estimators': beta_ridge_complete,
                'Std.error': se_complete,
                'T.stats': t_stat,
                'P.vals': pval
            })
            return summary_rigde

        else:

            # Lasso regression
            penalized_model = Lasso(alpha=alpha_scalar_, fit_intercept=intercept_, random_state=random_seed_)
            penalized_model.fit(X_mat, y_vect)
            regularized_estimators = penalized_model.coef_

            # Addition or not of intercept
            if intercept_ == True:
                regularized_intercept = penalized_model.intercept_
                results_complete = np.insert(regularized_estimators, 0, regularized_intercept)
            elif intercept_ == False:
                results_complete = regularized_estimators
            else:
                raise ValueError(
                    "The value of <<intercept>> is not correct. <<intercept>> must take a value among 'True' or 'False'.")

            # summary results of lasso regression
            summary_lasso = pd.DataFrame({
                'Variables': predictors_names,
                'Estimators': results_complete,
            })
            return summary_lasso

    # Fonction to compute residuals sum of square
    @staticmethod
    def residual_sum_square(beta_, X_, y_, intercept_):

        """
        Objective:
        ----------
        Fonction to compute residuals sum of square.

        Parameters:
        ---------
        beta_ --> (list, np.ndarray) : vector of coefficient regression.
        X_ --> (pd.DataFrame) : matrix of predictors variables.
        y_ --> (list, np.ndarray) : vector of target predictors.
        intercept_ --> (float) : intercept of the model.

        Output:
        -------
        rss --> (float) : the value of residuals sum of square.
        """

        # Check of <<beta>>
        Check.vector_(obj=beta_, name='beta_')

        # Check of <<y>>
        Check.vector_(obj=y_, name='y_')

        # Check of <<intercept>>
        Check.intercept_s_(obj=intercept_)

        # Check of <<X>>
        if not isinstance(X_, (list, pd.DataFrame, np.ndarray)):
            raise TypeError("The type of <<X_>> is not correct. <<X_>> must be of type pd.DataFrame, np.array, list.")

        # compute of residuals sum of square
        y_chapo = X_ @ beta_ + intercept_ * np.ones(shape=(X_.shape[0], 1))
        rss = np.sum((y_ - y_chapo) ** 2)

        # output of residuals sum of square
        return rss