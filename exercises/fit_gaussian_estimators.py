from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mean = 10
    var = 1
    X = np.random.normal(mean,var,1000)
    Qu1 = UnivariateGaussian()
    Qu1.fit(X)
    print(Qu1.mu_, Qu1.var_)

    # Question 2 - Empirically showing sample mean is consistent
    Qu2 = UnivariateGaussian()
    distance = []
    i = 10
    while i <= 1000:
        Y = X[:i]
        Qu2.fit(Y) 
        abs_dist = np.abs(Qu2.mu_ - mean)
        distance.append(abs_dist)
        i += 10
    
    axis_x = np.linspace(10,1000,100).astype(int)
    
    go.Figure([go.Scatter(x=axis_x, y=distance, mode='markers+lines')],
          layout=go.Layout(title=r"$\text{Absolute Distance Between The Estimated- And True Value of The Expectation, As a Function of The Sample Size}$", 
                  xaxis_title="$m\\text{ - number of samples}$", 
                  yaxis_title="r$\distance\mu$",
                  height=300)).show()
    
    # Question 3 - Plotting Empirical PDF of fitted model
    axis_x = X
    axis_y = Qu1.pdf(X)
    go.Figure([go.Scatter(x=axis_x, y=distance, mode='markers')],
          layout=go.Layout(title=r"$\text{Empirical PDF Function Under The Fitted Model}$", 
                  xaxis_title="$m\\text{ - samples}$", 
                  yaxis_title="r$\PDF of X$",
                  height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]
    X = np.random.multivariate_normal(mu, cov, 1000)
    Qu4 = MultivariateGaussian()
    Qu4.fit(X)
    print("mu:", Qu4.mu_)
    print("cov:", Qu4.cov_)

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
