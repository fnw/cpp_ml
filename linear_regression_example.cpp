#include <iostream>
#include <random>
#include "Util/random_utils.hpp"
#include "Linear/linear_regression.hpp"

int main()
{
    int numSamples = 500;
    Eigen::ArrayXXf train_X = 50.0 * randomUniform(numSamples, 1);
    Eigen::ArrayXXf train_y = 3.0 * train_X + 2.0 + randomNormal(numSamples, 1, 0, 1.5);

    Eigen::ArrayXXf test_X = 50.0 * randomUniform(numSamples, 1);
    Eigen::ArrayXXf test_y = 3.0 * test_X + 2.0;

    LinearRegression r;

    // Fit via normal equation
    r.fit(train_X, train_y);
    Eigen::ArrayXXf pred = r.predict(test_X);

    std::cout << "MEAN ABSOLUTE ERROR, linear least squares: " << (test_y - pred).abs().mean() << std::endl;

    // Fit via gradient descent
    r.fit(train_X, train_y, false, 0.000001);
    pred = r.predict(test_X);

    std::cout << "MEAN ABSOLUTE ERROR, gradient descent: " << (test_y - pred).abs().mean() << std::endl;

    return 0;
}