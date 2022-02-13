#include <iostream>
#include <random>
#include "Util/random_utils.hpp"
#include "Linear/logistic_regression.hpp"

int main()
{
    //TODO: Make a more interesting/harder example. I really miss numpy's facilites here.
    int samplesPerClass = 100;
    Eigen::ArrayXXf train_class0 = randomUniform(samplesPerClass, 2, -2, 1);
    Eigen::ArrayXXf train_class1 = randomUniform(samplesPerClass, 2, 1.5, 3);
    Eigen::ArrayXXf train_X(2 * samplesPerClass, 2);
    train_X << train_class0, train_class1;

    Eigen::ArrayXXf labels0 = Eigen::ArrayXXf::Zero(samplesPerClass, 1);
    Eigen::ArrayXXf labels1 = Eigen::ArrayXXf::Ones(samplesPerClass, 1);
    Eigen::ArrayXXf train_y(2 * samplesPerClass, 1);
    train_y << labels0, labels1;

    auto lr = LogisticRegression(250000);
    lr.fit(train_X, train_y, 0.0000001);

    Eigen::ArrayXXf test_class0 = randomUniform(samplesPerClass, 2, -2, 1);
    Eigen::ArrayXXf test_class1 = randomUniform(samplesPerClass, 2, 1.5, 3);
    Eigen::ArrayXXf test_X(2 * samplesPerClass, 2);
    test_X << test_class0, test_class1;

    Eigen::ArrayXXf test0 = Eigen::ArrayXXf::Zero(samplesPerClass, 1);
    Eigen::ArrayXXf test1 = Eigen::ArrayXXf::Ones(samplesPerClass, 1);
    Eigen::ArrayXXf test_y(2 * samplesPerClass, 1);
    test_y << test0, test1;

    Eigen::ArrayXXf pred = lr.predict(test_X);

    std::cout << "ACCURACY: " << ((pred == test_y).cast<int>().sum()) / (2.0*samplesPerClass) << std::endl;

    return 0;
}