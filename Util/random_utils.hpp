#include <random>
#include "../Eigen/Dense"

template<typename T> T sampleFromNormal(T mean, T std, std::normal_distribution<T>& nd)
{
  static std::mt19937 rng;
  
  return nd(rng);
}

//Random uniform in range [0, 1]
Eigen::ArrayXXf randomUniform(int rows, int cols)
{
    Eigen::ArrayXXf arr = Eigen::ArrayXXf::Random(rows, cols);
    return (arr + 1)/2;
}

// Random uniform in range [low, high]
Eigen::ArrayXXf randomUniform(int rows, int cols, float low, float high)
{
    return (high - low) * randomUniform(rows, cols) + low;
}

Eigen::ArrayXXf randomNormal(int rows, int cols, float mean, float std)
{
    Eigen::ArrayXXf arr = Eigen::ArrayXXf::Zero(rows, cols);
    std::normal_distribution<float> nd(mean, std);

    for(int i = 0 ; i < rows; ++i)
    {
        for(int j = 0 ; j < cols; ++j)
        {
            arr(i, j) = sampleFromNormal(mean, std, nd);
        }
    }
    return arr;
}
