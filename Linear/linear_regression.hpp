#include "../Eigen/Dense"

class LinearRegression
{
    private:
        Eigen::ArrayXXf weights;
        int nExamples;
        int nDims;
        bool hasBeenFit;
        int maxIter;
        float learningRate; 
    
    public:    
        LinearRegression(int maxIter=250000) : weights(Eigen::ArrayXXf()), nExamples(0), nDims(0), hasBeenFit(false), maxIter(maxIter), learningRate(0) {};
        void fit(const Eigen::ArrayXXf& x, const Eigen::ArrayXXf& y, bool lls=true, float lr=0.0001);
        Eigen::ArrayXXf predict(const Eigen::ArrayXXf& x);


};

void LinearRegression::fit(const Eigen::ArrayXXf& x, const Eigen::ArrayXXf& y, bool lls, float lr)
{
    nExamples = x.rows();
    nDims = x.cols();
    learningRate = lr;

    Eigen::ArrayXXf fitX = Eigen::ArrayXXf::Ones(nExamples, nDims+1);
    fitX.block(0, 1, nExamples, nDims) = x;
    
    if(lls)
    {
        Eigen::MatrixXf solution = fitX.matrix().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y.matrix());
        weights = solution.array();
    }
    else
    {
        Eigen::ArrayXXf solution = Eigen::ArrayXXf::Zero(nDims + 1, 1);
        Eigen::ArrayXXf error = Eigen::ArrayXXf::Ones(nDims + 1, 1);

        for(int count = 0; count < maxIter && error.abs().mean() > 0.001; ++count)
        {
            Eigen::ArrayXXf pred = (fitX.matrix() * solution.matrix()).array();
            
            error = (fitX.colwise() * ((pred - y)(Eigen::all, 0))).colwise().sum();

            solution -= learningRate * error.transpose();
        }

        weights = solution;
    }
}

Eigen::ArrayXXf LinearRegression::predict(const Eigen::ArrayXXf& x)
{
    auto nIn = x.rows();
    Eigen::ArrayXXf extendedX = Eigen::ArrayXXf::Ones(nIn, nDims+1);
    extendedX.block(0, 1, nIn, nDims) = x;

    return extendedX.matrix() * weights.matrix();
}