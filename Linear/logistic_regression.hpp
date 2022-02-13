#include "../Eigen/Dense"
#include <cmath>

Eigen::ArrayXXf logisticFunction(Eigen::ArrayXXf& x)
{
    return 1.0/(1 + (-x).exp());
}

class LogisticRegression
{
    private:
        Eigen::ArrayXXf weights;
        int nExamples;
        int nDims;
        bool hasBeenFit;
        int maxIter;
        float learningRate; 
    
    public:    
        LogisticRegression(int maxIter=250000) : weights(Eigen::ArrayXXf()), nExamples(0), nDims(0), hasBeenFit(false), maxIter(maxIter), learningRate(0) {};
        void fit(const Eigen::ArrayXXf& x, const Eigen::ArrayXXf& y, float lr=0.0001);
        Eigen::ArrayXXf predict_proba(const Eigen::ArrayXXf& x);
        Eigen::ArrayXXf predict(const Eigen::ArrayXXf& x);


};

void LogisticRegression::fit(const Eigen::ArrayXXf& x, const Eigen::ArrayXXf& y, float lr)
{
    nExamples = x.rows();
    nDims = x.cols();
    learningRate = lr;

    Eigen::ArrayXXf fitX = Eigen::ArrayXXf::Ones(nExamples, nDims+1);
    fitX.block(0, 1, nExamples, nDims) = x;
    
    Eigen::ArrayXXf solution = Eigen::ArrayXXf::Zero(nDims + 1, 1);
    Eigen::ArrayXXf error = Eigen::ArrayXXf::Ones(nDims + 1, 1);

    for(int count = 0; count < maxIter && error.abs().mean() > 0.001; ++count)
    {
        Eigen::ArrayXXf pred = (fitX.matrix() * solution.matrix()).array();
        pred = logisticFunction(pred);
        
        error = (fitX.colwise() * ((y - pred)(Eigen::all, 0))).colwise().sum();

        solution += learningRate * error.transpose();
    }

    weights = solution;
    
}

Eigen::ArrayXXf LogisticRegression::predict_proba(const Eigen::ArrayXXf& x)
{
    auto nIn = x.rows();
    Eigen::ArrayXXf extendedX = Eigen::ArrayXXf::Ones(nIn, nDims+1);
    extendedX.block(0, 1, nIn, nDims) = x;

    Eigen::ArrayXXf logits = extendedX.matrix() * weights.matrix();
    return logisticFunction(logits);
}

Eigen::ArrayXXf LogisticRegression::predict(const Eigen::ArrayXXf& x)
{
    Eigen::ArrayXXf preds = predict_proba(x);
    preds = (preds <= 0.5).select(0, preds);
    preds = (preds > 0.5).select(1, preds);

    return preds;
}
