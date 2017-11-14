using MathNet.Numerics.LinearAlgebra;

namespace MLSharp.LinearRegression
{
    public interface ILinearRegression
    {
        void Fit(Matrix<double> trainFeatures, Matrix<double> trainResults);
    }
}