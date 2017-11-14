using System;
using MathNet.Numerics.LinearAlgebra;

namespace MLSharp.LinearRegression
{
    public class GradientDescentLinearRegression : ILinearRegression
    {
        private Matrix<double> _theta = Matrix<double>.Build.DenseOfArray(new double[,]
        {
            {0, 0}
        });

        public void Fit(Matrix<double> trainFeatures, Matrix<double> trainResults)
        {
            var alfa = 0.01;
            for (int i = 0; i < 10000; i++)
            {
                var theta0 = _theta.At(0, 0) - (alfa * 1 / (trainResults.RowCount) * (trainFeatures * _theta.Transpose() - trainResults).Sum());
                var theta1 = _theta.At(0, 1) - (alfa * 1 / (trainResults.RowCount) * ((trainFeatures * _theta.Transpose()- trainResults).Transpose()) * trainFeatures).Sum();

                _theta = Matrix<double>.Build.DenseOfArray(new[,]
                {
                    {theta0, theta1}
                });

                var cost = CalculateCost(trainFeatures, trainResults, _theta);
            }
        }

        public Matrix<double> Predict(Matrix<double> features)
        {
            return features * _theta.Transpose();
        }

        private static double CalculateCost(Matrix<double> trainFeatures, Matrix<double> trainResults, Matrix<double> theta)
        {
            var matrix = (trainFeatures * theta.Transpose()- trainResults);
            var cost = 1 / (2.0f * trainFeatures.RowCount) * matrix * matrix.Transpose();
            return cost[0, 0];
        }
    }
}