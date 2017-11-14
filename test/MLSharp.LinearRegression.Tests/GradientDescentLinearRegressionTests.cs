using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Xunit;

namespace MLSharp.LinearRegression.Tests
{
    public class GradientDescentLinearRegressionTests
    {
        [Fact]
        public void Test()
        {
            var features = DenseMatrix.Build.DenseOfArray(new double[,]
            {
                {1, 1.0f},
                {1, 2.0f},
                {1, 3.0f},
                {1, 4.0f},
                {1, 5.0f},
                {1, 6.0f},
                {1, 7.0f},
                {1, 8.0f},
            });
            var result = DenseMatrix.Build.DenseOfArray(new double[,]
            {
                {2.0f},
                {3.0f},
                {4.0f},
                {5.0f},
                {6.0f},
                {7.0f},
                {8.0f},
                {9.0f},

            });
            var lineraRegression = new GradientDescentLinearRegression();
            lineraRegression.Fit(features, result);
            var prediction = lineraRegression.Predict(DenseMatrix.Build.DenseOfArray(new double[,]
            {
                {1, 101.0f}
            }));
        }
    }
}
