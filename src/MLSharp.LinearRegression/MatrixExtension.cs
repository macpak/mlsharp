using MathNet.Numerics.LinearAlgebra;

namespace MLSharp.LinearRegression
{
    public static class MatrixExtension
    {
        public static double Sum(this Matrix<double> matrix)
        {
            double sum = 0;
            for (int i = 0; i < matrix.RowCount; i++)
            for (int j = 0; j < matrix.ColumnCount; j++)
                sum += matrix.At(i, j);
            return sum;
        }
    }
}