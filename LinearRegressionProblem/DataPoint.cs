using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace LinearRegressionProblem
{
    public class DataPoint
    {
        [LoadColumn(0)]
        public float x { get; set; }
        [LoadColumn(1)]
        public float y { get; set; }
    }


    public class PredictModel
    {
        [LoadColumn(0)]
        public float Y_Predict { get; set; }
        [LoadColumn(1)]
        public float Y_test { get; set; }
    }
}
