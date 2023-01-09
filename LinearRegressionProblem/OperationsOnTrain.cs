using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace LinearRegressionProblem
{
    internal class OperationsOnTrain
    {
        public void TrainSet()
        {
            Console.WriteLine("Hello World!");
            var mlcontext = new MLContext();
            var lines = System.IO.File.ReadAllLines("C:/Users/Admin/Desktop/WebPractice/MachineLearning/LinearRegression/LinearRegressionProblem/train.csv").Skip(1).TakeWhile(t => t != null);

            List<DataPoint> itemlist = new List<DataPoint>();
            // Create a small dataset as an IEnumerable.
            foreach (var item in lines)
            {
                var values = item.Split(',');
                itemlist.Add(new DataPoint()
                {
                    x = float.Parse(values[0]),
                    y = float.Parse(values[1])

                });

            }
            Console.WriteLine("********Values from list before removing null*********");
            /*
            foreach (var item in itemlist)
            {
                Console.WriteLine(item.x + "        " + item.y);
            }
            */
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Before---" + itemlist.Count);
            Console.ResetColor();
            for (int i = 0; i < itemlist.Count; i++)
            {
                foreach (var item in itemlist.ToList())
                {
                    var itemToRemove = itemlist.FirstOrDefault(r => r.x == 0);

                    if (itemToRemove != null)
                        itemlist.Remove(itemToRemove);
                }
            }
            Console.WriteLine("----------------------------------After Removing null values----------------------------------------");
            /*
            foreach (var item in itemlist)
            {
                Console.WriteLine(item.x + "        " + item.y);
            }
            */
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("After---" + itemlist.Count);
            Console.ResetColor();
            /*string trainpath = @"C:/Users/Admin/Desktop/WebPractice/MachineLearning/LinearRegression/LinearRegressionProblem/trainWithoutNull.csv";
            StreamWriter sw = new StreamWriter(trainpath);
            CsvWriter cw = new CsvWriter(sw, CultureInfo.InvariantCulture);
            cw.WriteRecords(itemlist);
            */

            IDataView data1 = mlcontext.Data.LoadFromTextFile<DataPoint>(@"C:/Users/Admin/Desktop/WebPractice/MachineLearning/LinearRegression/LinearRegressionProblem/trainWithoutNull.csv", hasHeader: true, separatorChar: ',');
            var columnPair = new[]
            {
                new InputOutputColumnPair("x"),
                new InputOutputColumnPair("y")
            };
            var normalize = mlcontext.Transforms.NormalizeMeanVariance(columnPair,
                fixZero: false);
            var normalizeFixZero = mlcontext.Transforms.NormalizeMeanVariance(columnPair,
                fixZero: true);

            var normalizeTransform = normalize.Fit(data1);
            var transformedData = normalizeTransform.Transform(data1);
            var normalizeFixZeroTransform = normalizeFixZero.Fit(data1);
            var fixZeroData = normalizeFixZeroTransform.Transform(data1);
            var age = transformedData.GetColumn<float>("x");
            var salary = transformedData.GetColumn<float>("y");

            List<DataPoint> ListOfAllColumns = new List<DataPoint>();
            var preview = transformedData.Preview();
            //Headers
            foreach (var col in preview.Schema)
            {
                //Console.WriteLine(col.Index); 
                if (((col.Index % 2) == 0))
                {
                    string name = col.Name;
                    Console.Write(col.Name + "\t");
                }
            }

            Console.WriteLine();
            //All values
            //for (int j = 0; j < preview.RowView.Length; j++)
            //{
            //    for (int i = 1; i < 4; i++)
            //    {
            //        if (i % 2 != 0)
            //        {
            //            Console.Write(preview.RowView[j].Values[i].Value + "\t");
            //        }
            //    }
            //    Console.WriteLine();
            //}

            //adding 
            for (int j = 0; j < preview.RowView.Length; j++)
            {
                ListOfAllColumns.Add(new DataPoint()
                {
                    x = ((float)preview.RowView[j].Values[1].Value),
                    y = ((float)preview.RowView[j].Values[3].Value)
                });
            }
            Console.WriteLine("****************");
            /*
            foreach (var item in ListOfAllColumns)
            {
                Console.WriteLine(item.x + "              " + item.y);
            }
            */
            List<float> xvalue = new List<float>();

            for (int j = 0; j < preview.RowView.Length; j++)
            {
                xvalue.Add((float)preview.RowView[j].Values[1].Value);
            }

            List<float> yvalue = new List<float>();

            for (int j = 0; j < preview.RowView.Length; j++)
            {
                yvalue.Add((float)preview.RowView[j].Values[3].Value);
            }


            float[] X_train = xvalue.ToArray();
            float[] y_train = yvalue.ToArray();

            // Model

            var linearRegressor = new LinearRegressor();

            // model 
            linearRegressor.Fit(X_train, y_train);

            // Code for Test set

            IDataView data2 = mlcontext.Data.LoadFromTextFile<DataPoint>(@"C:/Users/Admin/Desktop/WebPractice/MachineLearning/LinearRegression/LinearRegressionProblem/testAfterNoNull.csv", hasHeader: true, separatorChar: ',');
            var columnPair1 = new[]
            {
                new InputOutputColumnPair("x"),
                new InputOutputColumnPair("y")
            };
            var normalize1 = mlcontext.Transforms.NormalizeMeanVariance(columnPair,
                fixZero: false);
            var normalizeFixZero1 = mlcontext.Transforms.NormalizeMeanVariance(columnPair,
                fixZero: true);

            var normalizeTransform1 = normalize.Fit(data2);
            var transformedDataTest = normalizeTransform1.Transform(data2);
            var normalizeFixZeroTransform1 = normalizeFixZero1.Fit(data2);
            var fixZeroData1 = normalizeFixZeroTransform1.Transform(data2);
            var age1 = transformedDataTest.GetColumn<float>("x");
            var salary1 = transformedDataTest.GetColumn<float>("y");

            List<DataPoint> ListOfAllColumns2 = new List<DataPoint>();
            var previewTest = transformedDataTest.Preview();
            //Headers
            foreach (var col in previewTest.Schema)
            {
                //Console.WriteLine(col.Index); 
                if (((col.Index % 2) == 0))
                {
                    string name = col.Name;
                    Console.Write(col.Name + "\t");
                }
            }
            Console.WriteLine();

            for (int j = 0; j < previewTest.RowView.Length; j++)
            {
                ListOfAllColumns2.Add(new DataPoint()
                {
                    x = ((float)previewTest.RowView[j].Values[1].Value),
                    y = ((float)previewTest.RowView[j].Values[3].Value)
                });
            }
            Console.WriteLine("**************************");
            /*
            foreach (var item in ListOfAllColumns2)
            {
                Console.WriteLine(item.x + "              " + item.y);
            }
            */

            List<float> xvalueTest = new List<float>();

            for (int j = 0; j < previewTest.RowView.Length; j++)
            {
                xvalueTest.Add((float)previewTest.RowView[j].Values[1].Value);
            }

            List<float> yvalueTest = new List<float>();

            for (int j = 0; j < previewTest.RowView.Length; j++)
            {
                yvalueTest.Add((float)previewTest.RowView[j].Values[3].Value);
            }

            float[] x_test = xvalueTest.ToArray();
            float[] y_test = yvalueTest.ToArray();

            // Finding y predict ::
            //var linearRegressor = new LinearRegressor();

            var predictions = linearRegressor.Predict(x_test);

            float[] y_predict = predictions.ToArray();
            List<PredictModel>  PredictList = new List<PredictModel>();

            for (int j = 0; j < previewTest.RowView.Length; j++)
            {
                PredictList.Add(new PredictModel()
                {
                    Y_Predict = y_predict[j],
                    Y_test = y_test[j]
                });
            }

            string path1 = @"C:/Users/Admin/Desktop/WebPractice/MachineLearning/LinearRegression/LinearRegressionProblem/yPredict_yTest.csv";
            using (StreamWriter sw = new StreamWriter(path1))
            {
                using CsvWriter cw = new CsvWriter(sw, System.Globalization.CultureInfo.InvariantCulture);
                cw.WriteRecords(PredictList);
            }
            // End of Test set

            // to check model we test accuracy

            // normalize test.csv
            //we get x_test,y_test
            //now we pass X_test in 161 to predict we get y_predict as output

            // replace X with x test we will get y predict
            //compare by mean absolute error (ytest - yPredict)/N
            //1 means 100% error, o means no error

            Console.WriteLine("-------------------------------------------------------------------------------");
            Console.WriteLine("Predictions:");
            Console.WriteLine($"{string.Join(", ", predictions.Select(p => p.ToString()))}");


            // calculating the accuracy
            int N = PredictList.Count;

            float[] Accuracy = {};
            List<float> AccuracyList = new List<float>();
            for (int j = 0; j < predictions.Length; j++)
            {
                AccuracyList.Add((y_test[j] - y_predict[j]));
            }

            float AccuracyValue = 0;

            Console.WriteLine("*******************Calculating Accuary by Root Mean Square!********************");
            foreach( var item in AccuracyList)
            {
                //Console.WriteLine(item.ToString());
                AccuracyValue += item;
            }

            double Square = (Math.Pow(AccuracyValue, 2))/N;
            Console.WriteLine("Root Mean Square : "+Square);
        }
    }
}
