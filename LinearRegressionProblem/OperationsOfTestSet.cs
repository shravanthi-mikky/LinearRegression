using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace LinearRegressionProblem
{
    public class OperationsOfTestSet
    {
        public void TestSet()
        {
            Console.WriteLine("Hello World!");
            var mlcontext = new MLContext();
            var lines = System.IO.File.ReadAllLines("C:/Users/Admin/Desktop/WebPractice/MachineLearning/LinearRegression/LinearRegressionProblem/test.csv").Skip(1).TakeWhile(t => t != null);

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
            foreach (var item in itemlist)
            {
                Console.WriteLine(item.x + "        " + item.y);
            }
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("Before---" + itemlist.Count);
            Console.ResetColor();

           /* string testpath = @"C:/Users/Admin/Desktop/WebPractice/MachineLearning/LinearRegression/LinearRegressionProblem/testAfterNoNull.csv";
            StreamWriter sw = new StreamWriter(testpath);
            CsvWriter cw = new CsvWriter(sw, CultureInfo.InvariantCulture);
            cw.WriteRecords(itemlist);
           */

            string path = @"C:/Users/Admin/Desktop/WebPractice/MachineLearning/LinearRegression/LinearRegressionProblem/testAfterNoNull.csv";
            using (StreamWriter sw = new StreamWriter(path))
            {
                using CsvWriter cw = new CsvWriter(sw, System.Globalization.CultureInfo.InvariantCulture);
                cw.WriteRecords(itemlist);
            }



            // data Normalization

            
        }
    }
}
