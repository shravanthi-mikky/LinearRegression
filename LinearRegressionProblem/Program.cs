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
    internal class Program
    {
        static void Main(string[] args)
        {
            OperationsOnTrain operationsOnTrain = new OperationsOnTrain();
            OperationsOfTestSet operationsOfTestSet = new OperationsOfTestSet();

            while (true)
            {
                Console.WriteLine("Enter the option");
                int option = Convert.ToInt32(Console.ReadLine());
                switch (option)
                {
                    case 1:
                        operationsOnTrain.TrainSet();
                        break;

                    case 2:
                        operationsOfTestSet.TestSet();
                        break;

                }

            }


        }
    }
}
