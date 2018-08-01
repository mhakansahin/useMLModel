using Microsoft.ML;
using System;

namespace useModel
{
    class Program
    {
        private static PredictionModel<ProjectData, ProjectPrediction> machineLearningModel;

        static void Main(string[] args)
        {
            PredictionModel<ProjectData, ProjectPrediction> model = predictedLabel();
            var prediction = model.Predict(TestData.Setosa);
            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.ReadLine();
        }

        private static PredictionModel<ProjectData, ProjectPrediction> predictedLabel()
        {
            machineLearningModel = PredictionModel.ReadAsync<ProjectData, ProjectPrediction>(@"./MachineLearningModel/Model.zip").Result;
            return machineLearningModel;
        }
    }
}
