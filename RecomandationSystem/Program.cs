using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace RecomandationSystem {
    class Program {
        static void Main (string[] args) {

            MLContext mlContext = new MLContext ();

            (IDataView trainingDataView, IDataView testDataView) = LoadData (mlContext);

            ITransformer model = BuildAndTrainModel (mlContext, trainingDataView);

            EvaluateModel (mlContext, testDataView, model);

            SaveModel (mlContext, trainingDataView.Schema, model);

            UseModelForSinglePredictionFromLocalModel (mlContext);
        }

        // Load data
        public static (IDataView training, IDataView test) LoadData (MLContext mlContext) {
            var trainingDataPath = Path.Combine (Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine (Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating> (trainingDataPath, hasHeader : true, separatorChar: ',');
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating> (testDataPath, hasHeader : true, separatorChar: ',');

            return (trainingDataView, testDataView);
        }

        // Build and train model
        public static ITransformer BuildAndTrainModel (MLContext mlContext, IDataView trainingDataView) {
            // Add data transformations
            // <SnippetDataTransformations>
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey (outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append (mlContext.Transforms.Conversion.MapValueToKey (outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            var options = new MatrixFactorizationTrainer.Options {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append (mlContext.Recommendation ().Trainers.MatrixFactorization (options));

            Console.WriteLine ("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit (trainingDataView);

            return model;
        }

        // Evaluate model
        public static void EvaluateModel (MLContext mlContext, IDataView testDataView, ITransformer model) {
            Console.WriteLine ("=============== Evaluating the model ===============");
            var prediction = model.Transform (testDataView);
            var metrics = mlContext.Regression.Evaluate (prediction, labelColumnName: "Label", scoreColumnName: "Score");         
            Console.WriteLine ("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString ());
            Console.WriteLine ("RSquared: " + metrics.RSquared.ToString ());
        }


        public static void UseModelForSinglePredictionFromLocalModel (MLContext mlContext) {
            Console.WriteLine ("=============== Making a prediction ===============");
            var modelPath = Path.Combine (Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

            DataViewSchema modelSchema;

            ITransformer trainedModel = mlContext.Model.Load (modelPath, out modelSchema);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction> (trainedModel);

            var testInput = new MovieRating { userId = 6, movieId = 55 };

            var movieRatingPrediction = predictionEngine.Predict (testInput);

            if (Math.Round (movieRatingPrediction.Score, 1) > 3.5) {
                Console.WriteLine ("Movie " + testInput.movieId + " is recommended for user " + testInput.userId);
            } else {
                Console.WriteLine ("Movie " + testInput.movieId + " is not recommended for user " + testInput.userId);
            }
        }
        //Save model
        public static void SaveModel (MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model) {
            // Save the trained model to .zip file
            // <SnippetSaveModel>
            var modelPath = Path.Combine (Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

            Console.WriteLine ("=============== Saving the model to a file ===============");
            mlContext.Model.Save (model, trainingDataViewSchema, modelPath);
            // </SnippetSaveModel>
        }
    }

}

//  public static (IDataView training, IDataView test) LoadData (MLContext mlContext) {

//             var DataPath = Path.Combine (Environment.CurrentDirectory, "Data", "ml25", "ratings.csv");
//             IDataView DataSourceView = mlContext.Data.LoadFromTextFile<MovieRating> (DataPath, hasHeader : true, separatorChar: ',');

//             var split = mlContext.Data.TrainTestSplit (DataSourceView, testFraction : 0.4);
//             var trainingData = mlContext.Data
//                 .CreateEnumerable<MovieRating> (split.TrainSet, reuseRowObject : false);

//             var testData = mlContext.Data
//                 .CreateEnumerable<MovieRating> (split.TestSet, reuseRowObject : false);

//             var trainingDataView = split.TrainSet; //mlContext.Data.LoadFromEnumerable (trainingData);
//             var testDataView = split.TestSet; // mlContext.Data.LoadFromEnumerable (testData);

//             return (trainingDataView, testDataView);
//         }