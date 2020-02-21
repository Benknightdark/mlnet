using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace RecomandationSystem {
    class Program {
        static void Main (string[] args) {

            // // Create MLContext to be shared across the model creation workflow objects 
            // // <SnippetMLContext>
             MLContext mlContext = new MLContext ();
            // // </SnippetMLContext>

            // // Load data
            // // <SnippetLoadDataMain>
          //   (IDataView trainingDataView, IDataView testDataView) = LoadData (mlContext);
            // // </SnippetLoadDataMain>

            // // Build & train model
            // // <SnippetBuildTrainModelMain>
         //   ITransformer model = BuildAndTrainModel (mlContext, trainingDataView);
            // // </SnippetBuildTrainModelMain>

            // // Evaluate quality of model
            // // <SnippetEvaluateModelMain>
          //   EvaluateModel (mlContext, testDataView, model);
            // // </SnippetEvaluateModelMain>

            // // Use model to try a single prediction (one row of data)
            // // <SnippetUseModelMain>
            UseModelForSinglePrediction (mlContext);
            // // </SnippetUseModelMain>

            // // Save model
            // // <SnippetSaveModelMain>
           //  SaveModel (mlContext, trainingDataView.Schema, model);
            // // </SnippetSaveModelMain>
        }

        // Load data
        public static (IDataView training, IDataView test) LoadData (MLContext mlContext) {

            var DataPath = Path.Combine (Environment.CurrentDirectory, "Data", "ml25", "ratings.csv");
            IDataView DataSourceView = mlContext.Data.LoadFromTextFile<MovieRating> (DataPath, hasHeader : true, separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit (DataSourceView, testFraction : 0.4);
            var trainingData = mlContext.Data
                .CreateEnumerable<MovieRating> (split.TrainSet, reuseRowObject : false);

            var testData = mlContext.Data
                .CreateEnumerable<MovieRating> (split.TestSet, reuseRowObject : false);

            var trainingDataView = split.TrainSet; //mlContext.Data.LoadFromEnumerable (trainingData);
            var testDataView = split.TestSet; // mlContext.Data.LoadFromEnumerable (testData);

            return (trainingDataView, testDataView);
        }

        // Build and train model
        public static ITransformer BuildAndTrainModel (MLContext mlContext, IDataView trainingDataView) {
            // Add data transformations
            // <SnippetDataTransformations>
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey (outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append (mlContext.Transforms.Conversion.MapValueToKey (outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));
            // </SnippetDataTransformations>

            // Set algorithm options and append algorithm
            // <SnippetAddAlgorithm>
            var options = new MatrixFactorizationTrainer.Options {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 50,
                ApproximationRank = 128,
                Alpha = 0.6
            };

            var trainerEstimator = estimator.Append (mlContext.Recommendation ().Trainers.MatrixFactorization (options));
            // </SnippetAddAlgorithm>

            // <SnippetFitModel>
            Console.WriteLine ("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit (trainingDataView);

            return model;
            // </SnippetFitModel>
        }

        // Evaluate model
        public static void EvaluateModel (MLContext mlContext, IDataView testDataView, ITransformer model) {
            // Evaluate model on test data & print evaluation metrics
            // <SnippetTransform>
            Console.WriteLine ("=============== Evaluating the model ===============");
            var prediction = model.Transform (testDataView);
            // </SnippetTransform>

            // <SnippetEvaluate>
            var metrics = mlContext.Regression.Evaluate (prediction, labelColumnName: "Label", scoreColumnName: "Score");
            // </SnippetEvaluate>

            // <SnippetPrintMetrics>
            Console.WriteLine ("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString ());
            Console.WriteLine ("RSquared: " + metrics.RSquared.ToString ());
            // </SnippetPrintMetrics>
        }

        // Use model for single prediction
        public static void UseModelForSinglePrediction (MLContext mlContext) {//, ITransformer model
            ITransformer trainedModel;
            var ModelPath = Path.Combine (Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");
             trainedModel = mlContext.Model.Load (ModelPath, out var modelSchema);
            Console.WriteLine ("=============== Making a prediction ===============");
            // var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction> (model);
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction> (trainedModel);
            var testInput = new MovieRating { userId = 6, movieId = 10 };

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

        public static IEnumerable<MovieRating> GetTestMovieData (MLContext mlContext, IDataView testDataView) {

            // Create an IEnumerable of HousingData objects from IDataView
            IEnumerable<MovieRating> housingDataEnumerable =
                mlContext.Data.CreateEnumerable<MovieRating> (testDataView, reuseRowObject : true);
            return housingDataEnumerable;

        }
    }

}