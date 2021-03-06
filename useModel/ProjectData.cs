﻿using Microsoft.ML.Runtime.Api;

namespace useModel
{
    public class ProjectData
    {
        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;

        [Column("4")]
        [ColumnName("Label")]
        public string Label;
    }

    public class ProjectPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
