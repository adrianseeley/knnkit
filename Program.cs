public class Sample
{
    public float[] input;
    public float[] output;

    public Sample(float[] input, float[] output)
    {
        this.input = input;
        this.output = output;
    }
}

public static class FeatureReduction
{
    public static List<int> Homogeneous(List<Sample> samples, List<int>? removedInputIndices = null)
    {
        int inputCount = samples[0].input.Length;
        List<int> homogeneousInputIndices = new List<int>();
        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
        {
            if (removedInputIndices != null && removedInputIndices.Contains(inputIndex))
            {
                continue;
            }
            bool homogeneous = true;
            float value = samples[0].input[inputIndex];
            for (int sampleIndex = 1; sampleIndex < samples.Count; sampleIndex++)
            {
                if (samples[sampleIndex].input[inputIndex] != value)
                {
                    homogeneous = false;
                    break;
                }
            }
            if (homogeneous)
            {
                homogeneousInputIndices.Add(inputIndex);
            }
        }
        if (removedInputIndices != null)
        {
            homogeneousInputIndices = homogeneousInputIndices.Concat(removedInputIndices).OrderBy(index => index).ToList();
        }
        return homogeneousInputIndices;
    }

    public static List<Sample> RemoveIndices(List<Sample> samples, List<int> removedInputIndices)
    {
        int inputCount = samples[0].input.Length;
        List<Sample> rSamples = new List<Sample>();
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            float[] rInput = new float[inputCount - removedInputIndices.Count];
            int rInputIndex = 0;
            for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
            {
                if (removedInputIndices.Contains(inputIndex))
                {
                    continue;
                }
                rInput[rInputIndex] = samples[sampleIndex].input[inputIndex];
                rInputIndex++;
            }
            rSamples.Add(new Sample(rInput, samples[sampleIndex].output));
        }
        return rSamples;
    }
}

public static class Statistics
{
    public static int Argmax(float[] values)
    {
        int argmax = 0;
        float max = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                argmax = i;
                max = values[i];
            }
        }
        return argmax;
    }

    public static float Mean(float[] values)
    {
        float sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sum += values[i];
        }
        return sum / values.Length;
    }

    public static float Variance(float[] values, float mean)
    {
        float sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sum += MathF.Pow(values[i] - mean, 2);
        }
        return sum / values.Length;
    }

    public static float StandardDeviation(float variance)
    {
        return MathF.Sqrt(variance);
    }

    public static float ZScore(float value, float mean, float standardDeviation)
    {
        return (value - mean) / standardDeviation;
    }
}

public static class Normalize
{
    public delegate List<Sample> NormalizeFunction(List<Sample> samples);

    public static List<NormalizeFunction> NormalizeFunctions = new List<NormalizeFunction>
    {
        None,
        ZeroOne,
        ZScore,
        ZScoreZeroOne,
        RankZeroOne
    };

    public static (float[] inputMins, float[] inputMaxs, float[] inputRanges) GetBounds(List<Sample> samples)
    {
        int inputCount = samples[0].input.Length;
        float[] inputMins = new float[inputCount];
        float[] inputMaxs = new float[inputCount];
        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
        {
            inputMins[inputIndex] = float.MaxValue;
            inputMaxs[inputIndex] = float.MinValue;
        }
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
            {
                if (samples[sampleIndex].input[inputIndex] < inputMins[inputIndex])
                {
                    inputMins[inputIndex] = samples[sampleIndex].input[inputIndex];
                }
                if (samples[sampleIndex].input[inputIndex] > inputMaxs[inputIndex])
                {
                    inputMaxs[inputIndex] = samples[sampleIndex].input[inputIndex];
                }
            }
        }
        float[] inputRanges = new float[inputCount];
        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
        {
            inputRanges[inputIndex] = inputMaxs[inputIndex] - inputMins[inputIndex];
        }
        return (inputMins, inputMaxs, inputRanges);
    }

    public static List<Sample> None(List<Sample> samples)
    {
        return samples;
    }

    public static List<Sample> ZeroOne(List<Sample> samples)
    {
        (float[] inputMins, float[] inputMaxs, float[] inputRanges) = GetBounds(samples);
        List<Sample> nSamples = new List<Sample>();
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            float[] nInput = new float[samples[sampleIndex].input.Length];
            for (int inputIndex = 0; inputIndex < samples[sampleIndex].input.Length; inputIndex++)
            {
                nInput[inputIndex] = (samples[sampleIndex].input[inputIndex] - inputMins[inputIndex]) / inputRanges[inputIndex];
            }
            nSamples.Add(new Sample(nInput, samples[sampleIndex].output));
        }
        return nSamples;
    }

    public static List<Sample> ZScore(List<Sample> samples)
    {
        int inputCount = samples[0].input.Length;
        float[] inputMeans = new float[inputCount];
        float[] inputVariances = new float[inputCount];
        float[] inputStandardDeviations = new float[inputCount];
        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
        {
            float[] inputValues = samples.Select(sample => sample.input[inputIndex]).ToArray();
            float mean = Statistics.Mean(inputValues);
            float variance = Statistics.Variance(inputValues, mean);
            float standardDeviation = Statistics.StandardDeviation(variance);
            inputMeans[inputIndex] = mean;
            inputVariances[inputIndex] = variance;
            inputStandardDeviations[inputIndex] = standardDeviation;
        }
        List<Sample> nSamples = new List<Sample>();
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            Sample sample = samples[sampleIndex];
            float[] nInput = new float[inputCount];
            for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
            {
                nInput[inputIndex] = Statistics.ZScore(sample.input[inputIndex], inputMeans[inputIndex], inputStandardDeviations[inputIndex]);
            }
            nSamples.Add(new Sample(nInput, sample.output));
        }
        return nSamples;
    }

    public static List<Sample> ZScoreZeroOne(List<Sample> samples)
    {
        return ZeroOne(ZScore(samples));
    }

    public static List<Sample> RankZeroOne(List<Sample> samples)
    {
        int inputCount = samples[0].input.Length;
        List<float[]> inputRanks = new List<float[]>(inputCount);
        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
        {
            float[] inputRank = samples.Select(sample => sample.input[inputIndex]).Distinct().OrderBy(value => value).ToArray();
            inputRanks.Add(inputRank);
        }
        List<Sample> nSamples = new List<Sample>();
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            Sample sample = samples[sampleIndex];
            float[] nInput = new float[inputCount];
            for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
            {
                float[] inputRank = inputRanks[inputIndex];
                if (inputRank.Length == 1)
                {
                    nInput[inputIndex] = 0;
                    continue;
                }
                int closestRankIndex = -1;
                float closestRankDistance = float.PositiveInfinity;
                for (int rankIndex = 0; rankIndex < inputRank.Length; rankIndex++)
                {
                    float rankDistance = MathF.Abs(sample.input[inputIndex] - inputRank[rankIndex]);
                    if (rankDistance < closestRankDistance)
                    {
                        closestRankIndex = rankIndex;
                        closestRankDistance = rankDistance;
                    }
                }
                nInput[inputIndex] = (float)closestRankIndex / (float)(inputRank.Length - 1);
            }
            nSamples.Add(new Sample(nInput, sample.output));
        }
        return nSamples;
    }
}

public static class Threshold
{
    public static float[] FindAll(List<Sample> samples)
    {
        int inputCount = samples[0].input.Length;
        HashSet<float> thresholds = new HashSet<float>();
        for (int aIndex = 0; aIndex < samples.Count; aIndex++)
        {
            for (int bIndex = aIndex + 1; bIndex < samples.Count; bIndex++)
            {
                for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
                {
                    float a = samples[aIndex].input[inputIndex];
                    float b = samples[bIndex].input[inputIndex];
                    float difference = MathF.Abs(b - a);
                    thresholds.Add(difference);
                }
            }
        }
        return thresholds.OrderBy(threshold => threshold).ToArray();
    }
}

public static class Neighbours
{
    public static List<(Sample sample, int index, float distance)> Find(List<Sample> samples, float[] testInput, float[]? inputWeights, float[]? trainDistanceWeights, float exponent, float threshold = 0.0f, int ignoreSampleIndex = -1)
    {
        int inputCount = samples[0].input.Length;
        (Sample sample, int index, float distance)[] neighbours = new (Sample sample, int index, float distance)[samples.Count];
        Parallel.For(0, samples.Count, sampleIndex =>
        {
            if (sampleIndex == ignoreSampleIndex)
            {
                neighbours[sampleIndex] = (samples[sampleIndex], sampleIndex, float.PositiveInfinity);
                return;
            }
            Sample neighbour = samples[sampleIndex];
            float distance = 0;
            for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
            {
                float componentDistance = MathF.Abs(neighbour.input[inputIndex] - testInput[inputIndex]);
                if (componentDistance <= threshold)
                {
                    continue;
                }
                componentDistance = MathF.Pow(componentDistance, exponent);
                if (inputWeights != null)
                {
                    componentDistance *= inputWeights[inputIndex];
                }
                distance += componentDistance;
            }
            distance = MathF.Pow(distance, 1.0f / exponent);
            if (trainDistanceWeights != null)
            {
                distance *= trainDistanceWeights[sampleIndex];
            }
            neighbours[sampleIndex] = (samples[sampleIndex], sampleIndex, distance);
        });
        return neighbours.OrderBy(neighbour => neighbour.distance).ToList();
    }
}

public static class Aggregate
{
    public const float EPSILON = 0.000001f;

    public delegate float[] AggregateFunction(List<(Sample sample, int index, float distance)> neighbours, float[]? sampleContributionWeights, int k);

    public static List<AggregateFunction> AggregateFunctions = new List<AggregateFunction>
    {
        Average,
        InverseNormalizedRatio,
        Reciprocal,
        ReciprocalNormalizedRatio
    };

    public static float[] Average(List<(Sample sample, int index, float distance)> neighbours, float[]? sampleContributionWeights, int k)
    {
        int outputCount = neighbours[0].sample.output.Length;
        float[] output = new float[outputCount];
        float weightSum = 0f;
        for (int neighbourIndex = 0; neighbourIndex < k; neighbourIndex++)
        {
            (Sample sample, int index, float distance) neighbour = neighbours[neighbourIndex];
            float weight = 1f;
            if (sampleContributionWeights != null)
            {
                weight *= sampleContributionWeights[neighbour.index];
            }
            weightSum += weight;
            for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
            {
                output[outputIndex] += neighbours[neighbourIndex].sample.output[outputIndex] * weight;
            }
        }
        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
        {
            output[outputIndex] /= weightSum;
        }
        return output;
    }

    public static float[] InverseNormalizedRatio(List<(Sample sample, int index, float distance)> neighbours, float[]? sampleContributionWeights, int k)
    {
        int outputCount = neighbours[0].sample.output.Length;
        float[] output = new float[outputCount];
        float maxDistance = neighbours.Take(k).Max(neighbour => neighbour.distance);
        float weightSum = 0f;
        for (int neighbourIndex = 0; neighbourIndex < k; neighbourIndex++)
        {
            (Sample sample, int index, float distance) neighbour = neighbours[neighbourIndex];
            float weight = 1f - (neighbour.distance / (maxDistance + EPSILON));
            if (sampleContributionWeights != null)
            {
                weight *= sampleContributionWeights[neighbour.index];
            }
            weightSum += weight;
            for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
            {
                output[outputIndex] += neighbours[neighbourIndex].sample.output[outputIndex] * weight;
            }
        }
        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
        {
            output[outputIndex] /= weightSum;
        }
        return output;
    }

    public static float[] Reciprocal(List<(Sample sample, int index, float distance)> neighbours, float[]? sampleContributionWeights, int k)
    {
        int outputCount = neighbours[0].sample.output.Length;
        float[] output = new float[outputCount];
        float weightSum = 0f;
        for (int neighbourIndex = 0; neighbourIndex < k; neighbourIndex++)
        {
            (Sample sample, int index, float distance) neighbour = neighbours[neighbourIndex];
            float weight = 1f / (neighbour.distance + EPSILON);
            if (sampleContributionWeights != null)
            {
                weight *= sampleContributionWeights[neighbour.index];
            }
            weightSum += weight;
            for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
            {
                output[outputIndex] += neighbours[neighbourIndex].sample.output[outputIndex] * weight;
            }
        }
        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
        {
            output[outputIndex] /= weightSum;
        }
        return output;
    }

    public static float[] ReciprocalNormalizedRatio(List<(Sample sample, int index, float distance)> neighbours, float[]? sampleContributionWeights, int k)
    {
        int outputCount = neighbours[0].sample.output.Length;
        float[] output = new float[outputCount];
        float maxDistance = neighbours.Take(k).Max(neighbour => neighbour.distance);
        float weightSum = 0f;
        for (int neighbourIndex = 0; neighbourIndex < k; neighbourIndex++)
        {
            (Sample sample, int index, float distance) neighbour = neighbours[neighbourIndex];
            float weight = 1f / (neighbour.distance / (maxDistance + EPSILON));
            if (sampleContributionWeights != null)
            {
                weight *= sampleContributionWeights[neighbour.index];
            }
            weightSum += weight;
            for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
            {
                output[outputIndex] += neighbours[neighbourIndex].sample.output[outputIndex] * weight;
            }
        }
        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++)
        {
            output[outputIndex] /= weightSum;
        }
        return output;
    }
}

public static class Fitness
{
    public delegate float ErrorFunction(List<Sample> samples, List<float[]> predictions);

    public static float ArgmaxMatchError(List<Sample> samples, List<float[]> predictions)
    {
        int match = 0;
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            if (Statistics.Argmax(predictions[sampleIndex]) == Statistics.Argmax(samples[sampleIndex].output))
            {
                match++;
            }
        }
        return 1f - (match / samples.Count);
    }

    public static float RootMeanSquaredError(List<Sample> samples, List<float[]> predictions)
    {
        float sum = 0;
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            float[] prediction = predictions[sampleIndex];
            float[] output = samples[sampleIndex].output;
            for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
            {
                sum += MathF.Pow(prediction[outputIndex] - output[outputIndex], 2);
            }
        }
        return MathF.Sqrt(sum / samples.Count);
    }

    public static float MeanAbsoluteError(List<Sample> samples, List<float[]> predictions)
    {
        float sum = 0;
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            float[] prediction = predictions[sampleIndex];
            float[] output = samples[sampleIndex].output;
            for (int outputIndex = 0; outputIndex < output.Length; outputIndex++)
            {
                sum += MathF.Abs(prediction[outputIndex] - output[outputIndex]);
            }
        }
        return sum / samples.Count;
    }
}

public static class Optimize
{
    public static (List<int> removedInputIndices, int k, Normalize.NormalizeFunction normalizeFunction, float threshold, float exponent, Aggregate.AggregateFunction aggregateFunction)? Primary(string filename, List<Sample> samples, Fitness.ErrorFunction errorFunction, int maxK, int maxExponent)
    {
        // create a best tracker
        (List<int> removedInputIndices, int k, Normalize.NormalizeFunction normalizeFunction, float threshold, float exponent, Aggregate.AggregateFunction aggregateFunction)? bestSolution = null;
        float bestFitness = float.PositiveInfinity;

        // create a log
        TextWriter log = new StreamWriter(filename, append: false);

        // writer header
        log.WriteLine("removedInputIndices,k,normalizeFunction,threshold,exponent,aggregateFunction,fitness");
        log.Flush();

        // first we remove any homogeneous inputs
        List<int> removedInputIndices = FeatureReduction.Homogeneous(samples);
        samples = FeatureReduction.RemoveIndices(samples, removedInputIndices);
        string removedInputIndicesString = "[" + string.Join(" ", removedInputIndices) + "]";

        // make a list of viable k values
        List<int> ks = Enumerable.Range(1, maxK).ToList();

        // make a list of viable exponents
        List<float> exponents = Enumerable.Range(1, maxExponent).Select(value => (float)value).ToList();

        // iterate through normalization styles
        foreach (Normalize.NormalizeFunction normalization in Normalize.NormalizeFunctions)
        {
            // normalize dataset
            List<Sample> nSamples = normalization(samples);

            // find all the thresholds (depends on the normalization)
            float[] thresholds = Threshold.FindAll(nSamples);

            // iterate thresholds
            foreach (float threshold in thresholds)
            {
                // iterate through exponents
                foreach (float exponent in exponents)
                {
                    // create a list of neighbours for all samples
                    List<List<(Sample sample, int index, float distance)>> allNeighbours = new List<List<(Sample sample, int index, float distance)>>();

                    // iterate through samples
                    for (int sampleIndex = 0; sampleIndex < nSamples.Count; sampleIndex++)
                    {
                        // get sample to predict
                        Sample nSample = nSamples[sampleIndex];

                        // get the neighbours
                        List<(Sample sample, int index, float distance)> neighbours = Neighbours.Find(nSamples, nSample.input, null, null, exponent, threshold, sampleIndex);

                        // store neighbours
                        allNeighbours.Add(neighbours);
                    }

                    // iterate through aggregators
                    foreach (Aggregate.AggregateFunction aggregateFunction in Aggregate.AggregateFunctions)
                    {
                        // track predictions for each k value
                        List<List<float[]>> kPredictions = new List<List<float[]>>(ks.Count);
                        for (int kIndex = 0; kIndex < ks.Count; kIndex++)
                        {
                            kPredictions.Add(new List<float[]>(nSamples.Count));
                        }

                        // iterate through samples
                        for (int sampleIndex = 0; sampleIndex < nSamples.Count; sampleIndex++)
                        {
                            // get sample neighbours
                            List<(Sample sample, int index, float distance)> neighbours = allNeighbours[sampleIndex];

                            // iterate through k values
                            Parallel.For(0, ks.Count, kIndex =>
                            {
                                // get k value
                                int k = ks[kIndex];

                                // make prediction
                                float[] prediction = aggregateFunction(neighbours, null, k);

                                // stash prediction
                                kPredictions[kIndex].Add(prediction);
                            });
                        }

                        // compute error for each k value
                        float[] kErrors = new float[ks.Count];
                        Parallel.For(0, ks.Count, kIndex =>
                        {
                            kErrors[kIndex] = errorFunction(samples, kPredictions[kIndex]);
                        });

                        // iterate kErrors
                        for (int kIndex = 0; kIndex < ks.Count; kIndex++)
                        {
                            // get k value
                            int k = ks[kIndex];

                            // get error
                            float error = kErrors[kIndex];

                            // log
                            log.WriteLine($"{removedInputIndicesString},{k},{normalization.Method.Name},{threshold},{exponent},{aggregateFunction.Method.Name},{error}");
                            log.Flush();

                            // console update
                            Console.WriteLine($"RII: {removedInputIndicesString}, K: {k}, N: {normalization.Method.Name}, T: {threshold}, E: {exponent}, A: {aggregateFunction.Method.Name}, E: {error}");

                            // if this error is better than best
                            if (error < bestFitness)
                            {
                                // update best
                                bestSolution = (removedInputIndices, k, normalization, threshold, exponent, aggregateFunction);
                                bestFitness = error;
                            }
                        }
                    }
                }
            }
        }

        // close log
        log.Close();

        // return best
        return bestSolution;
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        string logFilename = "log.csv";
        int maxK = 25;
        int maxExponent = 25;
        List<Sample> samples = new List<Sample>();
        (List<int> removedInputIndices, int k, Normalize.NormalizeFunction normalizeFunction, float threshold, float exponent, Aggregate.AggregateFunction aggregateFunction)? bestSolution = Optimize.Primary(logFilename, samples, Fitness.ArgmaxMatchError, maxK, maxExponent);
    }
}