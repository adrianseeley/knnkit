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

public static class Threshold
{
    public static float[] FindAll(List<Sample> samples, int maxThresholds = -1)
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
        if (maxThresholds == -1)
        {
            return thresholds.OrderBy(threshold => threshold).ToArray();
        }
        else
        {
            int step = (int)Math.Ceiling((float)thresholds.Count / (float)maxThresholds);
            return thresholds.OrderBy(threshold => threshold).Where((threshold, index) => index % step == 0).ToArray();
        }
    }
}

public class Program
{
    public static List<Sample> ReadMNIST(string filename, int max = -1)
    {
        List<Sample> samples = new List<Sample>();
        string[] lines = File.ReadAllLines(filename);
        for (int lineIndex = 1; lineIndex < lines.Length; lineIndex++) // skip headers
        {
            string line = lines[lineIndex].Trim();
            if (line.Length == 0)
            {
                continue; // skip empty lines
            }
            string[] parts = line.Split(',');
            int labelInt = int.Parse(parts[0]);
            float[] labelOneHot = new float[10];
            labelOneHot[labelInt] = 1;
            float[] input = new float[parts.Length - 1];
            for (int i = 1; i < parts.Length; i++)
            {
                input[i - 1] = float.Parse(parts[i]) / 255f;
            }
            samples.Add(new Sample(input, labelOneHot));
            if (max != -1 && samples.Count >= max)
            {
                break;
            }
        }
        return samples;
    }

    public static List<int> FindHomogeneousInputIndices(List<Sample> samples)
    {
        int inputCount = samples[0].input.Length;
        List<int> homogeneousInputIndices = new List<int>();
        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
        {
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
        return homogeneousInputIndices;
    }

    public static List<Sample> RemoveInputIndices(List<Sample> samples, List<int> inputIndices)
    {
        int inputCount = samples[0].input.Length;
        List<Sample> rSamples = new List<Sample>();
        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            float[] rInput = new float[inputCount - inputIndices.Count];
            int rInputIndex = 0;
            for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
            {
                if (inputIndices.Contains(inputIndex))
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

    public static List<(Sample sample, int index, float distance)> FindNeighbours(List<Sample> samples, float[] testInput, float[]? inputWeights, float[]? trainDistanceWeights, float distanceExponent, float absoluteDifferenceThreshold = 0.0f, int ignoreSampleIndex = -1)
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
                if (componentDistance <= absoluteDifferenceThreshold)
                {
                    continue;
                }
                componentDistance = MathF.Pow(componentDistance, distanceExponent);
                if (inputWeights != null)
                {
                    componentDistance *= inputWeights[inputIndex];
                }
                distance += componentDistance;
            }
            distance = MathF.Pow(distance, 1.0f / distanceExponent);
            if (trainDistanceWeights != null)
            {
                distance *= trainDistanceWeights[sampleIndex];
            }
            neighbours[sampleIndex] = (samples[sampleIndex], sampleIndex, distance);
        });
        return neighbours.OrderBy(neighbour => neighbour.distance).ToList();
    }

    public static float[] Aggregate(List<(Sample sample, int index, float distance)> neighbours, float[]? sampleContributionWeights, int k, float maxDistanceMultiplier = 1f, float weightExponent = 1f)
    {
        int outputCount = neighbours[0].sample.output.Length;
        float[] output = new float[outputCount];
        float maxDistance = neighbours.Take(k).Max(neighbour => neighbour.distance) * maxDistanceMultiplier;
        float weightSum = 0f;
        for (int neighbourIndex = 0; neighbourIndex < k; neighbourIndex++)
        {
            (Sample sample, int index, float distance) neighbour = neighbours[neighbourIndex];
            float weight = MathF.Pow(1f - (neighbour.distance / (maxDistance + EPSILON)), weightExponent);
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

    public const float EPSILON = 0.000001f;

    public static void Main(string[] args)
    {
        int k = 8;
        float absoluteDifferenceThreshold = 0.0f;
        int distanceExponent = 8;

        List<Sample> samples = ReadMNIST("d:/data/mnist_train.csv", max: 1000);
        List<int> homogeneousInputIndices = FindHomogeneousInputIndices(samples);
        samples = RemoveInputIndices(samples, homogeneousInputIndices);
        List<int> samplesArgmax = samples.Select(sample => Argmax(sample.output)).ToList();

        float maxDistanceMultiplierLow = 1.0f;
        float maxDistanceMultitplierHigh = 25.0f;
        float maxDistanceMultiplierStep = 0.1f;
        float weightExponentLow = 1.0f;
        float weightExponentHigh = 25.0f;
        float weightExponentStep = 0.1f;

        Dictionary<float, Dictionary<float, int>> corrects = new Dictionary<float, Dictionary<float, int>>();
        List<float> maxDistanceMultipliers = new List<float>();
        List<float> weightExponents = new List<float>();
        for (float maxDistanceMultiplier = maxDistanceMultiplierLow; maxDistanceMultiplier <= maxDistanceMultitplierHigh; maxDistanceMultiplier += maxDistanceMultiplierStep)
        {
            maxDistanceMultipliers.Add(maxDistanceMultiplier);
        }
        for (float weightExponent = weightExponentLow; weightExponent <= weightExponentHigh; weightExponent += weightExponentStep)
        {
            weightExponents.Add(weightExponent);
        }
        for (float maxDistanceMultiplier = maxDistanceMultiplierLow; maxDistanceMultiplier <= maxDistanceMultitplierHigh; maxDistanceMultiplier += maxDistanceMultiplierStep)
        {
            for (float weightExponent = weightExponentLow; weightExponent <= weightExponentHigh; weightExponent += weightExponentStep)
            {
                corrects[maxDistanceMultiplier] = new Dictionary<float, int>();
                corrects[maxDistanceMultiplier][weightExponent] = 0;
            }
        }

        for (int sampleIndex = 0; sampleIndex < samples.Count; sampleIndex++)
        {
            Console.Write($"{sampleIndex + 1}/{samples.Count}\r");
            Sample sample = samples[sampleIndex];
            int argmaxActual = Argmax(sample.output);
            List<(Sample sample, int index, float distance)> neighbours = FindNeighbours(samples, sample.input, null, null, distanceExponent, absoluteDifferenceThreshold, ignoreSampleIndex: sampleIndex);
            foreach (float maxDistanceMultiplier in maxDistanceMultipliers)
            {
                Parallel.ForEach(weightExponents, weightExponent =>
                {
                    float[] prediction = Aggregate(neighbours, null, k, maxDistanceMultiplier, weightExponent);
                    int argmaxPrediction = Argmax(prediction);
                    if (argmaxPrediction == argmaxActual)
                    {
                        corrects[maxDistanceMultiplier][weightExponent]++;
                    }
                });
            }
        }

        TextWriter log = new StreamWriter("log.csv");
        log.WriteLine("maxDistanceMultiplier,weightExponent,correct");
        foreach (float maxDistanceMultiplier in maxDistanceMultipliers)
        {
            foreach (float weightExponent in weightExponents)
            {
                log.WriteLine($"{maxDistanceMultiplier},{weightExponent},{corrects[maxDistanceMultiplier][weightExponent]}");
            }
        }
    }
}