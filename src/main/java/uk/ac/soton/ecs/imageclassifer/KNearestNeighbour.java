package uk.ac.soton.ecs.imageclassifer;

import java.util.*;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.approximate.FloatNearestNeighboursKDTree;
import org.openimaj.util.function.Operation;
import org.openimaj.util.pair.IntFloatPair;
import org.openimaj.util.parallel.Parallel;
import org.openimaj.util.parallel.partition.RangePartitioner;

public class KNearestNeighbour 
{
	protected VFSGroupDataset<FImage> trainingSet;
	protected Map<FloatFV, String> featureVectors;

	final public static int DIMENSION = 16;

	public static void main(String[] args) throws FileSystemException
	{
		System.out.println("Loading datasets...");

		VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>(args[0], ImageUtilities.FIMAGE_READER);
		VFSListDataset<FImage> testingSet = new VFSListDataset<>(args[1], ImageUtilities.FIMAGE_READER);

		System.out.println("Training the classifier...");

		KNearestNeighbour classifier = new KNearestNeighbour(trainingSet);

		classifier.train();

		System.out.println("Classifing training set...");

		int i = 0;

		for(FImage image : testingSet)
		{
			System.out.println("Guess for " + testingSet.getID(i) + ": " + classifier.classify(image));

			i++;

			if(i > 50)
			{
				break;
			}
		}
    }

	public KNearestNeighbour(VFSGroupDataset<FImage> trainingSet)
	{
		this.trainingSet = trainingSet;
	}

	protected FloatFV getFeatureVector(FImage image)
	{
		ResizeProcessor resizer = new ResizeProcessor(KNearestNeighbour.DIMENSION, KNearestNeighbour.DIMENSION, false);

		int dimension = Math.min(image.getWidth(), image.getHeight());

		FImage square = image.extractCenter(dimension, dimension).processInplace(resizer).normalise();

		return Utilities.zeroMean(square);
	}

	public void train()
	{
		this.featureVectors = new HashMap<>();

		for(final Map.Entry<String, VFSListDataset<FImage>> group : this.trainingSet.entrySet())
		{
			Parallel.forEachPartitioned(new RangePartitioner<FImage>(group.getValue()), new Operation<Iterator<FImage>>() 
			{
				@Override
				public void perform(Iterator<FImage> iterator) 
				{
					while(iterator.hasNext())
					{
						KNearestNeighbour.this.featureVectors.put(KNearestNeighbour.this.getFeatureVector(iterator.next()), group.getKey());
					}
				}
			});
		}
	}

	public String classify(FImage image)
	{
		FloatNearestNeighboursKDTree.Factory factory = new FloatNearestNeighboursKDTree.Factory();

		float[][] converted = new float[this.featureVectors.size()][(int) Math.pow(KNearestNeighbour.DIMENSION, 2)];

		int i = 0;

		for(Map.Entry<FloatFV, String> entry : this.featureVectors.entrySet())
		{
			converted[i] = entry.getKey().values;
			i++;
		}

		FloatNearestNeighboursKDTree nn = factory.create(converted);

		IntFloatPair nearestNeighbour = nn.searchNN(this.getFeatureVector(image).values);

		return this.featureVectors.get(new FloatFV(converted[nearestNeighbour.getFirst()]));
	}
}
