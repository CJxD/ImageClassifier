package uk.ac.soton.ecs.imageclassifer;

import java.io.File;
import java.util.*;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.approximate.FloatNearestNeighboursKDTree;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.ml.training.BatchTrainer;
import org.openimaj.util.pair.IntFloatPair;

/**
 * K-Nearest-Neighbour classifier using scaled-down images as the method of feature abstraction.
 * 
 * @author sl17g12
 *
 */
public class KNearestNeighbour
	implements
	Classifier<String, FImage>,
	BatchTrainer<Annotated<FImage, String>>
{
	protected VFSGroupDataset<FImage> trainingSet;
	protected Map<FloatFV, String> annotatedFeatures;

	final public static int DIMENSION = 16;
	final public static int K_DEFAULT = 5;

	private int K = 1;

	public static void main(String[] args) throws FileSystemException
	{
		if(args.length < 2)
			throw new IllegalArgumentException("Usage: KNearestNeighbour <training uri> <testing uri>");

		System.out.println("Loading datasets...");

		File trainingFile = new File(args[0]);
		File testingFile = new File(args[1]);

		VFSGroupDataset<FImage> training = new VFSGroupDataset<>(
			trainingFile.getAbsolutePath(),
			ImageUtilities.FIMAGE_READER);
		VFSListDataset<FImage> testing = new VFSListDataset<>(
			testingFile.getAbsolutePath(),
			ImageUtilities.FIMAGE_READER);

		System.out.println("Training the classifier...");

		KNearestNeighbour classifier = new KNearestNeighbour(5);

		classifier.train(AnnotatedObject.createList(training));

		System.out.println("Classifing testing set...");

		int i = 0;
		for(FImage image : testing)
		{
			System.out.println(testing.getID(i++) + " => " + classifier.classify(image));
		}
	}

	/**
	 * Create a K-Nearest-Neighbour classifier with the default K value
	 */
	public KNearestNeighbour()
	{
		this.K = K_DEFAULT;
	}

	/**
	 * Create a K-Nearest-Neighbour classifier
	 * 
	 * @param k Number of neighbours to classify against
	 */
	public KNearestNeighbour(int k)
	{
		this.K = k;
	}

	@Override
	public void train(List<? extends Annotated<FImage, String>> data)
	{
		annotatedFeatures = new HashMap<>();

		// Project each image down to a small-scale feature vector
		for (Annotated<FImage, String> a : data) {
			annotatedFeatures.put(
				getFeatureVector(a.getObject()),
				a.getAnnotations().toArray(new String[1])[0]);
		}
	}

	@Override
	public ClassificationResult<String> classify(FImage image)
	{
		float[][] converted = new float[annotatedFeatures.size()][(int) Math.pow(KNearestNeighbour.DIMENSION, 2)];

		int i = 0;

		for(Map.Entry<FloatFV, String> entry : annotatedFeatures.entrySet())
		{
			converted[i] = entry.getKey().values;
			i++;
		}

		FloatNearestNeighboursKDTree.Factory factory = new FloatNearestNeighboursKDTree.Factory();
		FloatNearestNeighboursKDTree nn = factory.create(converted);

		// Find the K nearest neighbours
		List<IntFloatPair> neighbours = nn.searchKNN(getFeatureVector(image).values, K);

		// Create a frequency table of neighbours
		Hashtable<String, Integer> frequency = new Hashtable<>(K);
		// List the total distances to the neighbours
		Hashtable<String, Float> distance = new Hashtable<>(K);
		float totalDist = 0f;

		for(IntFloatPair neighbour : neighbours)
		{
			String clazz = annotatedFeatures.get(new FloatFV(converted[neighbour.first]));
			
			// Add 1 to the frequency count
			Integer currentFreq = frequency.get(clazz);
			frequency.put(clazz, currentFreq == null ? 1 : ++currentFreq);
			
			// Add distance to the cumulative distance
			Float currentDist = distance.get(clazz);
			distance.put(clazz, currentDist == null ? neighbour.second : currentDist + neighbour.second);
			totalDist += neighbour.second;
		}

		// Find the most likely class by taking the average class of the nearest neighbours
		// Checks both frequency and average distance from point
		String clazz = "unknown";
		int count = 0;
		float dist = 0f;
		
		for(Entry<String, Integer> e : frequency.entrySet())
		{
			float aveDist = distance.get(e.getKey()) / e.getValue();
			
			if(count < e.getValue() ||
					count == e.getValue() && dist >= aveDist)
			{
				clazz = e.getKey();
				count = e.getValue();
				dist = aveDist;
			}
		}

		// Weighting function
		float weight = ((((float) count) / neighbours.size()) + (dist / totalDist)) / 2;
		
		PrintableClassificationResult<String> result = new PrintableClassificationResult<String>();
		result.put(clazz, weight);

		return result;
	}

	/**
	 * Convert an image into a {@link DIMENSION} by {@link DIMENSION} feature vector.
	 * 
	 * @param image
	 * @return Flattened pixels of resized image
	 */
	protected FloatFV getFeatureVector(FImage image)
	{
		ResizeProcessor resizer = new ResizeProcessor(KNearestNeighbour.DIMENSION, KNearestNeighbour.DIMENSION, false);

		int dimension = Math.min(image.getWidth(), image.getHeight());

		// Crop the image to a square, scale, and normalise.
		FImage square = image.extractCenter(dimension, dimension).processInplace(resizer).normalise();

		// Return the zero-meaned feature vector
		return Utilities.zeroMean(square);
	}
}
