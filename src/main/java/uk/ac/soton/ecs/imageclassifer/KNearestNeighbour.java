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
import org.openimaj.util.function.Operation;
import org.openimaj.util.pair.IntFloatPair;
import org.openimaj.util.parallel.Parallel;

/**
 * K-Nearest-Neighbour classifier using scaled-down images as the method of feature abstraction.
 * 
 * @author sl17g12
 *
 */
public class KNearestNeighbour implements Classifier<String, FImage>, BatchTrainer<Annotated<FImage, String>>
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
	public KNearestNeighbour() {
		this.K = K_DEFAULT;
	}
	
	/**
	 * Create a K-Nearest-Neighbour classifier
	 * @param k Number of neighbours to classify against
	 */
	public KNearestNeighbour(int k) {
		this.K = k;
	}
	
	@Override
	public void train(List<? extends Annotated<FImage, String>> data)
	{
		annotatedFeatures = new HashMap<>();

		// Project each image down to a small-scale feature vector
		Parallel.forEach(
			data,
			new Operation<Annotated<FImage, String>>()
			{
				@Override
				public void perform(Annotated<FImage, String> a)
				{
					annotatedFeatures.put(
						getFeatureVector(a.getObject()),
						a.getAnnotations().toArray(new String[1])[0]);
				}
			}
		);
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
		Hashtable<Integer, Integer> frequency = new Hashtable<>(K);
		
		for (IntFloatPair neighbour : neighbours) {
			// Add 1 to the frequency count
			Integer currentFreq = frequency.get(neighbour.first);
			frequency.put(neighbour.first, currentFreq == null ? 1 : ++currentFreq);
		}
		
		// Find the most likely class by taking the average class of the nearest neighbours
		String clazz = "unknown";
		int count = 0;
		for (Entry<Integer, Integer> e : frequency.entrySet()) {
			if (count < e.getValue()) {
				clazz = annotatedFeatures.get(new FloatFV(converted[e.getKey()]));
				count = e.getValue();
			}
		}
		
		// Put the class, and the proportion of this class to other classes in the result
		PrintableClassificationResult<String> result = new PrintableClassificationResult<String>();
		result.put(clazz, (float) count / neighbours.size());
		
		return result;
	}
	
	/**
	 * Convert an image into a {@link DIMENSION} by {@link DIMENSION} feature vector.
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
