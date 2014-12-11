package uk.ac.soton.ecs.imageclassifer;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import de.bwaldvogel.liblinear.SolverType;

/**
 * Bag of Visual Words classifier given a grouped dataset of training images, and a list dataset of testing images, SiftBoVW
 * will create sift interest points for an image, and train a lib linear annotator
 * 
 * @author Sam Lavers
 */
public class SIFTBoVW implements ClassificationAlgorithm
{
	protected int codebookSize = 500;
	protected int patchSize = 8;
	protected int patchSeparation = patchSize / 2;

	protected BagOfVisualWords<byte[]> quantiser;
	protected LiblinearAnnotator<FImage, String> annotator;
	protected Map<FImage, LocalFeatureList<Keypoint>> featureCache;

	public static void main(String[] args) throws FileSystemException, FileNotFoundException
	{
		Utilities.runClassifier(new SIFTBoVW(), "PyramidSift", args);
	}

	/**
	 * Train the classifier
	 * @param data The training set
	 */
	@Override
	public void train(List<? extends Annotated<FImage, String>> data)
	{
		this.featureCache = new HashMap<>();
		
		trainQuantiser(data);
		trainAnnotator(data);
	}

	/**
	 * Classify an image
	 * @param image The image
	 * @return The result
	 */
	@Override
	public ClassificationResult<String> classify(FImage image)
	{
		if(quantiser == null)
			throw new IllegalStateException("Classifier is not trained");
		if(annotator == null)
			throw new IllegalStateException("Annotator is not trained");

		PrintableClassificationResult<String> result = new PrintableClassificationResult<>(PrintableClassificationResult.BEST_RESULT);

		for(ScoredAnnotation<String> a : annotator.annotate(image))
		{
			result.put(a.annotation, a.confidence);
		}
		return result;
	}

	/**
	 * Trains the Bag of Visual Words with a K-means-generated codebook.
	 * 
	 * @param trainingData
	 */
	protected void trainQuantiser(List<? extends Annotated<FImage, String>> data)
	{
		List<LocalFeatureList<Keypoint>> allFeatures = new ArrayList<>();
		
		int i = 0;
		
		// Load a list of ImagePatch features
		for(Annotated<FImage, String> image : data)
		{
			System.out.println("training quanitzer " + ++i);
			
			allFeatures.add(getFeatures(image.getObject()));
		}

		// Populate a DataSource with the ImagePatches
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<Keypoint, byte[]>(
			allFeatures);

		// Create n centroids to act as a codebook for the bag of visual words
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(codebookSize);
		ByteCentroidsResult centroids = km.cluster(datasource);

		// Any inputs will be quantised to the nearest ImagePatch centroid
		quantiser = new BagOfVisualWords<byte[]>(centroids.defaultHardAssigner());
	}

	/**
	 * Trains the liblinear annotator with the trained quantiser.
	 * 
	 * @param trainingData
	 */
	protected void trainAnnotator(List<? extends Annotated<FImage, String>> data)
	{
		if(quantiser == null)
			throw new IllegalStateException("Quantiser is not trained");

		FeatureExtractor<SparseIntFV, FImage> extractor = new FeatureExtractor<SparseIntFV, FImage>()
		{
			@Override
			public SparseIntFV extractFeature(FImage image)
			{
				return quantiser.aggregate(getFeatures(image));
			}
		};

		// Train the annotator to make associations between certain "words" and image classes

		annotator = new LiblinearAnnotator<>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		annotator.train(data);
	}

	/**
	 * Gets the SIFT interest points for an image
	 * @param image The image
	 * @return SIFT interest points
	 */
	protected LocalFeatureList<Keypoint> getFeatures(FImage image)
	{
		LocalFeatureList<Keypoint> cached = this.featureCache.get(image);
		
		if(cached != null)
		{
			return cached;
		}
		
		DoGSIFTEngine engine = new DoGSIFTEngine();	
		LocalFeatureList<Keypoint> features = engine.findFeatures(image);

		this.featureCache.put(image, features);

		return features;
	}
}
