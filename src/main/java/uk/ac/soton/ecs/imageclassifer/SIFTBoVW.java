package uk.ac.soton.ecs.imageclassifer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureVectorProvider;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.training.BatchTrainer;

import com.stromberglabs.jopensurf.SURFInterestPoint;
import com.stromberglabs.jopensurf.Surf;

import de.bwaldvogel.liblinear.SolverType;

/**
 * Bag of Visual Words classifier Given a grouped dataset of training images, and a list dataset of testing images, BoVW
 * will use K-means on overlapping 8x8 image patches in each image to generate a 'codebook' for the quantiser, then
 * liblinear annotation to classify the input.
 * 
 * @author cw17g12
 * 
 */
public class SIFTBoVW implements ClassificationAlgorithm
{

	protected int codebookSize = 500;
	protected int patchSize = 8;
	protected int patchSeparation = patchSize / 2;

	protected BagOfVisualWords<byte[]> quantiser;
	protected LiblinearAnnotator<FImage, String> annotator;

	public static void main(String[] args) throws FileSystemException, FileNotFoundException
	{
		Utilities.runClassifier(new SIFTBoVW(), "PyramidSift", args);
	}

	@Override
	public void train(List<? extends Annotated<FImage, String>> data)
	{
		this.featureCache = new HashMap<>();
		
		trainQuantiser(data);
		trainAnnotator(data);
	}

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
	
	protected Map<FImage, LocalFeatureList<Keypoint>> featureCache;

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
