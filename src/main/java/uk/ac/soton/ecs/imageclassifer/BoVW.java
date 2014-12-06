package uk.ac.soton.ecs.imageclassifer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
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
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;

import de.bwaldvogel.liblinear.SolverType;

/**
 * Bag of Visual Words classifier Given a grouped dataset of training images, and a list dataset of testing images, BoVW
 * will use K-means on overlapping 8x8 image patches in each image to generate a 'codebook' for the quantiser, then
 * liblinear annotation to classify the input.
 * 
 * @author cw17g12
 * 
 */
public class BoVW
{

	protected int codebookSize = 500;
	protected int patchSize = 8;
	protected int patchSeparation = patchSize / 2;

	protected BagOfVisualWords<float[]> quantiser;
	protected LiblinearAnnotator<FImage, String> annotator;

	public static void main(String[] args) throws FileSystemException
	{
		if(args.length < 2)
			throw new IllegalArgumentException("Usage: BoVW <training uri> <testing uri>");

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
		
		BoVW bovw = new BoVW();

		// Sampler hack to fix the Java inheritance of VFSListDataset
		bovw.train(GroupedUniformRandomisedSampler.sample(training, 30));
		
		System.out.println("Classifing testing set...");

		int i = 0;
		for(FImage image : testing)
		{
			System.out.println(testing.getID(i++) + " => " + bovw.classify(image));
		}
	}

	/**
	 * Represents a small image patch as a feature vector and a spatial location.
	 * 
	 * @author cw17g12
	 *
	 */
	protected class ImagePatch
		implements
		LocalFeature<SpatialLocation, FloatFV>,
		LocalFeatureVectorProvider<SpatialLocation, FloatFV>
	{

		private FloatFV fv;
		private SpatialLocation location;

		public ImagePatch(int x, int y, FImage patch)
		{
			this.fv = Utilities.zeroMean(patch.normalise());
			this.location = new SpatialLocation(x, y);
		}

		@Override
		public void readASCII(Scanner in) throws IOException
		{
			throw new UnsupportedOperationException();
		}

		@Override
		public String asciiHeader()
		{
			throw new UnsupportedOperationException();
		}

		@Override
		public void readBinary(DataInput in) throws IOException
		{
			throw new UnsupportedOperationException();
		}

		@Override
		public byte[] binaryHeader()
		{
			throw new UnsupportedOperationException();
		}

		@Override
		public void writeASCII(PrintWriter out) throws IOException
		{
			throw new UnsupportedOperationException();
		}

		@Override
		public void writeBinary(DataOutput out) throws IOException
		{
			throw new UnsupportedOperationException();
		}

		@Override
		public FloatFV getFeatureVector()
		{
			return fv;
		}

		@Override
		public SpatialLocation getLocation()
		{
			return location;
		}

	}

	/**
	 * Classifies each image in the test data
	 * 
	 * @param testData
	 * @return Map of image to each category and its confidence
	 */
	public Map<FImage, List<ScoredAnnotation<String>>> classify(ListDataset<FImage> testData)
	{
		if(quantiser == null)
			throw new IllegalStateException("Classifier is not trained");
		if(annotator == null)
			throw new IllegalStateException("Annotator is not trained");
		
		// Annotate each image in the test data with scored annotations
		Map<FImage, List<ScoredAnnotation<String>>> result = new HashMap<>(testData.size());
		for(FImage image : testData)
		{
			result.put(image, annotator.annotate(image));
		}

		return result;
	}
	
	/**
	 * Classifies a single image
	 * 
	 * @param image
	 * @return Each category and its confidence
	 */
	public List<ScoredAnnotation<String>> classify(FImage image)
	{
		if(quantiser == null)
			throw new IllegalStateException("Classifier is not trained");
		if(annotator == null)
			throw new IllegalStateException("Annotator is not trained");

		return annotator.annotate(image);
	}

	/**
	 * Trains the quantiser and the annotator for classification.
	 * 
	 * @param trainingData
	 */
	public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingData)
	{
		trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingData, 30));
		trainAnnotator(trainingData);
	}

	/**
	 * Trains the Bag of Visual Words with a K-means-generated codebook.
	 * 
	 * @param trainingData
	 */
	protected void trainQuantiser(Dataset<FImage> trainingData)
	{
		List<LocalFeatureList<ImagePatch>> allFeatures = new ArrayList<>();

		// Load a list of ImagePatch features
		for(FImage image : trainingData)
		{
			allFeatures.add(getPatches(image));
		}

		// Populate a DataSource with the ImagePatches
		DataSource<float[]> datasource = new LocalFeatureListDataSource<ImagePatch, float[]>(
			allFeatures);

		// Create n centroids to act as a codebook for the bag of visual words
		FloatKMeans km = FloatKMeans.createKDTreeEnsemble(codebookSize);
		FloatCentroidsResult centroids = km.cluster(datasource);

		// Any inputs will be quantised to the nearest ImagePatch centroid
		quantiser = new BagOfVisualWords<float[]>(centroids.defaultHardAssigner());
	}

	/**
	 * Trains the liblinear annotator with the trained quantiser.
	 * 
	 * @param trainingData
	 */
	protected void trainAnnotator(GroupedDataset<String, ListDataset<FImage>, FImage> trainingData)
	{
		if(quantiser == null)
			throw new IllegalStateException("Quantiser is not trained");

		FeatureExtractor<SparseIntFV, FImage> extractor = new FeatureExtractor<SparseIntFV, FImage>() {

			@Override
			public SparseIntFV extractFeature(FImage image)
			{
				// Quantise the ImagePatches in the input to the nearest centroid
				return quantiser.aggregate(getPatches(image));
			}

		};

		// Train the annotator to make associations between certain "words" and image classes
		annotator = new LiblinearAnnotator<>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		annotator.train(trainingData);
	}

	/**
	 * Segments an image into centre-meaned, normalised patches. Will crop any pixels from the right and bottom of the
	 * image that do not divide into {@link #patchSeparation}.
	 * 
	 * @param image
	 * @return list of {@link #patchSize} by {@link #patchSize} patches
	 */
	public LocalFeatureList<ImagePatch> getPatches(FImage image)
	{
		int patchesX = (image.width / patchSeparation) - 1;
		int patchesY = (image.height / patchSeparation) - 1;

		// Image size must be a multiple of patchSeparation
		Rectangle crop = new Rectangle(0, 0, patchSeparation * patchesX + 1, patchSeparation * patchesY + 1);

		image = image.extractROI(crop);

		// Extract patchesX * patchesY patches of size patchSize
		LocalFeatureList<ImagePatch> patches = new MemoryLocalFeatureList<ImagePatch>(patchesX * patchesY);
		FImage temp;
		for(int y = 0; y < image.height; y += patchSeparation)
		{
			for(int x = 0; x < image.width; x += patchSeparation)
			{

				temp = image.extractROI(x, y, patchSize, patchSize);

				// Normalise and mean-centre the pixels
				// Converts image to feature vector
				patches.add(new ImagePatch(x, y, temp));

			}
		}

		return patches;
	}
}
