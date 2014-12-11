package uk.ac.soton.ecs.imageclassifer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
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
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.AnnotatedObject;
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
public class BoVW implements ClassificationAlgorithm
{

	protected int codebookSize = 500;
	protected int patchSize = 8;
	protected int patchSeparation = patchSize / 2;

	protected BagOfVisualWords<float[]> quantiser;
	protected LiblinearAnnotator<FImage, String> annotator;

	public static void main(String[] args) throws FileSystemException, FileNotFoundException
	{
		Utilities.runClassifier(new BoVW(), "ImagePatches", args);
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
			this.fv = new FloatFV(patch.getFloatPixelVector());
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
	 * Train the classifier
	 * @param data The training set
	 */
	@Override
	public void train(List<? extends Annotated<FImage, String>> data)
	{
		trainQuantiser(data);
		trainAnnotator(data);
	}

	/**
	 * Classify an image
	 * @param image The image
	 * @return The classification result
	 */
	@Override
	public ClassificationResult<String> classify(FImage image)
	{
		if(quantiser == null)
			throw new IllegalStateException("Classifier is not trained");
		if(annotator == null)
			throw new IllegalStateException("Annotator is not trained");

		PrintableClassificationResult<String> result = new PrintableClassificationResult<>(PrintableClassificationResult.BEST_RESULT);
		
		for (ScoredAnnotation<String> a : annotator.annotate(image)) {
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
		List<LocalFeatureList<ImagePatch>> allFeatures = new ArrayList<>();

		// Load a list of ImagePatch features
		for(Annotated<FImage, String> image : data)
		{
			allFeatures.add(getPatches(image.getObject()));
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
	protected void trainAnnotator(List<? extends Annotated<FImage, String>> data)
	{
		if(quantiser == null)
			throw new IllegalStateException("Quantiser is not trained");

		FeatureExtractor<SparseIntFV, FImage> extractor = new FeatureExtractor<SparseIntFV, FImage>() {

			@Override
			public SparseIntFV extractFeature(FImage image)
			{
				// Quantise the ImagePatches in the input to the nearest centroid
				LocalFeatureList<ImagePatch> patches = getPatches(image);
				SparseIntFV test = quantiser.aggregate(patches);
				return test;
			}

		};

		// Train the annotator to make associations between certain "words" and image classes
		annotator = new LiblinearAnnotator<>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		annotator.train(data);
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
