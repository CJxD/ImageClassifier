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
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
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
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import com.stromberglabs.jopensurf.SURFInterestPoint;
import com.stromberglabs.jopensurf.Surf;

import de.bwaldvogel.liblinear.SolverType;

/**
 * SURF interest points image classifier using bag of words, lib linear annotator and homgenous kernel maps
 * 
 * @author Sam Lavers
 */
public class SURFBoVW implements ClassificationAlgorithm
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

		SURFBoVW bovw = new SURFBoVW();

		bovw.train(AnnotatedObject.createList(training));

		System.out.println("Classifing testing set...");

		int i = 0;
		for(FImage image : testing)
		{
			System.out.print(testing.getID(i++) + " => ");
			System.out.println(bovw.classify(image));

			if(i > 10)
			{
				break;
			}
		}
	}

	/**
	 * Represents a surf interest point as a local feature
	 * 
	 * @author Sam Lavers
	 */
	protected class SurfInterestPoint
		implements
		LocalFeature<SpatialLocation, FloatFV>,
		LocalFeatureVectorProvider<SpatialLocation, FloatFV>
	{

		private FloatFV fv;
		private SpatialLocation location;

		public SurfInterestPoint(SURFInterestPoint point)
		{
			this.location = new SpatialLocation(point.getX(), point.getY());
			this.fv = new FloatFV(point.getDescriptor());
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
	 * @param data The trainign set
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
	
	protected Map<FImage, LocalFeatureList<SurfInterestPoint>> featureCache;

	/**
	 * Trains the Bag of Visual Words with a K-means-generated codebook.
	 * 
	 * @param trainingData
	 */
	protected void trainQuantiser(List<? extends Annotated<FImage, String>> data)
	{
		List<LocalFeatureList<SurfInterestPoint>> allFeatures = new ArrayList<>();
		
		int i = 0;
		
		// Load a list of ImagePatch features
		for(Annotated<FImage, String> image : data)
		{
			System.out.println("training quanitzer " + ++i);
			
			allFeatures.add(getFeatures(image.getObject()));
		}

		// Populate a DataSource with the ImagePatches
		DataSource<float[]> datasource = new LocalFeatureListDataSource<SurfInterestPoint, float[]>(
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

		FeatureExtractor<DoubleFV, FImage> extractor = new FeatureExtractor<DoubleFV, FImage>()
		{
			@Override
			public DoubleFV extractFeature(FImage image)
			{
				SparseIntFV potato = quantiser.aggregate(getFeatures(image));
				
				return potato.normaliseFV();
			}
		};
		
		HomogeneousKernelMap homo = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		extractor = homo.createWrappedExtractor(extractor);

		// Train the annotator to make associations between certain "words" and image classes

		annotator = new LiblinearAnnotator<>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		annotator.train(data);
	}

	/**
	 * Gets the SURF interest points for a given image
	 * @param image The image
	 * @return SURF interest points
	 */
	protected LocalFeatureList<SurfInterestPoint> getFeatures(FImage image)
	{
		LocalFeatureList<SurfInterestPoint> cached = this.featureCache.get(image);
		
		if(cached != null)
		{
			return cached;
		}
		
		Surf surf = new Surf(ImageUtilities.createBufferedImage(image));

		List<SURFInterestPoint> points = surf.getUprightInterestPoints();

		LocalFeatureList<SurfInterestPoint> features = new MemoryLocalFeatureList<SurfInterestPoint>(points.size());

		for(SURFInterestPoint point : points)
		{
			features.add(new SurfInterestPoint(point));
		}
		
		this.featureCache.put(image, features);

		return features;
	}
}
