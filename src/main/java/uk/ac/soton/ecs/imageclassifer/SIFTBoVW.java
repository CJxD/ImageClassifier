package uk.ac.soton.ecs.imageclassifer;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
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

		SIFTBoVW bovw = new SIFTBoVW();

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
		
		return Utilities.scoredListToResult(annotator.annotate(image));
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

		FeatureExtractor<DoubleFV, FImage> extractor = new FeatureExtractor<DoubleFV, FImage>()
		{
			@Override
			public DoubleFV extractFeature(FImage image)
			{
				return quantiser.aggregate(getFeatures(image)).normaliseFV();
			}
		};
		
		HomogeneousKernelMap homo = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
		extractor = homo.createWrappedExtractor(extractor);

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
