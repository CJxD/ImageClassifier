package uk.ac.soton.ecs.imageclassifer;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

/**
 * Image classifier implementation using dense pyramid sift features and a lib linear annotator
 * @author Sam Lavers
 */
public class PyramidSift implements ClassificationAlgorithm
{
	protected LiblinearAnnotator<FImage, String> annotator;	

	public static void main(String[] args)
	{
		Utilities.runClassifier(new PyramidSift(), "PyramidSift", args);
	}
	
	/**
	 * Train the classifier from a list of annotated images
	 * @param data The training set
	 */
	@Override
	public void train(List<? extends Annotated<FImage, String>> data)
	{
		// Setup PDSIFT
		
		DenseSIFT dsift = new DenseSIFT(5, 7);
		
		final PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		
		final HardAssigner<byte[], float[], IntFloatPair> assigner = this.trainQuantiser(data, pdsift);
		
		// Create a feature extractor which uses BoVW to generate the feature vector
		
		FeatureExtractor<DoubleFV, FImage> extractor = new FeatureExtractor<DoubleFV, FImage>()
		{
			@Override
			public DoubleFV extractFeature(FImage image)
			{
				pdsift.analyseImage(image);

				BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

				BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(bovw, 2, 2);

				return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
			}
		};
		
		// Train the annotator

		this.annotator = new LiblinearAnnotator<FImage, String>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		this.annotator.train(data);
	}
	
	/**
	 * Trains the quantiser using an existing PDSIFT instance and a training dataset
	 * @param data The training set
	 * @param pdsift The current PDSIFT instance
	 * @return The hard assigner
	 */
	protected HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(List<? extends Annotated<FImage, String>> data, PyramidDenseSIFT<FImage> pdsift)
	{
		List<LocalFeatureList<ByteDSIFTKeypoint>> allimages = new ArrayList<>();

		for(Annotated<FImage, String> annotated : data)
		{
			pdsift.analyseImage(annotated.getObject());
			allimages.add(pdsift.getByteKeypoints(0.005f));
		}
		
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allimages);

		ByteKMeans kmeans = ByteKMeans.createKDTreeEnsemble(300);
		ByteCentroidsResult result = kmeans.cluster(datasource);

		return result.defaultHardAssigner();
	}
	
	/**
	 * Classify an image
	 * @param image The image
	 * @return The classification 
	 */
	@Override
	public ClassificationResult<String> classify(FImage image)
	{
		return Utilities.scoredListToResult(annotator.annotate(image));
	}
}
