package uk.ac.soton.ecs.imageclassifer;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.FImage2DoubleFV;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.ml.annotation.ScoredAnnotation;

/**
 * Collection of generic utilities
 * @author Sam Lavers
 */
public class Utilities
{	
	/**
	 * Returns a new zero-meaned feature vector from an existing feature fector
	 * @param feature Existing feature vector
	 * @return Zero-meaned version of feature
	 */
	public static FloatFV zeroMean(DoubleFV feature)
	{
		FloatFV converted = new FloatFV(feature.values.length);
		
		double sum = 0;
		
		for(double value : feature.values)
		{
			sum += value;
		}
		
		double mean = sum / feature.values.length;
		
		for(int i = 0; i < feature.values.length; i++)
		{
			converted.values[i] = (float) (feature.values[i] - mean);
		}
		
		return converted;
	}
	
	/**
	 * Convienience method for zero-meaning an FImage
	 * @param image The image
	 * @return Zero mean feature vector
	 */
	public static FloatFV zeroMean(FImage image)
	{
		return Utilities.zeroMean(FImage2DoubleFV.INSTANCE.extractFeature(image));
	}
	
	/**
	 * Converts a list of scored annotations from an annotator to a classification result
	 * @param annotations The list of scored annotations
	 * @return The classification result
	 */
	public static <T> ClassificationResult<T> scoredListToResult(List<ScoredAnnotation<T>> annotations)
	{
		PrintableClassificationResult<T> result = new PrintableClassificationResult<>();

		for(ScoredAnnotation<T> annotation : annotations)
		{
			result.put(annotation.annotation, annotation.confidence);
		}
		
		return result;
	}
	
	/**
	 * Convienience method for testing a classifier from the command line
	 * @param classifier The classifier
	 * @param classifierName The name of the classifier for results filename
	 * @param args Command line arguments
	 */
	public static void runClassifier(ClassificationAlgorithm classifier, String classifierName, String[] args)
	{
		if(args.length < 2)
		{
			throw new IllegalArgumentException("Usage: " + classifierName + " <training uri> <testing uri>");
		}
		
		// Open results file
		
		System.out.println("Opening results file...");
		
		PrintWriter writer = null;
		
		try
		{
			writer = new PrintWriter("results/classify/" + classifierName + ".txt", "UTF-8");
		}
		catch(IOException e)
		{
			System.err.println("Can't open results file!");
			System.exit(1);
		}
		
		// Load datasets

		File trainingFile = new File(args[0]);
		File testingFile = new File(args[1]);

		System.out.println("Loading datasets...");
		
		 VFSGroupDataset<FImage> training = null;
		VFSListDataset<FImage> testing = null;
		
		try
		{
			training = new VFSGroupDataset<>(trainingFile.getAbsolutePath(), ImageUtilities.FIMAGE_READER);
			testing = new VFSListDataset<>(testingFile.getAbsolutePath(),ImageUtilities.FIMAGE_READER);
		}
		catch(FileSystemException e)
		{
			System.err.println("Couldn't load dataset: " + e.getMessage());
			System.exit(1);
		}
		
		// Train classifier

		System.out.println("Training the classifier...");
		
		classifier.train(AnnotatedObject.createList(training));
		
		// Classify testing set & write results

		System.out.println("Classifing testing set...");
			
		int i = 0;
		for(FImage image : testing)
		{
			System.out.println("Classifying image " + i);
			
			writer.print(testing.getID(i++) + " => ");
			writer.println(classifier.classify(image));
		}
			
		writer.close();
	}
}
