package uk.ac.soton.ecs.imageclassifier;

import java.io.File;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.AnnotatedObject;

import uk.ac.soton.ecs.imageclassifer.*;

/**
 * 
 * Tests classification algorithms on a subset of the training data
 * 
 * @author cw17g12
 *
 */
public class AccuracyTest
{
	public static void main(String[] args) throws FileSystemException
	{
		new AccuracyTest(new RandomGuesser(), 10, 10);
	}

	public AccuracyTest(ClassificationAlgorithm alg, int numTraining, int numTesting) throws FileSystemException
	{
		System.out.println("Loading Images...");
		File trainingFile = new File("imagesets/training");

		VFSGroupDataset<FImage> data = new VFSGroupDataset<>(
			trainingFile.getAbsolutePath(),
			ImageUtilities.FIMAGE_READER);

		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(data, numTraining, 0, numTesting);

		System.out.println("Performing Training...");
		alg.train(AnnotatedObject.createList(splits.getTrainingDataset()));

		/*
		 * Iterate over each image in the group dataset and check that the
		 * classified result is the same as the group name
		 */
		int correct = 0;
		for(Entry<String, ListDataset<FImage>> e : splits.getTestDataset().entrySet())
		{
			for(FImage image : e.getValue())
			{
				ClassificationResult<String> c = alg.classify(image);

				/*
				 * Iterate over results to find the one with the highest confidence
				 */
				double confidence = 0;
				String mostLikely = "unknown";
				for(String clazz : c.getPredictedClasses())
				{
					double conf = c.getConfidence(clazz);
					if(conf > confidence)
					{
						mostLikely = clazz;
						confidence = conf;
					}
				}

				/*
				 * Show expected and classified values
				 */
				System.out.printf("Expected: %s, Returned: %s\n",
					e.getKey(), mostLikely);
				if(mostLikely.equals(e.getKey()))
					correct++;
			}
		}

		/*
		 * Show percentage error
		 */
		System.out.printf("Accuracy: %f%%\n", correct * 100f / splits.getTestDataset().numInstances());
	}
}
