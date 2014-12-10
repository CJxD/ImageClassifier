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

public class AccuracyTest
{
	public static void main(String[] args) throws FileSystemException {
		new AccuracyTest(new SIFTBoVW());
	}
	
	public AccuracyTest(ClassificationAlgorithm alg) throws FileSystemException {
		File trainingFile = new File("imagesets/training");
		
		VFSGroupDataset<FImage> data = new VFSGroupDataset<>(
			trainingFile.getAbsolutePath(),
			ImageUtilities.FIMAGE_READER);
		
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(data, 10, 0, 10);
		
		alg.train(AnnotatedObject.createList(splits.getTrainingDataset()));
		
		int correct = 0;
		for (Entry<String, ListDataset<FImage>> e : splits.getTestDataset().entrySet()) {
			for (FImage image : e.getValue()) {
				ClassificationResult<String> c = alg.classify(image);
				
				double confidence = 0;
				String mostLikely = "unknown";
				for (String clazz : c.getPredictedClasses())
				{
					double conf = c.getConfidence(clazz);
					if (conf > confidence) {
						mostLikely = clazz;
						confidence = conf;
					}
				}
				
				System.out.printf("Expected: %s, Returned: %s\n",
					e.getKey(), mostLikely);
				if (mostLikely.equals(e.getKey())) correct++;
			}
		}
		
		System.out.printf("Accuracy: %f%%\n", correct * 100f / splits.getTestDataset().numInstances());
	}
}
