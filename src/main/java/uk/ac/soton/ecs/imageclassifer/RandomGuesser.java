package uk.ac.soton.ecs.imageclassifer;

import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotated;

/**
 * 
 * Literally just guesses classes randomly
 * 
 * @author cw17g12
 *
 */
public class RandomGuesser implements ClassificationAlgorithm
{
	private HashSet<String> classes = new HashSet<>();
	
	public static void main(String[] args) throws FileSystemException, FileNotFoundException
	{
		Utilities.runClassifier(new RandomGuesser(), "Random", args);
	}
	
	@Override
	public ClassificationResult<String> classify(FImage object)
	{
		int clazz = new Random().nextInt(classes.size());
		
		PrintableClassificationResult<String> result = new PrintableClassificationResult<>(PrintableClassificationResult.BEST_RESULT);
		result.put(classes.toArray(new String[0])[clazz], 1);
		
		return result;
	}

	@Override
	public void train(List<? extends Annotated<FImage, String>> data)
	{
		// Make a set of known classes
		for (Annotated<FImage, String> d : data) {
			classes.add(d.getAnnotations().iterator().next());
		}
		
		// Give it enough time to train
		long finish = System.currentTimeMillis() + 30000;
		while (System.currentTimeMillis() < finish) {}
	}

}
