package uk.ac.soton.ecs.imageclassifer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.util.function.Operation;
import org.openimaj.util.parallel.Parallel;

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
		if(args.length < 2)
			throw new IllegalArgumentException("Usage: AccuracyTest <num training> <num testing>");
		
		int numTraining = Integer.parseInt(args[0]), numTesting = Integer.parseInt(args[1]);
		
		ArrayList<AccuracyTest> tests = new ArrayList<>();
		
		// Initialise accuracy testers
		tests.add(new AccuracyTest(new RandomGuesser(), numTraining, numTesting));
		tests.add(new AccuracyTest(new BoVW(), numTraining, numTesting));
		tests.add(new AccuracyTest(new SURFBoVW(), numTraining, numTesting));
		tests.add(new AccuracyTest(new SIFTBoVW(), numTraining, numTesting));
		tests.add(new AccuracyTest(new PyramidSift(), numTraining, numTesting));
		
		// Run
		Parallel.forEach(tests, new Operation<AccuracyTest>() {

			@Override
			public void perform(AccuracyTest test)
			{
				test.run();
			}
			
		});
		
		// Save results
		File saveDir = new File("results/accuracy");
		for (AccuracyTest t : tests) {
			try
			{
				PrintWriter pw = new PrintWriter(new File(saveDir + t.alg.getClass().getSimpleName()));
				pw.write(t.getResults());
				pw.close();
			}
			catch(FileNotFoundException ex)
			{
				ex.printStackTrace();
			}
		}
	
	}
	
	private GroupedDataset<String, ListDataset<FImage>, FImage> trainingSet, testingSet;
	private ClassificationAlgorithm alg;
	private String testname;
	private StringBuilder results;

	public AccuracyTest(ClassificationAlgorithm alg, int numTraining, int numTesting) throws FileSystemException
	{
		File trainingFile = new File("imagesets/training");

		VFSGroupDataset<FImage> data = new VFSGroupDataset<>(
			trainingFile.getAbsolutePath(),
			ImageUtilities.FIMAGE_READER);

		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<>(data, numTraining, 0, numTesting);

		trainingSet = splits.getTrainingDataset();
		testingSet = splits.getTestDataset();
		this.alg = alg;
		this.testname = "[" + alg.getClass().getSimpleName() + "] ";
	}
	
	/**
	 * Run training and show results to console
	 */
	public void run() {
		run(false);
	}
	
	private void run(boolean quiet) {
		if (!quiet) {
			System.out.println(testname + "Performing Training...");
		}
		
		alg.train(AnnotatedObject.createList(trainingSet));

		if (!quiet) {
			System.out.println(testname + "Running Tests...");
		}
		
		/*
		 * Iterate over each image in the group dataset and check that the
		 * classified result is the same as the group name
		 */
		int correct = 0;
		for(Entry<String, ListDataset<FImage>> e : testingSet.entrySet())
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
				String expected = "Expected: " + e.getKey() + ", Returned: " + mostLikely;
				results.append(expected);
				results.append('\n');
				
				if (!quiet) {
					System.out.println(testname + expected);
				}
				
				if(mostLikely.equals(e.getKey()))
					correct++;
			}
		}

		/*
		 * Show percentage error
		 */
		String accuracy = "Accuracy: " + correct * 100f / testingSet.numInstances();
		results.append("=============================");
		results.append(accuracy);
		results.append('\n');
		
		if (!quiet) {
			System.out.println(testname + accuracy);
		}
	}
	
	/**
	 * Return test results as a string.
	 * Runs the testing session if not already run.
	 * 
	 * @return Results of test
	 */
	public String getResults() {
		if (results == null) run(true);
		return results.toString();
	}
}
