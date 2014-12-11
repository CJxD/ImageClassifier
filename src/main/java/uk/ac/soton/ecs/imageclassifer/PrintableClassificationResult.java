package uk.ac.soton.ecs.imageclassifer;

import java.util.Iterator;

import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.ml.annotation.ScoredAnnotation;

/**
 * Represents a classification result that can be output as a string
 * 
 * @author Chris Watts
 * @param <CLASS> The annotation type
 */
public class PrintableClassificationResult<CLASS> extends BasicClassificationResult<CLASS> implements
	Iterable<ScoredAnnotation<CLASS>>
{
	public static final int RESULT_LIST = 0x0;
	public static final int BEST_RESULT = 0x1;

	private int type;

	/**
	 * Create a printable classification result with the default type (list)
	 */
	public PrintableClassificationResult()
	{
		this(RESULT_LIST);
	}

	/**
	 * Create a printable classificaiton result with the specified type
	 * @param type The type
	 */
	public PrintableClassificationResult(int type)
	{
		this.type = type;
	}

	/**
	 * Implements the iterator interface
	 */
	@Override
	public Iterator<ScoredAnnotation<CLASS>> iterator()
	{
		return new Iterator<ScoredAnnotation<CLASS>>() {

			private Iterator<CLASS> it = getPredictedClasses().iterator();

			@Override
			public boolean hasNext()
			{
				return it.hasNext();
			}

			@Override
			public ScoredAnnotation<CLASS> next()
			{
				CLASS clazz = it.next();
				return new ScoredAnnotation<CLASS>(clazz, (float) getConfidence(clazz));
			}

			@Override
			public void remove()
			{
				throw new UnsupportedOperationException();
			}

		};
	}

	/**
	 * Get the most confident result the classifier output
	 * @return The best confidence result
	 */
	public String bestResult()
	{
		double confidence = 0f;
		CLASS best = null;

		for(CLASS clazz : getPredictedClasses())
		{
			double c = getConfidence(clazz);
			if(c > confidence)
			{
				best = clazz;
				confidence = c;
			}
		}
		
		return best == null ? "unknown" : best.toString();
	}

	/**
	 * Get a list of all the results
	 * @return The results
	 */
	public String resultList()
	{
		StringBuilder sb = new StringBuilder();

		for(CLASS clazz : getPredictedClasses())
		{
			sb.append("\t");
			sb.append(clazz.toString());
			sb.append(" (");
			sb.append(getConfidence(clazz));
			sb.append(")\n");
		}

		return sb.toString();
	}

	/**
	 * Output as string depending on the type of the result
	 * @return The result
	 */
	@Override
	public String toString()
	{
		if(type == BEST_RESULT)
		{
			return bestResult();
		}
		else
		{
			return resultList();
		}
	}
}
