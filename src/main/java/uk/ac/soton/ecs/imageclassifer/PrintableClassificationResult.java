package uk.ac.soton.ecs.imageclassifer;

import java.util.Iterator;

import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.ml.annotation.ScoredAnnotation;

public class PrintableClassificationResult<CLASS> extends BasicClassificationResult<CLASS> implements Iterable<ScoredAnnotation<CLASS>>
{
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
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		for (CLASS clazz : getPredictedClasses())
		{
			sb.append("\t");
			sb.append(clazz.toString());
			sb.append(" (");
			sb.append(getConfidence(clazz));
			sb.append(")\n");
		}
		
		return sb.toString();
	}
}
