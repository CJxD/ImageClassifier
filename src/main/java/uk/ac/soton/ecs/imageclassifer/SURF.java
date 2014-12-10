package uk.ac.soton.ecs.imageclassifer;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Iterator;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.ml.annotation.bayes.NaiveBayesAnnotator;
import org.openimaj.ml.training.BatchTrainer;
import org.openimaj.util.function.Operation;
import org.openimaj.util.parallel.Parallel;
import org.openimaj.util.parallel.partition.RangePartitioner;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FeatureVector;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;

import com.stromberglabs.jopensurf.*;

public class SURF implements ClassificationAlgorithm
{
	public static void main(String[] args) throws FileSystemException
	{
		if(args.length < 2)
		{
			throw new IllegalArgumentException("Usage: SURF <training set uri> <testing set uri>");
		}
		
		VFSGroupDataset<FImage> trainingSet = new VFSGroupDataset<>(args[0], ImageUtilities.FIMAGE_READER);
		VFSListDataset<FImage> testingSet = new VFSListDataset<>(args[1], ImageUtilities.FIMAGE_READER);

		SURF classifier = new SURF();
		
		System.out.println("training...");

		classifier.train(AnnotatedObject.createList(trainingSet));
		
		System.out.println("classifying...");

		int i = 0;

		for(FImage image : testingSet)
		{
			if(i > 10)
			{
				break;
			}
			
			System.out.println(testingSet.getID(i++) + " " + classifier.classify(image));
		}
    }
	
	protected NaiveBayesAnnotator<FImage, String> bayes;
	protected BagOfVisualWords<float[]> quantiser;
	
	private class SurfInterestPoint implements LocalFeature<SpatialLocation, FloatFV>
	{
		protected SpatialLocation location;
		protected FloatFV vector;
		
		public SurfInterestPoint(SURFInterestPoint point)
		{
			this.location = new SpatialLocation(point.getX(), point.getY());
			this.vector = new FloatFV(point.getDescriptor());
		}

		@Override
		public void readASCII(Scanner in) throws IOException
		{
			// TODO Auto-generated method stub
			
		}

		@Override
		public String asciiHeader()
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public void readBinary(DataInput in) throws IOException
		{
			// TODO Auto-generated method stub
			
		}

		@Override
		public byte[] binaryHeader()
		{
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public void writeASCII(PrintWriter out) throws IOException
		{
			// TODO Auto-generated method stub
			
		}

		@Override
		public void writeBinary(DataOutput out) throws IOException
		{
			// TODO Auto-generated method stub
			
		}

		@Override
		public FloatFV getFeatureVector()
		{
			return this.vector;
		}

		@Override
		public SpatialLocation getLocation()
		{
			return this.location;
		}
	
	}

	public SURF()
	{
		this.bayes = new NaiveBayesAnnotator<>(new FeatureExtractor<FloatFV, FImage>()
		{
			@Override
			public FloatFV extractFeature(FImage image)
			{	
				Surf surf = new Surf(ImageUtilities.createBufferedImage(image));
				
				List<SURFInterestPoint> points = surf.getUprightInterestPoints();
				
				for(SURFInterestPoint point : points)
				{
					
				}
				
				/*
				 * FIXME
				 */
				return null;
			}
		}, NaiveBayesAnnotator.Mode.ALL);
	}
	
	@Override
	public void train(List<? extends Annotated<FImage, String>> data)
	{
		/*
		 * FIXME
		 */
		this.quantiser = new BagOfVisualWords<float[]>(null);
		Parallel.forEach(data, new Operation<Annotated<FImage, String>>()
		{
			@Override
			public void perform(Annotated<FImage, String> object)
			{
				Surf surf = new Surf(ImageUtilities.createBufferedImage(object.getObject()));

				List<SURFInterestPoint> points = surf.getUprightInterestPoints();
				
				for(SURFInterestPoint point : points)
				{ 
					// SURF.this.bayes.train(new AnnotatedObject<FloatFV, String>(new FloatFV(point.getDescriptor()), object.getAnnotations().iterator().next()));
				}
			}
		});
	}

	@Override
	public ClassificationResult<String> classify(FImage object)
	{
		this.bayes.annotate(object);
		return null;
	}
}
