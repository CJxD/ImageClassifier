package uk.ac.soton.ecs.imageclassifer;

import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.FImage2FloatFV;
import org.openimaj.image.processor.PixelProcessor;

public class Utilities
{
	public static FloatFV zeroMean(FImage image)
	{
		final float mean = image.sum() / (image.getWidth() * image.getHeight());

		FImage processed = image.process(new PixelProcessor<Float>()
		{
			@Override
			public Float processPixel(Float pixel)
			{
				return pixel - mean;
			}
		});

		return FImage2FloatFV.INSTANCE.extractFeature(processed);
	}
}
