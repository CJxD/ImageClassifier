package uk.ac.soton.ecs.imageclassifer;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

public class App 
{
    public static void main(String[] args) 
	{
        MBFImage image = new MBFImage(320,70, ColourSpace.RGB);

        image.fill(RGBColour.WHITE);
        		        
        image.drawText("Hello World", 10, 60, HersheyFont.CURSIVE, 50, RGBColour.BLACK);

        image.processInplace(new FGaussianConvolve(2f));
        
        DisplayUtilities.display(image);
    }
}
