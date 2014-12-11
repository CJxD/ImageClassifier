package uk.ac.soton.ecs.imageclassifer;

import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.training.BatchTrainer;

/**
 * Represents a classification algorithm which can be trained on a set of images and classify an image.
 * All annotations are strings
 * @author Sam Lavers
 */
public interface ClassificationAlgorithm extends Classifier<String, FImage>, BatchTrainer<Annotated<FImage, String>>
{

}
