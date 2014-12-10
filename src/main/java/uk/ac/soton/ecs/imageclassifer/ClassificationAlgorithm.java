package uk.ac.soton.ecs.imageclassifer;

import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.training.BatchTrainer;

public interface ClassificationAlgorithm extends Classifier<String, FImage>, BatchTrainer<Annotated<FImage, String>>
{

}
