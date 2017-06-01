// Example : neural network learning
// usage: prog training_data_file testing_data_file

// For use with test / training datasets : optical_ex

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2010 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/******************************************************************************/

// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 3823
#define ATTRIBUTES_PER_SAMPLE 64
#define NUMBER_OF_TESTING_SAMPLES 1797

#define NUMBER_OF_CLASSES 10

// N.B. classes are integer handwritten digits in range 0-9

/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat data, Mat classes,
                       int n_samples )
{
    float tmp;

    // if we can't read the input file then return 0
    FILE* f = fopen( filename, "r" );
    if( !f )
    {
        printf("ERROR: cannot read file %s\n",  filename);
        return 0; // all not OK
    }

    // for each sample in the file

    for(int line = 0; line < n_samples; line++)
    {

        // for each attribute on the line in the file

        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
        {
            if (attribute < 64)
            {

                // first 64 elements (0-63) in each line are the attributes

                fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute) = tmp;
                // printf("%f,", data.at<float>(line, attribute));

            }
            else if (attribute == 64)
            {

                // attribute 65 is the class label {0 ... 9}

                fscanf(f, "%f,", &tmp);
                classes.at<float>(line, (int) tmp) = 1.0;
                // printf("%f\n", classes.at<float>(line, 0));

            }
        }
    }

    fclose(f);

    return 1; // all OK
}

/******************************************************************************/

int main( int argc, char** argv )
{
    // lets just check the version first

    printf ("OpenCV version %s (%d.%d.%d)\n",
            CV_VERSION,
            CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);

    // define training data storage matrices (one for attribute examples, one
    // for classifications)

    Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, NUMBER_OF_CLASSES, CV_32FC1);

    // define testing data storage matrices

    Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat testing_classifications = Mat::zeros(NUMBER_OF_TESTING_SAMPLES, NUMBER_OF_CLASSES, CV_32FC1);

    // define classification output vector

    Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
    Point max_loc = Point(0,0);

    // load training and testing data sets

    if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {
        // define the parameters for the neural network (MLP)

        // set the network to be 3 layer 64->10->10
        // - one input node per attribute in a sample
        // - 10 hidden nodes
        // - one output node per class

        // note that the OpenCV neural network (MLP) implementation does not
        // support categorical variables explicitly.
        // So, instead of the output class label, we will use
        // a binary vector of {0,0 ... 1,0,0} components (one element by class)
        // for training and therefore, MLP will give us a vector of "probabilities"
        // at the prediction stage - the highest probability can be accepted
        // as the "winning" class label output by the network

        int layers_d[] = { ATTRIBUTES_PER_SAMPLE, 10,  NUMBER_OF_CLASSES};
        Mat layers = Mat(1,3,CV_32SC1);
        layers.at<int>(0,0) = layers_d[0];
        layers.at<int>(0,1) = layers_d[1];
        layers.at<int>(0,2) = layers_d[2];

        // create the network using a sigmoid function with alpha and beta
        // parameters 0.6 and 1 specified respectively (refer to manual)

        CvANN_MLP* nnetwork = new CvANN_MLP;
        nnetwork->create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

        // set the training parameters

        CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams(

                                           // terminate the training after either 1000
                                           // iterations or a very small change in the
                                           // network wieghts below the specified value

                                           cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10000, 0.000001),

                                           // use backpropogation for training

                                           CvANN_MLP_TrainParams::BACKPROP,

                                           // co-efficents for backpropogation training
                                           // (refer to manual)

                                           0.1,
                                           0.1);

        // train the neural network (using training data)

        printf( "\nUsing training database: %s\n", argv[1]);

        int iterations = nnetwork->train(training_data, training_classifications, Mat(), Mat(), params);

        printf( "Training iterations: %i\n\n", iterations);

        // perform classifier testing and report results

        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0,0,0,0,0,0,0};

        printf( "\nUsing testing database: %s\n\n", argv[2]);

        for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
        {

            // extract a row from the testing matrix

            test_sample = testing_data.row(tsample);

            // run neural network prediction

            nnetwork->predict(test_sample, classificationResult);

            // The NN gives out a vector of probabilities for each class
            // We take the class with the highest "probability"
            // for simplicity (but we really should also check separation
            // of the different "probabilities" in this vector - what if
            // two classes have very similar values ?)

            minMaxLoc(classificationResult, 0, 0, 0, &max_loc);

            printf("Testing Sample %i -> class result (digit %d)\n", tsample, max_loc.x);

            // if the corresponding location in the testing classifications
            // is not "1" (i.e. this is the correct class) then record this

            if (!(testing_classifications.at<float>(tsample, max_loc.x)))
            {
                // if they differ more than floating point error => wrong class

                wrong_class++;

                false_positives[(int) max_loc.x]++;

            }
            else
            {

                // otherwise correct

                correct_class++;
            }
        }

        printf( "\nResults on the testing database: %s\n"
                "\tCorrect classification: %d (%g%%)\n"
                "\tWrong classifications: %d (%g%%)\n",
                argv[2],
                correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
                wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

        for (int i = 0; i < NUMBER_OF_CLASSES; i++)
        {
            printf( "\tClass (digit %d) false postives 	%d (%g%%)\n", i,
                    false_positives[i],
                    (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
        }

        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    return -1;
}
/******************************************************************************/
