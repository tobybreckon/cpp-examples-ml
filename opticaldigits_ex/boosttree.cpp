// Example : boosted tree learning
// usage: prog training_data_file testing_data_file

// For use with test / training datasets : opticaldigits_ex

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2011 School of Engineering, Cranfield University
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
                classes.at<float>(line, 0) = tmp;
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
    Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

    //define testing data storage matrices

    Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

    // load training and testing data sets

    if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // As currently boosted tree classifier in OpenCV can only be trained
        // for 2-class problems, we transform the training database by
        // "unrolling" each training sample as many times as the number of
        // classes (10) that we have.
        //
        //  In "unrolling" we add an additional attribute to each training
        //	sample that contains the classification - here 10 new samples
        //  are added for every original sample, one for each possible class
        //	but only one with the correct class as an additional attribute
        //  value has a new binary class of 1, all the rest of the new samples
        //  have a new binary class of 0.
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Mat new_data = Mat(NUMBER_OF_TRAINING_SAMPLES*NUMBER_OF_CLASSES, ATTRIBUTES_PER_SAMPLE + 1, CV_32F );
        Mat new_responses = Mat(NUMBER_OF_TRAINING_SAMPLES*NUMBER_OF_CLASSES, 1, CV_32S );

        // 1. unroll the training samples

        printf( "\nUnrolling the database...");
        fflush(NULL);
        for(int i = 0; i < NUMBER_OF_TRAINING_SAMPLES; i++ )
        {
            for(int j = 0; j < NUMBER_OF_CLASSES; j++ )
            {
                for(int k = 0; k < ATTRIBUTES_PER_SAMPLE; k++ )
                {

                    // copy over the attribute data

                    new_data.at<float>((i * NUMBER_OF_CLASSES) + j, k) = training_data.at<float>(i, k);

                }

                // set the new attribute to the original class

                new_data.at<float>((i * NUMBER_OF_CLASSES) + j, ATTRIBUTES_PER_SAMPLE) = (float) j;

                // set the new binary class

                if ( ( (int) training_classifications.at<float>( i, 0)) == j)
                {
                    new_responses.at<int>((i * NUMBER_OF_CLASSES) + j, 0) = 1;
                }
                else
                {
                    new_responses.at<int>((i * NUMBER_OF_CLASSES) + j, 0) = 0;
                }
            }
        }
        printf( "Done\n");

        // 2. Unroll the type mask

        // define all the attributes as numerical
        // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
        // that can be assigned on a per attribute basis

        Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 2, 1, CV_8U );
        var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

        // this is a classification problem (i.e. predict a discrete number of class
        // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
        // *** the last (new) class indicator attribute, as well
        // *** as the new (binary) response (class) are categorical

        var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;
        var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE + 1, 0) = CV_VAR_CATEGORICAL;

        // define the parameters for training the boosted trees

        // weights of each classification for classes
        // N.B. in the "unrolled" data we have an imbalance in the training examples

        float priors[] = {( NUMBER_OF_CLASSES - 1),1};
        //float priors[] = {1,1};

        // set the boost parameters

        CvBoostParams params = CvBoostParams(CvBoost::REAL,  // boosting type
                                             100,			 // number of weak classifiers
                                             0.95,   		 // trim rate

                                             // trim rate is a threshold (0->1)
                                             // used to eliminate samples with
                                             // boosting weight < 1.0 - (trim rate)
                                             // from the next round of boosting
                                             // Used for computational saving only.

                                             25, 	  // max depth of trees
                                             false,  // compute surrogate split, no missing data
                                             priors );

        // as CvBoostParams inherits from CvDTreeParams we can also set generic
        // parameters of decision trees too (otherwise they use the defaults)

        params.max_categories = 15; 	// max number of categories (use sub-optimal algorithm for larger numbers)
        params.min_sample_count = 5; 	// min sample count
        params.cv_folds = 1;					// cross validation folds
        params.use_1se_rule = false; 			// use 1SE rule => smaller tree
        params.truncate_pruned_tree = false; 	// throw away the pruned tree branches
        params.regression_accuracy = 0.0; 		// regression accuracy: N/A here


        // train boosted tree classifier (using training data)

        printf( "\nUsing training database: %s\n\n", argv[1]);
        printf( "Training .... (this may take several minutes) .... ");
        fflush(NULL);

        CvBoost* boostTree = new CvBoost;

        boostTree->train( new_data, CV_ROW_SAMPLE, new_responses, Mat(), Mat(), var_type,
                          Mat(), params, false);
        printf( "Done.");

        // perform classifier testing and report results

        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0,0,0,0,0,0,0};
        Mat weak_responses = Mat( 1, boostTree->get_weak_predictors()->total, CV_32F );
        Mat new_sample = Mat( 1,  ATTRIBUTES_PER_SAMPLE + 1, CV_32F );
        int best_class = 0; // best class returned by weak classifier
        double max_sum;	 // highest score for a given class

        printf( "\nUsing testing database: %s\n\n", argv[2]);

        for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
        {

            // extract a row from the testing matrix

            test_sample = testing_data.row(tsample);

            // convert it to the new "un-rolled" format of input

            for(int k = 0; k < ATTRIBUTES_PER_SAMPLE; k++ )
            {
                new_sample.at<float>( 0, k) = test_sample.at<float>(0, k);
            }

            // run boosted tree prediction (for N classes and take the
            // maximal response of all the weak classifiers)

            max_sum = INT_MIN; // maximum starts off as Min. Int.

            for(int c = 0; c < NUMBER_OF_CLASSES; c++ )
            {
                // set the additional attribute to original class

                new_sample.at<float>(0, ATTRIBUTES_PER_SAMPLE) = (float) c;

                // run prediction (getting also the responses of the weak classifiers)
                // - N.B. here we have to use CvMat() casts and take the address of temporary
                // in order to use the available call that gives us the weak responses
                // For this reason we also have to pass a NULL pointer for the missing data

                boostTree->predict(&CvMat((new_sample)), NULL, &CvMat(weak_responses));

                // obtain the sum of the responses from the weak classifiers

                Scalar responseSum = sum( weak_responses );

                // record the "best class" - i.e. one with maximal response
                // from weak classifiers

                if( responseSum.val[0] > max_sum)
                {
                    max_sum = (double) responseSum.val[0];
                    best_class = c;
                }
            }


            printf("Testing Sample %i -> class result (digit %d)\n", tsample, best_class);

            // if the prediction and the (true) testing classification are the same
            // (N.B. openCV uses a floating point decision tree implementation!)

            if (fabs(((float) (best_class)) - testing_classifications.at<float>( tsample, 0))
                    >= FLT_EPSILON)
            {
                // if they differ more than floating point error => wrong class

                wrong_class++;

                false_positives[best_class]++;

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

        // all matrix memory free by destructors

        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    return -1;
}
/******************************************************************************/
