// Example : decision tree learning
// usage: prog training_data_file testing_data_file

// For use with test / training datasets : dt_example2

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2010 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 449
#define ATTRIBUTES_PER_SAMPLE 30  // not the first two as patient ID and class
#define NUMBER_OF_TESTING_SAMPLES 120

static char CLASSES[2] = {'B', 'M'};  // class B = 0, class M = 1

/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat data, Mat classes,
                       int n_samples )
{
    char tmpc;
    float tmpf;

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

        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 2); attribute++)
        {
            if (attribute == 0)
            {
                fscanf(f, "%f,", &tmpf);

                // ignore attribute 0 (as it's the patient ID)

                continue;
            }
            else if (attribute == 1)
            {

                // attribute 2 (in the database) is the classification
                // record 1 = M = malignant
                // record 0 = B = benign

                fscanf(f, "%c,", &tmpc);

                switch(tmpc)
                {
                case 'M':
                    classes.at<float>(line, 0) = 1.0;
                    break;
                case 'B':
                    classes.at<float>(line, 0) = 0.0;
                    break;
                default:
                    printf("ERROR: unexpected class in file %s\n",  filename);
                    return 0; // all not OK
                }

                // printf("%c,", tmpc);
            }
            else
            {
                fscanf(f, "%f,", &tmpf);
                data.at<float>(line, (attribute - 2)) = (float) tmpf;
                // printf("%f,", data.at<float>(line, (attribute - 2)));
            }
        }
        fscanf(f, "\n");
        // printf("\n");
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

    Mat training_data =
        Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

    //define testing data storage matrices

    Mat testing_data =
        Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat testing_classifications =
        Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

    // define all the attributes as numerical
    // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
    // that can be assigned on a per attribute basis

    Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
    var_type = Scalar(CV_VAR_NUMERICAL); // all inputs are numerical

    // this is a classification problem (i.e. predict a discrete number of class
    // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

    var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

    CvDTreeNode* resultNode; // node returned from a prediction

    // load training and testing data sets

    if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {
        // define the parameters for training the decision tree

        float priors[] = { 1, 1 }; // weights of each classification for classes
        // 0 = B = benign, 1 = M = malignant

        CvDTreeParams params = CvDTreeParams(8, // max depth
                                             5, // min sample count
                                             0, // regression accuracy: N/A here
                                             false, // compute surrogate split, no missing data
                                             15, // max number of categories (use sub-optimal algorithm for larger numbers)
                                             10, // the number of cross-validation folds
                                             true, // use 1SE rule => smaller tree
                                             false, // throw away the pruned tree branches
                                             priors // the array of priors, the bigger weight, the more attention
                                             // to the maligant cases
                                             // (i.e. a case will be judjed to be maligant with bigger chance)
                                            );


        // train decision tree classifier (using training data)

        printf( "\nUsing training database: %s\n\n", argv[1]);
        CvDTree* dtree = new CvDTree;

        dtree->train(training_data, CV_ROW_SAMPLE,
                     training_classifications,
                     Mat(), Mat(), var_type, Mat(), params);

        // perform classifier testing and report results

        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        int m_class_fp = 0;
        int b_class_fp = 0;

        printf( "\nUsing testing database: %s\n\n", argv[2]);

        for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
        {

            // extract a row from the testing matrix

            test_sample = testing_data.row(tsample);

            // run decision tree prediction

            resultNode = dtree->predict(test_sample, Mat(), false);

            printf("Testing Sample %i -> class result %c\n", tsample, CLASSES[(int) (resultNode->value)]);

            // if the prediction and the (true) testing classification are the same
            // (N.B. openCV uses a floating point decision tree implementation!)

            if (fabs(resultNode->value - testing_classifications.at<float>(tsample, 0))
                    >= FLT_EPSILON)
            {
                // if they differ more than floating point error => wrong class

                wrong_class++;

                // if the result class is different from 1.0 (M class label) by
                // more than floating point error => B class false +ve

                if (fabs(resultNode->value - 1.0) >= FLT_EPSILON)
                {
                    b_class_fp++;
                }
                else
                {

                    // otherwise it's an

                    m_class_fp++;
                }

            }
            else
            {

                // otherwise correct

                correct_class++;
            }
        }

        printf( "\nResults on the testing database: %s\n"
                "\tCorrect classification: %d (%g%%)\n"
                "\tWrong classifications: %d (%g%%)\n"
                "\tM false +ve classifications: %d (%g%%)\n"
                "\tB false +ve classifications: %d (%g%%)\n",
                argv[2],
                correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
                wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES,
                m_class_fp, (double) m_class_fp*100/NUMBER_OF_TESTING_SAMPLES,
                b_class_fp, (double) b_class_fp*100/NUMBER_OF_TESTING_SAMPLES );

        // all matrix memory free by destructors


        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    return -1;
}
/******************************************************************************/
