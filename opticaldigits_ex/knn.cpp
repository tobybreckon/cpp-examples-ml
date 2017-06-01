// Example : weighted knn digit classification
// usage: prog training_data_file testing_data_file

// For use with test / training datasets : opticaldigits_ex

// Copyright (c) 2013 Toby Breckon, toby.breckon@durham.ac.uk
// School of Engineering and Computing Sciences, Durham University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
using namespace cv;            // OpenCV API is in the C++ "cv" namespace

#include <cstdio>
using namespace std;

/******************************************************************************/
// global definitions

#define NUMBER_OF_TRAINING_SAMPLES 3823
#define ATTRIBUTES_PER_SAMPLE 64
#define NUMBER_OF_TESTING_SAMPLES 1797

#define NUMBER_OF_CLASSES 10 // digits 0->9

// "self load" data from CSV file in Mat() objects
// filename = file to load
// data = training or testing attributes (1 sample per row)
// responses =  training or testing classes (1 sample per row)
// n_samples = number of samples in the set

int read_data_from_csv(const char* filename, Mat &data, Mat &responses, int n_samples );

/******************************************************************************/

int main( int argc, char** argv )
{
    // define data set objects

        Mat training_data;
        Mat training_responses;

        Mat testing_data;
        Mat testing_responses;

    // load training and testing data sets (either from command line or *.{test|train} files

    if (((argc > 1) && (!(read_data_from_csv(argv[1],
                          training_data, training_responses, NUMBER_OF_TRAINING_SAMPLES))
                    && !(read_data_from_csv(argv[2],
                          testing_data, testing_responses, NUMBER_OF_TESTING_SAMPLES))))
        ||            (!(read_data_from_csv("optdigits.train",
                          training_data, training_responses, NUMBER_OF_TRAINING_SAMPLES))
                    && !(read_data_from_csv("optdigits.test",
                          testing_data, testing_responses, NUMBER_OF_TESTING_SAMPLES)))
        )
    {

        CvKNearest knn; // knn classifier object

        // train kNN classifier (using training data)

        knn.train(training_data, training_responses, Mat(), false, 32, false);

        // perform classifier testing and report results

        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        Mat false_positives = Mat::zeros(NUMBER_OF_CLASSES, 1, CV_32S);
        float result;

        // for each test example i the test set

        for (int tsample = 0; tsample < testing_data.rows; tsample++)
        {

            // extract a row from the testing matrix

            test_sample = testing_data.row(tsample);

            // run kNN classificaation (for k = 7)

            result = knn.find_nearest(test_sample, 7);

            printf("Test Example %i -> class result (digit %i)\n",
                    tsample, ((int) result));

            // if the prediction and the (true) testing classification are the same
            // (within the bounds of floating point error for cross-platfom safety)

            if (fabs(result - testing_responses.at<float>(tsample, 0))
                >= FLT_EPSILON)
            {
                // if they differ more than floating point error => wrong class

                wrong_class++;
                false_positives.at<int>((int) result, 0)++;

            } else {

                // otherwise correct

                correct_class++;
            }
        }

        printf( "\nResults on the testing database: %s\n"
                "\tCorrect classification: %d (%g%%)\n"
                "\tWrong classification: %d (%g%%)\n",
                (argc > 1) ? argv[2] : "optdigits.test",
                correct_class, (double) correct_class*100/testing_data.rows,
                wrong_class, (double) wrong_class*100/testing_data.rows);

        for (unsigned int c = 0; c < NUMBER_OF_CLASSES; c++)
        {
            printf( "\tClass (digit %i) false positives 	%d (%g%%)\n", c,
                    false_positives.at<int>(c,0),
                    (((double) false_positives.at<int>(c,0))*100)
                                                    /testing_data.rows);
        }

        // on MS Windows wait to exit prompt
        #ifdef WIN32
            getchar();
        #endif // WIN32

        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    printf("usage: %s filename.train filename.test\n", argv[0]);
    printf("Failed to load training and testing data from specified files\n");
    return -1;
}
/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat &data, Mat &responses, int n_samples )
{
    data = Mat(n_samples, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    responses = Mat(n_samples, 1, CV_32FC1);

    float tmp;

    // if we can't read the input file then return 0
    FILE* f = fopen( filename, "r" );
    if( !f )
    {
        printf("ERROR: cannot read file %s\n",  filename);
        return 1; // all not OK
    }

    // for each sample in the file

    for(int line = 0; line < n_samples; line++)
    {

        // for each attribute on the line in the file

        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
        {
            if (attribute < ATTRIBUTES_PER_SAMPLE)
            {

                // first 64 elements (0-63) in each line are the attributes

                fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute) = tmp;
                // printf("%f,", data.at<float>(line, attribute));

            }
            else if (attribute == ATTRIBUTES_PER_SAMPLE)
            {

                // attribute 65 is the class label {0 ... 9}

                fscanf(f, "%f,", &tmp);
                responses.at<float>(line, 0) = tmp;
                // printf("%f\n", classes.at<float>(line, 0));

            }
        }
    }

    fclose(f);

    return 0; // all OK
}

/******************************************************************************/
