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

#define NUMBER_OF_CLASSES 10 // digits 0->9

/******************************************************************************/

int main( int argc, char** argv )
{
    // define data loading objects

    CvMLData training_loader;
    CvMLData testing_loader;

    // load training and testing data sets (either from command line or *.{test|train} files

    if (((argc > 1) && (!(training_loader.read_csv(argv[1]))
                    && !(testing_loader.read_csv(argv[2]))))
        ||            (!(training_loader.read_csv("optdigits.train"))
                    && !(testing_loader.read_csv("optdigits.test")))
        )
    {

        CvKNearest knn; // knn classifier object

        // retrieve data from data loaders

        Mat training_data =
        (Mat(training_loader.get_values())).colRange(0,64); // 0->63 = attributes

        training_loader.set_response_idx(64); // 65th value is the classification
        Mat training_responses = training_loader.get_responses();

        Mat testing_data =
        (Mat((testing_loader.get_values())).colRange(0,64)); // 0->63 = attributes

        testing_loader.set_response_idx(64); // 65th value is the classification
        Mat testing_responses = testing_loader.get_responses();

        // train kNN classifier (using training data)

        knn.train(training_data, training_responses, Mat(), false, 32, false);

        // perform classifier testing and report results

        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        Mat false_positives = Mat::zeros(NUMBER_OF_CLASSES, 1, CV_32S);
        Mat neighbourResponses, dists, results, weighted_results;
        double minVal, maxVal; // dummy variables for using minMaxLoc()
        Point result_class_location;
        int result_class; // resulting class with highest weighted knn score

        // for each test example i the test set

        for (int tsample = 0; tsample < testing_data.rows; tsample++)
        {

            // extract a row from the testing matrix

            test_sample = testing_data.row(tsample);

            // zero weighted results on each test iteration

            weighted_results = Mat::zeros(NUMBER_OF_CLASSES, 1, CV_32F);

            // run kNN classification (for k = 7)

            knn.find_nearest(test_sample, 7, results, neighbourResponses, dists);

            // perform weighted sum for all the classes that occur in the responses
            // from the k nearest neighbours based on distance from query sample

            for(int i=0; i < neighbourResponses.cols; i++)
            {
                weighted_results.at<float>((int) neighbourResponses.at<float>(0,i), 0) += 1.0 / pow((dists.at<float>(0,i)),2.0);
            }

            // find the class with the maximum weighted sum (as the maximal y co-ordinate
            // of the resulting weighted_results matrix

            minMaxLoc(weighted_results, &minVal, &maxVal, 0, &result_class_location);
            result_class = result_class_location.y; // resulting class is in col location

            printf("Test Example %i -> class result (digit %i)\n",
                    tsample, ((int) result_class));

            // if the prediction and the (true) testing classification are the same
            // (within the bounds of floating point error for cross-platfom safety)

            if (fabs(((float) result_class) - testing_responses.at<float>(tsample, 0))
                >= FLT_EPSILON)
            {
                // if they differ more than floating point error => wrong class

                wrong_class++;
                false_positives.at<int>(result_class, 0)++;

            } else {

                // otherwise correct

                correct_class++;
            }
        }

        printf( "\nResults on the testing database: %s\n"
                "\tCorrect classification: %d (%g%%)\n"
                "\tWrong classifications: %d (%g%%)\n",
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
