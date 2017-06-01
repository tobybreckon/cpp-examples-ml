// Example : decision tree learning
// usage: prog training_data_file testing_data_file

// For use with test / training datasets : dt_example1

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2010 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/******************************************************************************/
// global definitions (for speed and ease of use)

#define NUMBER_OF_TRAINING_SAMPLES 1383
#define ATTRIBUTES_PER_SAMPLE 6  // not the last as this is the class
#define NUMBER_OF_TESTING_SAMPLES 345

#define NUMBER_OF_CLASSES 4 // classes 0->3
static char* CLASSES[NUMBER_OF_CLASSES] =
{(char *) "unacc", (char *) "acc", (char *) "good", (char *) "vgood"};

/******************************************************************************/

// a basic hash function from: http://www.cse.yorku.ca/~oz/hash.html

int hash(char *str)
{
    int hash = 5381;
    int c;

    while ((c = (*str++)))
    {
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

/******************************************************************************/

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat data, Mat classes,
                       int n_samples )
{
    char tmp_buf[10];
    int i = 0;
    char c;

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
            // last attribute is the class

            if (attribute == 6)
            {
                c = '\0';
                for(i=0; c != '\n'; i++)
                {
                    c = fgetc(f);
                    tmp_buf[i] = c;
                }
                tmp_buf[i - 1] = '\0';
                //printf("%s\n", tmp_buf);

                // find the class number and record this

                for (int i = 0; i < NUMBER_OF_CLASSES; i++)
                {
                    if (strcmp(CLASSES[i], tmp_buf) == 0)
                    {
                        classes.at<float>(line, 0) = (float) i;
                    }
                }
            }
            else
            {

                // for all other attributes just read in the string value
                // and use a hash function to convert to to a float
                // (N.B. openCV uses a floating point decision tree implementation!)

                c = '\0';
                for(i=0; c != ','; i++)
                {
                    c = fgetc(f);
                    tmp_buf[i] = c;
                }
                tmp_buf[i - 1] = '\0';
                data.at<float>(line, attribute) = (float) hash(tmp_buf);

                //printf("%s,", tmp_buf);
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

    // define all the attributes as categorical (i.e. categories)
    // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
    // that can be assigned on a per attribute basis

    // this is a classification problem (i.e. predict a discrete number of class
    // outputs) so also the last (+1) output var_type element to CV_VAR_CATEGORICAL

    Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
    var_type = Scalar(CV_VAR_CATEGORICAL); // all inputs are categorical

    CvDTreeNode* resultNode; // node returned from a prediction

    // load training and testing data sets

    if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
            read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
    {
        // define the parameters for training the decision tree

        float priors[] = { 1, 1, 1, 1 }; // weights of each classification for classes
        //float priors[] = { 70, 22, 4, 4 }; // weights of each classification for classes

        CvDTreeParams params = CvDTreeParams(25, // max depth
                                             10, // min sample count
                                             0, // regression accuracy: N/A here
                                             false, // compute surrogate split, no missing data
                                             25, // max number of categories (use sub-optimal algorithm for larger numbers)
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

        dtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
                     Mat(), Mat(), var_type, Mat(), params);

        // perform classifier testing and report results

        Mat test_sample;
        int correct_class = 0;
        int wrong_class = 0;
        int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0};

        printf( "\nUsing testing database: %s\n\n", argv[2]);

        for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
        {

            // extract a row from the testing matrix

            test_sample = testing_data.row(tsample);

            // run decision tree prediction

            resultNode = dtree->predict(test_sample, Mat(), false);

            printf("Testing Sample %i -> class result %s\n", tsample, CLASSES[(int) (resultNode->value)]);

            // if the prediction and the (true) testing classification are the same
            // (N.B. openCV uses a floating point decision tree implementation!)

            if (fabs(resultNode->value - testing_classifications.at<float>(tsample, 0))
                    >= FLT_EPSILON)
            {
                // if they differ more than floating point error => wrong class

                wrong_class++;

                false_positives[(int) resultNode->value]++;

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
            printf( "\tClass %s false postives 	%d (%g%%)\n", CLASSES[i],
                    false_positives[i],
                    (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
        }

        // all matrix memory free by destructors

        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    printf("usage: %s training_data_file testing_data_file\n", argv[0]);
    return -1;
}
/******************************************************************************/
