// Example : decision tree variable importance
// usage: prog tree.{yml|.xml}

// For use with any test / training datasets

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2011 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

// Copyright (c) 2011 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include <cv.h>       // opencv general include file
#include <ml.h>		  // opencv machine learning include file

using namespace cv; // OpenCV API is in the C++ "cv" namespace

#include <stdio.h>

/*****************************************************************************/

// prints out the relative importance of the variables (i.e. attributes) used
// for decision tree classification

// Based on the mushroom.cpp example from OpenCV 1.0

int print_variable_importance(CvDTree* dtree)
{
    const Mat var_importance = dtree->get_var_importance();

    if( var_importance.empty() )
    {
        printf( "Error: Variable importance can not be retrieved\n" );
        return -1;
    }

    for(int i = 0; i < var_importance.cols*var_importance.rows; i++ )
    {
        double val = var_importance.at<double>(0,i);
        printf( "var #%d", i );
        printf( ": %g%%\n", val*100. );
    }

	return 1;
}

/*****************************************************************************/

int main( int argc, char** argv )
{

	// check we have enough command line arguments

	if (argc == 2)
	{
		// define a decision tree object

		CvDTree* dtree = new CvDTree;

		// load tree structure from XML file

		dtree->load(argv[1]);

		// extract (and display) variable importance information

		if (print_variable_importance(dtree)){
			return 0; // all OK
		} else {
			return -1; // all not OK
		}

    } else {

    // not OK : main returns -1

	printf("usage: %s decision_tree_filename.xml\n", argv[0]);
    return -1;

    }
}
/******************************************************************************/
