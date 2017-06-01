// Example : genetic algorithm example interface to camera / video
// usage: prog {<video_name>}

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2012 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include <cv.h>   		// open cv general include file
#include <highgui.h>	// open cv GUI include file
#include <iostream>		// standard C++ I/O

using namespace cv; // OpenCV API is in the C++ "cv" namespace

/******************************************************************************/

#define CAMERA_TO_USE 0

/******************************************************************************/

#include "ga.h"     // GA class header file

/******************************************************************************/

// callback funtion for mouse to select a region of the image and store that selection
// in global variables origin and selection (acknowledgement: opencv camsiftdemo.cpp)

static bool selectObject = false;
static Point origin;
static Rect selection;
static bool selectionComplete = false;

void onMouseSelect( int event, int x, int y, int, void* image)
{
    if( selectObject && !selectionComplete)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, ((Mat *) image)->cols, ((Mat *) image)->rows);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        if (!selectionComplete)
        {
            origin = Point(x,y);
            selection = Rect(x,y,0,0);
            selectObject = true;
        }
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            selectionComplete = true;
        break;
    }
}

/******************************************************************************/

int main( int argc, char** argv )
{

  Mat img, roi, selected, display;	// image object
  VideoCapture cap;                 // capture object

  const string windowName = "GA Input / Output"; // window name
  const string windowName2 = "Selected Region / Object"; // window name

  bool keepProcessing = true;	// loop control flag
  char  key;			        // user input
  int  EVENT_LOOP_DELAY = 200;	// delay for GUI window (slow down for visualization

  int mutation_i = 3;          // x 0.01 == mutation rate
  int crossover_i = 40;         // x 0.01 == crossover rate
  int population = 100;         // population size

  // if command line arguments are provided try to read image/video_name
  // otherwise default to capture from attached H/W camera

    if(
	  ( argc == 2 && (cap.open(argv[1]) == true )) ||
	  ( argc != 2 && (cap.open(CAMERA_TO_USE) == true))
	  )
    {
      // create window object (use flag=0 to allow resize, 1 to auto fix size)

      namedWindow(windowName, 0);
      namedWindow(windowName2, 0);
      setMouseCallback( windowName, onMouseSelect, &img);
      createTrackbar( "M x 0.01", windowName, &mutation_i, 100);
      createTrackbar( "C x 0.01", windowName, &crossover_i, 100);
      createTrackbar( "P", windowName, &population, 1000);

      // GA object

      correlationGA *GA = NULL;
      Gene_t bestResult;

	  // start main loop

	  while (keepProcessing) {

		  // if capture object in use (i.e. video/camera)
		  // get image from capture object until we select
		  // an object/region

		  if (cap.isOpened() && selected.empty()) {

			  cap >> img;
			  if(img.empty()){
				if (argc == 2){
					std::cerr << "End of video file reached" << std::endl;
				} else {
					std::cerr << "ERROR: cannot get next frame from camera"
						      << std::endl;
				}
				exit(0);
			  }

		  }

          // draw interactive display effect

          if( selectObject && selection.width > 0 && selection.height > 0 )
          {
            roi = img(selection);
            bitwise_not(roi, roi);

          } else if ( selectionComplete && selection.width > 0 && selection.height > 0 ){

            selected = roi.clone();

          }

		  // if we have no GA but now have a selection then create GA

		  if ((GA == NULL) && (!(selected.empty())))
          {
              GA = new correlationGA(population, (0.01 * crossover_i), (0.01 * mutation_i), img, selected);
          }

		  // display image in window

          display = img.clone();

		  if (!(selected.empty()))
		  {
                imshow(windowName2, selected);

                // if we have a GA then draw the current best output

                if (GA != NULL)
                {
                    bestResult = GA->returnMaximal();
                    rectangle(display, Rect(bestResult.x, bestResult.y, selection.width, selection.height),
                            Scalar(0,0,255), 2);
                    GA->createNextGeneration(); // get next generation, N.B. no stopping criteria
                }
		  }
          imshow(windowName, display);
		  key = waitKey(EVENT_LOOP_DELAY);

		  if (key == 'x'){

	   		// if user presses "x" then exit

			  	std::cout << "Keyboard exit requested : exiting now - bye!"
				  		  << std::endl;
	   			keepProcessing = false;
		  } else if (key == 'r'){
		        std::cout << "\n\n*** reset\n"
				  		  << std::endl;
                delete GA;
                GA = NULL;
                selected = Mat();
                selectionComplete = false;

		  }

	  }

	  // the camera will be deinitialized automatically in VideoCapture destructor

      // all OK : main returns 0

      return 0;
    }

    // not OK : main returns -1

    return -1;
}
/******************************************************************************/
