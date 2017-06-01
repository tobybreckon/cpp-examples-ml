// Example : select a subset of lines in a specified input file
// between a specified min and max line numbers INCLUSIVE
// (also removing any empty lines in the file - i.e. no chars apart from "\n")

// usage: prog min max input_file output_file
// where min and max are integer line numbers from the input file (range 1 to N)

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2009 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

/******************************************************************************/

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define LINELENGTHMAX 5000 // all file lines less than 5000 chars

/******************************************************************************/

int main( int argc, char** argv )
{

	vector<char *> inputlines; 				// vector of input lines
	vector<char *>::iterator outline;		// iterator for above
	
	char * line = NULL;						// tmp pointer for line memory

	// check we have the correct number of arguments
	
	if (argc < 5){
		printf("usage: %s min max input_file output_file\n", argv[0]);
		exit(0); 
	}
	
	// get min / max line numbers
	
	int minL = min(atoi(argv[1]), atoi(argv[2]));
	int maxL = max(atoi(argv[1]), atoi(argv[2]));

	int lineN = 0;
			
	// open input file
	
	FILE* fi = fopen( argv[3], "r" );
	if( !fi ){
		printf("ERROR: cannot read input file %s\n",  argv[1]);
		return -1; // all not OK
	}

	// open output file
	
	FILE* fw = fopen( argv[4], "w" );
	if( !fw ){
		printf("ERROR: cannot read output file %s\n",  argv[2]);
		return -1; // all not OK
	}

	// read in all the lines of the file (allocating fresh memory for each)
	
	while (!feof(fi))
	{
		line = (char *) malloc(LINELENGTHMAX * sizeof(char));
		fscanf(fi, "%[^\n]\n", line);
		inputlines.push_back(line);
	}

	// output seleted lines to output file
	
	for(outline = inputlines.begin(); outline < inputlines.end(); outline++)
	{
		if ((lineN >= minL) && (lineN <= maxL))
		{
			fprintf(fw, "%s\n", *outline); 
		}
		lineN++;
		
		free((void *) *outline); // free memory also
	}

	// close files
	
	fclose(fi);
	fclose(fw);
	
	return 1; // all OK
}
/******************************************************************************/
