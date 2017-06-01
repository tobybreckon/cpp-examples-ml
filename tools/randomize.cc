// Example : randomize the lines in a specified input file
// (also removing any empty lines in the file - i.e. no chars apart from "\n")

// usage: prog input_file output_file

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
	
	// open input file
	
	FILE* fi = fopen( argv[1], "r" );
	if( !fi ){
		printf("ERROR: cannot read input file %s\n",  argv[1]);
		return -1; // all not OK
	}

	// open output file
	
	FILE* fw = fopen( argv[2], "w" );
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

	// shuffle input file lines

	// "This algorithm is described in section 3.4.2 of Knuth (D. E. Knuth, 
	// The Art of Computer Programming. Volume 2: Seminumerical Algorithms, 
	// second edition. Addison-Wesley, 1981). Knuth credits Moses and 
	// Oakford (1963) and Durstenfeld (1964)." 
	// - SGI STL manual, http://www.sgi.com/tech/stl/random_shuffle.html

	random_shuffle(inputlines.begin(), inputlines.end());

	// output all of the lines to output file
	
	for(outline = inputlines.begin(); outline < inputlines.end(); outline++)
	{
		fprintf(fw, "%s\n", *outline); 
		free((void *) *outline); // free memory also
	}

	// close files
	
	fclose(fi);
	fclose(fw);
	
	return 1; // all OK
}
/******************************************************************************/
