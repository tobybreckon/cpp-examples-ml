// Example : check type sizes on your platform
// usage: prog

// taken from: http://home.att.net/~jackklein/c/inttypes.html
// (c) 2008 By Jack Klein.

/*****************************************************************************/

#include <stdio.h>
#include <limits.h>

volatile int char_min = CHAR_MIN;

int main(void)
{
    printf("\n\n       Character Types\n");
    printf("Number of bits in a character: %d\n",
        CHAR_BIT);
    printf("Size of character types is %d byte\n",
        (int)sizeof(char));
    printf("Signed char min: %d max: %d\n",
        SCHAR_MIN, SCHAR_MAX);
    printf("Unsigned char min: 0 max: %u\n",
        (unsigned int)UCHAR_MAX);

    printf("Default char is ");
    if (char_min < 0)
        printf("signed\n");
    else if (char_min == 0)
        printf("unsigned\n");
    else
        printf("non-standard\n");
		printf("*** This is %d bit character representation\n",
       		(int)sizeof(char) * 8);
	
    printf("\n\n        Short Int Types\n");
    printf("Size of short int types is %d bytes\n",
        (int)sizeof(short));
    printf("Signed short min: %d max: %d\n",
        SHRT_MIN, SHRT_MAX);
    printf("Unsigned short min: 0 max: %u\n",
        (unsigned int)USHRT_MAX);

    printf("\n           Int Types\n");
    printf("Size of int types is %d bytes\n",
        (int)sizeof(int));
    printf("Signed int min: %d max: %d\n",
        INT_MIN, INT_MAX);
    printf("Unsigned int min: 0 max: %u\n",
        (unsigned int)UINT_MAX);
	printf("*** This is %d bit representation\n",
       (int)sizeof(int) * 8);

    printf("\n        Long Int Types\n");
    printf("Size of long int types is %d bytes\n",
        (int)sizeof(long));
    printf("Signed long min: %ld max: %ld\n",
        LONG_MIN, LONG_MAX);
    printf("Unsigned long min: 0 max: %lu\n",
        ULONG_MAX);

	// mild addition by Toby Breckon, toby.breckon@cranfield.ac.uk
	
	printf("\n\n        Float Types\n");
    printf("Size of float types is %d bytes\n",
        (int)sizeof(float));
	printf("*** This is %d bit representation\n",
       (int)sizeof(float) * 8);
 	printf("\n        Double Types\n");
    printf("Size of float types is %d bytes\n\n",
        (int)sizeof(double));
	
    return 0;
}

/*****************************************************************************/