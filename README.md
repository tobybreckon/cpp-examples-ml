# C++ Machine Learning OpenCV 2.4.x Teaching Examples

OpenCV C/C++ Interface Machine Learning legacy 2.4.x interface examples used for teaching, instruction and reference over the years (2010-2013) -
_(that for a long time, the in absence of other fully worked examples for the OpenCV machine learning components became the de facto reference for the use of these OpenCV routines)_

**Uses older C++ interface to OpenCV ML library, with additional code** - _as these examples pre-date the new C++ OpenCV 3.x ML interface._

All tested with OpenCV 2.4.x and GCC (Linux) and known to work with MS Visual Studio 200x on Win32 / Win64.

N.B. due to changes in the OpenCV API _these do not generically work with OpenCV > 2.4.x_ by default.

---

Demo source code is provided _"as is"_ to aid your learning and understanding.

If I taught you between 2006 and 2010+ at [Cranfield University](http://www.cranfield.ac.uk) or [ESTIA](http://www.estia.fr) - these are the examples from class.

---

In each sub-directory:

+ .cpp file(s) - code for the example
+ .name file - an explanation of the data and its source
+ .data file - the original and complete set of data (CSV file format)
+ .train file - the data to be used for training (CSV file format)
+ .test file - the data to be used for testing (CSV file format)
+ .xml, .yml - example data files for testing some tools

All dataset examples are taken and reproduced from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/).

---

If referencing these examples in your own work please use:
```
@TechReport{breckon2010,
  author =       {Breckon, T.P. and Barnes, S.E.},
  title =        {Machine Learning - MSc Course Notes},
  institution =  {Cranfield University},
  year =         {2010},
  address =      {Bedfordshire, UK},
}
```

---

If you find any bugs please raise an issue (or better still submit a pull request, please) - toby.breckon@durham.ac.uk

_"may the source be with you"_ - anon.
