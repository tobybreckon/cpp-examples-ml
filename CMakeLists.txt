# this file is setup to build everything for the ML examples

cmake_minimum_required (VERSION 2.6)
cmake_policy(SET CMP0037 OLD)
set( CMAKE_CXX_FLAGS "-O3 -Wall ${CMAKE_CXX_FLAGS}" )
# linux specific stuff

IF ( UNIX )
   set( CMAKE_PREFIX_PATH "/opt/opencv-2.4" )
   set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
   MESSAGE( "LINUX CONFIG" )
ENDIF ( UNIX )

# windows (inc. 64-bit specific stuff)

IF ( WIN32 )
   set( CMAKE_PREFIX_PATH "C:/OpenCV2.4/build" )
   set( OpenCV_DIR "C:/OpenCV2.4/build" )
   MESSAGE( "WINDOWS CONFIG" )
ENDIF ( WIN32 )

find_package( OpenCV 2.4.13 REQUIRED )
# MESSAGE ( "OPENCV CONFIG" )
# MESSAGE ( ${OpenCV_LIBS} )

project(decisiontree)
add_executable(./handwritten_ex/decisiontree ./handwritten_ex/decisiontree.cpp)
target_link_libraries( ./handwritten_ex/decisiontree ${OpenCV_LIBS} )

project(neuralnetwork)
add_executable(./handwritten_ex/neuralnetwork ./handwritten_ex/neuralnetwork.cpp)
target_link_libraries( ./handwritten_ex/neuralnetwork ${OpenCV_LIBS} )

project(svm)
add_executable(./handwritten_ex/svm ./handwritten_ex/svm.cpp)
target_link_libraries( ./handwritten_ex/svm ${OpenCV_LIBS} )

project(ga_interface)
add_executable(./ga_ex/ga_interface ./ga_ex/ga_interface.cpp)
target_link_libraries( ./ga_ex/ga_interface ${OpenCV_LIBS} )

project(decisiontree)
add_executable(./dt_example1/decisiontree ./dt_example1/decisiontree.cpp)
target_link_libraries( ./dt_example1/decisiontree ${OpenCV_LIBS} )

project(decisiontree2)
add_executable(./dt_example2/decisiontree ./dt_example2/decisiontree.cpp)
target_link_libraries( ./dt_example2/decisiontree ${OpenCV_LIBS} )

project(boosttree)
add_executable(./opticaldigits_ex/boosttree ./opticaldigits_ex/boosttree.cpp)
set_target_properties(./opticaldigits_ex/boosttree PROPERTIES COMPILE_FLAGS "-fpermissive")
target_link_libraries( ./opticaldigits_ex/boosttree ${OpenCV_LIBS} )

project(decisiontree3)
add_executable(./opticaldigits_ex/decisiontree ./opticaldigits_ex/decisiontree.cpp)
target_link_libraries( ./opticaldigits_ex/decisiontree ${OpenCV_LIBS} )

project(extremerandomforest3)
add_executable(./opticaldigits_ex/extremerandomforest ./opticaldigits_ex/extremerandomforest.cpp)
target_link_libraries( ./opticaldigits_ex/extremerandomforest ${OpenCV_LIBS} )

project(randomforest)
add_executable(./opticaldigits_ex/randomforest ./opticaldigits_ex/randomforest.cpp)
target_link_libraries( ./opticaldigits_ex/randomforest ${OpenCV_LIBS} )

project(svm2)
add_executable(./opticaldigits_ex/svm ./opticaldigits_ex/svm.cpp)
target_link_libraries( ./opticaldigits_ex/svm ${OpenCV_LIBS} )

project(knn)
add_executable(./opticaldigits_ex/knn ./opticaldigits_ex/knn.cpp)
target_link_libraries( ./opticaldigits_ex/knn ${OpenCV_LIBS} )

project(knn_weighted)
add_executable(./opticaldigits_ex/knn_weighted ./opticaldigits_ex/knn_weighted.cpp)
target_link_libraries( ./opticaldigits_ex/knn_weighted ${OpenCV_LIBS} )

project(normalbayes)
add_executable(./opticaldigits_ex/normalbayes ./opticaldigits_ex/normalbayes.cpp)
target_link_libraries( ./opticaldigits_ex/normalbayes ${OpenCV_LIBS} )

project(neuralnetwork)
add_executable(./opticaldigits_ex/neuralnetwork ./opticaldigits_ex/neuralnetwork.cpp)
target_link_libraries( ./opticaldigits_ex/neuralnetwork ${OpenCV_LIBS} )

project(normalbayes)
add_executable(./other_ex/normalbayes ./other_ex/normalbayes.cpp)
target_link_libraries( ./other_ex/normalbayes ${OpenCV_LIBS} )

project(decisiontree)
add_executable(./speech_ex/decisiontree ./speech_ex/decisiontree.cpp)
target_link_libraries( ./speech_ex/decisiontree ${OpenCV_LIBS} )

project(svm)
add_executable(./speech_ex/svm ./speech_ex/svm.cpp)
target_link_libraries( ./speech_ex/svm ${OpenCV_LIBS} )

project(dt_varimportance)
add_executable(./tools/dt_varimportance ./tools/dt_varimportance.cc)
target_link_libraries( ./tools/dt_varimportance ${OpenCV_LIBS} )

project(randomize)
add_executable(./tools/randomize tools/randomize.cc)

project(selectlines)
add_executable(./tools/selectlines tools/selectlines.cc)

project(typechecker)
add_executable(./tools/typechecker tools/typechecker.cc)
